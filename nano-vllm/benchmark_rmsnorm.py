"""Compare existing torch.compile RMSNorm with explicit Triton kernels.

The parent never initializes CUDA. Model cases run in fresh processes, with
reversed backend order on alternating trials. No package synchronization needed.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def environment():
    import hashlib
    import torch
    import triton
    source = hashlib.sha256()
    for path in sorted((Path(__file__).resolve().parent / "nanovllm").rglob("*.py")):
        source.update(str(path.relative_to(Path(__file__).resolve().parent)).encode())
        source.update(path.read_bytes())
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda,
            "nanovllm_source_sha256": source.hexdigest()}


def kernel_worker(args):
    import torch
    from triton.testing import do_bench_cudagraph
    from nanovllm.layers.layernorm import RMSNorm

    # Many deliberate dtype/layout cases must not exhaust Dynamo's default
    # cache limit and silently turn the compiled baseline into eager PyTorch.
    torch._dynamo.config.cache_size_limit = 128
    torch.manual_seed(42)
    rows = []
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16):
            cases = []
            for count in (1, 8, 128, 1024):
                x = torch.randn(count, 1024, device="cuda", dtype=dtype)
                cases.extend([("hidden", x, None), ("hidden", x, torch.randn_like(x))])
            for tokens in (1, 32, 256):
                qkv = torch.randn(tokens, 32, 128, device="cuda", dtype=dtype)
                cases.extend([("q_view", qkv[:, :16], None), ("k_view", qkv[:, 16:24], None)])
            for layout, x, residual in cases:
                old = RMSNorm(x.shape[-1], backend="torch").to(device="cuda", dtype=dtype)
                new = RMSNorm(x.shape[-1], backend="triton").to(device="cuda", dtype=dtype)
                old.weight.copy_(torch.randn_like(old.weight))
                new.weight.copy_(old.weight)
                functions = {"torch": lambda: old(x, residual), "triton": lambda: new(x, residual)}
                for fn in functions.values():
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
                a, b = functions["torch"](), functions["triton"]()
                if residual is not None:
                    torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)
                    a, b = a[0], b[0]
                eps = torch.finfo(dtype).eps
                torch.testing.assert_close(a, b, rtol=2 * eps, atol=2 * eps)
                times = {"torch": [], "triton": []}
                for trial in range(3):
                    for backend in (("torch", "triton") if trial % 2 == 0 else ("triton", "torch")):
                        times[backend].append(1000 * do_bench_cudagraph(functions[backend], rep=30))
                old_us, new_us = (statistics.median(times[k]) for k in ("torch", "triton"))
                row = {
                    "dtype": str(dtype).removeprefix("torch."), "layout": layout,
                    "shape": list(x.shape), "strides": list(x.stride()),
                    "residual": residual is not None,
                    "torch_us": old_us, "triton_us": new_us, "speedup": old_us / new_us,
                    "max_abs_difference": (a.float() - b.float()).abs().max().item(),
                    "relative_l2": ((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12)).item(),
                    "exact_fraction": (a == b).float().mean().item(),
                    "samples_us": times,
                }
                rows.append(row)
                print(f"{row['dtype']:8} {layout:7} {str(row['shape']):17} add={row['residual']} "
                      f"{old_us:.2f} -> {new_us:.2f} us ({row['speedup']:.2f}x)", flush=True)
    save(Path(args.output_dir) / "kernels.json", {"environment": environment(), "rows": rows})


def model_worker(args):
    import torch
    from nanovllm import LLM, SamplingParams
    from speculative_example import DRAFT, DRAFT_REVISION, PROMPTS, resolve_model

    torch._dynamo.config.cache_size_limit = 128
    torch.manual_seed(42)
    model = resolve_model(args.model, args.cache_dir)
    draft = resolve_model(DRAFT, args.cache_dir, DRAFT_REVISION) if args.mode in ("eagle", "eagle_graphs") else None
    backends = {"rms_norm_backend": args.backend} if args.experiment == "rmsnorm" else {"qk_rope_backend": args.backend}
    llm = LLM(model, draft_model=draft, **backends,
              speculative_cuda_graphs=args.mode == "eagle_graphs",
              num_speculative_tokens=getattr(args, "draft_tokens", 3),
              enforce_eager=args.mode != "graphs", tensor_parallel_size=1,
              max_num_seqs=8 if args.mode == "graphs" else 1,
              max_model_len=1024, max_num_batched_tokens=1024,
              gpu_memory_utilization=getattr(args, "gpu_memory_utilization", .12))
    try:
        prompts = [llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        ) for text in PROMPTS]
        params = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
        # Warm the exact workload, including draft repair and terminal shapes.
        for prompt in prompts:
            llm.generate([prompt], params, use_tqdm=False)
        samples, outputs = [], {}
        for repeat in range(args.repeats):
            for index in [(repeat + i) % len(prompts) for i in range(len(prompts))]:
                torch.cuda.synchronize()
                before_replays = dict(llm.speculator.graph_replays) if llm.speculator is not None else {}
                begin = perf_counter()
                output = llm.generate([prompts[index]], params, use_tqdm=False)[0]
                torch.cuda.synchronize()
                seconds = perf_counter() - begin
                assert len(output["token_ids"]) == args.max_tokens
                previous = outputs.setdefault(str(index), output)
                if previous["token_ids"] != output["token_ids"]:
                    raise AssertionError("Greedy output changed between repeated runs of one backend")
                sample = {"repeat": repeat, "prompt": index, "seconds": seconds}
                if llm.speculative_stats:
                    s = llm.speculative_stats[0]
                    sample.update({k: s[k] for k in ("accepted_tokens", "proposed_tokens", "target_forwards")})
                    sample["graph_replays"] = {k: value - before_replays[k]
                                               for k, value in llm.speculator.graph_replays.items()}
                samples.append(sample)
        totals = [sum(s["seconds"] for s in samples if s["repeat"] == r) for r in range(args.repeats)]
        result = {"environment": environment(), "backend": args.backend, "mode": args.mode,
                  "experiment": args.experiment, "backend_settings": backends,
                  "speculative_cuda_graphs": args.mode == "eagle_graphs",
                  "num_speculative_tokens": getattr(args, "draft_tokens", 3),
                  "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", .12),
                  "max_model_len": 1024, "prompt_token_ids": prompts,
                  "trial": args.trial, "model": model, "draft": draft,
                  "max_tokens": args.max_tokens, "repeats": args.repeats,
                  "eos_token_id": llm.tokenizer.eos_token_id,
                  "samples": samples, "repeat_seconds": totals, "outputs": outputs,
                  "tokens_per_second": len(prompts) * args.max_tokens / statistics.median(totals)}
        save(Path(args.output_dir) / f"{args.mode}-{args.backend}-{args.trial}.json", result)
        print(f"{args.mode:6} {args.backend:6} trial {args.trial}: {result['tokens_per_second']:.1f} tokens/s", flush=True)
    finally:
        llm.exit()


def summarize(args):
    folder = Path(args.output_dir)
    summary = {"method": "Median of warmed fixed-token runs; includes prefill, engine work and output decoding; excludes input tokenization/load/warmup",
               "max_tokens": args.max_tokens, "experiment": args.experiment, "model_rows": []}
    if args.suite in ("all", "kernels"):
        data = json.loads((folder / "kernels.json").read_text())
        fields = [k for k in data["rows"][0] if k != "samples_us"]
        with (folder / "kernels.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in data["rows"])
        summary["kernel_environment"] = data["environment"]
        summary["kernel_speedup_range"] = [min(r["speedup"] for r in data["rows"]),
                                             max(r["speedup"] for r in data["rows"])]
    if args.suite in ("all", "model"):
        for mode in args.modes:
            runs = {backend: [json.loads((folder / f"{mode}-{backend}-{trial}.json").read_text())
                              for trial in range(args.trials)] for backend in ("torch", "triton")}
            for results in runs.values():
                for result in results:
                    if (result.get("experiment", "rmsnorm") != args.experiment
                            or result["max_tokens"] != args.max_tokens
                            or result["repeats"] != args.repeats):
                        raise ValueError("Saved experiment/tokens/repeats differ from requested arguments")
            hashes = {result["environment"].get("nanovllm_source_sha256")
                      for results in runs.values() for result in results}
            if "kernel_environment" in summary:
                hashes.add(summary["kernel_environment"].get("nanovllm_source_sha256"))
            if "model_environment" in summary:
                hashes.add(summary["model_environment"].get("nanovllm_source_sha256"))
            if len(hashes) != 1 or None in hashes:
                raise ValueError("Benchmark files have missing/mixed source hashes; rerun the requested suite")
            rates, spreads = {}, {}
            for backend, results in runs.items():
                times = [t for result in results for t in result["repeat_seconds"]]
                rates[backend] = 3 * args.max_tokens / statistics.median(times)
                spreads[backend] = [3 * args.max_tokens / max(times), 3 * args.max_tokens / min(times)]
            matches = [runs["torch"][0]["outputs"][str(i)]["token_ids"] == result["outputs"][str(i)]["token_ids"]
                       for backend in runs for result in runs[backend] for i in range(3)]
            differences = []
            through_eos = []
            eos = runs["torch"][0]["eos_token_id"]
            def truncate(tokens):
                return tokens[:tokens.index(eos) + 1] if eos in tokens else tokens
            for backend in runs:
                for result in runs[backend]:
                    for i in range(3):
                        a = runs["torch"][0]["outputs"][str(i)]["token_ids"]
                        b = result["outputs"][str(i)]["token_ids"]
                        through_eos.append(truncate(a) == truncate(b))
                        if a != b:
                            first = next(j for j, (x, y) in enumerate(zip(a, b)) if x != y)
                            differences.append({"backend": backend, "trial": result["trial"], "prompt": i,
                                                "generated_index": first, "torch_token": a[first], "other_token": b[first]})
            row = {"mode": mode, "torch_tokens_per_second": rates["torch"],
                   "triton_tokens_per_second": rates["triton"],
                   "speedup": rates["triton"] / rates["torch"],
                   "exact_greedy_match": all(matches), "exact_through_eos": all(through_eos),
                   "first_differences": differences, "tokens_per_second_range": spreads}
            if mode == "eagle":
                row["acceptance_rates"] = {}
                row["target_forwards"] = {}
                for backend in runs:
                    samples = [s for result in runs[backend] for s in result["samples"]]
                    proposed = sum(s["proposed_tokens"] for s in samples)
                    row["acceptance_rates"][backend] = sum(s["accepted_tokens"] for s in samples) / proposed if proposed else 0.0
                    row["target_forwards"][backend] = sum(s["target_forwards"] for s in samples)
            summary["model_rows"].append(row)
            summary["model_environment"] = runs["torch"][0]["environment"]
            print(f"{mode}: {rates['torch']:.1f} -> {rates['triton']:.1f} tokens/s "
                  f"({row['speedup']:.3f}x), exact tokens: {all(matches)}")
        with (folder / "model.csv").open("w", newline="") as handle:
            fields = ["mode", "torch_tokens_per_second", "triton_tokens_per_second", "speedup", "exact_greedy_match", "exact_through_eos"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in summary["model_rows"])
    save(folder / "summary.json", summary)
    fused_qk = args.experiment == "qk_rope"
    title = "Fused Q/K RMSNorm + RoPE" if fused_qk else "RMSNorm"
    report = [f"# {title} benchmark results", "", summary["method"], "",
              "The torch backend is the pre-existing @torch.compile implementation.",
              ("The triton backend fuses Q/K normalization and RoPE in the target only; other norms, the EAGLE draft, attention, and cache writes use the original path."
               if fused_qk else "The triton backend replaces both plain and residual-add RMSNorm, including Q/K norms."), ""]
    if "kernel_speedup_range" in summary:
        low, high = summary["kernel_speedup_range"]
        report += [f"Kernel GPU-time speed ratios range from {low:.2f}x to {high:.2f}x.",
                   "Measured with CUDA graphs; see kernels.csv for every dtype, layout, and shape.", ""]
        report += ["| Dtype | Maximum absolute difference | Maximum relative L2 error | Minimum exact element fraction |",
                   "| --- | ---: | ---: | ---: |"]
        for dtype in ("float16", "bfloat16"):
            rows = [r for r in data["rows"] if r["dtype"] == dtype]
            report.append(f"| {dtype} | {max(r['max_abs_difference'] for r in rows):.6g} | "
                          f"{max(r['relative_l2'] for r in rows):.6g} | {min(r['exact_fraction'] for r in rows):.6f} |")
        report.append("")
    if summary["model_rows"]:
        report += ["| Mode | Compiled torch tokens/s | Triton tokens/s | Ratio | Exact full token stream | Exact through EOS |",
                   "| --- | ---: | ---: | ---: | --- | --- |"]
        for row in summary["model_rows"]:
            report.append(f"| {row['mode']} | {row['torch_tokens_per_second']:.1f} | "
                          f"{row['triton_tokens_per_second']:.1f} | {row['speedup']:.3f}x | {row['exact_greedy_match']} | {row['exact_through_eos']} |")
    report += ["", "Model trials use fresh processes with reversed backend order on alternating trials.",
               "Each prompt produces the same fixed number of tokens (EOS ignored for timing).",
               "Ratios above 1 favor Triton. Small differences inside run-to-run spread are inconclusive.",
               "BF16/FP16 reduction rounding can change close token decisions; equality is reported, not assumed.",
               "These results cover this workload and GPU, not all batch sizes or context lengths.", ""]
    for row in summary["model_rows"]:
        if row["mode"] == "eagle":
            rates = row["acceptance_rates"]
            forwards = row["target_forwards"]
            report += [f"EAGLE acceptance: torch {rates['torch']:.1%}, Triton {rates['triton']:.1%}. "
                       f"Total target forwards across timed samples: {forwards['torch']} versus {forwards['triton']}.",
                       "Normalization rounding can change draft proposals even when final tokens match through EOS. "
                       "This generation comparison therefore includes changes in speculation efficiency, alongside kernel and dispatch costs.", ""]
    env = summary.get("model_environment", summary.get("kernel_environment"))
    report += [f"Environment: {env}", "",
               "Microbenchmarks measure graph-replayed GPU work. Generation also includes Python dispatch, allocations, transfers, and engine decisions.",
               "The baseline uses a larger Dynamo cache limit to keep all intentional dtype/layout cases compiled, without eager fallback.", ""]
    if summary["model_rows"] and all(r["speedup"] < 1 for r in summary["model_rows"]):
        report += ["The custom backend did not improve full generation in these measurements. The default remains torch.", ""]
    if fused_qk:
        report += ["The fused Q/K backend remains opt-in. The measured gains apply to this batch-one, three-prompt workload; "
                   "small changes, particularly EAGLE's, need cautious interpretation alongside timing ranges and acceptance changes.", ""]
    make_plot(folder, summary)
    report += ["![Kernel and generation comparison](comparison.png)", ""]
    (folder / "REPORT.md").write_text("\n".join(report))


def make_plot(folder, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    has_kernels = "kernel_speedup_range" in summary
    has_model = bool(summary["model_rows"])
    count = int(has_kernels) + int(has_model)
    fig, axes = plt.subplots(1, count, figsize=(7 * count, 4.8), squeeze=False)
    panel = 0
    if has_kernels:
        rows = json.loads((folder / "kernels.json").read_text())["rows"]
        fused_qk = summary["experiment"] == "qk_rope"
        chosen = [r for r in rows if r["dtype"] == "bfloat16" and (
            (fused_qk and r["shape"][0] in (1, 8, 256, 1024))
            or (r["layout"] == "hidden" and r["shape"][0] in (1, 1024) and r["residual"])
            or (r["layout"] == "q_view" and r["shape"][0] in (1, 256)))]
        labels = [f"{r['layout']} {r['shape']}" + (" + residual" if r["residual"] else "") for r in chosen]
        ax = axes[0, panel]
        positions = np.arange(len(chosen))
        ax.barh(positions - .18, [r["torch_us"] for r in chosen], .36, label="torch.compile")
        ax.barh(positions + .18, [r["triton_us"] for r in chosen], .36, label="Triton")
        ax.set_yticks(positions, labels, fontsize=8)
        ax.set_xlabel("GPU microseconds — lower is better")
        ax.set_title("BF16 Q/K RMSNorm + RoPE" if fused_qk else "Selected BF16 RMSNorm kernels")
        ax.legend()
        panel += 1
    if has_model:
        rows = summary["model_rows"]
        ax = axes[0, panel]
        positions = np.arange(len(rows))
        ax.bar(positions - .18, [r["torch_tokens_per_second"] for r in rows], .36, label="torch.compile")
        ax.bar(positions + .18, [r["triton_tokens_per_second"] for r in rows], .36, label="Triton")
        ax.set_xticks(positions, [r["mode"] for r in rows])
        ax.set_ylabel("Generated tokens/s — higher is better")
        ax.set_title("Qwen3-0.6B generation, including prefill")
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)


def main(experiment="rmsnorm", kernel_benchmark=None):
    parser = argparse.ArgumentParser(description=("Compare fused Q/K RMSNorm + RoPE against the original compiled path." if experiment == "qk_rope" else __doc__))
    parser.add_argument("--suite", choices=["all", "kernels", "model"], default="all")
    parser.add_argument("--modes", nargs="+", choices=["eager", "graphs", "eagle"], default=["eager", "graphs", "eagle"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE", str(Path(__file__).resolve().parents[2] / "hf_cache")))
    parser.add_argument("--output-dir", default=f"{experiment}_results")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild tables and plots from saved results without GPU work")
    parser.add_argument("--worker", choices=["kernels", "model"], help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["torch", "triton"], help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=["eager", "graphs", "eagle"], help=argparse.SUPPRESS)
    parser.add_argument("--trial", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.experiment = experiment
    if min(args.repeats, args.trials, args.max_tokens) < 1 or args.max_tokens > 512:
        parser.error("repeats/trials must be positive; max-tokens must be between 1 and 512")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args)
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.worker:
        ((kernel_benchmark or kernel_worker) if args.worker == "kernels" else model_worker)(args)
        return
    command = [sys.executable, str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    if args.suite in ("all", "kernels"):
        subprocess.run([*command, "--worker", "kernels"], check=True)
    if args.suite in ("all", "model"):
        for trial in range(args.trials):
            for mode in args.modes:
                for backend in (("torch", "triton") if trial % 2 == 0 else ("triton", "torch")):
                    subprocess.run([*command, "--worker", "model", "--mode", mode,
                                    "--backend", backend, "--trial", str(trial)], check=True)
    summarize(args)


if __name__ == "__main__":
    main()
