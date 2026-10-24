"""Compare ordinary CUDA graphs, eager EAGLE, and EAGLE CUDA graphs."""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from benchmark_rmsnorm import model_worker, save


MODES = ("graphs", "eagle", "eagle_graphs")
LABELS = {"graphs": "Ordinary + CUDA graphs", "eagle": "EAGLE eager",
          "eagle_graphs": "EAGLE + CUDA graphs"}


def gpu_snapshot(gpu):
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,name,memory.used,utilization.gpu",
         "--format=csv,noheader"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def summarize(args):
    folder = Path(args.output_dir)
    runs = {mode: [json.loads((folder / f"{mode}-{args.qk_rope_backend}-{trial}.json").read_text())
                   for trial in range(args.trials)] for mode in MODES}
    reference = runs["graphs"][0]
    hashes = {r["environment"]["nanovllm_source_sha256"] for results in runs.values() for r in results}
    if len(hashes) != 1:
        raise ValueError("Saved cases use different engine sources; rerun all cases")
    for mode, results in runs.items():
        for trial, result in enumerate(results):
            if (result["mode"] != mode or result["trial"] != trial
                    or result["max_tokens"] != args.max_tokens or result["repeats"] != args.repeats
                    or result["num_speculative_tokens"] != args.draft_tokens
                    or result["prompt_token_ids"] != reference["prompt_token_ids"]
                    or result["backend_settings"] != {"qk_rope_backend": args.qk_rope_backend}
                    or result["model"] != reference["model"]
                    or result["environment"] != reference["environment"]):
                raise ValueError("Saved cases do not match the requested workload/environment")
    rows = []
    eos = reference["eos_token_id"]
    def truncate(tokens):
        return tokens[:tokens.index(eos) + 1] if eos in tokens else tokens
    for mode, results in runs.items():
        seconds = [t for result in results for t in result["repeat_seconds"]]
        count = len(reference["prompt_token_ids"]) * args.max_tokens
        rate = count / statistics.median(seconds)
        exact, through_eos, differences = [], [], []
        for result in results:
            for index, expected in reference["outputs"].items():
                a, b = expected["token_ids"], result["outputs"][index]["token_ids"]
                exact.append(a == b)
                through_eos.append(truncate(a) == truncate(b))
                if a != b:
                    first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                    differences.append({"trial": result["trial"], "prompt": index, "index": first,
                                        "reference_token": a[first] if first < len(a) else None,
                                        "other_token": b[first] if first < len(b) else None})
        row = {"mode": mode, "tokens_per_second": rate,
               "range": [count / max(seconds), count / min(seconds)],
               "exact_full_stream": all(exact), "exact_through_eos": all(through_eos),
               "first_differences": differences}
        if mode != "graphs":
            samples = [s for result in results for s in result["samples"]]
            proposed = sum(s["proposed_tokens"] for s in samples)
            row["acceptance_rate"] = sum(s["accepted_tokens"] for s in samples) / proposed if proposed else 0.
            row["target_forwards"] = sum(s["target_forwards"] for s in samples)
            row["graph_replays"] = {key: sum(s["graph_replays"][key] for s in samples) for key in ("target", "draft")}
            if mode == "eagle_graphs" and args.max_tokens > 1 and not row["graph_replays"]["target"]:
                raise AssertionError("Speculative graph benchmark did not replay target graphs")
        rows.append(row)
    baseline = rows[0]["tokens_per_second"]
    for row in rows:
        row["ratio_to_ordinary_graphs"] = row["tokens_per_second"] / baseline
    summary = {"rows": rows, "environment": reference["environment"],
               "model": reference["model"], "draft": runs["eagle_graphs"][0]["draft"],
               "physical_gpu": args.gpu, "qk_rope_backend": args.qk_rope_backend,
               "max_tokens": args.max_tokens, "repeats": args.repeats, "trials": args.trials,
               "draft_tokens": args.draft_tokens,
               "speculative_graphs_vs_eager": rows[2]["tokens_per_second"] / rows[1]["tokens_per_second"],
               "eager_vs_graph_speculative_exact": all(
                   result["outputs"] == runs["eagle"][0]["outputs"] for result in runs["eagle_graphs"])}
    save(folder / "summary.json", summary)
    with (folder / "comparison.csv").open("w", newline="") as handle:
        fields = ["mode", "tokens_per_second", "ratio_to_ordinary_graphs", "exact_full_stream", "exact_through_eos"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    report = ["# Speculative decoding with CUDA graphs", "",
              "Qwen3-0.6B BF16, one request at a time, three cached chat prompts.",
              f"Q/K backend: {args.qk_rope_backend}; all other norms use the original compiled path.", "",
              "| Mode | Tokens/s | Observed range | Ratio to ordinary graphs | Exact full stream | Exact through EOS |",
              "| --- | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        low, high = row["range"]
        report.append(f"| {LABELS[row['mode']]} | {row['tokens_per_second']:.1f} | {low:.1f}–{high:.1f} | "
                      f"{row['ratio_to_ordinary_graphs']:.3f}x | {row['exact_full_stream']} | {row['exact_through_eos']} |")
        print(f"{LABELS[row['mode']]}: {row['tokens_per_second']:.1f} tokens/s "
              f"({row['ratio_to_ordinary_graphs']:.3f}x ordinary graphs)")
    report += ["", f"EAGLE graph/eager ratio: {summary['speculative_graphs_vs_eager']:.3f}x.",
               f"EAGLE graph/eager saved outputs identical: {summary['eager_vs_graph_speculative_exact']}.", ""]
    for row in rows[1:]:
        report += [f"{LABELS[row['mode']]}: acceptance {row['acceptance_rate']:.1%}; "
                   f"target forwards {row['target_forwards']}; graph replays {row['graph_replays']}."]
    relation = "faster" if rows[2]["ratio_to_ordinary_graphs"] > 1 else "slower"
    report += ["", f"In this workload, EAGLE with graphs was {relation} than ordinary decoding with graphs.", "",
               f"Median of {args.trials * args.repeats} warmed three-prompt repeat durations, "
               "using fresh processes and reversed case order on alternating trials.",
               f"Each request produces {args.max_tokens} tokens, ignoring EOS for timing. "
               "Includes prefill, engine overhead and output decoding; excludes loading, capture, warmup and input tokenization.",
               "The range is min/max observed repeat throughput, not a confidence interval. Small gains require cautious interpretation.",
               "Output equality is reported separately through EOS and over forced continuations. "
               "Floating-point differences can change draft acceptance and near-tied target choices.",
               "This is a partial-graph implementation: initial prefill, CPU acceptance, draft token .item(), "
               "target sampling and request bookkeeping remain outside capture. "
               "Target verification (including LM head), small draft repairs and single-token rollout forwards are captured.", "",
               "The ordinary baseline uses the existing decoder-only capture; its LM head and sampling execute afterward.", "",
               f"Environment: {reference['environment']}",
               f"Physical GPU: {args.gpu}. Before/after GPU snapshots are in gpu_state.json. "
               "This machine is shared; measurements are workload-specific.", "",
               "![Throughput comparison](comparison.png)", ""]
    (folder / "REPORT.md").write_text("\n".join(report))
    make_plot(folder, rows)


def make_plot(folder, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [row["tokens_per_second"] for row in rows]
    errors = [[value - row["range"][0] for value, row in zip(values, rows)],
              [row["range"][1] - value for value, row in zip(values, rows)]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh([LABELS[row["mode"]] for row in rows], values, xerr=errors,
                   color=["#4477aa", "#aaaaaa", "#ee7733"], capsize=4)
    ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=12)
    ax.set_xlim(0, max(row["range"][1] for row in rows) * 1.18)
    ax.set_xlabel("Generated tokens/s, including prefill (higher is better)")
    ax.set_title("Qwen3-0.6B: ordinary vs speculative CUDA graphs")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HUB_CACHE", str(Path(__file__).resolve().parents[2] / "hf_cache")))
    parser.add_argument("--qk-rope-backend", choices=["torch", "triton"], default="triton")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None,
                        help="Default: budget 8 GiB above current GPU use, capped at 95%%")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--output-dir", default="speculative_graph_results")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--trial", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.max_tokens, args.repeats, args.trials, args.draft_tokens) < 1 or args.max_tokens > 512:
        parser.error("counts must be positive and max-tokens at most 512")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("GPU memory utilization must be between 0 and 1")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args)
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.worker:
        if args.gpu_memory_utilization is None:
            import torch
            free, total = torch.cuda.mem_get_info()
            if free < 9 * 1024 ** 3:
                raise RuntimeError("Need at least 9 GiB free GPU memory for this benchmark")
            args.gpu_memory_utilization = min(.95, (total - free + 8 * 1024 ** 3) / total)
        args.backend, args.experiment = args.qk_rope_backend, "qk_rope"
        model_worker(args)
        return
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    states = []
    for trial in range(args.trials):
        for mode in MODES if trial % 2 == 0 else reversed(MODES):
            state = {"mode": mode, "trial": trial, "before": gpu_snapshot(args.gpu)}
            subprocess.run([*command, "--worker", "--mode", mode, "--trial", str(trial)], check=True)
            state["after"] = gpu_snapshot(args.gpu)
            states.append(state)
            save(Path(args.output_dir) / "gpu_state.json", states)
    summarize(args)


if __name__ == "__main__":
    main()
