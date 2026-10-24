"""Compare ordinary nano-vLLM and EAGLE-3 in separate GPU processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


DRAFT = "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge"
DRAFT_REVISION = "5fe9c43a50078119bfa38e5f30969475be314315"
PROMPTS = [
    "List the prime numbers below 30.",
    "Explain why a language model uses a KV cache in two sentences.",
    "Write a Python function that returns the factorial of a nonnegative integer.",
]


def resolve_model(name, cache_dir, revision=None):
    from huggingface_hub import snapshot_download
    if Path(name).is_dir():
        return str(Path(name).resolve())
    return snapshot_download(name, revision=revision, cache_dir=cache_dir,
                             allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])


def worker(args):
    import torch
    from transformers import AutoTokenizer
    from nanovllm import LLM, SamplingParams

    torch.manual_seed(0)
    model = resolve_model(args.model, args.cache_dir)
    draft = resolve_model(args.draft_model, args.cache_dir, args.draft_revision) if args.mode == "eagle" else None
    tokenizer = AutoTokenizer.from_pretrained(model)
    messages = args.prompt or PROMPTS
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) for text in messages]
    llm = LLM(model, draft_model=draft, num_speculative_tokens=args.draft_tokens,
              rms_norm_backend=args.rms_norm_backend,
              qk_rope_backend=args.qk_rope_backend,
              speculative_cuda_graphs=args.speculative_cuda_graphs and args.mode == "eagle",
              enforce_eager=True, tensor_parallel_size=1, max_num_seqs=1,
              max_model_len=args.max_model_len, max_num_batched_tokens=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization)
    try:
        # Exercise prompt, rollout, verification, repair, and end-of-budget shapes.
        # Compilation is lazy in the existing RMSNorm/RoPE layers.
        warmup = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Count from one to ten."}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        llm.generate([warmup], SamplingParams(temperature=args.temperature, max_tokens=32, ignore_eos=True), use_tqdm=False)
        outputs, stats, seconds = [], [], []
        for prompt in prompts:
            torch.cuda.synchronize()
            start = perf_counter()
            output = llm.generate([prompt], SamplingParams(
                temperature=args.temperature, max_tokens=args.max_tokens,
                ignore_eos=args.ignore_eos,
            ), use_tqdm=False)[0]
            torch.cuda.synchronize()
            seconds.append(perf_counter() - start)
            outputs.append(output)
            if llm.speculative_stats:
                stats.extend(llm.speculative_stats)
        result = {
            "mode": args.mode, "model": model, "draft_model": draft,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "temperature": args.temperature, "max_tokens": args.max_tokens,
            "rms_norm_backend": args.rms_norm_backend,
            "qk_rope_backend": args.qk_rope_backend,
            "speculative_cuda_graphs": args.speculative_cuda_graphs and args.mode == "eagle",
            "num_speculative_tokens": args.draft_tokens,
            "prompts": messages, "outputs": outputs, "seconds": seconds,
            "tokens_per_second": sum(len(o["token_ids"]) for o in outputs) / sum(seconds),
            "speculative_stats": stats,
        }
        path = Path(args.output_dir) / f"{args.mode}.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{args.mode}: {result['tokens_per_second']:.1f} tokens/s (includes prefill, excludes load/warmup)")
        for text, output in zip(messages, outputs):
            print(f"\nPrompt: {text}\n{output['text']}")
        if stats:
            proposed = sum(s["proposed_tokens"] for s in stats)
            accepted = sum(s["accepted_tokens"] for s in stats)
            forwards = sum(s["target_forwards"] for s in stats)
            print(f"\nDraft acceptance: {accepted}/{proposed}; target forwards: {forwards}")
        print(f"Result: {path}")
    finally:
        llm.exit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["compare", "baseline", "eagle"], default="compare")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--draft-model", default=DRAFT)
    parser.add_argument("--draft-revision", default=DRAFT_REVISION)
    parser.add_argument("--cache-dir", default=os.environ.get(
        "HF_HUB_CACHE", str(Path(__file__).resolve().parents[2] / "hf_cache")))
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index (PCI_BUS_ID order)")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--rms-norm-backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--qk-rope-backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--speculative-cuda-graphs", action="store_true",
                        help="Capture EAGLE forwards; this example's ordinary baseline remains eager. "
                             "Use benchmark_speculative_graphs.py to compare ordinary CUDA graphs.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prompt", action="append", help="Repeat to replace the three default prompts")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.12)
    parser.add_argument("--output-dir", default="speculative_results")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # Set before importing torch; children inherit exactly the same mapping.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.mode != "compare":
        worker(args)
        return
    for mode in ["baseline", "eagle"]:
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        *sys.argv[1:], "--mode", mode], check=True)
    folder = Path(args.output_dir)
    baseline = json.loads((folder / "baseline.json").read_text())
    eagle = json.loads((folder / "eagle.json").read_text())
    matches = [a["token_ids"] == b["token_ids"] for a, b in zip(baseline["outputs"], eagle["outputs"])]
    differences = []
    if args.temperature == 0:
        for i, (a, b) in enumerate(zip(baseline["outputs"], eagle["outputs"])):
            a, b = a["token_ids"], b["token_ids"]
            if a == b:
                continue
            first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            differences.append({"prompt_index": i, "generated_index": first,
                                "baseline_token": a[first] if first < len(a) else None,
                                "eagle_token": b[first] if first < len(b) else None})
    summary = {
        "greedy_token_match": matches if args.temperature == 0 else None,
        "first_differences": differences,
        "baseline_tokens_per_second": baseline["tokens_per_second"],
        "eagle_tokens_per_second": eagle["tokens_per_second"],
        "speedup": eagle["tokens_per_second"] / baseline["tokens_per_second"],
    }
    (folder / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSpeed ratio: {summary['speedup']:.2f}x (EAGLE / baseline)")
    if args.temperature == 0:
        if not all(matches):
            raise AssertionError(f"Greedy token mismatch: {differences}; inspect saved outputs. "
                                 "BF16 batch-shape rounding can change close argmax choices; "
                                 "see SPECULATIVE_DECODING.md.")
        print(f"PASS: exact greedy token match on all {len(matches)} prompts")
    else:
        print("Sampling run: equal seeds do not imply equal token sequences with speculative decoding.")


if __name__ == "__main__":
    main()
