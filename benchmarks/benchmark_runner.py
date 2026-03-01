#!/usr/bin/env python3
"""
benchmarks/benchmark_runner.py
─────────────────────────────────────────────────────────────────────────────
Runs throughput + latency benchmarks against the vLLM-ascend server and logs
results to MLflow.

Usage:
    python benchmarks/benchmark_runner.py \
        --run-name baseline_fp16_tp8 \
        --dataset sharegpt \
        --num-prompts 1100 \
        --request-rate 10 \
        --server-url http://localhost:8000

Environment:
    MLFLOW_TRACKING_URI   — MLflow server (default: http://localhost:5000)
    MLFLOW_EXPERIMENT     — Experiment name (default: vllm-ascend-qwen3-32b)
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from statistics import median, quantiles

import mlflow
import requests

# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float          # Time to first token (seconds)
    tpot_s: float          # Time per output token (seconds)
    total_latency_s: float
    success: bool
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    run_name: str
    num_prompts: int
    request_rate: float
    total_time_s: float
    # Throughput
    throughput_req_s: float
    throughput_tok_s: float
    # TTFT
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    # TPOT
    tpot_p50_ms: float
    tpot_p90_ms: float
    tpot_p99_ms: float
    # Totals
    total_prompt_tokens: int
    total_completion_tokens: int
    success_rate: float


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_sharegpt(path: str, n: int = 1100) -> list[str]:
    """Load ShareGPT conversation prompts."""
    if not Path(path).exists():
        print(f"[warn] ShareGPT dataset not found at {path}, using synthetic prompts.")
        return [
            f"Explain concept number {i} in detail: " + "word " * 50
            for i in range(n)
        ]
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data[:n]:
        convs = item.get("conversations", [])
        if convs:
            prompts.append(convs[0].get("value", "")[:2000])
    return prompts[:n]


def load_synthetic_longctx(n: int = 100, input_len: int = 8192) -> list[str]:
    """Synthetic long-context prompts for TTFT stress testing."""
    base = "This is a synthetic long-context prompt for benchmarking. " * 100
    return [base[:input_len] for _ in range(n)]


# ── Single request ───────────────────────────────────────────────────────────

def send_request(server_url: str, prompt: str, max_tokens: int = 256) -> RequestResult:
    """Send a streaming chat-completion request and measure TTFT + TPOT."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    try:
        t0 = time.perf_counter()
        ttft = None
        completion_tokens = 0

        with requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if line == b"data: [DONE]":
                    break
                chunk = line.decode().removeprefix("data: ")
                try:
                    data = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                delta = data["choices"][0]["delta"].get("content", "")
                if delta and ttft is None:
                    ttft = time.perf_counter() - t0
                if delta:
                    completion_tokens += 1

        total = time.perf_counter() - t0
        tpot = (total - (ttft or 0)) / max(completion_tokens - 1, 1)
        prompt_tokens = len(prompt.split())  # rough estimate

        return RequestResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_s=ttft or 0.0,
            tpot_s=tpot,
            total_latency_s=total,
            success=True,
        )
    except Exception as e:
        return RequestResult(
            prompt_tokens=0,
            completion_tokens=0,
            ttft_s=0.0,
            tpot_s=0.0,
            total_latency_s=0.0,
            success=False,
            error=str(e),
        )


# ── Load test orchestrator ───────────────────────────────────────────────────

def run_benchmark(
    server_url: str,
    prompts: list[str],
    request_rate: float,
    run_name: str,
    max_workers: int = 32,
) -> BenchmarkSummary:
    interval = 1.0 / request_rate
    results: list[RequestResult] = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for i, prompt in enumerate(prompts):
            futures.append(pool.submit(send_request, server_url, prompt))
            if i < len(prompts) - 1:
                time.sleep(interval)

        for fut in as_completed(futures):
            results.append(fut.result())

    total_time = time.perf_counter() - t_start
    successful = [r for r in results if r.success]

    if not successful:
        raise RuntimeError("All requests failed — is the server running?")

    ttfts  = sorted(r.ttft_s * 1000 for r in successful)
    tpots  = sorted(r.tpot_s * 1000 for r in successful)
    p = lambda data, q: quantiles(data, n=100)[q - 1]

    total_tok = sum(r.completion_tokens for r in successful)

    return BenchmarkSummary(
        run_name=run_name,
        num_prompts=len(prompts),
        request_rate=request_rate,
        total_time_s=total_time,
        throughput_req_s=len(successful) / total_time,
        throughput_tok_s=total_tok / total_time,
        ttft_p50_ms=median(ttfts),
        ttft_p90_ms=p(ttfts, 90),
        ttft_p99_ms=p(ttfts, 99),
        tpot_p50_ms=median(tpots),
        tpot_p90_ms=p(tpots, 90),
        tpot_p99_ms=p(tpots, 99),
        total_prompt_tokens=sum(r.prompt_tokens for r in successful),
        total_completion_tokens=total_tok,
        success_rate=len(successful) / len(results),
    )


# ── MLflow logging ───────────────────────────────────────────────────────────

def log_to_mlflow(summary: BenchmarkSummary, tags: dict):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "vllm-ascend-qwen3-32b"))

    with mlflow.start_run(run_name=summary.run_name, tags=tags):
        mlflow.log_metrics({
            "throughput_req_s":  summary.throughput_req_s,
            "throughput_tok_s":  summary.throughput_tok_s,
            "ttft_p50_ms":       summary.ttft_p50_ms,
            "ttft_p90_ms":       summary.ttft_p90_ms,
            "ttft_p99_ms":       summary.ttft_p99_ms,
            "tpot_p50_ms":       summary.tpot_p50_ms,
            "tpot_p90_ms":       summary.tpot_p90_ms,
            "tpot_p99_ms":       summary.tpot_p99_ms,
            "success_rate":      summary.success_rate,
            "total_output_tokens": summary.total_completion_tokens,
        })
        mlflow.log_params(tags)
        print(f"\n[MLflow] Run '{summary.run_name}' logged.")
        print(f"  Throughput: {summary.throughput_tok_s:.0f} tok/s")
        print(f"  TTFT p50:   {summary.ttft_p50_ms:.1f} ms")
        print(f"  TPOT p50:   {summary.tpot_p50_ms:.2f} ms")
        print(f"  Success:    {summary.success_rate:.1%}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name",     default="run",               help="MLflow run name")
    p.add_argument("--dataset",      choices=["sharegpt","longctx"], default="sharegpt")
    p.add_argument("--dataset-path", default="data/ShareGPT_V3_unfiltered_cleaned_split.json")
    p.add_argument("--num-prompts",  type=int, default=1100)
    p.add_argument("--request-rate", type=float, default=10.0)
    p.add_argument("--server-url",   default="http://localhost:8000")
    p.add_argument("--tp",           default="8",  help="Tensor parallel size (for tagging)")
    p.add_argument("--quant",        default="awq", help="Quantization method (for tagging)")
    p.add_argument("--output-json",  default="results/benchmark_result.json")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[+] Loading {args.dataset} prompts ({args.num_prompts})...")
    if args.dataset == "sharegpt":
        prompts = load_sharegpt(args.dataset_path, args.num_prompts)
    else:
        prompts = load_synthetic_longctx(args.num_prompts)

    print(f"[+] Running benchmark → {args.server_url} @ {args.request_rate} req/s")
    summary = run_benchmark(
        server_url=args.server_url,
        prompts=prompts,
        request_rate=args.request_rate,
        run_name=args.run_name,
    )

    tags = {"tp_size": args.tp, "quantization": args.quant, "dataset": args.dataset}
    log_to_mlflow(summary, tags)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"[+] Results saved → {args.output_json}")


if __name__ == "__main__":
    main()
