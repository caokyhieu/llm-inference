"""
scripts/run_sweep.py
─────────────────────────────────────────────────────────────────────────────
Ray-Train + W&B hyperparameter sweep for vLLM-Ascend serving optimization.

Tests the search space defined in configs/sweep_config.yaml and reports
optimal (max_model_len, block_size, max_num_batched_tokens, tp_size) to
both MLflow and Weights & Biases.

Usage:
    # Single-node Ray (no cluster needed)
    python scripts/run_sweep.py

    # Multi-node Ray cluster
    ray start --head
    python scripts/run_sweep.py --ray-address auto

    # W&B integration
    WANDB_API_KEY=<key> python scripts/run_sweep.py --use-wandb
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import mlflow
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


# ── Search space ─────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "max_model_len":           tune.choice([8192, 16384, 32768]),
    "max_num_batched_tokens":  tune.choice([4096, 8192, 16384]),
    "block_size":              tune.choice([16, 32]),
    "enable_chunked_prefill":  tune.choice([True, False]),
    "enable_prefix_caching":   tune.choice([True, False]),
    "tp_size":                 tune.choice([4, 8]),
    "max_num_seqs":            tune.choice([128, 256, 512]),
    "kv_cache_dtype":          tune.choice(["auto", "fp8_e4m3"]),
    "speculation_tokens":      tune.choice([0, 3, 5]),
}

MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/models/Qwen3-32B-AWQ")
SERVER_URL  = os.environ.get("SERVER_URL", "http://localhost:8000")
NUM_PROMPTS = int(os.environ.get("BENCHMARK_PROMPTS", "200"))


# ── Trial function ───────────────────────────────────────────────────────────

def run_trial(config: dict):
    """
    Launch a vLLM server with the given config, benchmark it, and report metrics.
    Uses subprocess to isolate server from Ray worker.
    """
    import requests

    # Build vLLM command
    port = 8000 + int(ray.get_runtime_context().get_assigned_resources().get("CPU", 0)) % 100
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model",                   MODEL_PATH,
        "--device",                  "npu",
        "--tensor-parallel-size",    str(config["tp_size"]),
        "--quantization",            "awq",
        "--dtype",                   "bfloat16",
        "--gpu-memory-utilization",  "0.95",
        "--max-model-len",           str(config["max_model_len"]),
        "--block-size",              str(config["block_size"]),
        "--max-num-batched-tokens",  str(config["max_num_batched_tokens"]),
        "--max-num-seqs",            str(config["max_num_seqs"]),
        "--kv-cache-dtype",          config["kv_cache_dtype"],
        "--port",                    str(port),
        "--host",                    "0.0.0.0",
    ]
    if config["enable_chunked_prefill"]:
        cmd.append("--enable-chunked-prefill")
    if config["enable_prefix_caching"]:
        cmd.append("--enable-prefix-caching")
    if config["speculation_tokens"] > 0:
        cmd += ["--speculative-method", "ngram",
                "--num-speculative-tokens", str(config["speculation_tokens"])]

    proc = None
    try:
        # Start server
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Wait for server readiness
        url = f"http://localhost:{port}/health"
        for _ in range(60):
            try:
                if requests.get(url, timeout=2).ok:
                    break
            except Exception:
                pass
            time.sleep(3)
        else:
            tune.report(throughput_tok_s=0.0, ttft_p50_ms=9999.0)
            return

        # Run quick benchmark (200 prompts to keep each trial < 3 min)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            result_path = f.name

        bench = subprocess.run([
            "python", "benchmarks/benchmark_runner.py",
            "--server-url",   f"http://localhost:{port}",
            "--num-prompts",  str(NUM_PROMPTS),
            "--request-rate", "20",
            "--run-name",     f"sweep_{port}",
            "--output-json",  result_path,
        ], capture_output=True, timeout=180)

        if bench.returncode == 0 and Path(result_path).exists():
            with open(result_path) as f:
                res = json.load(f)
            tune.report(
                throughput_tok_s=res.get("throughput_tok_s", 0),
                ttft_p50_ms=res.get("ttft_p50_ms", 9999),
                tpot_p50_ms=res.get("tpot_p50_ms", 9999),
                success_rate=res.get("success_rate", 0),
            )
        else:
            tune.report(throughput_tok_s=0.0, ttft_p50_ms=9999.0)

    except Exception as e:
        print(f"[trial error] {e}")
        tune.report(throughput_tok_s=0.0, ttft_p50_ms=9999.0)
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=10)


# ── W&B callback ─────────────────────────────────────────────────────────────

def get_wandb_callback():
    try:
        from ray.air.integrations.wandb import WandbLoggerCallback
        return WandbLoggerCallback(
            project="vllm-ascend-qwen3",
            log_config=True,
        )
    except ImportError:
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ray-address",  default=None, help="Ray cluster address")
    p.add_argument("--num-samples",  type=int, default=30)
    p.add_argument("--max-parallel", type=int, default=2)
    p.add_argument("--use-wandb",    action="store_true")
    args = p.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    callbacks = []
    if args.use_wandb:
        cb = get_wandb_callback()
        if cb:
            callbacks.append(cb)

    # ASHA scheduler: aggressive early stopping of bad configs
    scheduler = ASHAScheduler(
        metric="throughput_tok_s",
        mode="max",
        max_t=10,
        grace_period=2,
        reduction_factor=3,
    )

    tuner = tune.Tuner(
        tune.with_resources(run_trial, {"CPU": 2}),
        param_space=SEARCH_SPACE,
        tune_config=tune.TuneConfig(
            metric="throughput_tok_s",
            mode="max",
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_parallel,
            scheduler=scheduler,
        ),
        run_config=ray.air.RunConfig(
            name="vllm_ascend_sweep",
            storage_path="/workspace/results/ray_sweep",
            callbacks=callbacks,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="throughput_tok_s", mode="max")

    print("\n========== BEST CONFIGURATION ==========")
    print(f"Throughput: {best.metrics['throughput_tok_s']:.0f} tok/s")
    print(f"TTFT p50:   {best.metrics['ttft_p50_ms']:.1f} ms")
    print("Config:")
    for k, v in best.config.items():
        print(f"  {k}: {v}")

    # Log best to MLflow
    mlflow.set_experiment("vllm-ascend-qwen3-32b")
    with mlflow.start_run(run_name="sweep_best"):
        mlflow.log_params(best.config)
        mlflow.log_metrics({
            "throughput_tok_s": best.metrics["throughput_tok_s"],
            "ttft_p50_ms": best.metrics["ttft_p50_ms"],
        })

    # Save best config
    best_path = Path("/workspace/results/best_config.json")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_path, "w") as f:
        json.dump(best.config, f, indent=2)
    print(f"\n[+] Best config saved → {best_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
