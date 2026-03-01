#!/usr/bin/env bash
# scripts/run_baseline.sh — Unoptimized baseline for comparison
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-32B-AWQ}"
TP="${TP_SIZE:-8}"
PORT="${PORT:-8000}"

echo "[baseline] Starting FP16 baseline, TP=$TP ..."

exec python -m vllm.entrypoints.openai.api_server \
  --model          "$MODEL_PATH" \
  --device         npu \
  --tensor-parallel-size "$TP" \
  --dtype          bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len  8192 \
  --block-size     16 \
  --max-num-seqs   128 \
  --port           "$PORT" \
  --host           0.0.0.0 \
  --trust-remote-code
