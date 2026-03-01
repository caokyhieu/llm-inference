#!/usr/bin/env bash
# scripts/run_optimized.sh — Launch vLLM-ascend with full optimization stack
# O1: block_size=32  O2: FP8 KV  O3: chunked+prefix  O4: AWQ  O5: HCCL
# O6: mem pool  O7: speculative decoding
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-32B-AWQ}"
TP="${TP_SIZE:-8}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_BATCHED="${MAX_BATCHED_TOKENS:-8192}"
MAX_SEQS="${MAX_NUM_SEQS:-256}"
SWAP_SPACE="${SWAP_SPACE:-16}"

echo "========================================================="
echo " vLLM-Ascend Optimized Serving"
echo " Model:  $MODEL_PATH"
echo " TP:     $TP  |  Port: $PORT"
echo " Len:    $MAX_MODEL_LEN  |  Batch: $MAX_BATCHED"
echo "========================================================="

# Wait for MLflow
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
  echo "[+] Waiting for MLflow at $MLFLOW_TRACKING_URI ..."
  until curl -sf "${MLFLOW_TRACKING_URI}/health" > /dev/null; do sleep 2; done
  echo "[+] MLflow ready."
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model          "$MODEL_PATH" \
  --device         npu \
  --tensor-parallel-size "$TP" \
  --quantization   awq \
  --dtype          bfloat16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len  "$MAX_MODEL_LEN" \
  --block-size     32 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --max-num-seqs   "$MAX_SEQS" \
  --swap-space     "$SWAP_SPACE" \
  --speculative-method ngram \
  --num-speculative-tokens 5 \
  --enable-metrics \
  --metrics-port   9090 \
  --port           "$PORT" \
  --host           0.0.0.0 \
  --trust-remote-code
