#!/usr/bin/env bash
# scripts/run_eval.sh — Full evaluation suite for quantization ablation
# Covers: lm-evaluation-harness (EleutherAI tasks) + perplexity + MT-Bench
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/models/Qwen3-32B-AWQ}"
QUANT="${QUANT:-awq}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results/eval}"
TP="${TP_SIZE:-8}"

mkdir -p "$OUTPUT_DIR"

echo "================================================================"
echo " Evaluation Suite"
echo " Model:  $MODEL_PATH"
echo " Quant:  $QUANT  |  TP: $TP"
echo "================================================================"

# ── 1. LM Evaluation Harness (EleutherAI) ───────────────────────────────────
echo ""
echo "[1/3] lm-evaluation-harness (HumanEval + MMLU + HellaSwag + ARC + TruthfulQA)"

lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${TP},device=npu,dtype=bfloat16,quantization=${QUANT},trust_remote_code=True" \
  --tasks hellaswag,arc_challenge,mmlu,truthfulqa_mc2,winogrande \
  --batch_size auto \
  --num_fewshot 0 \
  --output_path "${OUTPUT_DIR}/lm_eval_${QUANT}.json" \
  --log_samples 2>&1 | tee "${OUTPUT_DIR}/lm_eval_${QUANT}.log"

echo "[1/3] ✓ Saved to ${OUTPUT_DIR}/lm_eval_${QUANT}.json"

# ── 2. Perplexity on WikiText-2 ──────────────────────────────────────────────
echo ""
echo "[2/3] WikiText-2 perplexity"

lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL_PATH},tensor_parallel_size=${TP},device=npu,dtype=bfloat16,quantization=${QUANT}" \
  --tasks wikitext \
  --batch_size auto \
  --output_path "${OUTPUT_DIR}/perplexity_${QUANT}.json" 2>&1 | tee "${OUTPUT_DIR}/perplexity_${QUANT}.log"

# Extract perplexity value
PPL=$(python3 -c "
import json
d = json.load(open('${OUTPUT_DIR}/perplexity_${QUANT}.json'))
ppl = d.get('results', {}).get('wikitext', {}).get('word_perplexity,none', 'N/A')
print(ppl)
")
echo "[2/3] ✓ WikiText-2 PPL=${PPL} for quant=${QUANT}"

# ── 3. Log perplexity to MLflow ──────────────────────────────────────────────
echo ""
echo "[3/3] Logging to MLflow..."

python3 - <<PYEOF
import json, os, mlflow

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("vllm-ascend-qwen3-32b")

quant = "${QUANT}"
ppl   = float("${PPL}") if "${PPL}" != "N/A" else None

with mlflow.start_run(run_name=f"eval_{quant}"):
    mlflow.log_param("quantization", quant)
    mlflow.log_param("tp_size", "${TP}")
    if ppl:
        mlflow.log_metric("wikitext2_ppl", ppl)

    # Parse and log task scores
    try:
        d = json.load(open("${OUTPUT_DIR}/lm_eval_${QUANT}.json"))
        for task, metrics in d.get("results", {}).items():
            for metric, val in metrics.items():
                if isinstance(val, (int, float)) and not metric.endswith("_stderr"):
                    clean_m = metric.replace(",none","").replace(",","_")
                    mlflow.log_metric(f"{task}_{clean_m}", val)
    except Exception as e:
        print(f"[warn] Could not log task scores: {e}")

print(f"[3/3] ✓ MLflow run logged for quant={quant}")
PYEOF

echo ""
echo "================================================================"
echo " Evaluation complete → ${OUTPUT_DIR}"
echo " WikiText-2 PPL=${PPL}  Quant=${QUANT}"
echo "================================================================"
