################################################################################
# vllm-ascend-opt — Optimized vLLM serving on Huawei Ascend 910B
# Base: Ascend CANN 8.0 + torch-npu 2.3 + vllm-ascend 0.10
#
# Build:
#   docker build -t vllm-ascend-opt:0.10 .
#
# Run (8x 910B NPU cluster, Atlas 800I):
#   docker run --device /dev/davinci0 ... --device /dev/davinci7 \
#     --device /dev/davinci_manager \
#     --device /dev/devmm_svm \
#     --device /dev/hisi_hdc \
#     -v /usr/local/Ascend:/usr/local/Ascend \
#     -p 8000:8000 -p 5000:5000 \
#     vllm-ascend-opt:0.10 bash scripts/run_optimized.sh
################################################################################

FROM quay.io/ascend/cann:8.0.0-910b-ubuntu22.04

LABEL maintainer="Huawei 2012 Labs <ai-systems@huawei.com>"
LABEL description="vLLM inference stack optimized for Ascend 910B NPU"

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev git curl wget jq htop \
    && rm -rf /var/lib/apt/lists/*

# ── Python packages ──────────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    "vllm-ascend>=0.10.0" \
    "torch-npu>=2.3.0" \
    "autoawq>=0.2.0" \
    "auto-gptq>=0.7.0" \
    "hqq>=0.2.0" \
    "llmcompressor>=0.4.0" \
    "mlflow>=2.12.0" \
    "wandb>=0.17.0" \
    "ray[tune]>=2.20.0" \
    "lm-eval>=0.4.3" \
    "arize-phoenix>=4.0.0" \
    "locust>=2.28.0" \
    "transformers>=4.45.0" \
    "accelerate>=0.31.0" \
    "datasets>=2.19.0" \
    "sentencepiece>=0.2.0" \
    "jsonlines>=4.0.0"

# ── Ascend environment ────────────────────────────────────────────────────────
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:$LD_LIBRARY_PATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:$PATH

# ── vLLM-Ascend tuning ───────────────────────────────────────────────────────
# O1: vector-unit aligned block size (32 = 256B / (2 × 128))
ENV VLLM_NPU_BLOCK_SIZE=32
# O6: NPU memory allocator tuning
ENV PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:256,garbage_collection_threshold:0.8"
# O4/O5: paged attention NPU path + HCCL
ENV VLLM_USE_NPU_PAGED_ATTN=1
ENV HCCL_ALGO="level0:fullmesh;level1:fullmesh"
ENV HCCL_BUFFSIZE=512
ENV HCCL_EXEC_TIMEOUT=600
ENV HCCL_TIMEOUT=1800

WORKDIR /workspace
COPY . /workspace/

RUN chmod +x scripts/*.sh

EXPOSE 8000 5000 9090

CMD ["bash", "scripts/run_optimized.sh"]
