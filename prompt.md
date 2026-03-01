You are an expert AI research engineer specializing in LLM inference optimization. Use the loaded AI-Research-SKILLs library, focusing on the "vLLM Inference & Serving" skill (12-inference-serving/vllm).

Research objective: Optimize vLLM for high-throughput, low-latency LLM serving on Huawei Ascend NPU chips (e.g., 910B/Atlas 800I cluster). Target models: Qwen3-32B, DeepSeek-R1 (AWQ/W4A16 quantized).

Step-by-step workflow:
1. **Setup Environment**: Provision Docker with vllm-ascend (pip install vllm-ascend>=0.10, CANN 8.0+, torch-npu). Use multi-NPU config (--device npu --tensor-parallel-size 8 --gpu-memory-utilization 0.95).
2. **Benchmark Baseline**: Run ShareGPT-1.1K dataset eval (throughput tokens/s, TTFT, TPOT, mem usage). Compare vs. A100/H100 baselines from vLLM docs.
3. **Apply Optimizations**:
   - Quantization: AWQ/SmoothQuant; tune swap_space=16GiB.
   - Scheduling: Prefix caching, FP8 KV-cache, disaggregated prefill/decode.
   - NPU-specific: PYTORCH_NPU_ALLOC_CONF, HCCL comms, PID-like dynamic batching (adapt your llm-d scheduler [cite:11]).
   - Custom kernels: Fuse attention via torch_npu if possible.
4. **Hyperparam Sweep**: Grid search --max-model-len, --enable-chunked-prefill; use MLflow/W&B for logging.
5. **Ablation Study**: Test TP/DP/EP scaling (1-8 NPUs); anomaly detection on logs.
6. **Generate Report**: Plots (throughput vs. batch), PPTX summary, code repo PR to vllm-ascend. Quantify gains (aim 20-30% over baseline [web:19]).

Provision cloud sim (Huawei ModelArts/AutoDL 910B instance), execute experiments, iterate on failures. Output: Full reproducible notebook, key findings table, next experiments.
