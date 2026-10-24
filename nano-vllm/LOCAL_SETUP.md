# Local setup with uv

Run these commands from `nano-vllm/` or use `python setup_env.py` at the workspace root.
Nano-vLLM, quantization, and the VLM lab share the root `.venv` and `uv.lock`.
This setup uses Python 3.12 and the
official FlashAttention 2.7.3 wheel for PyTorch 2.6 with CUDA 12 on Linux x86_64.
This wheel works with the machine's glibc 2.31; the newer 2.8.3.post2 wheel
requires glibc 2.32 or later.

```bash
python ../setup_env.py
```

If this machine's configured proxy fails, use
`NO_PROXY='*' no_proxy='*' python ../setup_env.py`. TLS certificate verification stays
enabled. `requirements-h100.txt` is retained for legacy standalone installs;
the root lockfile is authoritative for the shared workspace. After the wheel
has been prepared, `uv sync --locked` works from this directory or the root.

Download Qwen3-0.6B into the existing shared Hugging Face cache:

```bash
NO_PROXY='*' no_proxy='*' uv run --no-sync hf download Qwen/Qwen3-0.6B --cache-dir ../../hf_cache
```

Run the example on GPU 0:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-sync python example.py
```

`--no-sync` uses the shared environment installed above. Ordinary
`uv run --locked` also works and checks against the root lockfile.
The example defaults to Qwen3-0.6B and the
`hf_cache/` directory next to `llm-inference/`. Override the cache with
`--cache-dir` or `HF_HUB_CACHE`; override the model with `--model`, which accepts
a Hugging Face repository ID or a local snapshot directory.

For a shorter learning run:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 uv run --no-sync python example.py --max-tokens 32
```

The example uses eager execution, one GPU, and a GPU memory utilization budget
of 0.2. You can change the budget with `--gpu-memory-utilization`. The first
run still compiles some individual layers with `torch.compile`, so startup
can take longer than subsequent runs.

Run the fused softmax tutorial with the same environment:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run --no-sync python test_fused_softmax.py
```

The script uses device selection and kernel launching compatible with Triton
3.2. It checks its output against `torch.softmax`, prints benchmark results,
and saves a PNG chart and CSV measurements under `softmax_results/`.
The benchmark requires `matplotlib` and `pandas`, included in
`requirements-h100.txt`.
