# Triton RMSNorm experiment

This experiment replaces RMSNorm and residual-add + RMSNorm with a small explicit
Triton implementation. The original `@torch.compile` methods remain unchanged
and are the default. This compares against the compiled implementation that
nano-vLLM already used, rather than an unfused eager baseline.

## Select a backend

```python
llm = LLM(
    model_path,
    rms_norm_backend="triton",  # "torch" selects the original compiled methods
    enforce_eager=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.2,
)
```

The runner sets the backend on every target RMSNorm, including Q/K normalization,
and on the EAGLE-3 head when one is loaded. Selection happens before warmup and
CUDA-graph capture. The backend adds no checkpoint parameters and does not change
weight loading.

From `nano-vllm/`, using the existing environment and cached models:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  uv run --no-sync python example.py --rms-norm-backend triton --max-tokens 64

HF_HUB_OFFLINE=1 uv run --no-sync python speculative_example.py \
  --mode eagle --rms-norm-backend triton --max-tokens 64
```

## Implementation

- [kernels/rms_norm.py](nanovllm/kernels/rms_norm.py) contains the kernel and
  input/output wrapper.
- [layers/layernorm.py](nanovllm/layers/layernorm.py) selects the new backend or
  calls the existing compiled methods.
- [ModelRunner](nanovllm/engine/model_runner.py) applies the configuration before
  executing either model.

One GPU program processes one row. For widths at most 256 with at least 256
rows, a program processes four rows to reduce the number of small workgroups.
The reduction runs along the last dimension, using eight warps for larger or
grouped rows and four for smaller single rows. Reduction order matters for
low-precision model behavior; this choice was checked on real model features as
well as random tensors. Masking handles both non-power-of-two widths and a final
partial group of rows.

Inputs can have one to four dimensions. The wrapper pads leading-dimension
metadata to three axes; the kernel maps each flattened row back to the original
strides. This directly supports Q/K views into packed QKV tensors without making
a contiguous input copy. Outputs are contiguous and inputs are not modified.

For residual-add + RMSNorm, the computation is:

```text
s = float32(x) + float32(residual)
residual_out = cast_to_input_dtype(s)
variance = mean(s * s)
normalized = s * rsqrt(variance + epsilon)
out = cast_to_input_dtype(normalized * float32(weight))
```

Plain RMSNorm uses `s = float32(x)` and omits `residual_out`.

The residual output is the unnormalized sum, which the following decoder layer
needs. It is a separate output tensor, not an in-place update of the input.

### Match the compiled precision behavior

The original Python code appears to cast normalized values to FP16/BF16 before
multiplying by the weight. Inspection of this environment's TorchInductor-generated
kernels showed that the compiler eliminates that intermediate round and retains
FP32 through the multiply. The custom kernel matches that executed behavior and
rounds at the final output store. It also uses CUDA libdevice's reciprocal square
root, as the generated baseline does, rather than Triton's generic `tl.rsqrt`.
A literal intermediate round caused substantially more one-ULP differences during
development.

Parallel reductions can still differ in rounding. Tests compare against both
the original compiled methods and an FP32 computation, with dtype-appropriate
tolerances. Full generation compares exact greedy token IDs separately; numerical
closeness alone does not establish identical text on every prompt.

### Supported scope

The explicit kernel supports CUDA FP16/BF16 inputs, row widths 1–16,384, and
contiguous FP16/BF16/FP32 weight vectors on the same device. Residual inputs must
match shape, dtype, and device, but can have different strides. Empty batches
return empty outputs without a launch.

This is an inference kernel. Use `torch.inference_mode()` for standalone calls;
it has no custom backward implementation. The engine already runs model execution
under inference mode. The original backend remains available for other uses.

## Reproduce the benchmark

### Measured outcome

On this machine's H100 PCIe, using Qwen3-0.6B in BF16, the explicit Triton backend
did not improve full generation. The original compiled backend remains the default.

| Generation mode | Original tokens/s | Triton tokens/s | Change |
| --- | ---: | ---: | ---: |
| Ordinary eager | 57.0 | 43.9 | -22.9% |
| Ordinary CUDA graphs | 416.7 | 403.4 | -3.2% |
| EAGLE-3 eager | 86.5 | 68.4 | -20.9% |

Individual GPU kernels ranged from 0.83× to 1.57× the original speed. The strongest
gain was a larger BF16 Q-normalization view; small decode shapes were generally
slower. The original `torch.compile` path already generates fused Triton kernels,
so writing Triton explicitly does not by itself remove additional operations.
Python dispatch and allocation costs also contribute to the full eager path.

All 18 tests passed with the Triton backend selected. Benchmark tokens matched
through EOS on all three prompts, but some forced continuations after EOS differed.
EAGLE acceptance changed from 44.6% to 38.8%, increasing target forwards from 510
to 552 across timed samples. Its throughput difference therefore includes changed
draft proposals as well as kernel execution costs. See the
[full report](rmsnorm_results/REPORT.md) for raw comparisons and environment details.

### Commands

```bash
# GPU 0 by default. Kernels, ordinary eager generation, ordinary CUDA graphs,
# and EAGLE-3 generation, comparing the two norm backends.
HF_HUB_OFFLINE=1 uv run --no-sync python -u benchmark_rmsnorm.py

# Kernel timings only, or a selected generation mode.
uv run --no-sync python benchmark_rmsnorm.py --suite kernels
HF_HUB_OFFLINE=1 uv run --no-sync python benchmark_rmsnorm.py \
  --suite model --modes graphs

# Rebuild the default run's tables/plot without GPU execution.
uv run --no-sync python benchmark_rmsnorm.py --summarize-only
```

Results are written under `rmsnorm_results/`:

| Artifact | Contents |
| --- | --- |
| [REPORT.md](rmsnorm_results/REPORT.md) | Measured results and comparison plot |
| [kernels.csv](rmsnorm_results/kernels.csv) | Every dtype, shape, layout, and residual case |
| [model.csv](rmsnorm_results/model.csv) | Generation rates, ratios, and exact-token comparison |
| `kernels.json` | Timing samples, strides, environment, and numerical differences |
| `<mode>-<backend>-<trial>.json` | Raw generation timings and outputs for each fresh process |
| [summary.json](rmsnorm_results/summary.json) | Aggregated measurements and run-to-run ranges |

Kernel timings use CUDA-graph replay to measure GPU work with Python dispatch
excluded. The benchmark warms both implementations and alternates measurement
order. It covers hidden width 1,024 and strided head width 128, with FP16 and BF16.
It raises Dynamo's specialization-cache limit so the intentional shape/dtype
sweep does not silently push the old backend into eager fallback.

Generation uses the same three chat prompts as our speculative example, tokenized
before timing. Every request generates 64 tokens with EOS ignored, so both backends
do a fixed amount of work. Both backends warm the actual workload first. Defaults
are three repeats in each of two fresh-process trials, with backend order reversed
on the second trial. The reported tokens/s includes prefill and generation, while
excluding model loading, tokenization, compilation, and warmup.

`eager` and `graphs` use ordinary decoding; `eagle` uses our existing eager EAGLE-3
path. Compare norm backends within each mode. This is a batch-one experiment, not
a concurrency or long-context serving benchmark. Raw timings are retained so a
small ratio can be assessed against run-to-run variation.

A faster GPU microkernel does not necessarily improve generation. The full path
also includes Python wrapper/launch overhead, allocations, copies, matrix
projections, attention, and scheduling. Use the full-generation result to decide
whether the backend benefits a particular workload.

## Correctness checks

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  NANOVLLM_RUN_GPU_TESTS=1 NANOVLLM_RMS_NORM_BACKEND=triton \
  uv run --no-sync python -m unittest discover -s tests -v
```

The new norm tests cover FP16/BF16, packed Q/K slices, noncontiguous outer and
column strides, masked widths, partial row groups, zero inputs, empty batches,
input immutability, and exact residual sums. The backend environment variable
selects Triton in the existing EAGLE reference/cache/EOS tests, reusing those
integration checks. It does not change the library's default backend.
