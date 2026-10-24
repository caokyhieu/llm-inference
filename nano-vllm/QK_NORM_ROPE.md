# Fused Q/K RMSNorm + RoPE

This experiment fuses Q normalization, K normalization, and rotary position
encoding into one Triton launch. It compares against the original separately
compiled operations. The standalone RMSNorm experiment remains independent:
hidden-state norms use the original backend unless explicitly changed.

## Measured result

On the H100 PCIe with Qwen3-0.6B in BF16:

| Generation mode | Original tokens/s | Fused tokens/s | Change |
| --- | ---: | ---: | ---: |
| Ordinary eager | 57.0 | 62.2 | +9.2% |
| Ordinary CUDA graphs | 417.6 | 449.2 | +7.6% |
| EAGLE-3 eager | 86.5 | 87.9 | +1.6% |

The combined GPU operation improved in all 12 measured dtype/shape cases, with
speed ratios from 1.14× to 3.46×. BF16 at one token took 1.93 µs versus 6.06 µs
for the original path. Fusion removes normalized Q/K intermediate tensors and
combines their normalization and rotation into one launch. This gives a clearer
benefit than the earlier replacement of RMSNorm alone.

The ordinary-generation timing ranges were separated in both modes. The EAGLE
gain is small and should be interpreted cautiously: draft acceptance changed
from 44.6% to 38.8%, with 510 versus 552 target forwards across timed samples.
That comparison includes changes in speculation efficiency, not just kernel cost.

All benchmark prompts matched through EOS in both trials. Forced continuation
after EOS differed for one prompt; the full fixed-token streams are therefore
not identical. These results cover three prompts and batch-one generation, not
all contexts or serving workloads. The backend remains opt-in.

## Enable it

```python
from nanovllm import LLM

llm = LLM(model_path, qk_rope_backend="triton")
```

The default is `qk_rope_backend="torch"`. Backend selection happens before model
warmup and CUDA-graph capture. Checkpoints and weight loading are unchanged.

From `nano-vllm/`, using the existing environment and cached model:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  uv run --no-sync python example.py --qk-rope-backend triton --max-tokens 64

HF_HUB_OFFLINE=1 uv run --no-sync python speculative_example.py \
  --mode eagle --qk-rope-backend triton --max-tokens 64
```

## What the kernel does

The [attention path](nanovllm/models/qwen3.py) splits the QKV projection into
strided Q, K, and V views. The
[fused kernel](nanovllm/kernels/qk_norm_rope.py) reads Q and K directly from those
views and produces contiguous normalized, rotated Q and K. It leaves V and the
packed projection intact. The existing attention implementation then writes
K/V to the cache and calls FlashAttention.

Each GPU program handles one token/head pair. A single launch covers both the
query heads and key heads, using the corresponding learned norm weights.
For each head:

```text
x32 = float32(x)
inverse_rms = rsqrt(mean(x32 * x32) + epsilon)
normalized = cast_to_input_dtype(x32 * inverse_rms * float32(weight))
left, right = split_halves(float32(normalized))
output = cast_to_input_dtype(concat(left*cos - right*sin,
                                  right*cos + left*sin))
```

The intermediate cast is intentional: the original normalization writes an
FP16/BF16 tensor before RoPE reads it. Fusion keeps that round in registers,
without writing the normalized intermediate to GPU memory. FP32 reduction and
CUDA libdevice reciprocal square root match the existing compiled norm's
computation. Floating-point reduction and fused arithmetic can still produce
small differences; tests check values and generation comparisons record tokens.

For power-of-two head widths, the kernel swaps the two normalized halves in
registers. Other even widths use a second input load for the partner half;
this supports masking with the installed Triton 3.2, which lacks `tl.gather`.

## Scope

- CUDA FP16/BF16 inference, even head widths from 2 through 512.
- Packed or independently strided token/head axes; contiguous head columns.
- Contiguous Q/K weight vectors with the same FP16/BF16/FP32 dtype.
- Existing FP32 cosine/sine cache and integer positions, including strided
  positions. Callers must supply positions within the cache's bounds.
- Empty token batches and CUDA-graph replay are supported. No backward pass.

Qwen attention modules without Q/K normalization retain their original RoPE
path. The EAGLE draft has no Q/K norm and retains its original implementation;
enabling this backend changes the Qwen target only. Combining this flag with
`rms_norm_backend="triton"` also changes other RMSNorms, so the benchmark keeps
that independent option at its original `"torch"` setting.

## Benchmark and validation

```bash
# All kernel and generation comparisons, on physical GPU 0 by default.
HF_HUB_OFFLINE=1 uv run --no-sync python -u benchmark_qk_rope.py

# Only the combined Q/K norm + RoPE operation.
uv run --no-sync python benchmark_qk_rope.py --suite kernels

# Regenerate tables and plot from the saved default run, without GPU work.
uv run --no-sync python benchmark_qk_rope.py --summarize-only

# Real checkpoint/reference/cache checks with fusion enabled.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  NANOVLLM_RUN_GPU_TESTS=1 NANOVLLM_QK_ROPE_BACKEND=triton \
  uv run --no-sync python -m unittest discover -s tests -v
```

[benchmark_qk_rope.py](benchmark_qk_rope.py) reuses the process isolation,
generation worker, reporting, and plotting from [benchmark_rmsnorm.py](benchmark_rmsnorm.py).
Kernel cases cover FP16/BF16, Qwen3-0.6B's 16 query heads and 8 KV heads of width
128, and 1–1,024 tokens, using packed views. CUDA-graph timing excludes Python
dispatch. The full-generation comparison includes dispatch and engine work.

Generation uses the existing three chat prompts, one request at a time, each
producing 64 greedy tokens with EOS ignored for a fixed token budget. Each
backend warms the actual workload. Two fresh-process trials, with reversed
backend order, each contain three repeats. Tokens/s includes prefill and
excludes model loading, compilation, warmup, and tokenization.

See [the generated report](qk_rope_results/REPORT.md),
[kernel measurements](qk_rope_results/kernels.csv), and
[summary with timing ranges and token comparisons](qk_rope_results/summary.json).
Raw per-process JSON files retain timings, outputs, model paths, package versions,
and a hash of the engine source. The report distinguishes complete fixed-token
streams from comparisons through EOS, and records EAGLE acceptance changes.

The five new GPU tests cover compiled and independent references, different
head counts and widths, masking, strided inputs and positions, input immutability,
zeros, empty batches, and CUDA-graph replay with changing inputs/positions.
Existing target, draft, cache rollback, EOS, and sampling checks are reused.
The full suite passed all 23 tests with `NANOVLLM_QK_ROPE_BACKEND=triton`.
