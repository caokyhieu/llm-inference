"""Inference-only RMSNorm and residual-add + RMSNorm for CUDA FP16/BF16.

One program handles one row, or four small rows for prefill. Leading dimensions
may be strided, including Q/K views of packed QKV. Inputs are never modified.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _row_offset(row, D1: tl.constexpr, D2: tl.constexpr,
                S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr):
    return (row // (D1 * D2)) * S0 + ((row // D2) % D1) * S1 + (row % D2) * S2


@triton.jit
def _rms_norm_kernel(
    X, R, W, Y, R_OUT,
    x_col_stride, r_col_stride,
    rows, N: tl.constexpr, EPS: tl.constexpr,
    D1: tl.constexpr, D2: tl.constexpr,
    SX0: tl.constexpr, SX1: tl.constexpr, SX2: tl.constexpr,
    SR0: tl.constexpr, SR1: tl.constexpr, SR2: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr, BLOCK: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK)[None, :]
    mask = (row[:, None] < rows) & (cols < N)
    x_offset = _row_offset(row, D1, D2, SX0, SX1, SX2)[:, None]
    x = tl.load(X + x_offset + cols * x_col_stride, mask, other=0).to(tl.float32)
    if HAS_RESIDUAL:
        r_offset = _row_offset(row, D1, D2, SR0, SR1, SR2)[:, None]
        residual = tl.load(R + r_offset + cols * r_col_stride, mask, other=0).to(tl.float32)
        x = x + residual
        tl.store(R_OUT + row[:, None] * N + cols, x, mask)
    variance = tl.sum(x * x, axis=1) / N
    normalized = x * libdevice.rsqrt(variance[:, None] + EPS)
    # Match the existing TorchInductor-generated kernel: normalization and the
    # weight multiply stay FP32, then round at the store. The compiler removes
    # the intermediate low-precision round in layernorm.py's Python expression.
    weight = tl.load(W + cols, cols < N, other=0).to(tl.float32)
    tl.store(Y + row[:, None] * N + cols, normalized * weight, mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
             residual: torch.Tensor | None = None):
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton RMSNorm requires CUDA FP16/BF16 input")
    if not 1 <= x.ndim <= 4 or x.shape[-1] == 0 or x.shape[-1] > 16384:
        raise ValueError("Triton RMSNorm supports rank 1–4 and row widths from 1 to 16384")
    if (weight.shape != (x.shape[-1],) or not weight.is_contiguous()
            or weight.device != x.device or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        raise ValueError("RMSNorm weight must be a contiguous vector on the input device")
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype or residual.device != x.device):
        raise ValueError("Residual must match the input shape, dtype, and device")
    tensors = (x, weight) if residual is None else (x, weight, residual)
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise RuntimeError("Triton RMSNorm is inference-only; use torch.inference_mode()")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    residual_out = torch.empty_like(out) if residual is not None else None
    n = x.shape[-1]
    rows = x.numel() // n
    if rows:
        block = triton.next_power_of_2(n)
        block_rows = 4 if n <= 256 and rows >= 256 else 1
        # Pad outer dimensions to three axes. Scalar constexprs also work with
        # the project's older Triton 3.2, without tuple-constexpr indexing.
        padding = 4 - x.ndim
        outer = (1,) * padding + tuple(x.shape[:-1])
        x_strides = (0,) * padding + tuple(x.stride()[:-1])
        r_strides = (0,) * padding + tuple(residual.stride()[:-1]) if residual is not None else (0, 0, 0)
        _rms_norm_kernel[(triton.cdiv(rows, block_rows),)](
            x, residual if residual is not None else x, weight, out,
            residual_out if residual_out is not None else out,
            x.stride(-1), residual.stride(-1) if residual is not None else 0,
            rows, n, eps, outer[1], outer[2], *x_strides, *r_strides,
            residual is not None, block, block_rows,
            # Reduction parallelism affects FP32 summation order. Eight warps
            # for hidden rows also tracks the compiled baseline more closely
            # on real model activations than a four-warp hidden reduction.
            num_warps=8 if n >= 1024 or block_rows > 1 else 4,
        )
    return out if residual is None else (out, residual_out)
