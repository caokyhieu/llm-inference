"""Fuse Q/K RMSNorm and split-half RoPE; leave attention/cache handling intact."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _qk_norm_rope_kernel(
    Q, K, QW, KW, CACHE, POS, Q_OUT, K_OUT,
    QS0: tl.constexpr, QS1: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
    PS: tl.constexpr, CS: tl.constexpr,
    Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    if head < Q_HEADS:
        x_ptr = Q + token * QS0 + head * QS1
        w_ptr = QW
        out = Q_OUT + (token * Q_HEADS + head) * N
    else:
        head = head - Q_HEADS
        x_ptr = K + token * KS0 + head * KS1
        w_ptr = KW
        out = K_OUT + (token * K_HEADS + head) * N
    x = tl.load(x_ptr + cols, cols < N, other=0).to(tl.float32)
    weight = tl.load(w_ptr + cols, cols < N, other=0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / N
    inverse_rms = libdevice.rsqrt(variance + EPS)
    norm = x * inverse_rms * weight
    # The original norm writes FP16/BF16 before RoPE loads it as FP32. Keep
    # that rounding boundary even though this kernel has no intermediate store.
    norm = norm.to(Q.dtype.element_ty).to(tl.float32)
    if N == BLOCK:
        left, right = tl.split(tl.trans(tl.reshape(norm, (2, BLOCK // 2))))
        partner = tl.reshape(tl.trans(tl.join(right, left)), (BLOCK,))
    else:
        # Triton 3.2 has no gather. Masked widths use a second input load to
        # obtain the other half while preserving the full-width reduction order.
        swapped = (cols + N // 2) % N
        other_x = tl.load(x_ptr + swapped, cols < N, other=0).to(tl.float32)
        other_w = tl.load(w_ptr + swapped, cols < N, other=0).to(tl.float32)
        partner = (other_x * inverse_rms * other_w).to(Q.dtype.element_ty).to(tl.float32)
    position = tl.load(POS + token * PS)
    frequency = cols % (N // 2)
    cosine = tl.load(CACHE + position * CS + frequency, cols < N, other=0)
    sine = tl.load(CACHE + position * CS + N // 2 + frequency, cols < N, other=0)
    rotated = tl.where(cols < N // 2, -partner, partner)
    tl.store(out + cols, norm * cosine + rotated * sine, cols < N)


def qk_norm_rope(q, k, q_weight, k_weight, cos_sin_cache, positions, eps=1e-6):
    """Normalize/rotate [tokens, heads, width] packed views without input copies.

    CUDA FP16/BF16, even widths up to 512, contiguous last dimension. Positions
    must index valid cache rows (as in the existing RoPE path). Inference only.
    Outputs are contiguous; all inputs, including the packed V view, stay intact.
    """
    if (q.ndim != 3 or k.ndim != 3 or not q.is_cuda
            or q.dtype not in (torch.float16, torch.bfloat16)
            or k.dtype != q.dtype or k.device != q.device
            or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]
            or min(q.shape[1], k.shape[1]) < 1
            or q.stride(-1) != 1 or k.stride(-1) != 1):
        raise ValueError("Q/K must be CUDA FP16/BF16 [tokens, heads, width] with matching tokens/width and contiguous columns")
    tokens, q_heads, width = q.shape
    if not 2 <= width <= 512 or width % 2:
        raise ValueError("Q/K head width must be even and between 2 and 512")
    for weight in (q_weight, k_weight):
        if (weight.shape != (width,) or weight.device != q.device or not weight.is_contiguous()
                or weight.dtype != q_weight.dtype
                or weight.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            raise ValueError("Norm weights must be contiguous vectors of the same floating-point dtype on the Q/K device")
    if (cos_sin_cache.ndim != 3 or cos_sin_cache.shape[1:] != (1, width)
            or cos_sin_cache.dtype != torch.float32 or cos_sin_cache.device != q.device
            or not cos_sin_cache.is_contiguous()):
        raise ValueError("RoPE cache must be contiguous CUDA FP32 [positions, 1, width]")
    if (positions.shape != (tokens,) or positions.device != q.device
            or positions.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Positions must be an integer vector on the Q/K device")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, q_weight, k_weight, cos_sin_cache)):
        raise RuntimeError("Fused Q/K norm + RoPE is inference-only; use torch.inference_mode()")
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if tokens:
        _qk_norm_rope_kernel[(tokens, q_heads + k.shape[1])](
            q, k, q_weight, k_weight, cos_sin_cache, positions, q_out, k_out,
            *q.stride()[:2], *k.stride()[:2], positions.stride(0), cos_sin_cache.stride(0),
            q_heads, k.shape[1], width, eps, triton.next_power_of_2(width), num_warps=4,
        )
    return q_out, k_out
