"""Benchmark fused Q/K RMSNorm + RoPE using the existing benchmark harness."""

from pathlib import Path
import statistics

from benchmark_rmsnorm import environment, main, save


def kernel_worker(args):
    import torch
    from triton.testing import do_bench_cudagraph
    from nanovllm.kernels.qk_norm_rope import qk_norm_rope
    from nanovllm.layers.layernorm import RMSNorm
    from nanovllm.layers.rotary_embedding import RotaryEmbedding

    torch._dynamo.config.cache_size_limit = 128
    torch.manual_seed(42)
    rows = []
    with torch.inference_mode():
        for dtype in (torch.float16, torch.bfloat16):
            for tokens in (1, 4, 8, 32, 256, 1024):
                packed = torch.randn(tokens, 32, 128, device="cuda", dtype=dtype)
                q, k = packed[:, :16], packed[:, 16:24]
                q_norm = RMSNorm(128).to(device="cuda", dtype=dtype)
                k_norm = RMSNorm(128).to(device="cuda", dtype=dtype)
                q_norm.weight.copy_(torch.randn_like(q_norm.weight))
                k_norm.weight.copy_(torch.randn_like(k_norm.weight))
                rope = RotaryEmbedding(128, 128, 4096, 1000000).to("cuda")
                positions = torch.randint(4096, (tokens,), device="cuda")
                functions = {
                    "torch": lambda: rope(positions, q_norm(q), k_norm(k)),
                    "triton": lambda: qk_norm_rope(q, k, q_norm.weight, k_norm.weight,
                                                  rope.cos_sin_cache, positions, q_norm.eps),
                }
                for fn in functions.values():
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
                expected, actual = functions["torch"](), functions["triton"]()
                for a, b in zip(expected, actual):
                    tolerance = 4 * torch.finfo(dtype).eps
                    torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
                times = {"torch": [], "triton": []}
                for trial in range(3):
                    for backend in (("torch", "triton") if trial % 2 == 0 else ("triton", "torch")):
                        times[backend].append(1000 * do_bench_cudagraph(functions[backend], rep=30))
                old_us, new_us = (statistics.median(times[key]) for key in ("torch", "triton"))
                a = torch.cat([tensor.flatten() for tensor in expected]).float()
                b = torch.cat([tensor.flatten() for tensor in actual]).float()
                row = {"dtype": str(dtype).removeprefix("torch."), "layout": "qk_rope",
                       "shape": list(q.shape), "k_shape": list(k.shape),
                       "strides": list(q.stride()), "residual": False,
                       "torch_us": old_us, "triton_us": new_us, "speedup": old_us / new_us,
                       "max_abs_difference": (a - b).abs().max().item(),
                       "relative_l2": ((a - b).norm() / a.norm().clamp_min(1e-12)).item(),
                       "exact_fraction": (a == b).float().mean().item(), "samples_us": times}
                rows.append(row)
                print(f"{row['dtype']:8} tokens={tokens:4} Q16/K8/128: "
                      f"{old_us:.2f} -> {new_us:.2f} us ({row['speedup']:.2f}x)", flush=True)
    save(Path(args.output_dir) / "kernels.json", {"environment": environment(), "rows": rows})


if __name__ == "__main__":
    main(experiment="qk_rope", kernel_benchmark=kernel_worker)
