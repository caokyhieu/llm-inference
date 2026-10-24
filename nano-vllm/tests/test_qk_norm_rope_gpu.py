"""NANOVLLM_RUN_GPU_TESTS=1 enables fused Q/K normalization + RoPE checks."""

import os
import unittest

import torch

from nanovllm.kernels.qk_norm_rope import qk_norm_rope
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import RotaryEmbedding


@unittest.skipUnless(os.environ.get("NANOVLLM_RUN_GPU_TESTS") == "1", "GPU tests are opt-in")
class FusedQKRoPETests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        previous = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = 128
        cls.addClassCleanup(setattr, torch._dynamo.config, "cache_size_limit", previous)

    @torch.inference_mode()
    def check_case(self, tokens, q_heads, k_heads, width, dtype, transposed=False, weight_dtype=None):
        torch.manual_seed(53)
        packed = torch.randn(tokens * 2, q_heads + 2 * k_heads, width, device="cuda", dtype=dtype)
        q = packed[::2, :q_heads]
        k = packed[::2, q_heads:q_heads + k_heads]
        if transposed:
            q = q.transpose(0, 1).contiguous().transpose(0, 1)
            k = k.transpose(0, 1).contiguous().transpose(0, 1)
        saved_q, saved_k, saved_packed = q.clone(), k.clone(), packed.clone()
        q_norm = RMSNorm(width, eps=1e-5).to(device="cuda", dtype=weight_dtype or dtype)
        k_norm = RMSNorm(width, eps=1e-5).to(device="cuda", dtype=weight_dtype or dtype)
        q_norm.weight.copy_(torch.randn_like(q_norm.weight))
        k_norm.weight.copy_(torch.randn_like(k_norm.weight))
        rope = RotaryEmbedding(width, width, 4096, 1000000).to("cuda")
        positions = torch.randint(4096, (tokens * 2,), device="cuda", dtype=torch.int64)[::2]
        if tokens:
            positions[0] = 4095
        actual = qk_norm_rope(q, k, q_norm.weight, k_norm.weight,
                              rope.cos_sin_cache, positions, q_norm.eps)
        expected = rope(positions, q_norm(q), k_norm(k))
        cosine, sine = rope.cos_sin_cache[positions].chunk(2, dim=-1)
        for result, compiled, x, norm in zip(actual, expected, (q, k), (q_norm, k_norm)):
            # Independent eager reference with the same intermediate dtype boundary.
            normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)
            normalized = (normalized * norm.weight.float()).to(dtype).float()
            left, right = normalized.chunk(2, dim=-1)
            reference = torch.cat((left * cosine - right * sine,
                                   right * cosine + left * sine), dim=-1).to(dtype)
            tolerance = 4 * torch.finfo(dtype).eps
            torch.testing.assert_close(result, compiled, rtol=tolerance, atol=tolerance)
            torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)
            self.assertTrue(result.is_contiguous())
        torch.testing.assert_close(q, saved_q, atol=0, rtol=0)
        torch.testing.assert_close(k, saved_k, atol=0, rtol=0)
        torch.testing.assert_close(packed, saved_packed, atol=0, rtol=0)

    def test_packed_and_masked_head_shapes(self):
        for dtype in (torch.float16, torch.bfloat16):
            for shape in ((1, 16, 8, 128), (4, 16, 8, 128), (37, 16, 8, 128),
                          (257, 16, 8, 128), (3, 5, 2, 80), (7, 4, 1, 64),
                          (2, 2, 1, 512), (2, 1, 1, 2), (5, 3, 1, 256)):
                with self.subTest(dtype=dtype, shape=shape):
                    self.check_case(*shape, dtype)

    def test_noncontiguous_outer_dimensions_and_empty(self):
        self.check_case(17, 5, 3, 128, torch.bfloat16, transposed=True)
        self.check_case(17, 5, 3, 128, torch.bfloat16, transposed=True, weight_dtype=torch.float32)
        self.check_case(0, 16, 8, 128, torch.float16)

    @torch.inference_mode()
    def test_cuda_graph_replay_uses_new_inputs_and_positions(self):
        q = torch.randn(1, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 8, 128, device="cuda", dtype=q.dtype)
        weight = torch.ones(128, device="cuda", dtype=q.dtype)
        rope = RotaryEmbedding(128, 128, 4096, 1000000).to("cuda")
        positions = torch.zeros(1, device="cuda", dtype=torch.int32)
        def run():
            return qk_norm_rope(q, k, weight, weight, rope.cos_sin_cache, positions)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for position in (1, 255, 256, 4095):
            q.normal_()
            k.normal_()
            positions.fill_(position)
            expected = run()
            graph.replay()
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    @torch.inference_mode()
    def test_zero_inputs(self):
        q = torch.zeros(3, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.zeros(3, 8, 128, device="cuda", dtype=q.dtype)
        weight = torch.ones(128, device="cuda", dtype=q.dtype)
        rope = RotaryEmbedding(128, 128, 32, 1000000).to("cuda")
        positions = torch.tensor([0, 1, 31], device="cuda")
        for result in qk_norm_rope(q, k, weight, weight, rope.cos_sin_cache, positions):
            self.assertEqual(torch.count_nonzero(result).item(), 0)

    def test_invalid_inputs_and_backward(self):
        q = torch.ones(1, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = q[:, :8]
        weight = torch.ones(128, device="cuda", dtype=q.dtype)
        rope = RotaryEmbedding(128, 128, 32, 1000000).to("cuda")
        positions = torch.zeros(1, device="cuda", dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            qk_norm_rope(q.requires_grad_(), k, weight, weight, rope.cos_sin_cache, positions)
        with self.assertRaisesRegex(ValueError, "Positions"):
            qk_norm_rope(q.detach(), k, weight, weight, rope.cos_sin_cache, positions.float())
        with self.assertRaisesRegex(ValueError, "Norm weights"):
            qk_norm_rope(q.detach(), k, weight, weight.float(), rope.cos_sin_cache, positions)


if __name__ == "__main__":
    unittest.main()
