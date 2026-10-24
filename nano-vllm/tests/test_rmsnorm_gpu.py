"""NANOVLLM_RUN_GPU_TESTS=1 enables CUDA correctness checks."""

import os
import unittest

import torch

from nanovllm.kernels.rms_norm import rms_norm
from nanovllm.layers.layernorm import RMSNorm


@unittest.skipUnless(os.environ.get("NANOVLLM_RUN_GPU_TESTS") == "1", "GPU tests are opt-in")
class TritonRMSNormTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        previous = torch._dynamo.config.cache_size_limit
        torch._dynamo.config.cache_size_limit = 128
        cls.addClassCleanup(setattr, torch._dynamo.config, "cache_size_limit", previous)

    @torch.inference_mode()
    def check_case(self, x, residual=None):
        torch.manual_seed(19)
        weight = torch.randn(x.shape[-1], dtype=x.dtype, device=x.device)
        old = RMSNorm(x.shape[-1]).to(device=x.device, dtype=x.dtype)
        old.weight.copy_(weight)
        saved = x.clone()
        saved_residual = residual.clone() if residual is not None else None
        actual = rms_norm(x, weight, residual=residual)
        expected = old(x, residual)
        if residual is not None:
            actual, actual_sum = actual
            expected, expected_sum = expected
            # This output must be the rounded SUM, not the normalized result.
            reference_sum = (saved.float() + saved_residual.float()).to(x.dtype)
            torch.testing.assert_close(actual_sum, reference_sum, rtol=0, atol=0)
            torch.testing.assert_close(actual_sum, expected_sum, rtol=0, atol=0)
            torch.testing.assert_close(residual, saved_residual, rtol=0, atol=0)
        reference = saved.float() + (saved_residual.float() if residual is not None else 0)
        reference = reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + 1e-6)
        # The existing compiled backend fuses away the intermediate dtype round.
        reference = (reference * weight.float()).to(x.dtype)
        eps = torch.finfo(x.dtype).eps
        torch.testing.assert_close(actual, expected, rtol=2 * eps, atol=2 * eps)
        torch.testing.assert_close(actual, reference, rtol=2 * eps, atol=2 * eps)
        torch.testing.assert_close(x, saved, rtol=0, atol=0)

    def test_contiguous_and_masked_widths(self):
        for dtype in (torch.float16, torch.bfloat16):
            for rows, width in ((1, 128), (257, 128), (1, 1024), (8, 1024), (257, 1024),
                                (33, 1537), (4, 4096), (2, 16384)):
                for fused in (False, True):
                    with self.subTest(dtype=dtype, shape=(rows, width), fused=fused):
                        x = torch.randn(rows, width, device="cuda", dtype=dtype)
                        self.check_case(x, torch.randn_like(x) if fused else None)

    def test_strided_qkv_views_and_residual(self):
        for dtype in (torch.float16, torch.bfloat16):
            qkv = torch.randn(17, 32, 128, device="cuda", dtype=dtype)
            q, k = qkv[:, :16], qkv[:, 16:24]
            self.check_case(q)
            self.check_case(k)
            self.check_case(k, torch.randn_like(k))
            # Exercise noncontiguous columns, outer axes, and different residual strides.
            x = torch.randn(3, 5, 7, 258, device="cuda", dtype=dtype).transpose(0, 2)[..., ::2]
            self.check_case(x, torch.randn(x.shape, device="cuda", dtype=dtype))

    @torch.inference_mode()
    def test_empty_vector_and_zero_inputs(self):
        self.check_case(torch.zeros(1, 1024, device="cuda", dtype=torch.bfloat16))
        self.check_case(torch.randn(1024, device="cuda", dtype=torch.float16))
        x = torch.empty(0, 1024, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(rms_norm(x, torch.ones(1024, device="cuda", dtype=x.dtype)).shape, x.shape)

    def test_backward_is_explicitly_unsupported(self):
        x = torch.ones(1, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            rms_norm(x, torch.ones(1024, device="cuda", dtype=x.dtype))


if __name__ == "__main__":
    unittest.main()
