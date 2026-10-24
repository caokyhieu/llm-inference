"""Opt in with NANOVLLM_RUN_GPU_TESTS=1; requires the two cached checkpoints."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from nanovllm import LLM, SamplingParams


def reference_draft(draft, embeddings, features):
    """Independent dense SDPA equations, without the paged attention or RoPE code."""
    weights = draft.state_dict()
    config = draft.config

    def linear(x, name):
        return F.linear(x, weights[name + ".weight"])

    def norm(x, name):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + config["rms_norm_eps"])
        return normalized.to(x.dtype) * weights[name + ".weight"]

    hidden = linear(features, "fc")
    x = torch.cat((norm(embeddings, "midlayer.input_layernorm"),
                   norm(hidden, "midlayer.hidden_norm")), dim=-1)
    n, d = x.shape[0], config["head_dim"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    q = linear(x, "midlayer.self_attn.q_proj").view(n, heads, d)
    k = linear(x, "midlayer.self_attn.k_proj").view(n, kv_heads, d)
    v = linear(x, "midlayer.self_attn.v_proj").view(n, kv_heads, d)
    inv_freq = 1. / config["rope_theta"] ** (torch.arange(0, d, 2, device=x.device).float() / d)
    angles = torch.arange(n, device=x.device).float()[:, None] * inv_freq[None]

    def rotate(tensor):
        first, second = tensor.float().chunk(2, dim=-1)
        cos, sin = angles.cos()[:, None], angles.sin()[:, None]
        return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(tensor.dtype)

    q = rotate(q).transpose(0, 1)[None]
    k = rotate(k).repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)[None]
    v = v.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)[None]
    attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    attention = attention[0].transpose(0, 1).reshape(n, heads * d)
    hidden = hidden + linear(attention, "midlayer.self_attn.o_proj")
    x = norm(hidden, "midlayer.post_attention_layernorm")
    mlp = F.silu(linear(x, "midlayer.mlp.gate_proj")) * linear(x, "midlayer.mlp.up_proj")
    return hidden + linear(mlp, "midlayer.mlp.down_proj")


@unittest.skipUnless(os.environ.get("NANOVLLM_RUN_GPU_TESTS") == "1", "GPU tests are opt-in")
class EagleCheckpointTests(unittest.TestCase):

    def assert_bf16_vectors_close(self, actual, expected, max_relative_error=.04):
        # Existing compiled nano-vLLM norms/activations fuse BF16 intermediates;
        # eager HF/SDPA rounds them separately. Compare per-token vectors, not
        # relative element errors around zero, and check token decisions below.
        actual, expected = actual.float(), expected.float()
        relative_error = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-6)
        cosine = F.cosine_similarity(actual, expected, dim=-1)
        self.assertLess(relative_error.max().item(), max_relative_error)
        self.assertGreater(cosine.min().item(), .999)

    @classmethod
    def setUpClass(cls):
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        cache = os.environ.get("HF_HUB_CACHE", str(Path(__file__).resolve().parents[3] / "hf_cache"))
        model = snapshot_download("Qwen/Qwen3-0.6B", cache_dir=cache, local_files_only=True)
        draft = snapshot_download("GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge",
                                  revision="5fe9c43a50078119bfa38e5f30969475be314315",
                                  cache_dir=cache, local_files_only=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(model)
        cls.llm = LLM(model, draft_model=draft,
                      gpu_memory_utilization=float(os.environ.get("NANOVLLM_GPU_MEMORY_UTILIZATION", ".12")),
                      speculative_cuda_graphs=os.environ.get("NANOVLLM_SPECULATIVE_CUDA_GRAPHS") == "1",
                      rms_norm_backend=os.environ.get("NANOVLLM_RMS_NORM_BACKEND", "torch"),
                      qk_rope_backend=os.environ.get("NANOVLLM_QK_ROPE_BACKEND", "torch"),
                      max_model_len=1024, max_num_batched_tokens=1024)
        cls.addClassCleanup(cls.llm.exit)
        cls.decoder = cls.llm.speculator
        cls.reference = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to("cuda").eval()
        cls.prompt = cls.tokenizer.apply_chat_template(
            [{"role": "user", "content": "List the prime numbers below 30."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.reference
        cls.llm.exit()

    @torch.inference_mode()
    def test_target_auxiliary_states_against_transformers(self):
        self.assertEqual(self.decoder.draft.capture_layers, (1, 13, 24))
        _, actual = self.decoder._target_forward(self.prompt, 0, prefill=True)
        output = self.reference(torch.tensor([self.prompt], device="cuda"),
                                output_hidden_states=True, logits_to_keep=1)
        for j, i in enumerate(self.decoder.draft.capture_layers):
            self.assert_bf16_vectors_close(actual[:, j * 1024:(j + 1) * 1024],
                                          output.hidden_states[i + 1][0])

    @torch.inference_mode()
    def test_draft_dense_reference_and_rejected_slot_overwrite(self):
        logits, features = self.decoder._target_forward(self.prompt, 0, prefill=True)
        # Real target features exercise the distribution the checkpoint was trained on.
        inputs = torch.tensor(self.prompt[1:] + logits.argmax(-1).tolist(), device="cuda")
        embeddings = self.decoder.target.model.embed_tokens(inputs)
        expected = reference_draft(self.decoder.draft, embeddings, features)
        full = self.decoder._draft_forward(inputs.tolist(), features, 0)
        self.assert_bf16_vectors_close(full, expected, .02)
        self.assertEqual(self.decoder.draft.next_token(full).tolist(),
                         self.decoder.draft.next_token(expected).tolist())
        self.decoder._draft_forward(inputs[:16].tolist(), features[:16], 0)
        # Write a rejected branch, then replace it with the correct branch.
        self.decoder._draft_forward([0] * 5, -features[16:], 16)
        actual = self.decoder._draft_forward(inputs[16:].tolist(), features[16:], 16)
        self.assert_bf16_vectors_close(actual, full[16:], .02)
        self.assertEqual(self.decoder.draft.next_token(actual).tolist(),
                         self.decoder.draft.next_token(expected[16:]).tolist())

    @torch.inference_mode()
    def test_target_rejected_slot_overwrite(self):
        prefix = self.prompt
        suffix = self.tokenizer.encode("The answer is 2, 3, 5.")
        self.decoder._target_forward(prefix, 0, prefill=True)
        self.decoder._target_forward([0] * len(suffix), len(prefix))
        actual, _ = self.decoder._target_forward(suffix, len(prefix))
        full, _ = self.decoder._target_forward(prefix + suffix, 0)
        self.assert_bf16_vectors_close(actual, full[-len(suffix):])
        self.assertEqual(actual.argmax(-1).tolist(), full[-len(suffix):].argmax(-1).tolist())
        expected = self.reference(torch.tensor([prefix + suffix], device="cuda")).logits[0, -len(suffix):]
        self.assertEqual(actual.argmax(-1).tolist(), expected.argmax(-1).tolist())

    @torch.inference_mode()
    def test_rollback_across_page_boundary_is_exact(self):
        # Both executions have the same matmul/query shapes, isolating rollback
        # from the BF16 rounding differences of batched vs tokenwise decoding.
        prefix = (self.prompt * 12)[:250]
        suffix = self.prompt[:8]
        logits, prefix_features = self.decoder._target_forward(prefix, 0, prefill=True)
        expected, suffix_features = self.decoder._target_forward(suffix, 250)
        self.decoder._target_forward(prefix, 0, prefill=True)
        self.decoder._target_forward([0] * 16, 250)  # Stale tail in page 1.
        actual, _ = self.decoder._target_forward(suffix, 250)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

        shifted = prefix[1:] + logits.argmax(-1).tolist()
        self.decoder._draft_forward(shifted, prefix_features, 0)
        expected = self.decoder._draft_forward(suffix, suffix_features, 250)
        self.decoder._draft_forward(shifted, prefix_features, 0)
        self.decoder._draft_forward([0] * 16, -suffix_features.repeat(2, 1), 250)
        actual = self.decoder._draft_forward(suffix, suffix_features, 250)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_budgets_draft_lengths_and_eos(self):
        params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
        expected = self.llm.generate([self.prompt], params, use_tqdm=False)[0]["token_ids"]
        old_k = self.decoder.config.num_speculative_tokens
        old_eos = self.decoder.config.eos
        try:
            for k in (1, 3, 5):
                self.decoder.config.num_speculative_tokens = k
                for budget in (1, 2, 4, 8):
                    with self.subTest(k=k, max_tokens=budget):
                        result = self.llm.generate([self.prompt], SamplingParams(
                            temperature=0, max_tokens=budget, ignore_eos=True,
                        ), use_tqdm=False)[0]["token_ids"]
                        self.assertEqual(result, expected[:budget])
            for index in (0, 2):
                self.decoder.config.eos = expected[index]
                result = self.llm.generate([self.prompt], SamplingParams(
                    temperature=0, max_tokens=32,
                ), use_tqdm=False)[0]["token_ids"]
                self.assertEqual(result, expected[:expected.index(expected[index]) + 1])
        finally:
            self.decoder.config.num_speculative_tokens = old_k
            self.decoder.config.eos = old_eos

    def test_sampled_generation_and_context_validation(self):
        result = self.llm.generate([self.prompt], SamplingParams(
            temperature=.6, max_tokens=8, ignore_eos=True,
        ), use_tqdm=False)[0]["token_ids"]
        self.assertEqual(len(result), 8)
        self.assertTrue(all(0 <= token < self.reference.config.vocab_size for token in result))
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.llm.generate([self.prompt], SamplingParams(max_tokens=1024), use_tqdm=False)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.llm.generate([[]], SamplingParams(), use_tqdm=False)

    def test_sampler_handles_mixed_greedy_and_sampled_rows(self):
        logits = torch.tensor([[0., 5., 1.], [-100., -100., 100.]], device="cuda")
        temperatures = torch.tensor([0., .6], device="cuda")
        tokens = self.llm.model_runner.sampler(logits, temperatures)
        self.assertEqual(tokens.tolist(), [1, 2])

    @torch.inference_mode()
    def test_graph_verification_replay_rollback_and_eager_agreement(self):
        if not self.decoder.target_graphs:
            self.skipTest("Enable NANOVLLM_SPECULATIVE_CUDA_GRAPHS=1")
        decoder = self.decoder
        for start in (len(self.prompt), 254):
            prefix = (self.prompt * 32)[:start]
            for count in (1, 2, 3, 4):
                suffix = self.prompt[:count]
                decoder._target_forward(prefix, 0, prefill=True)
                logits, features = decoder._target_forward(suffix, start)
                logits, features = logits.clone(), features.clone()
                # Leave rejected values beyond the accepted prefix, including
                # a page crossing. Replay must obey the new GPU sequence length.
                decoder._target_forward([0] * 8, start)
                actual, actual_features = decoder._target_forward(suffix, start)
                torch.testing.assert_close(actual, logits, atol=0, rtol=0)
                torch.testing.assert_close(actual_features, features, atol=0, rtol=0)
                with patch.object(decoder, "target_graphs", {}):
                    decoder._target_forward(prefix, 0, prefill=True)
                    eager, eager_features = decoder._target_forward(suffix, start)
                self.assert_bf16_vectors_close(logits, eager)
                self.assert_bf16_vectors_close(features, eager_features)
                self.assertEqual(logits.argmax(-1).tolist(), eager.argmax(-1).tolist())

    @torch.inference_mode()
    def test_graph_draft_repair_and_rollout_match_eager(self):
        if not self.decoder.draft_graphs:
            self.skipTest("Enable NANOVLLM_SPECULATIVE_CUDA_GRAPHS=1")
        decoder = self.decoder
        logits, features = decoder._target_forward(self.prompt, 0, prefill=True)
        inputs = self.prompt[1:] + logits.argmax(-1).tolist()

        def run():
            decoder._draft_forward(inputs[:-4], features[:-4], 0)
            repaired = decoder._draft_forward(inputs[-4:], features[-4:], len(inputs) - 4)
            repaired = repaired.clone()
            token = decoder.draft.next_token(repaired[-1:]).item()
            first = decoder._draft_forward([token], repaired[-1:], len(inputs)).clone()
            token = decoder.draft.next_token(first).item()
            second = decoder._draft_forward([token], first, len(inputs) + 1).clone()
            return repaired, first, second

        actual = run()
        with patch.object(decoder, "draft_graphs", {}):
            expected = run()
        for a, b in zip(actual, expected):
            self.assert_bf16_vectors_close(a, b, .02)
            self.assertEqual(decoder.draft.next_token(a).tolist(), decoder.draft.next_token(b).tolist())


if __name__ == "__main__":
    unittest.main()
