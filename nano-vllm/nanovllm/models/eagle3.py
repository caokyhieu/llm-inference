"""The one-layer SpecForge EAGLE-3 head; embeddings come from the target.

Checkpoint layout and equations:
https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/llama3_eagle.py
"""

import json
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import get_rope


class EagleAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        h, d = config["hidden_size"], config["head_dim"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = d
        # EAGLE concatenates normalized token embeddings and hidden features.
        self.q_proj = nn.Linear(2 * h, self.num_heads * d, bias=False)
        self.k_proj = nn.Linear(2 * h, self.num_kv_heads * d, bias=False)
        self.v_proj = nn.Linear(2 * h, self.num_kv_heads * d, bias=False)
        self.o_proj = nn.Linear(self.num_heads * d, h, bias=False)
        self.rotary_emb = get_rope(d, d, config["max_position_embeddings"], config["rope_theta"])
        self.attn = Attention(self.num_heads, d, d ** -0.5, self.num_kv_heads)

    def forward(self, positions, x):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v).flatten(1))


class EagleMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        h, intermediate = config["hidden_size"], config["intermediate_size"]
        self.gate_proj = nn.Linear(h, intermediate, bias=False)
        self.up_proj = nn.Linear(h, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, h, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class EagleDecoderLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        h, eps = config["hidden_size"], config["rms_norm_eps"]
        self.input_layernorm = RMSNorm(h, eps)
        self.hidden_norm = RMSNorm(h, eps)
        self.post_attention_layernorm = RMSNorm(h, eps)
        self.self_attn = EagleAttention(config)
        self.mlp = EagleMLP(config)

    def forward(self, positions, embeddings, hidden):
        x = torch.cat((self.input_layernorm(embeddings), self.hidden_norm(hidden)), dim=-1)
        hidden = hidden + self.self_attn(positions, x)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Eagle3(nn.Module):

    def __init__(self, path: str, target_config, max_model_len: int, block_size: int):
        super().__init__()
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        if (config.get("architectures") != ["LlamaForCausalLMEagle3"]
                or config["num_hidden_layers"] != 1
                or config["hidden_size"] != target_config.hidden_size
                or config["vocab_size"] != target_config.vocab_size
                or config.get("norm_output", False)
                or config.get("fc_norm", False)
                or config.get("rope_scaling") is not None
                or config.get("attention_bias", False)
                or config.get("mlp_bias", False)):
            raise ValueError("Expected a one-layer SpecForge EAGLE-3 head matching the target hidden size and vocabulary")
        if max_model_len > config["max_position_embeddings"]:
            raise ValueError("max_model_len exceeds the draft's position limit")
        self.config = config
        n = target_config.num_hidden_layers
        # SGLang's defaults capture BEFORE layers [2, n//2, n-3].
        self.capture_layers = (1, n // 2 - 1, n - 4)
        if len(set(self.capture_layers)) != 3 or min(self.capture_layers) < 0:
            raise ValueError("Target needs three distinct EAGLE-3 feature layers")
        h = config["hidden_size"]
        self.fc = nn.Linear(3 * h, h, bias=False)
        self.midlayer = EagleDecoderLayer(config)
        self.norm = RMSNorm(h, config["rms_norm_eps"])
        self.lm_head = nn.Linear(h, config["draft_vocab_size"], bias=False)
        self.register_buffer("d2t", torch.empty(config["draft_vocab_size"], dtype=torch.int64))
        self.register_buffer("t2d", torch.empty(config["vocab_size"], dtype=torch.bool))
        # Strict loading catches missing tensors and incompatible checkpoint variants.
        self.load_state_dict(load_file(str(path / "model.safetensors"), device="cpu"), strict=True)
        mapping = self.d2t + torch.arange(config["draft_vocab_size"], device=self.d2t.device)
        if mapping.min().item() < 0 or mapping.max().item() >= config["vocab_size"]:
            raise ValueError("Invalid draft-to-target vocabulary mapping")
        self.register_buffer("target_token_ids", mapping, persistent=False)
        blocks = (max_model_len + block_size - 1) // block_size
        shape = (blocks, block_size, config["num_key_value_heads"], config["head_dim"])
        self.midlayer.self_attn.attn.k_cache = torch.empty(shape)
        self.midlayer.self_attn.attn.v_cache = torch.empty(shape)

    def forward(self, positions, embeddings, hidden):
        # First extend uses target features (3H); rollout feeds back draft states (H).
        if hidden.shape[-1] == self.fc.in_features:
            hidden = self.fc(hidden)
        return self.midlayer(positions, embeddings, hidden)

    def next_token(self, hidden):
        draft_id = self.lm_head(self.norm(hidden)).argmax(dim=-1)
        return self.target_token_ids[draft_id]
