import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    draft_model: str | None = None
    num_speculative_tokens: int = 3
    rms_norm_backend: str = "torch"
    qk_rope_backend: str = "torch"
    speculative_cuda_graphs: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        if self.rms_norm_backend not in ("torch", "triton"):
            raise ValueError("rms_norm_backend must be 'torch' or 'triton'")
        if self.qk_rope_backend not in ("torch", "triton"):
            raise ValueError("qk_rope_backend must be 'torch' or 'triton'")
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculative_cuda_graphs and self.draft_model is None:
            raise ValueError("speculative_cuda_graphs requires a draft_model")
        if self.draft_model is not None:
            if not os.path.isdir(self.draft_model):
                raise ValueError("draft_model must be a local EAGLE-3 checkpoint directory")
            if self.tensor_parallel_size != 1:
                raise ValueError("EAGLE-3 currently supports tensor_parallel_size=1")
            if self.num_speculative_tokens < 1:
                raise ValueError("num_speculative_tokens must be positive")
            # Ordinary batch-decode graphs do not represent speculative query
            # positions. SpeculativeDecoder owns its optional dedicated captures.
            self.enforce_eager = True
            self.max_num_seqs = 1
