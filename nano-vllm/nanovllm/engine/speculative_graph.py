"""One fixed-query-length forward over a single sequence's paged KV cache.

Unlike ordinary decode graphs, rows here are positions in ONE sequence. The
maximum attention length is fixed, while GPU cumulative lengths carry the real
prefix length. Capture all shapes before serving: warmup writes dummy KV slots.
"""

import torch

from nanovllm.utils.context import set_context, reset_context


class CapturedSpeculativeForward:

    @torch.inference_mode()
    def __init__(self, function, count, feature_width, config, device, dtype):
        self.inputs = torch.zeros(count, dtype=torch.int64, device=device)
        self.offsets = torch.arange(count, device=device)
        self.positions = self.offsets.clone()
        self.q_lengths = torch.tensor([0, count], dtype=torch.int32, device=device)
        self.k_lengths = self.q_lengths.clone()
        blocks = (config.max_model_len + config.kvcache_block_size - 1) // config.kvcache_block_size
        self.blocks = torch.arange(blocks, dtype=torch.int32, device=device)[None]
        self.features = (torch.zeros(count, feature_width, device=device, dtype=dtype)
                         if feature_width else None)
        # Keep each capture's pool separate: target features can remain live
        # while a draft graph executes. Sharing their storage would corrupt them.
        self.graph = torch.cuda.CUDAGraph()
        set_context(True, self.q_lengths, self.k_lengths, count, config.max_model_len,
                    self.positions, block_tables=self.blocks)
        try:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    function(self.inputs, self.positions, self.features)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            with torch.cuda.graph(self.graph):
                self.outputs = function(self.inputs, self.positions, self.features)
        finally:
            reset_context()

    @torch.inference_mode()
    def replay(self, token_ids, start, features=None):
        self.inputs.copy_(torch.as_tensor(token_ids, dtype=torch.int64, device=self.inputs.device))
        torch.add(self.offsets, start, out=self.positions)
        self.k_lengths[1:].fill_(start + self.inputs.numel())
        if self.features is not None:
            self.features.copy_(features)
        self.graph.replay()
        # Graph-owned outputs are overwritten on its next replay. The decoder
        # consumes/copies them before then; callers retaining them must clone.
        return self.outputs
