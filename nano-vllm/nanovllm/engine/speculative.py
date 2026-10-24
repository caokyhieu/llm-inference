"""Sequential EAGLE-3: draft a chain, verify once, commit its matching prefix.

This path owns the target's cache while generate() runs. It deliberately avoids
the normal scheduler, shared prefix blocks, and trees. Optional CUDA graphs
capture fixed-shape forwards; acceptance and rollback stay CPU-controlled.
"""

from time import perf_counter

import torch
from tqdm.auto import tqdm

from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import set_context, reset_context


def verify_draft(proposals: list[int], target_tokens: list[int]) -> tuple[list[int], int]:
    """Keep matches, then emit the target's correction (or all-match bonus)."""
    if len(target_tokens) != len(proposals) + 1:
        raise ValueError("Verification needs one target choice per proposal plus a bonus")
    accepted = 0
    while accepted < len(proposals) and proposals[accepted] == target_tokens[accepted]:
        accepted += 1
    return proposals[:accepted] + [target_tokens[accepted]], accepted


def stop_at_eos(tokens: list[int], eos: int, ignore_eos: bool) -> list[int]:
    if not ignore_eos and eos in tokens:
        return tokens[:tokens.index(eos) + 1]
    return tokens


class SpeculativeDecoder:

    def __init__(self, runner, tokenizer):
        self.runner = runner
        self.target = runner.model
        self.draft = runner.draft_model
        self.tokenizer = tokenizer
        self.config = runner.config
        self.device = next(self.target.parameters()).device
        self.stats = []
        self.target_graphs = {}
        self.draft_graphs = {}
        self.graph_replays = {"target": 0, "draft": 0}
        if self.config.speculative_cuda_graphs:
            self._capture_graphs()

    @torch.inference_mode()
    def _capture_graphs(self):
        from nanovllm.engine.speculative_graph import CapturedSpeculativeForward

        def target(inputs, positions, features):
            hidden, auxiliary = self.target(inputs, positions, self.draft.capture_layers)
            return self.target.compute_logits(hidden, all_tokens=True), auxiliary

        def draft(inputs, positions, features):
            return self.draft(positions, self.target.model.embed_tokens(inputs), features)

        dtype = next(self.target.parameters()).dtype
        hidden = self.config.hf_config.hidden_size
        counts = range(1, min(self.config.num_speculative_tokens + 1, self.config.max_model_len) + 1)
        for count in counts:
            self.target_graphs[count] = CapturedSpeculativeForward(
                target, count, 0, self.config, self.device, dtype)
            self.draft_graphs[count, 3 * hidden] = CapturedSpeculativeForward(
                draft, count, 3 * hidden, self.config, self.device, dtype)
        self.draft_graphs[1, hidden] = CapturedSpeculativeForward(
            draft, 1, hidden, self.config, self.device, dtype)

    def close(self):
        self.target_graphs.clear()
        self.draft_graphs.clear()

    def _context(self, start: int, count: int):
        """Identity page table: position p lives at flat slot p in both caches."""
        end = start + count
        block_size = self.config.kvcache_block_size
        positions = torch.arange(start, end, device=self.device)
        q_lens = torch.tensor([0, count], dtype=torch.int32, device=self.device)
        k_lens = torch.tensor([0, end], dtype=torch.int32, device=self.device)
        blocks = torch.arange((end + block_size - 1) // block_size, dtype=torch.int32, device=self.device)[None]
        set_context(True, q_lens, k_lens, count, end, positions.to(torch.int32), block_tables=blocks)
        return positions

    def _target_forward(self, token_ids, start, prefill=False):
        if not prefill and len(token_ids) in self.target_graphs:
            self.graph_replays["target"] += 1
            return self.target_graphs[len(token_ids)].replay(token_ids, start)
        inputs = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        positions = self._context(start, len(token_ids))
        try:
            hidden, features = self.target(inputs, positions, self.draft.capture_layers)
            # Verification needs every row; prompt prefill needs only its last row.
            logits = self.target.compute_logits(hidden[-1:] if prefill else hidden, all_tokens=True)
            return logits, features
        finally:
            reset_context()

    def _draft_forward(self, token_ids, features, start):
        graph = self.draft_graphs.get((len(token_ids), features.shape[-1]))
        if graph is not None:
            self.graph_replays["draft"] += 1
            return graph.replay(token_ids, start, features)
        inputs = torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        positions = self._context(start, len(token_ids))
        try:
            return self.draft(positions, self.target.model.embed_tokens(inputs), features)
        finally:
            reset_context()

    def _sample_target(self, logits, temperature):
        if temperature == 0:
            return logits.argmax(dim=-1).tolist()
        temperatures = torch.full((logits.shape[0],), temperature, device=self.device)
        return self.runner.sampler(logits, temperatures).tolist()

    @torch.inference_mode()
    def _generate_one(self, prompt: list[int], params: SamplingParams):
        capacity = min(self.config.max_model_len,
                       self.config.num_kvcache_blocks * self.config.kvcache_block_size)
        if not prompt:
            raise ValueError("A speculative prompt must contain at least one token")
        if len(prompt) + params.max_tokens > capacity:
            raise ValueError(f"Prompt + max_tokens exceeds speculative cache/context capacity ({capacity})")
        eos = self.config.eos
        torch.cuda.synchronize()
        begin = perf_counter()
        logits, features = self._target_forward(prompt, 0, prefill=True)
        completion = self._sample_target(logits, params.temperature)
        prefill_end = perf_counter()
        # Pair feature[t] with token[t+1]. The last token is target-confirmed but
        # pending: it does not yet have target K/V. Draft positions are shifted by 1.
        repair_tokens = prompt[1:] + completion
        repair_features = features
        draft_start = 0
        target_length = len(prompt)
        rounds = []
        while len(completion) < params.max_tokens and (params.ignore_eos or completion[-1] != eos):
            remaining = params.max_tokens - len(completion)
            k = min(self.config.num_speculative_tokens, remaining - 1)
            proposals = []
            if k:
                # Repair speculative draft features using confirmed target features.
                hidden = self._draft_forward(repair_tokens, repair_features, draft_start)[-1:]
                for i in range(k):
                    proposal = self.draft.next_token(hidden).item()
                    proposals.append(proposal)
                    if i + 1 < k:
                        hidden = self._draft_forward([proposal], hidden, target_length + i)

            # Causal verification produces p(d1|prefix), p(d2|prefix,d1), ...,
            # plus p(bonus|prefix,d1,...,dk), in ONE target forward.
            logits, features = self._target_forward([completion[-1]] + proposals, target_length)
            choices = self._sample_target(logits, params.temperature)
            emitted, accepted = verify_draft(proposals, choices)
            emitted = stop_at_eos(emitted, eos, params.ignore_eos)
            rounds.append({
                "proposals": proposals, "target_choices": choices,
                "accepted": min(accepted, len(emitted)), "emitted": emitted,
                "target_cache_before": target_length,
                "target_cache_after": target_length + len(emitted),
            })
            completion.extend(emitted)
            # Rollback is logical: next attention sees only the committed prefix.
            # Old rejected slots are invisible and get overwritten on the next call.
            draft_start = target_length
            target_length += len(emitted)
            repair_tokens = emitted
            repair_features = features[:len(emitted)]

        torch.cuda.synchronize()
        end = perf_counter()
        proposed = sum(len(r["proposals"]) for r in rounds)
        accepted = sum(r["accepted"] for r in rounds)
        stats = {
            "prompt_tokens": len(prompt), "generated_tokens": len(completion),
            "target_forwards": 1 + len(rounds), "proposed_tokens": proposed,
            "accepted_tokens": accepted, "acceptance_rate": accepted / proposed if proposed else 0.,
            "prefill_seconds": prefill_end - begin, "decode_seconds": end - prefill_end,
            "decode_tokens_per_second": (len(completion) - 1) / (end - prefill_end),
            "rounds": rounds,
        }
        return completion, stats

    def generate(self, prompts, sampling_params, use_tqdm=True):
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("Provide one SamplingParams per prompt")
        self.stats = []
        outputs = []
        for prompt, params in tqdm(zip(prompts, sampling_params), total=len(prompts),
                                   desc="Speculating", disable=not use_tqdm):
            tokens = self.tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
            token_ids, stats = self._generate_one(tokens, params)
            self.stats.append(stats)
            outputs.append({"text": self.tokenizer.decode(token_ids), "token_ids": token_ids})
        return outputs
