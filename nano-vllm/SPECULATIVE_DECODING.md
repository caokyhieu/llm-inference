# EAGLE-3 speculative decoding

This optional teaching implementation runs `Qwen/Qwen3-0.6B` with
[`GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge`](https://huggingface.co/GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge).
It uses one GPU, eager execution by default, and a linear chain of greedy draft proposals.
Prompt lists are processed sequentially. Ordinary `LLM(...)` retains its existing
scheduler, prefix caching, tensor parallelism, and graph behavior.

For optional captured target/draft forwards and a comparison against ordinary
CUDA-graph decoding, see [Speculative CUDA graphs](SPECULATIVE_CUDA_GRAPHS.md).

## Run the comparison

From `nano-vllm/`, using the existing environment:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python speculative_example.py
```

Both checkpoints are already in this machine's `../../hf_cache/`. The script uses
physical GPU 0, fixes the draft revision to
`5fe9c43a50078119bfa38e5f30969475be314315`, and launches baseline and EAGLE in
separate processes. It warms up, compares exact greedy token IDs on three prompts,
and writes output, timings, and verification traces under `speculative_results/`.
Reported tokens/s includes prefill and generation, excluding model load and warmup;
both modes use eager execution. It is not a comparison against CUDA-graph decoding.
The head reduces target forwards; actual wall-clock speed depends on acceptance
and draft/verification overhead, especially with this small target.

```bash
# Inspect a short trace for one prompt.
HF_HUB_OFFLINE=1 uv run --no-sync python speculative_example.py \
  --mode eagle --prompt 'List the prime numbers below 30.' --max-tokens 32

# Longer-output equality diagnostic (can flag BF16 differences; see below).
HF_HUB_OFFLINE=1 uv run --no-sync python speculative_example.py \
  --max-tokens 300 --ignore-eos --prompt 'Write a detailed guide to Python lists.' \
  --output-dir speculative_results/long

# The target may also sample; proposals remain greedy.
HF_HUB_OFFLINE=1 uv run --no-sync python speculative_example.py \
  --mode eagle --temperature 0.6

uv run --no-sync python -m unittest discover -s tests -p 'test_speculative.py' -v

# Independent target/draft reference checks and cache/EOS/budget tests on GPU 0.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
  NANOVLLM_RUN_GPU_TESTS=1 uv run --no-sync python -m unittest discover \
  -s tests -p 'test_eagle3_gpu.py' -v
```

Omit `HF_HUB_OFFLINE=1` to download on another machine. `--cache-dir`, `--gpu`,
`--draft-tokens`, and `--max-model-len` are configurable. The example disables
Qwen's thinking mode so the generation is easy to inspect.

## API

```python
from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams

cache = "../../hf_cache"
target = snapshot_download("Qwen/Qwen3-0.6B", cache_dir=cache, local_files_only=True)
draft = snapshot_download(
    "GavinLucky/SGLang-EAGLE3-Qwen3-0.6B-SpecForge",
    revision="5fe9c43a50078119bfa38e5f30969475be314315",
    cache_dir=cache, local_files_only=True,
)
llm = LLM(target, draft_model=draft, num_speculative_tokens=3,
          gpu_memory_utilization=0.12, max_model_len=1024,
          max_num_batched_tokens=1024)
try:
    outputs = llm.generate(["The capital of France is"],
                           SamplingParams(temperature=0, max_tokens=64))
    print(outputs[0]["text"])
    print(llm.speculative_stats[0]["acceptance_rate"])
finally:
    llm.exit()
```

`draft_model` is a local checkpoint directory. Speculation disables ordinary
batch-decode captures and sets the sequence limit to one. Its dedicated captures
are enabled separately with `speculative_cuda_graphs=True`. Tensor parallelism above one and queued
`add_request()` / `step()` calls are explicitly unsupported on this path. The
whole prompt and generation budget must fit `max_model_len` and allocated target
pages. There is no speculative prefix sharing or chunked prefill. The loader
supports this one-layer SpecForge layout with matching target hidden size and
vocabulary, rather than arbitrary EAGLE variants.

## Read the code in this order

1. `nanovllm/engine/speculative.py`: `_generate_one()` contains the complete loop;
   `verify_draft()` defines acceptance and the correction/bonus token.
2. `nanovllm/models/qwen3.py`: optional `capture_layers` returns concatenated
   **unnormalized** features after target layers **1, 13, 24** (zero based).
   The ordinary forward still returns its usual tensor.
3. `nanovllm/models/eagle3.py`: three features are projected from 3,072 to 1,024
   dimensions. The head concatenates normalized token embeddings and features,
   runs one attention/MLP layer, and predicts among 32,000 draft tokens. Embeddings
   are shared with the target. The checkpoint's `d2t` stores **offsets**, so target
   token IDs are `draft_id + d2t[draft_id]`.
4. `_context()` in `speculative.py`: both models reuse the existing Triton KV
   writer and paged FlashAttention. Each has independent K/V tensors.

## One decoding round

After target prefill, the target chooses one token `x`. It is confirmed, but its
K/V is still pending, just like the last token in ordinary nano-vLLM decoding.

```text
Draft:          propose a, b, c after confirmed token x
Target input:   [x, a, b, c] in one causal forward
Target choices: [a, z, ..., ...]
Commit:         a, z              (accept a, reject b and c)
Next pending:   z
```

If every proposal matches, also emit the target's final **bonus** choice. The
first mismatch ends acceptance even if later choices happen to match.

The draft is shifted by one token: target feature at position `t` pairs with the
embedding of token `t+1`. During rollout it feeds its own hidden state back in.
After verification, the next draft extend overwrites that approximation with
features from the confirmed target path. Skipping this repair hurts acceptance.

## KV rollback

For clarity, sequence position `p` occupies flat slot `p`: page `p // 256`, offset
`p % 256`. The page table is an identity mapping. A target verification writes
K/V for `x` and all proposals. If only `a` is accepted, the committed cache ends
after `x, a`; `z` remains pending. Rejected slots can retain old bytes because
subsequent attention lengths exclude them. The next forward overwrites those
slots. The draft similarly resumes from its committed length before repairing
features. No copy or zeroing of the whole cache is needed.

Each JSON round records proposals, target choices, emitted tokens, accepted count,
and target cache lengths before/after, so you can follow this directly.

## Sampling correctness and limits

At temperature zero, accept the longest prefix matching target argmax choices.
This implements greedy speculative decoding; token equality is checked against
ordinary decoding. Different matrix shapes can introduce floating-point rounding
differences near an argmax tie, so the comparison deliberately fails on a mismatch.

For positive temperature, draft proposals are still deterministic, and each
verification row independently samples the target distribution. Accept while the
sample equals the proposal; at the first disagreement, emit that target sample.
This preserves the target's autoregressive distribution: reaching a row means its
preceding draft tokens were accepted. It is the deterministic-draft special case
of speculative sampling, with potentially lower acceptance than a probabilistic
draft. Unused verification samples consume RNG state, so equal seeds do not
produce identical sampled sequences across the two modes.

EOS is checked inside the emitted block, and the proposal length shrinks near the
generation limit. These rules prevent returning tokens beyond EOS or `max_tokens`.
No SGLang, vLLM, training framework, or new package is required at runtime.

GPU reference tests compare target features to Transformers, the draft equations
to dense PyTorch SDPA, and rejected-slot overwrite to recomputing the confirmed
prefix. Existing nano-vLLM norms and activations fuse BF16 intermediates, while
eager reference operations round separately; these tests use per-token relative
vector error and cosine similarity, plus exact token decisions. The example's
baseline comparison requires exact equality of the complete greedy token stream.

## Observed results and numerical limit

On this machine's H100 PCIe, with PyTorch 2.6.0, BF16 weights, three draft tokens,
and the default three prompts (up to 64 output tokens each):

| Mode | Output tokens | Target forwards | Generation tokens/s |
| --- | ---: | ---: | ---: |
| Ordinary eager decoding | 134 | 134 | 56.0 |
| EAGLE-3 eager decoding | 134 | 60 | 86.1 |

All 134 output token IDs matched. Draft acceptance was 76/168 (45.2%). This is one
local run, including prefill and excluding load/warmup, not a general speed claim.
Machine-readable results and per-round traces are in `speculative_results/`.

The 300-token Python-lists diagnostic **did not match exactly**: its first
divergence was generated index 116, before the first KV page boundary. A replay
found batched verification assigned both token 68 (`e`) and token 4803 (`like`)
the BF16 logit 21.375. Single-query decoding on the same confirmed prefix gave
`like` 21.375 and `e` 21.25. Argmax therefore changed at a rounding-sensitive
decision. `speculative_results/long/numerics.json` records the diagnostic.

The implementation does not promise bitwise equivalence to single-token BF16
execution on every prompt. The comparison reports and fails on such differences;
it does not silently accept a changed greedy sequence. Separately, the GPU
page-boundary test verifies target and draft rejected-slot overwrite with exact
tensor equality using matching computation shapes, including stale data beyond
the committed length.

Architecture references: [SpecForge EAGLE-3](https://github.com/sgl-project/SpecForge/blob/main/specforge/modeling/draft/llama3_eagle.py),
[SGLang draft model](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/llama_eagle3.py),
[SGLang target feature selection](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3.py).
