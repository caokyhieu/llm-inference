# CUDA graphs for EAGLE-3

`speculative_cuda_graphs=True` enables dedicated captures for speculative
verification and draft forwards. This keeps the existing sequential acceptance
and cache rollback algorithm. It is independent of ordinary batch-decode graphs.

## Measured result

On physical GPU 5 (H100 PCIe), Qwen3-0.6B BF16, using fused Q/K norm + RoPE in
every case:

| Mode | Tokens/s | Observed range | Ratio to ordinary CUDA graphs |
| --- | ---: | ---: | ---: |
| Ordinary CUDA graphs | 449.5 | 445.7–450.4 | 1.000× |
| EAGLE eager | 87.1 | 83.4–87.8 | 0.194× |
| EAGLE CUDA graphs | 622.0 | 611.8–624.4 | 1.384× |

Speculative graphs improved throughput by 38.4% over the current ordinary graph
path and by 7.14× over eager EAGLE. All full 64-token streams matched exactly
across modes and trials. Both EAGLE modes had the same 38.8% draft acceptance and
552 total target forwards across timed samples. The graph mode recorded 534
verification replays and 1,530 draft replays, confirming the captured paths ran.

These measurements cover three prompts and batch-one generation on a shared
GPU, with other resident allocations and intermittent utilization in the GPU
snapshots. They do not establish gains for larger batches or longer contexts.
The ordinary baseline captures its decoder; speculative verification captures
also include the LM head. Capture coverage is therefore part of this comparison.
The new backend remains opt-in.

## Enable it

```python
llm = LLM(
    target_path,
    draft_model=draft_path,
    num_speculative_tokens=3,
    speculative_cuda_graphs=True,
    qk_rope_backend="triton",
    max_model_len=1024,
    max_num_batched_tokens=1024,
)
```

The new flag defaults to `False`. Speculation still sets the runner's
`enforce_eager=True` to disable its ordinary batch-decode captures. The speculator
owns the separate graphs enabled by this flag; `enforce_eager` does not disable
them. Ordinary `LLM` usage and scheduling are unchanged.

The existing example accepts the flag:

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python speculative_example.py \
  --mode eagle --speculative-cuda-graphs --qk-rope-backend triton
```

Use a GPU/memory budget appropriate for the machine. The comparison benchmark
below can choose a budget based on current free GPU memory.

## Captured work

For three draft tokens, startup creates nine captures:

| Graphs | Input shapes | Captured operations |
| --- | --- | --- |
| Target verification | 1, 2, 3, or 4 query positions | Target decoder, auxiliary features, and vocabulary projection for every query |
| Draft feature repair | 1, 2, 3, or 4 tokens with 3H target features | Embedding lookup, feature projection, draft decoder |
| Draft rollout | One token with H draft features | Embedding lookup and draft decoder |

Long prompt prefill and initial draft extension stay eager. Shapes not captured
at startup fall back to eager execution, including longer repair calls. For a
different configured draft length, startup captures query counts up to K+1.

Target sampling, draft vocabulary projection/argmax, per-proposal `.item()`,
acceptance, EOS handling, statistics and logical cache-length updates remain
outside capture. These graphs accelerate the model forwards while preserving
the readable CPU loop; the entire speculative round is not one GPU graph.

## How replay stays correct

[speculative_graph.py](nanovllm/engine/speculative_graph.py) holds persistent token,
position, sequence-length, block-table and optional feature buffers. Every replay
copies new inputs, updates positions and the GPU cumulative key length, then
launches the captured forward.

Attention retains the existing paged, variable-length FlashAttention path. Its
Python maximum key length is fixed to `max_model_len` during capture. GPU
cumulative lengths contain the actual live prefix length. This allows growing
and rolled-back prefixes without recording a new graph or exposing rejected slots.
Query rows belong to one sequence, rather than independent batch members.

Capture warmup writes dummy K/V slots, so every capture is built before any user
request starts. Each graph owns a separate memory pool: target features must
remain live while a draft graph executes. Sharing their storage could overwrite
those features. Output tensors are graph-owned and overwritten on that graph's
next replay; the decoder consumes/copies them before reuse. Debugging code that
retains results across replays must clone them.

The existing cache rollback remains logical. The next attention call uses the
committed length; stale speculative slots are hidden and overwritten later.
[speculative.py](nanovllm/engine/speculative.py) selects the graph by query count
and draft feature width, leaving the acceptance algorithm unchanged.

## Compare against ordinary CUDA graphs

```bash
# Physical GPU 5 for the measured run; choose an available GPU on your machine.
HF_HUB_OFFLINE=1 uv run --no-sync python -u benchmark_speculative_graphs.py --gpu 5

# Original Q/K implementation in all three cases.
HF_HUB_OFFLINE=1 uv run --no-sync python benchmark_speculative_graphs.py \
  --gpu 5 --qk-rope-backend torch --output-dir speculative_graph_results/original_qk

# Rebuild the saved default run's report without GPU work.
uv run --no-sync python benchmark_speculative_graphs.py --gpu 5 --summarize-only
```

[benchmark_speculative_graphs.py](benchmark_speculative_graphs.py) reuses the
existing model benchmark worker. It compares ordinary CUDA graphs, eager EAGLE,
and EAGLE with graphs in fresh processes, reversing case order on the second
trial. All use the same Q/K backend (Triton by default), original hidden-state
norms, checkpoint, three chat prompts, and batch-one generation workload.

Each request generates 64 greedy tokens with EOS ignored for a fixed workload.
The exact prompts are tokenized before timing and warmed before measurement.
Three repeats per trial produce six three-prompt timing samples per case.
Throughput includes prefill, CPU engine work, and output decoding. Model loading,
capture, warmup and input tokenization are excluded. Ordinary graphs currently
capture the decoder; their LM head and token selection execute afterward. The
speculative verification captures include the LM head.

The benchmark reports exact tokens through EOS and across the full forced
continuation, acceptance, target forwards and actual graph replay counts. It
checks source hashes and workload settings before aggregation. Raw timings and
GPU utilization/memory snapshots are retained; this is a shared machine.

See [REPORT.md](speculative_graph_results/REPORT.md),
[summary.json](speculative_graph_results/summary.json), and
[comparison.csv](speculative_graph_results/comparison.csv) for measured results.

## Validation

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 \
  NANOVLLM_RUN_GPU_TESTS=1 NANOVLLM_SPECULATIVE_CUDA_GRAPHS=1 \
  NANOVLLM_QK_ROPE_BACKEND=triton NANOVLLM_GPU_MEMORY_UTILIZATION=.82 \
  uv run --no-sync python -m unittest discover -s tests -v
```

The memory fraction above was appropriate for GPU 5's existing allocations;
adjust it for your GPU. All 25 tests passed in this configuration. New checks
compare graphed target verification and draft repair/rollout against eager
execution, vary live prefix lengths, and test exact rejected-slot overwrites
across a page boundary. Existing target/HF and draft/SDPA reference checks,
sampling, budgets, EOS and kernel tests also pass. The original Q/K backend
additionally passed the seven pre-existing EAGLE integration checks with graphs.
