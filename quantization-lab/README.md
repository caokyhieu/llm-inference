# Qwen quantization lab

This is the quantization project from the inference-learning roadmap: compare
FP16, 8-bit LLM.int8, and 4-bit NF4 using the same cached Qwen3-0.6B checkpoint.
The lab uses the shared workspace-root `.venv` and `uv.lock`, together with
Nano-vLLM and the VLM lab. It uses Transformers and bitsandbytes;
nano-vLLM's existing linear layers assume ordinary floating-point weights.

## Start with the arithmetic

```bash
uv run --locked python toy_quantization.py
```

For simple symmetric, uniform quantization:

```text
scale = max(abs(weights)) / maximum_integer
integer_code = round(weight / scale)
reconstructed_weight = scale * integer_code
```

For example, `-0.3` becomes approximately `-0.299213` with the toy INT8 scale,
or `-0.285714` with its coarser INT4 scale. Rounding introduces error.
The toy uses Python lists to show the arithmetic; it does not pack their storage.
NF4 uses a non-uniform codebook, so its reconstruction differs from this toy.

## Run the model comparison

```bash
uv run --locked python compare.py
```

The default is physical GPU 0, batch size 1, 64 generated tokens per timing
repeat, and three repeats per format. The model must already be in
`../../hf_cache/`; the lab runs offline. Optional arguments include
`--gpu`, `--model`, `--cache-dir`, `--tokens`, `--repeats`, and `--output-dir`.

Each format runs in a fresh process. Results go to `results/`:

- `fp16.json`, `int8.json`, `nf4.json`: measurements, generated answers,
  versions, layer types, actual weight-storage dtypes, and KV-cache dtype.
- Matching `.log` files: loading and execution output.
- `summary.csv` and `summary.json`: comparison table.
- `comparison.png`: GPU allocation and decode throughput charts.

## Read the implementation in this order

1. `toy_quantization.py`: scale, rounding, and reconstruction error.
2. `measure.py`, `BitsAndBytesConfig`: select 8-bit or NF4 weight storage.
3. `measure.py`, `from_pretrained`: replace eligible linear layers during loading.
4. `measure.py`, `generate_fixed`: ordinary prefill and KV-cached decode.
5. Open the JSON reports and compare their `projection` and `kv_cache` fields.

| Mode | Linear layer | Weight tensor | Computation |
| --- | --- | --- | --- |
| FP16 | `Linear` | FP16 | FP16 |
| INT8 | `Linear8bitLt` | INT8 | LLM.int8 mixes INT8 with higher-precision outlier handling |
| NF4 | `Linear4bit` | Packed UINT8, two 4-bit codes per byte | NF4 weights with FP16 compute |

`dtype=torch.float16` sets the ordinary, unquantized modules' dtype.
`bnb_4bit_compute_dtype=torch.float16` sets the NF4 compute dtype. It does not
mean that LLM.int8 performs every operation in FP16.

## Interpret the measurements

Ideal weight storage is `number_of_weights * bits / 8` bytes. A quantized model
also stores scales and other metadata, and some modules remain unquantized.
The whole model therefore need not shrink by exactly 2x or 4x. The report's
example projection shows the actual packed weight tensor size; that field
excludes its separately stored quantization metadata.

Weight compression changes part of the total inference memory:

```text
inference memory = weights + KV cache + activations/workspaces
```

In this experiment the attention KV cache stays FP16 in all three formats.
Quantizing that cache is a separate experiment.

Smaller weights can reduce memory traffic. Quantized kernels also need to
interpret compressed values and apply scales, and LLM.int8 handles outliers.
The net speed depends on model size, shapes, batch size, and kernel support.
Use the measured latency alongside memory savings.

Measurement definitions:

- **GPU allocation after load:** PyTorch-allocated CUDA bytes after model loading,
  including quantization state. This excludes unrelated GPU processes and
  allocations outside PyTorch's allocator.
- **Peak inference allocation:** maximum PyTorch allocation during a timing
  repeat, including model storage, the KV cache, and temporary tensors.
- **TTFT:** wall time from an already-tokenized GPU input to its first greedy
  token, synchronized with the GPU. Tokenization and queueing are excluded.
- **Decode tokens/s:** subsequent tokens divided by their wall time, including
  this Python loop's dispatch and bookkeeping. This is not nano-vLLM throughput.
- **Load time:** local loading plus conversion/quantization; filesystem caching
  affects this measurement. The original checkpoint on disk is shared by all modes.

One warmup request precedes timing. The fixed-length timing loop ignores EOS
to give every mode equal work; separate answer checks honor EOS. Thinking is
disabled through Qwen3's chat template. Results summarize warmed inference;
initial kernel setup is outside the timed repeats.

## Check numerical changes and answers

`sample_text_loss` records next-token negative log-likelihood and perplexity on
one fixed, short text included in the source. Lower perplexity means higher
probability for that particular text. It is a sanity signal, not a general
quality benchmark or a representative evaluation dataset.

The three question-answer checks use strict normalized string matching and
save the full responses. An answer like `7 × 8 = 56` fails the requested
number-only format despite containing the correct number. Inspect answers
alongside the score; this count includes formatting compliance.

## Recreate the environment

First-time setup from this directory is `python ../setup_env.py`; see the
[shared setup](../README.md). Once prepared, the following command works from
this directory or the workspace root; both select the same environment.

```bash
uv sync --locked
```

If this machine's proxy interrupts downloads, direct HTTPS worked:

```bash
NO_PROXY='*' no_proxy='*' uv sync --locked
```

Official references: [Transformers bitsandbytes integration](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
and [bitsandbytes installation requirements](https://huggingface.co/docs/bitsandbytes/main/en/installation).
