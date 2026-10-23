# Triton tutorial lab

The ten Python examples under `tutorials/` were downloaded from the
[official tutorial gallery](https://triton-lang.org/main/getting-started/tutorials/).
They are unchanged. `SOURCE.json` records the archive URL and SHA-256.

This directory has its own Python 3.12 environment, PyTorch 2.14.1, and Triton
3.8.0. Nano-vLLM, quantization, and the VLM lab share the workspace-root PyTorch
2.6 / Triton 3.2 environment. This lab is excluded from that uv workspace because
its exact versions conflict with Nano-vLLM's compiled FlashAttention wheel.

## Run the examples

From this directory:

```bash
uv run --locked python run.py --list
uv run --locked python run.py all
uv run --locked python run.py 1 2 3
uv run --locked python run.py 6
```

The runner selects physical GPU 0 with PCI ordering. Use `--gpu 1` to select
another physical GPU. Each tutorial runs in a separate process and saves its
output under `results/quick/<tutorial-name>/`. Read `run.log`, the CSV tables,
the PNG charts, and any profiler output there. `results/quick/summary.json`
records passes, failures, hardware skips, versions, and elapsed times.

The default `--mode quick` retains the upstream kernels, autotuning, and
embedded correctness checks. It limits ordinary benchmark sweeps to their
first three points and reduces timing warmup/repetition durations. Attention
uses smaller benchmark inputs and adds explicit FP16 forward/backward and FP8
forward correctness checks. Persistent matmul validates two smaller sizes and
profiles the original FP16 benchmark with fewer repetitions. These short
measurements are for learning and checking execution, not definitive GPU
performance comparisons. `benchmarks.json` records the reduced report inputs.

Run the original full benchmark sweeps with:

```bash
uv run --locked python run.py all --mode full
```

Full runs retain the upstream timing settings and can take substantially
longer. Charts are saved rather than opening interactive windows. Use
`--timeout 7200` to increase the default one-hour limit per tutorial.

To execute an original source file directly, including its own CLI options:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg \
  uv run --locked python tutorials/09-persistent-matmul.py --prec fp8 -K 512
```

Direct execution retains upstream plotting/output behavior; the runner is
recommended for consistent output directories and headless plotting.

## Hardware coverage

| Tutorial | H100 support |
| --- | --- |
| 01 Vector addition | Supported |
| 02 Fused softmax | Supported |
| 03 Matrix multiplication | FP16 and FP8 |
| 04 Low-memory dropout | Supported |
| 05 Layer normalization | Forward and backward |
| 06 Fused attention | Supported Hopper variants; FP8 backward is not implemented upstream |
| 07 External libdevice functions | Supported |
| 08 Grouped GEMM | Supported, including TMA |
| 09 Persistent matmul | Hopper variants; CLC requires newer NVIDIA hardware |
| 10 Block-scaled matmul | Hardware skip: requires NVIDIA compute capability 10/11 or AMD CDNA4 |

## Recreate the environment

```bash
uv sync --locked
```

On this machine, the configured proxy can fail during downloads. Direct HTTPS
worked with certificate verification enabled:

```bash
NO_PROXY='*' no_proxy='*' uv sync --locked
```

`requirements.lock` is also provided for `uv pip install -r requirements.lock`.

## Runner adjustments

The attention tutorial's autotuning key does not include `STAGE`, even though
causal masking changes the valid tile sizes. The runner adds `STAGE` to the
key before running checks and benchmarks. This prevents a configuration tuned
for non-causal attention from being reused incorrectly for causal attention.
The existing numerical tolerances are unchanged.

The libdevice tutorial expects a full Triton checkout when constructing its
explicit bitcode-library path. The runner mirrors that directory layout under
`.cache/upstream/` and links to the installed Triton wheel's `libdevice.10.bc`.

Both adjustments are in `worker.py`; the official source files are unchanged.

## Verified run

On this machine's H100, the quick suite completed with **9 passed, 1 hardware
skip, and no failures**. This includes the attention forward/backward checks,
the grouped GEMM TMA implementation, and the supported persistent-matmul/TMA
variants. Tutorial 10 was skipped for the hardware reason listed above.
Full benchmark sweeps are available but have not been run as part of this setup.
