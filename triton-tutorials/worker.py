"""Execute one upstream tutorial, keeping its downloaded source unchanged."""

import inspect
import json
from pathlib import Path
import runpy
import shutil
import sys
import traceback

import torch
import triton
import triton.testing


def configure_reports(number, mode):
    original_run = triton.testing.Mark.run
    benchmark_records = []

    def run(mark, *args, **kwargs):
        benchmarks = mark.benchmarks
        if not isinstance(benchmarks, list):
            benchmarks = [benchmarks]
        if mode == "quick":
            if number == 6:
                benchmarks = [b for b in benchmarks if b.args["HEAD_DIM"] == 64]
            for benchmark in benchmarks:
                benchmark.x_vals = benchmark.x_vals[:3]
                if number == 6:
                    benchmark.x_vals = [128, 256]
                    original_shape = f"batch{benchmark.args['BATCH']}-head{benchmark.args['H']}"
                    benchmark.plot_name = benchmark.plot_name.replace(original_shape, "batch1-head2")
                    benchmark.args.update(BATCH=1, H=2)
            mark.benchmarks = benchmarks
        for benchmark in benchmarks:
            benchmark_records.append({
                "name": benchmark.plot_name, "x_names": benchmark.x_names,
                "x_vals": benchmark.x_vals, "providers": benchmark.line_vals,
                "args": benchmark.args,
            })
        Path("benchmarks.json").write_text(json.dumps(benchmark_records, indent=2, default=str) + "\n")
        kwargs.update(show_plots=False, print_data=True, save_path=".")
        result = original_run(mark, *args, **kwargs)
        import matplotlib.pyplot as plt
        plt.close("all")
        return result

    triton.testing.Mark.run = run
    if mode == "quick":
        original_bench = triton.testing.do_bench
        signature = inspect.signature(original_bench)

        def do_bench(*args, **kwargs):
            bound = signature.bind_partial(*args, **kwargs)
            bound.arguments["warmup"] = min(bound.arguments.get("warmup", 25), 1)
            bound.arguments["rep"] = min(bound.arguments.get("rep", 100), 5)
            return original_bench(*bound.args, **bound.kwargs)

        triton.testing.do_bench = do_bench


def attention_checks(namespace):
    for head_dim in (64, 128):
        for causal in (False, True):
            namespace["test_op"](
                Z=1, H=2, N_CTX=128, HEAD_DIM=head_dim, causal=causal,
                warp_specialize=False, mode="bwd", provider="triton-fp16",
            )
            print(f"PASS attention FP16 forward/backward: head_dim={head_dim}, causal={causal}", flush=True)
    for causal in (False, True):
        namespace["test_op"](
            Z=1, H=2, N_CTX=128, HEAD_DIM=64, causal=causal,
            warp_specialize=False, mode="fwd", provider="triton-fp8",
        )
        print(f"PASS attention FP8 forward: causal={causal}", flush=True)
    namespace["test_fp8_v_batch_head_descriptor"]()
    print("PASS attention FP8 batch/head descriptor", flush=True)


def external_source(source):
    # This tutorial demonstrates an explicit path relative to a full source
    # checkout. Mirror that layout in our cache and use the wheel's bitcode.
    root = Path(__file__).resolve().parent / ".cache" / "upstream"
    mirrored = root / "python" / "tutorials" / source.name
    mirrored.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, mirrored)
    library = root / "third_party" / "nvidia" / "backend" / "lib" / "libdevice.10.bc"
    library.parent.mkdir(parents=True, exist_ok=True)
    library.unlink(missing_ok=True)
    library.symlink_to(Path(triton.__file__).parent / "backends" / "nvidia" / "lib" / "libdevice.10.bc")
    return mirrored


def main():
    source, mode = Path(sys.argv[1]), sys.argv[2]
    number = int(source.name[:2])
    details = {
        "status": "running", "mode": mode, "torch": torch.__version__,
        "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0),
        "compute_capability": torch.cuda.get_device_capability(0),
    }
    print(json.dumps(details), flush=True)
    try:
        if number == 10 and torch.cuda.get_device_capability(0)[0] not in (10, 11):
            details.update(status="skipped", reason="block-scaled matmul requires NVIDIA compute capability 10/11; H100 is 9.0")
            print(details["reason"], flush=True)
        else:
            configure_reports(number, mode)
            if number == 6:
                namespace = runpy.run_path(str(source), run_name="tutorial")
                # STAGE controls causal masking and which tile sizes are valid.
                # Upstream's tuning key omits it, allowing an invalid tile from
                # a non-causal run to be reused by a causal run in this process.
                tuner = namespace["_attn_fwd"]
                if "STAGE" not in tuner.keys:
                    tuner.keys = [*tuner.keys, "STAGE"]
                details["adjustments"] = ["Include causal STAGE in the attention autotuning key"]
                attention_checks(namespace)
                namespace["bench_flash_attention"].run()
            elif number == 9 and mode == "quick":
                namespace = runpy.run_path(str(source), run_name="tutorial")
                namespace["validate"](128, 128, 128, torch.float16)
                namespace["validate"](512, 512, 512, torch.float16)
                profiler = namespace["proton"]
                profiler.start("matmul", hook="triton")
                profiler.deactivate()
                try:
                    namespace["bench"](128, torch.float16, reps=3, warmup_reps=2)
                finally:
                    profiler.finalize()
                namespace["show_profile"]("fp16", "matmul")
            else:
                if number == 7:
                    source = external_source(source)
                    details["adjustments"] = ["Mirror source-checkout layout for installed libdevice bitcode"]
                sys.argv = [str(source)]
                runpy.run_path(str(source), run_name="__main__")
            torch.cuda.synchronize()
    except BaseException as exc:
        details.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    if details["status"] == "running":
        details["status"] = "passed"
    Path("status.json").write_text(json.dumps(details, indent=2) + "\n")
    raise SystemExit(1 if details["status"] == "failed" else 0)


if __name__ == "__main__":
    main()
