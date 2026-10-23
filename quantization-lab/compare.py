"""Compare three weight formats on the same local Qwen checkpoint."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default=str(ROOT.parent.parent / "hf_cache"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=["fp16", "int8", "nf4"], default=["fp16", "int8", "nf4"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.tokens < 2 or args.repeats < 1:
        parser.error("--tokens must be at least 2 and --repeats must be positive")
    from huggingface_hub import snapshot_download
    path = Path(args.model).expanduser()
    model = str(path.resolve()) if path.is_dir() else snapshot_download(args.model, cache_dir=args.cache_dir, local_files_only=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1", MPLBACKEND="Agg")
    rows, failed = [], False
    for mode in args.modes:
        print(f"RUN {mode} on GPU {args.gpu}", flush=True)
        target = output / f"{mode}.json"
        target.unlink(missing_ok=True)
        with (output / f"{mode}.log").open("w") as log:
            process = subprocess.run(
                [sys.executable, str(ROOT / "measure.py"), "--mode", mode, "--model", model,
                 "--tokens", str(args.tokens), "--repeats", str(args.repeats), "--output", str(target)],
                env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
        if process.returncode:
            failed = True
            print(f"FAILED {mode}:\n{(output / f'{mode}.log').read_text(errors='replace')[-5000:]}", flush=True)
            continue
        report = json.loads(target.read_text())
        row = {key: report[key] for key in ("mode", "load_allocated_mib", "peak_inference_allocated_mib", "ttft_ms", "decode_tokens_per_second", "load_seconds", "exact_match_correct", "exact_match_total")}
        row["sample_text_perplexity"] = report["sample_text_loss"]["perplexity"]
        rows.append(row)
        print(f"PASS {mode}: {row['load_allocated_mib']:.1f} MiB, {row['decode_tokens_per_second']:.1f} decode tokens/s", flush=True)
    if rows:
        with (output / "summary.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
        for ax, field, label in zip(axes, ("load_allocated_mib", "decode_tokens_per_second"), ("GPU allocation after loading (MiB)", "Decode tokens/s (median)")):
            bars = ax.bar([row["mode"].upper() for row in rows], [row[field] for row in rows])
            ax.bar_label(bars, fmt="%.1f", padding=3)
            ax.set_ylabel(label)
            ax.margins(y=0.2)
        fig.suptitle(f"{args.model}: one GPU, batch size 1")
        fig.savefig(output / "comparison.png", dpi=150)
        plt.close(fig)
    print(f"Results: {output}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
