"""Run the downloaded official tutorials in separate GPU processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TUTORIALS = sorted((ROOT / "tutorials").glob("[0-9][0-9]-*.py"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tutorials", nargs="*", default=["all"], help="all, or numbers such as 1 2 6")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--gpu", default="0", help="physical GPU index with PCI ordering")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed per tutorial")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list:
        for source in TUTORIALS:
            print(source.stem)
        return
    if args.tutorials == ["all"]:
        selected = TUTORIALS
    else:
        try:
            numbers = {int(value) for value in args.tutorials}
        except ValueError:
            parser.error("choose 'all' or tutorial numbers from 1 to 10")
        if not numbers or not numbers <= {int(p.name[:2]) for p in TUTORIALS}:
            parser.error("tutorial numbers must be from 1 to 10")
        selected = [p for p in TUTORIALS if int(p.name[:2]) in numbers]

    result_root = ROOT / "results" / args.mode
    result_root.mkdir(parents=True, exist_ok=True)
    summary_path = result_root / "summary.json"
    environment = os.environ.copy()
    environment.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=args.gpu,
        MPLBACKEND="Agg", PYTHONUNBUFFERED="1",
        TRITON_CACHE_DIR=str(ROOT / ".cache" / "triton"),
    )
    failed = False
    for source in selected:
        destination = result_root / source.stem
        destination.mkdir(exist_ok=True)
        status_file = destination / "status.json"
        status_file.unlink(missing_ok=True)
        started = time.monotonic()
        print(f"RUN  {source.stem} ({args.mode})", flush=True)
        with (destination / "run.log").open("w") as log:
            try:
                process = subprocess.run(
                    [sys.executable, str(ROOT / "worker.py"), str(source), args.mode],
                    cwd=destination, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
                status = json.loads(status_file.read_text()) if status_file.exists() else {
                    "status": "failed", "reason": f"worker exited with code {process.returncode}"
                }
                if process.returncode != 0:
                    status["status"] = "failed"
            except subprocess.TimeoutExpired:
                status = {"status": "failed", "reason": f"exceeded {args.timeout}s timeout"}
        output = (destination / "run.log").read_text(errors="replace")
        if "❌" in output:
            status = {"status": "failed", "reason": "an upstream numerical comparison reported a mismatch"}
        status.update(seconds=round(time.monotonic() - started, 2), log=str(destination / "run.log"))
        status_file.write_text(json.dumps(status, indent=2) + "\n")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary[source.stem] = status
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"{status['status'].upper():5} {source.stem}: {status['seconds']}s {status.get('reason', '')}", flush=True)
        if status["status"] == "failed":
            failed = True
            print(output[-5000:], flush=True)
    print(f"Results: {summary_path}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
