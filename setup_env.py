"""Prepare the pinned FlashAttention wheel and sync the shared uv environment."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parent
WHEEL_NAME = "flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
# The wheel's internal METADATA version is 2.7.3. Match it locally so uv does
# not repeatedly reinstall a filename-only local version suffix.
LOCAL_WHEEL_NAME = "flash_attn-2.7.3-cp312-cp312-linux_x86_64.whl"
WHEEL_URL = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/" + WHEEL_NAME
SHA256 = "944f5c6ba4ff1dd9261ddde56dac3299596b2d1e5110b2cff474fafe03cfd991"


def verify(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != SHA256:
        raise ValueError(f"FlashAttention wheel checksum mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Only prepare the wheel")
    args = parser.parse_args()
    wheel = ROOT / ".wheels" / LOCAL_WHEEL_NAME
    if not wheel.exists():
        wheel.parent.mkdir(exist_ok=True)
        temporary = wheel.with_suffix(".whl.part")
        cached = Path("/tmp/nanovllm-wheels") / WHEEL_NAME
        try:
            if cached.is_file():
                print("Reusing the previously downloaded FlashAttention wheel.", flush=True)
                shutil.copyfile(cached, temporary)
            else:
                print(f"Downloading the official FlashAttention wheel: {WHEEL_URL}", flush=True)
                with urllib.request.urlopen(WHEEL_URL, timeout=120) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
            verify(temporary)
            temporary.replace(wheel)
        finally:
            temporary.unlink(missing_ok=True)
    verify(wheel)
    print("FlashAttention wheel SHA-256 verified.", flush=True)
    if not args.prepare_only:
        subprocess.run(["uv", "sync", "--locked"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
