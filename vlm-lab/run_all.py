"""Run the six small lessons; optionally include real Qwen internals."""

import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", action="store_true", help="Include optional Transformers lesson")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    lessons = sorted(directory.glob("[0-9][0-9]_*.py"))
    for lesson in lessons:
        if lesson.name.startswith("07_") and not args.qwen:
            continue
        print(f"\n=== {lesson.name} ===", flush=True)
        subprocess.run([sys.executable, str(lesson)], check=True)
    print("\nAll selected lessons completed successfully.")


if __name__ == "__main__":
    main()
