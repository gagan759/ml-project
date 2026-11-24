"""Download the deployed model weights from the GitHub release and verify their SHA-256.

    python scripts/download_weights.py        # the served models (experiments/deployment.json)
    python scripts/download_weights.py --all  # + the original study's reference models (to re-run the benchmark)

Files that already exist with the right hash are skipped. Uses only the Python standard library.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tomatoleaf.weights import DEPLOYMENT, MODELS_DIR, fetch, files_for, is_present, load_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true",
                        help="also the original study's reference models (needed only to re-run the final benchmark)")
    parser.add_argument("--defaults", action="store_true", help=argparse.SUPPRESS)  # old flag; now the default
    args = parser.parse_args()
    manifest = load_manifest()
    deployment = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    files = manifest["files"] if args.all else files_for(None, manifest, deployment)
    failed = 0
    for f in files:
        if is_present(f, MODELS_DIR, verify=True):
            print(f"ok      {f['path']}")
            continue
        print(f"fetch   {f['path']}")
        try:
            fetch(f, MODELS_DIR, on_progress=lambda done, total: print(
                f"\r  {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True))
            print()
        except Exception as exc:  # network problems or a bad file: report and continue with the next one
            print(f"\n  FAILED: {exc}")
            failed += 1
    print("all weights present and verified" if not failed else f"{failed} file(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
