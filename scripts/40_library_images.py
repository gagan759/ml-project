"""Fix for audit root cause RC5 (docs/AUDIT.md) — rebuild the disease-library images without changing their pixels.

scripts/03 saved frontend/public/samples/<key>-<n>.jpg by decoding and re-encoding at JPEG q92 WITHOUT the Display P3
colour profile, so the upload path treated them as sRGB and converted their colours (docs/AUDIT.md §5). Here every
library image is replaced by its own dataset file with only the metadata segments removed (lossless; the colour
profile is kept). Same photos as before (the mapping comes from scripts/34's audit, by nearest pixels).
Checks: decoded pixels equal the dataset file's; the API's upload path reports "Display P3 (unchanged)".
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from tomatoleaf.config import DATASET_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.preprocessing import load_canonical_array, prepare_upload, strip_jpeg_metadata  # noqa: E402
from tomatoleaf.utils import load_json  # noqa: E402


def main():
    audit = RESULTS_DIR / "audit_examples_before.json"
    rows = load_json(audit if audit.exists() else RESULTS_DIR / "audit_examples.json")["rows"]
    mapping = sorted({(r["file"], r["dataset_file"]) for r in rows if r["set"] == "library"})
    assert len(mapping) == 24, len(mapping)
    for web, rel in mapping:
        data = strip_jpeg_metadata((DATASET_DIR / rel).read_bytes())
        dst = ROOT / web
        dst.write_bytes(data)
        arr, info = prepare_upload(data)
        assert np.array_equal(arr, load_canonical_array(DATASET_DIR / rel)), web
        assert info["colour"] == "Display P3 (unchanged)", (web, info["colour"])
        print(f"{web}: {len(data) / 1024:.1f} KB from {rel} (pixels identical, Display P3 kept)")


if __name__ == "__main__":
    main()
