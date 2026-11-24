"""Audit, step 1 — every built-in example image through the real HTTP API, for every served model.

Why: the app was reported to misclassify some of its own example images. This script checks the whole path a user
takes, not a shortcut: the file is read from frontend/public, posted as multipart/form-data to POST /api/predict
(FastAPI TestClient, same app object as uvicorn), decoded by the upload preprocessing and answered by the served
model. Nothing is hard-coded: the true label comes from the gallery manifest (held-out test photos, byte-identical
dataset files) or from the library file name, and is checked against the dataset by MD5 / nearest pixels.

Example sets:
- gallery: frontend/public/test-samples (one held-out TEST photo per class; the Diagnose page gallery in both apps);
- library: frontend/public/samples (3 training photos per class, re-encoded at JPEG q92 by scripts/03 without
  their colour profile; shown in the disease library and on the landing page);
- library_original: the dataset file each library image was made from (same photo, original bytes).

Writes experiments/results/audit_examples.json (or --out NAME). Run again after any model change.
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402
from tomatoleaf.config import DATA_DIR, DATASET_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.preprocessing import load_canonical_array  # noqa: E402
from tomatoleaf.utils import save_json  # noqa: E402

PUBLIC = ROOT / "frontend" / "public"


def main():
    out_name = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "audit_examples.json"
    index = list(csv.DictReader(open(DATA_DIR / "splits.csv", encoding="utf-8")))
    folds = {r["photo_id"]: r["fold"] for r in csv.DictReader(open(DATA_DIR / "cv_folds.csv", encoding="utf-8"))}
    by_md5 = {r["md5"]: r for r in index}
    by_class: dict[str, list] = {}
    for r in index:
        by_class.setdefault(r["class_key"], []).append(r)

    examples = []
    manifest = json.loads((PUBLIC / "test-samples" / "manifest.json").read_text(encoding="utf-8"))
    for s in manifest["samples"]:
        path = PUBLIC / s["file"].lstrip("/")
        src = by_md5.get(hashlib.md5(path.read_bytes()).hexdigest())
        examples.append({"set": "gallery", "file": str(path.relative_to(ROOT)).replace("\\", "/"), "true": s["key"],
                         "dataset_file": src and src["path"], "dataset_class": src and src["class_key"],
                         "dataset_split": src and src["split"], "byte_identical": src is not None})
    cache = {}
    for f in sorted((PUBLIC / "samples").glob("*.jpg")):
        key = f.stem.rsplit("-", 1)[0]
        arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
        best = (np.inf, None)
        for r in by_class[key]:  # the library image is a re-encode of one photo of its own class
            if r["path"] not in cache:
                cache[r["path"]] = load_canonical_array(DATASET_DIR / r["path"]).astype(np.float32)
            d = float(np.mean((cache[r["path"]] - arr) ** 2))
            if d < best[0]:
                best = (d, r)
        r = best[1]
        meta = {"true": key, "dataset_file": r["path"], "dataset_class": r["class_key"], "dataset_split": r["split"],
                "cv_fold": folds.get(r["photo_id"]), "pixel_mse_to_source": round(best[0], 3)}
        examples.append({"set": "library", "file": str(f.relative_to(ROOT)).replace("\\", "/"), **meta})
        examples.append({"set": "library_original", "file": f"Dataset/.../{r['path']}", **meta,
                         "_path": str(DATASET_DIR / r["path"])})

    rows = []
    with TestClient(app) as client:
        served = [m["key"] for m in client.get("/api/models").json()["models"]]
        for ex in examples:
            path = Path(ex.pop("_path", None) or ROOT / ex["file"])
            data = path.read_bytes()
            for key in served:
                res = client.post(f"/api/predict?model={key}", files={"file": (path.name, data, "image/jpeg")})
                res.raise_for_status()
                j = res.json()
                p = j["prediction"]
                top = max(j["probabilities"], key=lambda q: q["probability"])
                answered = p["status"] in ("Healthy", "Diseased")
                rows.append({**ex, "model": key, "in_scope": ex["true"] in j["classes"], "predicted": p["key"],
                             "confidence": p["confidence"], "status": p["status"],
                             "threshold": j["confidence_threshold"],
                             "ood_ratio": None if j["ood"] is None else j["ood"]["ratio"],
                             "label_correct": p["key"] == ex["true"],
                             "outcome": ("correct" if p["key"] == ex["true"] else "WRONG") if answered else
                             p["status"].lower(),
                             "tallest_bar": top["key"], "label_differs_from_tallest_bar": top["key"] != p["key"],
                             "colour_handling": j["image"]["colour"]})
    summary = {}
    for (s, m) in sorted({(r["set"], r["model"]) for r in rows}):
        sub = [r for r in rows if r["set"] == s and r["model"] == m]
        summary[f"{s} | {m}"] = {o: sum(r["outcome"] == o for r in sub) for o in
                                 ("correct", "WRONG", "uncertain", "unusual image")}
        summary[f"{s} | {m}"]["wrong_out_of_scope"] = sum(r["outcome"] == "WRONG" and not r["in_scope"] for r in sub)
    save_json({"created": time.strftime("%Y-%m-%d %H:%M"), "served_models": served, "summary": summary, "rows": rows},
              RESULTS_DIR / out_name)
    for k, v in summary.items():
        print(f"{k:45s} {v}")


if __name__ == "__main__":
    main()
