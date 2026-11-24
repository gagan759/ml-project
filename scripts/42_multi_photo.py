"""Several photos of one leaf: does averaging their probabilities help? (out-of-fold data only, nothing fitted)

The dataset has leaves photographed 2-5 times (same group_id: same condition, timestamps seconds apart; grouped CV
keeps a group in one fold, so every member's out-of-fold prediction comes from the same networks, as for a new
leaf). For the served pipeline of each mode (chosen arm + its cross-fitted calibration, scripts/36) and its served
threshold, this compares, on the leaves with >= 2 photos:
- single photo: every photo judged on its own (what the app did so far);
- averaged: the leaf's calibrated probabilities averaged, then the same argmax and threshold (what the app does when
  the user adds several photos).
For the focused mode it also reports how many unsupported leaves (Early/Late Blight, Potassium) get an answer.
Writes experiments/results/multi_photo.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tomatoleaf.config import EXPERIMENTS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.data import load_splits  # noqa: E402
from tomatoleaf.utils import save_json  # noqa: E402

spec = importlib.util.spec_from_file_location("fp", ROOT / "scripts" / "36_final_pipeline.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

SERVED = {"5class": "focused_5class", "8class": "full_8class"}


def selective(p, y, t):
    keep = p.max(1) >= t
    return {"leaves_or_photos": int(len(y)), "accuracy": float((p.argmax(1) == y).mean()),
            "coverage": float(keep.mean()),
            "accuracy_answered": float((p.argmax(1)[keep] == y[keep]).mean()) if keep.any() else None}


def averaged(p, groups):
    ids = np.unique(groups)
    return ids, np.stack([p[groups == g].mean(0) for g in ids])


def main():
    deployment = {e["key"]: e for e in json.loads((EXPERIMENTS_DIR / "deployment.json").read_text())["models"]}
    splits = load_splits()
    group_of = dict(zip(splits.photo_id, splits.group_id))
    out = {}
    for task, key in SERVED.items():
        chosen = json.loads((RESULTS_DIR / f"final_pipeline_{task}.json").read_text())["chosen"]
        a = fp.evaluate_arm(task, chosen["arm"], chosen["recipe"], None)
        t = float(deployment[key]["confidence_threshold"])
        p, y = a["_p"], a["_y"]
        g = np.array([group_of[str(i)] for i in a["_d"]["pid"]])
        sizes = {k: (g == k).sum() for k in np.unique(g)}
        multi = np.array([sizes[k] > 1 for k in g])
        ids, pa = averaged(p[multi], g[multi])
        ya = np.array([y[multi][g[multi] == k][0] for k in ids])
        res = {"threshold": t, "single_photo": selective(p[multi], y[multi], t), "averaged": selective(pa, ya, t),
               "all_photos_single": selective(p, y, t)}
        if a["_po"] is not None:
            og = np.array([group_of[str(i)] for i in a["_d"]["opid"]])
            osz = {k: (og == k).sum() for k in np.unique(og)}
            om = np.array([osz[k] > 1 for k in og])
            _, poa = averaged(a["_po"][om], og[om])
            res["unsupported_answered"] = {"single_photo": float((a["_po"][om].max(1) >= t).mean()),
                                           "averaged": float((poa.max(1) >= t).mean()),
                                           "leaves": int(len(poa)), "photos": int(om.sum())}
        out[task] = res
        s, m = res["single_photo"], res["averaged"]
        print(f"{task}: {m['leaves_or_photos']} leaves / {s['leaves_or_photos']} photos | accuracy {s['accuracy']:.3f} -> "
              f"{m['accuracy']:.3f} | answered {s['coverage']:.2f} -> {m['coverage']:.2f} | correct when answered "
              f"{s['accuracy_answered']:.3f} -> {m['accuracy_answered']:.3f}"
              + (f" | unsupported answered {res['unsupported_answered']['single_photo']:.2f} -> "
                 f"{res['unsupported_answered']['averaged']:.2f}" if "unsupported_answered" in res else ""))
    save_json(out, RESULTS_DIR / "multi_photo.json")


if __name__ == "__main__":
    main()
