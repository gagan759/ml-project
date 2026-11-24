"""Photo quality checks (tomatoleaf.quality): limits from the training photos, then how well they work.

Limits: the 0.5th / 99.5th percentile of each measure over the TRAIN + VALIDATION photos (canonical 256x256 images, as
the models see them). Evaluation (no fitting):
- false flags on the held-out TEST photos (genuine dataset photos: should rarely be flagged);
- detection on test photos changed on purpose (blur, dark, over-exposed, heavy JPEG, other backgrounds, four
  leaves in one frame — the stress sets of scripts/38), on synthetic non-leaf probes and on the out-of-dataset photos
  of scripts/37 (other crops, everyday photos).
Writes experiments/quality_limits.json and experiments/results/quality_check.json.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tomatoleaf.config import DATASET_DIR, EXPERIMENTS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.data import load_splits  # noqa: E402
from tomatoleaf.preprocessing import load_canonical_array  # noqa: E402
from tomatoleaf.probes import far_ood_probes  # noqa: E402
from tomatoleaf.quality import CHECKS, HIGH, LOW, assess, measures  # noqa: E402
from tomatoleaf.utils import save_json  # noqa: E402


def flagged(images, lim):
    res = [assess(x, lim) for x in images]
    codes = {}
    for r in res:
        for i in r["issues"]:
            codes[i["code"]] = codes.get(i["code"], 0) + 1
    return {"photos": len(res), "flagged": float(np.mean([not r["ok"] for r in res])),
            "by_issue": {k: v / len(res) for k, v in sorted(codes.items())}}


def main():
    splits = load_splits()
    fit = splits[splits.split != "test"]
    m = [measures(load_canonical_array(DATASET_DIR / p)) for p in fit.path]
    names = sorted({c[0] for c in CHECKS})
    lim = {n: {"low": float(np.percentile([x[n] for x in m], LOW)), "high": float(np.percentile([x[n] for x in m], HIGH))}
           for n in names}
    save_json({"created": time.strftime("%Y-%m-%d %H:%M"), "fitted_on": f"{len(fit)} train+validation photos",
               "percentiles": [LOW, HIGH], "limits": lim,
               },
              EXPERIMENTS_DIR / "quality_limits.json")

    spec = importlib.util.spec_from_file_location("ft", ROOT / "scripts" / "38_final_test.py")
    ft = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft)
    test = splits[splits.split == "test"]
    X = np.stack([load_canonical_array(DATASET_DIR / p) for p in test.path])
    out = {"limits": lim, "genuine_test_photos": flagged(X, lim), "changed_test_photos": {}, "out_of_dataset": {}}
    for name, (imgs, _) in ft.stress_sets(X, list(test.class_key), np.random.default_rng(0)).items():
        out["changed_test_photos"][name] = flagged(imgs, lim)
    out["out_of_dataset"]["synthetic_probes"] = flagged(far_ood_probes()[1], lim)
    out["out_of_dataset"]["empty_paper"] = flagged(ft.paper_only(X[:60]), lim)
    for name, (imgs, _) in ft.external_sets().items():
        out["out_of_dataset"][name] = flagged(imgs, lim)
    save_json(out, RESULTS_DIR / "quality_check.json")
    print(f"genuine test photos flagged: {out['genuine_test_photos']['flagged']:.3f} {out['genuine_test_photos']['by_issue']}")
    for group in ("changed_test_photos", "out_of_dataset"):
        for k, v in out[group].items():
            print(f"  {k:22s} flagged {v['flagged']:.2f}  {({c: round(s, 2) for c, s in v['by_issue'].items()})}")


if __name__ == "__main__":
    main()
