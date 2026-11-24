"""Audit, step 2 — what the served models rely on, and how they answer each condition (test photos, no fitting).

1. Background shortcut, AVAILABILITY: colour statistics of the background pixels only (leaf masked out) ->
   logistic regression. Grouped 5-fold CV on train+val (the project's folds) and the test photos.
   Chance balanced accuracy = 1/8. Above chance = the capture sessions differ in paper/lighting.
2. Background shortcut, RELIANCE: the served models on each test photo as-is, with the background replaced by one
   global paper colour (leaf only), and with the leaf inpainted away (background only).
3. Answer behaviour per TRUE condition: share answered correctly, answered wrongly, "Uncertain", "Unusual image".
   For the focused model the three unsupported conditions show how often they get a confident wrong diagnosis.
4. Empty paper (leaf inpainted away, 40 test photos): should never be diagnosed.

Writes experiments/results/audit_behaviour.json. Nothing here is fitted on test photos except the logistic
regression in step 1, which is fitted on train+val and only scored on test.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf.config import DATA_DIR, DATASET_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.features import leaf_mask  # noqa: E402
from tomatoleaf.preprocessing import load_canonical_array  # noqa: E402
from tomatoleaf.utils import save_json  # noqa: E402

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def background_features(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bg = rgb[~mask] if (~mask).sum() >= 50 else rgb.reshape(-1, 3)
    hsv = cv2.cvtColor(bg[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0].astype(float)
    parts = [bg.mean(0), bg.std(0), np.percentile(bg, [10, 50, 90], axis=0).ravel(), hsv.mean(0), hsv.std(0),
             [1 - mask.mean()]]
    return np.concatenate([np.ravel(p) for p in parts])


def main(device: str = "cpu"):
    s = pd.read_csv(DATA_DIR / "splits.csv").merge(pd.read_csv(DATA_DIR / "cv_folds.csv")[["photo_id", "fold"]],
                                                    on="photo_id", how="left")
    imgs, masks = {}, {}
    for rel in s.path:
        rgb = load_canonical_array(DATASET_DIR / rel)
        imgs[rel] = rgb
        masks[rel] = cv2.dilate(leaf_mask(rgb).astype(np.uint8), KERNEL).astype(bool)

    # 1. availability
    X = np.stack([background_features(imgs[r], masks[r]) for r in s.path])
    y = s.label.to_numpy()
    tv = (s.split != "test").to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, class_weight="balanced"))
    folds = s.fold[tv].to_numpy()
    oof = np.zeros(tv.sum(), dtype=int)
    for f in np.unique(folds):
        clf.fit(X[tv][folds != f], y[tv][folds != f])
        oof[folds == f] = clf.predict(X[tv][folds == f])
    clf.fit(X[tv], y[tv])
    pred_test = clf.predict(X[~tv])
    out = {"created": time.strftime("%Y-%m-%d %H:%M"), "background_shortcut_availability": {
        "features": "background-pixel colour statistics only (RGB/HSV mean, std, percentiles; background area)",
        "oof_balanced_accuracy": balanced_accuracy_score(y[tv], oof), "oof_macro_f1": f1_score(y[tv], oof, average="macro"),
        "test_balanced_accuracy": balanced_accuracy_score(y[~tv], pred_test),
        "test_macro_f1": f1_score(y[~tv], pred_test, average="macro"), "chance_balanced_accuracy": 1 / 8}}
    print(out["background_shortcut_availability"])

    # 2-4. served models
    reg = ModelRegistry(device=device)
    test = s[s.split == "test"].reset_index(drop=True)
    paper = np.median(np.concatenate([imgs[r][~masks[r]] for r in test.path]), axis=0).astype(np.uint8)
    out["global_paper_colour"] = paper.tolist()

    def variant(rel, name):
        rgb, m = imgs[rel], masks[rel]
        if name == "original":
            return rgb
        if name == "leaf_only":
            leaf = rgb.copy()
            leaf[~m] = paper
            return leaf
        return cv2.inpaint(rgb, m.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)  # background only

    out["models"] = {}
    for key in [e["key"] for e in reg.entries]:
        pred = reg.load(key)
        keys, thr = pred.class_keys, reg.threshold_for(pred)
        res = {"threshold": thr, "reliance": {}, "per_condition": {}, "empty_paper": {}}
        rows = []
        for name in ("original", "leaf_only", "background_only"):
            decisions, confs = [], []
            for rel, true in zip(test.path, test.class_key):
                probs, ratio = pred.analyse(variant(rel, name))
                d = pred.decide(probs)
                ratio = pred.unusualness(ratio, d)
                conf = float(probs[d])
                status = ("unusual" if ratio is not None and ratio > 1 else "uncertain" if conf < thr else "answered")
                decisions.append(keys[d])
                confs.append(conf)
                if name == "original":
                    rows.append({"true": true, "pred": keys[d], "status": status})
                if name == "background_only" and len(res["empty_paper"].get("_s", [])) < 40:
                    res["empty_paper"].setdefault("_s", []).append((status, keys[d], conf))
            inscope = test.class_key.isin(keys).to_numpy()
            yt, dp = test.class_key.to_numpy()[inscope], np.array(decisions)[inscope]
            res["reliance"][name] = {"accuracy": float((yt == dp).mean()),
                                     "balanced_accuracy": balanced_accuracy_score(yt, dp),
                                     "macro_f1": f1_score(yt, dp, average="macro"),
                                     "mean_confidence": float(np.mean(np.array(confs)[inscope])),
                                     "predicted": pd.Series(dp).value_counts().to_dict()}
            print(key, name, {k: round(v, 3) for k, v in res["reliance"][name].items() if isinstance(v, float)})
        df = pd.DataFrame(rows)
        for true, g in df.groupby("true"):
            ans = g.status == "answered"
            res["per_condition"][true] = {"photos": len(g), "in_scope": true in keys,
                                          "answered_correct": float((ans & (g.pred == true)).mean()),
                                          "answered_wrong": float((ans & (g.pred != true)).mean()),
                                          "uncertain": float((g.status == "uncertain").mean()),
                                          "unusual": float((g.status == "unusual").mean())}
        ep = res["empty_paper"].pop("_s")
        res["empty_paper"] = {"images": len(ep), **{st: sum(e[0] == st for e in ep) for st in
                                                    ("answered", "uncertain", "unusual")},
                              "answered_as": pd.Series([e[1] for e in ep if e[0] == "answered"]).value_counts().to_dict()}
        out["models"][key] = res
        print(key, "per condition:", {k: (v["answered_correct"], v["answered_wrong"]) for k, v in
                                      res["per_condition"].items()}, "empty paper:", res["empty_paper"])
    save_json(out, RESULTS_DIR / "audit_behaviour.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
