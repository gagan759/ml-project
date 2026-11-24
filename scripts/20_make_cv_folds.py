"""Upgrade round, step 1 — grouped, stratified 5-fold cross-validation folds over the TRAIN + VALIDATION photos.

Why: the original study chose models on one validation split of 311 photos, with 1 potassium and 6 nitrogen photos.
One photo then moves validation macro-F1 by up to 0.125, so model choice was mostly noise. Cross-validation over the
1,813 train+val photos gives every photo an out-of-fold prediction. The 290 TEST photos are not touched: they are
scored once, at the very end (scripts/25_final_benchmark.py).

Rules: whole leaf groups stay in one fold (no near-duplicate shots on both sides); every class appears in every
fold's held-out part where possible; among 200 seeded StratifiedGroupKFold draws (labels only) we keep the one whose
per-class fold counts are most even.
Output: data/cv_folds.csv (photo rows with a `fold` column 0-4), experiments/results/cv_folds_summary.json
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from tomatoleaf.config import CLASS_KEYS, DATA_DIR, NUM_CLASSES, RESULTS_DIR, SEED
from tomatoleaf.data import load_splits
from tomatoleaf.utils import save_json

N_FOLDS, CANDIDATES = 5, 200


def main():
    splits = load_splits()
    pool = splits[splits.split.isin(["train", "val"])].reset_index(drop=True)
    y, groups = pool.label.to_numpy(), pool.group_id.to_numpy()
    counts = np.bincount(y, minlength=NUM_CLASSES)
    best = None
    for s in range(SEED, SEED + CANDIDATES):
        fold = np.empty(len(pool), dtype=int)
        for k, (_, held) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=s).split(pool, y, groups)):
            fold[held] = k
        per = np.stack([np.bincount(y[fold == k], minlength=NUM_CLASSES) for k in range(N_FOLDS)])
        missing = int((per == 0).sum())
        score = float((((per - counts / N_FOLDS) ** 2) / counts).sum())
        key = (missing, score)
        if best is None or key < best[0]:
            best = (key, s, fold, per)
    (missing, score), seed, fold, per = best
    pool["fold"] = fold
    assert pool.groupby("group_id").fold.nunique().max() == 1, "a leaf group crosses folds"
    assert not set(pool.photo_id) & set(splits[splits.split == "test"].photo_id), "a test photo entered the folds"
    pool.to_csv(DATA_DIR / "cv_folds.csv", index=False)
    summary = {"n_folds": N_FOLDS, "photos": len(pool), "seed": seed, "balance_score": score,
               "empty_class_fold_cells": missing,
               "per_fold": [{"fold": k, "photos": int((fold == k).sum()),
                             "per_class": {CLASS_KEYS[c]: int(per[k, c]) for c in range(NUM_CLASSES)}}
                            for k in range(N_FOLDS)],
               "rule": "StratifiedGroupKFold over train+val photos; groups = leaf groups; best of 200 seeds by label "
                       "balance only; test photos excluded"}
    save_json(summary, RESULTS_DIR / "cv_folds_summary.json")
    print(f"seed {seed}, balance {score:.3f}, empty cells {missing}")
    print(pd.DataFrame(per, columns=[k[:8] for k in CLASS_KEYS]))


if __name__ == "__main__":
    main()
