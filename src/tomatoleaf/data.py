"""Dataset indexing, leakage analysis, leaf grouping and the group-aware stratified split."""
import re
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import CLASSES, DATA_DIR, DATASET_DIR, IMAGE_EXTENSIONS, NUM_CLASSES, OFFICIAL_SPLITS, SEED
from .preprocessing import load_canonical_array

FILENAME_RE = re.compile(r"^IMG(\d{14})(?:_(\d+))?$", re.IGNORECASE)


def build_index(dataset_dir=DATASET_DIR) -> pd.DataFrame:
    """One row per image file in Variant-a, with its photo id (filename without the _N suffix)."""
    rows = []
    for split in OFFICIAL_SPLITS:
        for cls in CLASSES:
            folder = dataset_dir / split / cls["folder"]
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                match = FILENAME_RE.match(path.stem)
                if match is None:
                    raise ValueError(f"Unexpected file name: {path}")
                stamp, aug = match.groups()
                rows.append({
                    "path": path.relative_to(dataset_dir).as_posix(),
                    "official_split": split,
                    "label": cls["index"],
                    "class_key": cls["key"],
                    "photo_id": f"IMG{stamp}",
                    "aug_index": int(aug) if aug else 0,
                    "is_original": aug is None,
                    "timestamp": datetime.strptime(stamp, "%Y%m%d%H%M%S"),
                })
    return pd.DataFrame(rows)


def leakage_report(index: pd.DataFrame) -> dict:
    """How many photo families are spread across official splits, and how many val/test images leak."""
    families = index.groupby("photo_id")
    family_sizes = families.size()
    augmented = family_sizes[family_sizes > 1].index
    splits_per_family = families["official_split"].nunique()
    train_photos = set(index.loc[index.official_split == "train", "photo_id"])
    report = {
        "total_images": int(len(index)),
        "unique_photos": int(index.photo_id.nunique()),
        "augmented_families": int(len(augmented)),
        "augmented_copies": int((~index.is_original).sum()),
        "families_crossing_splits": int((splits_per_family.loc[augmented] > 1).sum()),
    }
    for split in ("val", "test"):
        part = index[index.official_split == split]
        report[f"{split}_images"] = int(len(part))
        report[f"{split}_images_with_train_sibling"] = int(part.photo_id.isin(train_photos).sum())
    per_class = []
    for cls in CLASSES:
        part = index[(index.official_split == "test") & (index.label == cls["index"])]
        per_class.append({"key": cls["key"], "test_images": int(len(part)),
                          "test_images_with_train_sibling": int(part.photo_id.isin(train_photos).sum())})
    report["per_class_test_leakage"] = per_class
    return report


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}

    def find(self, item):
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def assign_leaf_groups(photo_ids, duplicate_pairs) -> dict:
    """Merge photos that show the same physical leaf; returns photo_id -> integer group id."""
    uf = UnionFind(photo_ids)
    for a, b in duplicate_pairs:
        uf.union(a, b)
    roots = sorted({uf.find(p) for p in photo_ids})
    root_to_gid = {root: i for i, root in enumerate(roots)}
    return {p: root_to_gid[uf.find(p)] for p in photo_ids}


def _first_fold(n_splits, y, groups, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(sgkf.split(np.zeros(len(y)), y, groups))


def make_group_split(originals: pd.DataFrame, seed: int = SEED, candidates: int = 200):
    """Split original photos ~71/14/14 so that no leaf group crosses splits.

    Groups cannot be divided, so a single StratifiedGroupKFold draw can leave a class unbalanced. We draw
    `candidates` seeded splits, discard any with a class missing from a partition, and keep the one whose
    per-class val/test counts are closest to the 1/7 target (a chi-square-like score). Labels only — no model
    or test performance is involved. Returns (assignment Series, seed used, balance score).
    """
    y = originals.label.to_numpy()
    groups = originals.group_id.to_numpy()
    counts = np.bincount(y, minlength=NUM_CLASSES)
    target = counts / 7
    best = None
    for s in range(seed, seed + candidates):
        rest_idx, test_idx = _first_fold(7, y, groups, s)
        tr_rel, va_rel = _first_fold(6, y[rest_idx], groups[rest_idx], s)
        train_idx, val_idx = rest_idx[tr_rel], rest_idx[va_rel]
        if not all(len(np.unique(y[idx])) == NUM_CLASSES for idx in (train_idx, val_idx, test_idx)):
            continue
        n_val = np.bincount(y[val_idx], minlength=NUM_CLASSES)
        n_test = np.bincount(y[test_idx], minlength=NUM_CLASSES)
        score = float((((n_val - target) ** 2 + (n_test - target) ** 2) / counts).sum())
        if best is None or score < best[0]:
            best = (score, s, train_idx, val_idx, test_idx)
    if best is None:
        raise RuntimeError("Could not find a split with every class in every partition")
    score, s, train_idx, val_idx, test_idx = best
    assignment = pd.Series("train", index=originals.index)
    assignment.iloc[val_idx] = "val"
    assignment.iloc[test_idx] = "test"
    return assignment, s, score


def assert_no_group_leakage(splits: pd.DataFrame) -> None:
    """Fail loudly if any leaf group or photo appears in more than one split."""
    for col in ("group_id", "photo_id"):
        crossing = splits.groupby(col)["split"].nunique()
        bad = crossing[crossing > 1]
        if len(bad):
            raise AssertionError(f"{len(bad)} {col} values appear in more than one split")


def load_index() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "index.csv", parse_dates=["timestamp"])


def load_splits() -> pd.DataFrame:
    """Originals with their leaf group and clean split assignment (train/val/test)."""
    return pd.read_csv(DATA_DIR / "splits.csv", parse_dates=["timestamp"])


def load_images(relative_paths, dataset_dir=DATASET_DIR) -> np.ndarray:
    """Decode images into an N x 256 x 256 x 3 uint8 array (dataset is small enough to keep in RAM)."""
    out = np.empty((len(relative_paths), 256, 256, 3), dtype=np.uint8)
    for i, rel in enumerate(relative_paths):
        out[i] = load_canonical_array(dataset_dir / rel)
    return out
