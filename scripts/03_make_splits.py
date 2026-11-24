"""Phase 2a — leakage-free, group-aware, class-stratified split of the ORIGINAL photos.

A "leaf group" = an original photo + all its augmented copies (same photo_id) + any re-shots of the same leaf
(near-duplicate pairs from script 02), merged with union-find. Groups never cross splits.
Validation/test use original photos only. Also exports 3 training photos per class for the web app.

Outputs: data/splits.csv, data/class_map.json, experiments/results/split_summary.json,
         figures/eda/split_comparison.png, frontend/public/samples/<key>-<n>.jpg
"""
import numpy as np
import pandas as pd

from tomatoleaf.config import CLASSES, DATA_DIR, DATASET_DIR, FIGURES_DIR, RESULTS_DIR, ROOT, SEED
from tomatoleaf.data import assert_no_group_leakage, assign_leaf_groups, load_index, make_group_split
from tomatoleaf.plotting import SHORT_NAMES, SPLIT_COLORS, apply_style, plt, save
from tomatoleaf.preprocessing import strip_jpeg_metadata
from tomatoleaf.utils import save_json


def main():
    apply_style()
    index = load_index()
    index["index_row"] = np.arange(len(index))
    pairs = pd.read_csv(DATA_DIR / "near_duplicate_pairs.csv")

    originals = index[index.is_original].reset_index(drop=True)
    group_of = assign_leaf_groups(originals.photo_id.tolist(), zip(pairs.photo_a, pairs.photo_b))
    originals["group_id"] = originals.photo_id.map(group_of)
    assignment, used_seed, balance_score = make_group_split(originals, seed=SEED)
    originals["split"] = assignment.values
    assert_no_group_leakage(originals)
    originals.to_csv(DATA_DIR / "splits.csv", index=False)
    save_json({"classes": CLASSES}, DATA_DIR / "class_map.json")

    table = originals.pivot_table(index="label", columns="split", values="photo_id", aggfunc="count", fill_value=0)
    table = table[["train", "val", "test"]]
    official = index.pivot_table(index="label", columns="official_split", values="photo_id", aggfunc="count", fill_value=0)
    official = official[["train", "val", "test"]]
    summary = {
        "seed_used": used_seed,
        "candidate_splits": 200,
        "balance_score": balance_score,
        "leaf_groups": int(originals.group_id.nunique()),
        "photos": int(len(originals)),
        "largest_group": int(originals.group_id.value_counts().max()),
        "split_totals": {s: int((originals.split == s).sum()) for s in ("train", "val", "test")},
        "per_class": [{"key": c["key"], "name": c["name"], **{s: int(table.loc[c["index"], s]) for s in ("train", "val", "test")},
                       "groups": int(originals[originals.label == c["index"]].group_id.nunique())} for c in CLASSES],
    }
    save_json(summary, RESULTS_DIR / "split_summary.json")
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, tab) in zip(axes, [("Official split (images, incl. augmented copies)", official),
                                       ("Our group-aware split (original photos only)", table)]):
        left = np.zeros(len(tab))
        for split in ("train", "val", "test"):
            ax.barh(range(len(tab)), tab[split], left=left, color=SPLIT_COLORS[split], height=0.62,
                    edgecolor="#fcfcfb", linewidth=1.5, label=split)
            left += tab[split].to_numpy()
        for i, total in enumerate(left):
            ax.text(total, i, f" {int(total)}", va="center", fontsize=8, color="#52514e")
        ax.set_yticks(range(len(tab)), SHORT_NAMES)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")
    save(fig, FIGURES_DIR / "eda" / "split_comparison.png")

    samples_dir = ROOT / "frontend" / "public" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for c in CLASSES:
        pool = originals[(originals.split == "train") & (originals.label == c["index"])].path.to_numpy()
        for n, rel in enumerate(rng.choice(pool, size=3, replace=False), start=1):
            # lossless copy (metadata removed, Display P3 profile kept): a re-encode without the profile made the
            # upload path convert the colours (docs/AUDIT.md, RC5)
            (samples_dir / f"{c['key']}-{n}.jpg").write_bytes(strip_jpeg_metadata((DATASET_DIR / rel).read_bytes()))
    print("Exported web samples to", samples_dir)


if __name__ == "__main__":
    main()
