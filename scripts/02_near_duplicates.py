"""Phase 1b — detect re-shots of the same physical leaf (different files, taken seconds apart).

Signal 1: capture time (from the filename) within TIME_WINDOW_S seconds, same class.
Signal 2: cosine similarity of ImageNet-pretrained MobileNetV2 embeddings (no training on our data).
The similarity threshold is set from a reference distribution of pairs that CANNOT be re-shots
(same class, photographed on different days): we take its 99.9th percentile.

Outputs: data/near_duplicate_pairs.csv, experiments/results/near_duplicates.json, figures/eda/near_duplicates.png
"""
import numpy as np
import pandas as pd
import torch

from tomatoleaf.config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, SEED
from tomatoleaf.data import load_images, load_index
from tomatoleaf.models import build_mobilenetv2
from tomatoleaf.plotting import INK2, MUTED, SERIES, apply_style, plt, save
from tomatoleaf.preprocessing import eval_transform, to_image_tensor
from tomatoleaf.utils import save_json, set_seed

TIME_WINDOW_S = 120
REFERENCE_QUANTILE = 0.999


@torch.no_grad()
def embed(images: np.ndarray, device) -> np.ndarray:
    model = build_mobilenetv2(pretrained=True).to(device).eval()
    transform = eval_transform()
    out = []
    for start in range(0, len(images), 64):
        batch = torch.stack([transform(to_image_tensor(img)).as_subclass(torch.Tensor) for img in images[start:start + 64]])
        feats = model.features(batch.to(device)).mean(dim=(2, 3))
        out.append(torch.nn.functional.normalize(feats, dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    set_seed(SEED)
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = load_index()
    originals = index[index.is_original].reset_index(drop=True)
    images = load_images(originals.path)
    emb = embed(images, device)
    sim = emb @ emb.T

    labels = originals.label.to_numpy()
    seconds = originals.timestamp.astype("int64").to_numpy() // 10**9
    days = originals.timestamp.dt.date.to_numpy()
    iu, ju = np.triu_indices(len(originals), k=1)
    same_class = labels[iu] == labels[ju]
    dt = np.abs(seconds[iu] - seconds[ju])
    pair_sim = sim[iu, ju]

    reference = pair_sim[same_class & (days[iu] != days[ju])]
    threshold = float(np.quantile(reference, REFERENCE_QUANTILE))
    candidates = same_class & (dt <= TIME_WINDOW_S)
    merged = candidates & (pair_sim >= threshold)

    pairs = pd.DataFrame({
        "photo_a": originals.photo_id.to_numpy()[iu[merged]], "photo_b": originals.photo_id.to_numpy()[ju[merged]],
        "class_key": originals.class_key.to_numpy()[iu[merged]], "similarity": pair_sim[merged], "seconds_apart": dt[merged],
        "official_split_a": originals.official_split.to_numpy()[iu[merged]],
        "official_split_b": originals.official_split.to_numpy()[ju[merged]],
    }).sort_values("similarity", ascending=False)
    pairs.to_csv(DATA_DIR / "near_duplicate_pairs.csv", index=False)

    cross_split = int((pairs.official_split_a != pairs.official_split_b).sum())
    stats = {
        "time_window_s": TIME_WINDOW_S, "reference_quantile": REFERENCE_QUANTILE, "threshold": threshold,
        "reference_pairs": int(len(reference)), "reference_max": float(reference.max()),
        "candidate_pairs": int(candidates.sum()), "merged_pairs": int(len(pairs)),
        "merged_pairs_crossing_official_splits": cross_split,
        "photos_in_merged_pairs": int(len(set(pairs.photo_a) | set(pairs.photo_b))),
        "per_class_merged_pairs": pairs.class_key.value_counts().to_dict(),
    }
    save_json(stats, RESULTS_DIR / "near_duplicates.json")
    print(stats)

    # Figure: similarity distributions + example pairs around the threshold for visual verification
    fig = plt.figure(figsize=(11, 10.5))
    gs = fig.add_gridspec(4, 8, height_ratios=[1.6, 1, 1, 1], width_ratios=[1, 1, 0.22, 1, 1, 0.22, 1, 1],
                          hspace=0.45, wspace=0.06)
    ax = fig.add_subplot(gs[0, :])
    bins = np.linspace(min(reference.min(), pair_sim[candidates].min()), 1, 60)
    ax.hist(reference, bins=bins, density=True, histtype="step", color=SERIES[0], lw=2,
            label=f"Same class, different day (cannot be re-shots), n={len(reference):,}")
    ax.hist(pair_sim[candidates], bins=bins, density=True, histtype="step", color=SERIES[1], lw=2,
            label=f"Same class, taken <= {TIME_WINDOW_S}s apart, n={int(candidates.sum()):,}")
    ax.axvline(threshold, color=INK2, lw=1)
    ax.text(threshold, ax.get_ylim()[1] * 0.92, f"  threshold {threshold:.3f}\n  (99.9th pct of reference)", color=INK2, fontsize=8.5, va="top")
    ax.set_xlabel("Cosine similarity of ImageNet MobileNetV2 embeddings")
    ax.set_ylabel("Density")
    ax.set_title("Near-duplicate detection: embedding similarity of photo pairs")
    ax.legend(loc="upper left")

    above = pairs.head(3)
    near = pairs.tail(3)
    below_idx = np.flatnonzero(candidates & (pair_sim < threshold))
    below_idx = below_idx[np.argsort(-pair_sim[below_idx])][:3]
    pid_to_row = {p: i for i, p in enumerate(originals.photo_id)}
    examples = [(r.photo_a, r.photo_b, r.similarity, r.seconds_apart, "merged") for r in above.itertuples()]
    examples += [(r.photo_a, r.photo_b, r.similarity, r.seconds_apart, "merged, near threshold") for r in near.itertuples()]
    examples += [(originals.photo_id[iu[k]], originals.photo_id[ju[k]], pair_sim[k], dt[k], "not merged") for k in below_idx]
    for n, (a, b, s, secs, verdict) in enumerate(examples[:9]):
        row, col = 1 + n // 3, (n % 3) * 3
        for half, pid in enumerate((a, b)):
            axi = fig.add_subplot(gs[row, col + half])
            axi.imshow(images[pid_to_row[pid]])
            axi.set_axis_off()
            r = originals.iloc[pid_to_row[pid]]
            axi.set_title(f"{r.official_split} | {pid[-6:]}", fontsize=7, color=MUTED, loc="center", fontweight="normal")
            if half == 0:
                axi.text(1.03, -0.1, f"{verdict}: sim {s:.3f}, {int(secs)} s apart", transform=axi.transAxes,
                         ha="center", va="top", fontsize=7.5, color=INK2)
    save(fig, FIGURES_DIR / "eda" / "near_duplicates.png")


if __name__ == "__main__":
    main()
