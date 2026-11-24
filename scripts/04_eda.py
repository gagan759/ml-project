"""Phase 1c — exploratory data analysis.

Counts use the whole dataset; pixel-level statistics use TRAIN-split original photos only,
so nothing about the validation/test images influences preprocessing decisions.
Outputs: reports/figures/eda/*.png, experiments/results/eda_stats.json
"""
import cv2
import numpy as np

from tomatoleaf.config import CLASSES, DATASET_DIR, FIGURES_DIR, NUM_CLASSES, RESULTS_DIR, SEED
from tomatoleaf.data import load_images, load_index, load_splits
from tomatoleaf.features import leaf_mask
from tomatoleaf.plotting import (AXIS, CHANNEL_COLORS, INK, INK2, MUTED, SERIES, SHORT_NAMES, apply_style, plt, save)
from tomatoleaf.preprocessing import load_canonical_array
from tomatoleaf.utils import load_json, save_json

OUT = FIGURES_DIR / "eda"


def small_multiples(title, ylabel=None, xlabel=None, sharey=True):
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), sharex=True, sharey=sharey)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    for i, ax in enumerate(axes.flat):
        ax.set_title(SHORT_NAMES[i], fontsize=9.5)
        if xlabel and i >= 4:
            ax.set_xlabel(xlabel)
        if ylabel and i % 4 == 0:
            ax.set_ylabel(ylabel)
    return fig, axes.flat


def fig_class_distribution(index):
    images = index.groupby("label").size().reindex(range(NUM_CLASSES)).to_numpy()
    photos = index[index.is_original].groupby("label").size().reindex(range(NUM_CLASSES)).to_numpy()
    order = np.argsort(-images)
    y, h = np.arange(NUM_CLASSES), 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.barh(y - h / 2, images[order], height=h, color=SERIES[0], label="Image files (incl. augmented copies)")
    ax.barh(y + h / 2, photos[order], height=h, color=SERIES[1], label="Unique photographs")
    for yi, (a, b) in enumerate(zip(images[order], photos[order])):
        ax.text(a, yi - h / 2, f" {a}", va="center", fontsize=8, color=INK2)
        ax.text(b, yi + h / 2, f" {b}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [SHORT_NAMES[i] for i in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Count")
    ax.set_title(f"Class distribution: imbalance {images.max() / images.min():.1f}:1 by files, "
                 f"{photos.max() / photos.min():.1f}:1 by unique photographs")
    ax.legend(loc="lower right")
    save(fig, OUT / "class_distribution.png")


def fig_samples(train, images):
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(NUM_CLASSES, 6, figsize=(9, 12.4))
    for c in range(NUM_CLASSES):
        idx = rng.choice(np.flatnonzero(train.label.to_numpy() == c), size=6, replace=False)
        for j, k in enumerate(idx):
            ax = axes[c, j]
            ax.imshow(images[k])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[c, 0].set_ylabel(SHORT_NAMES[c], rotation=0, ha="right", va="center", fontsize=10, color=INK)
    fig.suptitle("Sample training photographs (one detached leaf on white paper, 256x256)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.16)
    save(fig, OUT / "samples_grid.png")


def fig_samples_compact(train, images):
    rng = np.random.default_rng(SEED + 2)
    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(14, 3.9))
    for c in range(NUM_CLASSES):
        for r, k in enumerate(rng.choice(np.flatnonzero(train.label.to_numpy() == c), size=2, replace=False)):
            axes[r, c].imshow(images[k])
            axes[r, c].set_axis_off()
        axes[0, c].set_title(SHORT_NAMES[c], fontsize=10, loc="center", fontweight="normal")
    fig.subplots_adjust(wspace=0.04, hspace=0.05)
    save(fig, OUT / "samples_compact.png")


def fig_augmentation_family(index):
    fam = index.groupby("photo_id").official_split.nunique()
    photo = sorted(fam[fam == 3].index)[0]
    members = index[index.photo_id == photo].sort_values("aug_index")
    fig, axes = plt.subplots(1, 8, figsize=(14, 2.4))
    for ax, row in zip(axes, members.itertuples()):
        ax.imshow(load_canonical_array(DATASET_DIR / row.path))
        ax.set_axis_off()
        name = "original" if row.aug_index == 0 else f"copy _{row.aug_index}"
        ax.set_title(f"{name}\nofficial {row.official_split}", fontsize=8.5,
                     color=INK if row.official_split == "train" else SERIES[7], fontweight="normal", loc="center")
    fig.suptitle(f"One photo ({photo}, {SHORT_NAMES[members.label.iloc[0]]}) and its 7 pre-made copies, "
                 "spread across train / val / test by the official split", x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, OUT / "augmentation_family.png")


def fig_image_properties(index, leakage):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    a1.set_axis_off()
    a1.text(0.0, 0.78, "256 x 256 px", fontsize=30, fontweight="bold", color=INK, transform=a1.transAxes)
    a1.text(0.0, 0.58, f"every one of the {len(index):,} images", fontsize=11, color=INK2, transform=a1.transAxes)
    qualities = ", ".join(f"q{k}: {v}" for k, v in leakage["jpeg_quality_counts"].items())
    a1.text(0.0, 0.40, f"3-channel RGB JPEG (estimated quality {qualities})", fontsize=10, color=INK2, transform=a1.transAxes)
    a1.text(0.0, 0.26, f"{leakage['corrupt_files']} corrupt files, "
            f"{leakage['byte_identical_duplicate_files']} byte-identical duplicates", fontsize=10, color=INK2, transform=a1.transAxes)
    a1.set_title("Image dimensions and format")
    mb = leakage["mean_bytes"]
    kinds = ["Original photos", "Augmented copies"]
    meta = np.array([mb["original"]["metadata"], mb["copy"]["metadata"]]) / 1024
    pix = np.array([mb["original"]["file"] - mb["original"]["metadata"], mb["copy"]["file"] - mb["copy"]["metadata"]]) / 1024
    a2.barh(kinds, pix, color=SERIES[0], height=0.5, label="Compressed image data", edgecolor="#fcfcfb", linewidth=1.5)
    a2.barh(kinds, meta, left=pix, color=SERIES[3], height=0.5, label="Metadata (EXIF, ICC, camera debug)",
            edgecolor="#fcfcfb", linewidth=1.5)
    for i in range(2):
        a2.text(pix[i] + meta[i], i, f"  {pix[i] + meta[i]:.0f} KB", va="center", fontsize=9, color=INK2)
    a2.set_xlabel("Mean file size (KB)")
    a2.invert_yaxis()
    a2.grid(axis="y", visible=False)
    a2.set_title("Why originals are ~20x larger: metadata, not pixels")
    a2.legend(loc="lower right")
    save(fig, OUT / "image_properties.png")


def fig_leakage(index):
    train_photos = set(index.loc[index.official_split == "train", "photo_id"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, split in zip(axes, ("val", "test")):
        part = index[index.official_split == split]
        leaked = part.photo_id.isin(train_photos)
        a = part[leaked].groupby("label").size().reindex(range(NUM_CLASSES), fill_value=0).to_numpy()
        b = part[~leaked].groupby("label").size().reindex(range(NUM_CLASSES), fill_value=0).to_numpy()
        y = np.arange(NUM_CLASSES)
        ax.barh(y, a, color=SERIES[7], height=0.62, label="Same photo also in train (leaked)", edgecolor="#fcfcfb", linewidth=1.5)
        ax.barh(y, b, left=a, color=SERIES[0], height=0.62, label="Photo not in train", edgecolor="#fcfcfb", linewidth=1.5)
        ax.set_yticks(y, SHORT_NAMES)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Images")
        ax.set_title(f"Official {split} split: {leaked.sum()}/{len(part)} images leaked ({100 * leaked.mean():.1f}%)")
    axes[1].legend(loc="lower right")
    save(fig, OUT / "official_split_leakage.png")


def pixel_statistics(train, images):
    k = NUM_CLASSES
    stats = {"rgb": np.zeros((k, 3, 256)), "gray_all": np.zeros((k, 256)), "gray_leaf": np.zeros((k, 256)),
             "hue": np.zeros((k, 180)), "sat": np.zeros((k, 256))}
    leaf_fraction, paper_value, masks = [], [], []
    channel_sum, channel_sq, n_pix = np.zeros(3), np.zeros(3), 0
    for img, c in zip(images, train.label.to_numpy()):
        mask = leaf_mask(img)
        masks.append(mask)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for ch in range(3):
            stats["rgb"][c, ch] += np.bincount(img[..., ch][mask], minlength=256)
        stats["gray_all"][c] += np.bincount(gray.ravel(), minlength=256)
        stats["gray_leaf"][c] += np.bincount(gray[mask], minlength=256)
        stats["hue"][c] += np.bincount(hsv[..., 0][mask], minlength=180)
        stats["sat"][c] += np.bincount(hsv[..., 1][mask], minlength=256)
        leaf_fraction.append(mask.mean())
        paper_value.append(float(np.median(hsv[..., 2][~mask])) if (~mask).any() else np.nan)
        f = img.reshape(-1, 3) / 255.0
        channel_sum += f.sum(0)
        channel_sq += (f ** 2).sum(0)
        n_pix += len(f)
    mean = channel_sum / n_pix
    std = np.sqrt(channel_sq / n_pix - mean ** 2)
    return stats, np.array(leaf_fraction), np.array(paper_value), masks, mean, std


def binned(hist, width):
    return hist.reshape(-1, width).sum(axis=-1)


def fig_rgb(stats):
    fig, axes = small_multiples("RGB channel distributions of leaf pixels (train originals)", "Density", "Pixel value")
    x = np.arange(0, 256, 4) + 2
    for c, ax in enumerate(axes):
        for ch, name in enumerate("RGB"):
            h = binned(stats["rgb"][c, ch], 4)
            ax.plot(x, h / h.sum() / 4, color=CHANNEL_COLORS[name], lw=1.6, label=name)
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, OUT / "rgb_distribution.png")


def fig_intensity(stats):
    fig, axes = small_multiples("Pixel intensity histograms: whole image vs leaf pixels only", "Density", "Grey level")
    x = np.arange(0, 256, 4) + 2
    for c, ax in enumerate(axes):
        whole, leaf = binned(stats["gray_all"][c], 4), binned(stats["gray_leaf"][c], 4)
        ax.plot(x, whole / whole.sum() / 4, color=MUTED, lw=1.6, label="Whole image")
        ax.plot(x, leaf / leaf.sum() / 4, color=SERIES[0], lw=1.6, label="Leaf only")
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, OUT / "intensity_histograms.png")


def fig_hue(stats):
    fig, axes = small_multiples("Hue distribution of leaf pixels (all-class average in grey)", "Density", "Hue (degrees)")
    hue = stats["hue"]
    x = np.arange(0, 180, 2) * 2 + 2
    avg = binned(hue.sum(0), 2)
    avg = avg / avg.sum()
    for c, ax in enumerate(axes):
        h = binned(hue[c], 2)
        ax.plot(x, avg, color=AXIS, lw=1.4)
        ax.plot(x, h / h.sum(), color=SERIES[0], lw=1.8)
        ax.set_xlim(0, 160)
    save(fig, OUT / "hue_distribution.png")


def fig_mean_images(train, images):
    fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(14, 2.3))
    for c, ax in enumerate(axes):
        ax.imshow(images[train.label.to_numpy() == c].mean(axis=0).astype(np.uint8))
        ax.set_axis_off()
        ax.set_title(SHORT_NAMES[c], fontsize=9, loc="center", fontweight="normal")
    fig.suptitle("Mean training image per class", x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, OUT / "mean_images.png")


def fig_masks(train, images, masks):
    rng = np.random.default_rng(SEED + 1)
    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(14, 3.9))
    for c in range(NUM_CLASSES):
        for r in range(2):
            k = rng.choice(np.flatnonzero(train.label.to_numpy() == c))
            img = images[k].copy()
            contours, _ = cv2.findContours(masks[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, (235, 104, 52), 2)
            axes[r, c].imshow(img)
            axes[r, c].set_axis_off()
            if r == 0:
                axes[r, c].set_title(SHORT_NAMES[c], fontsize=9, loc="center", fontweight="normal")
    fig.suptitle("Leaf masks (orange outline) used for the colour features: Otsu on saturation + dark pixels, cleaned",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    save(fig, OUT / "leaf_masks.png")


def fig_background_and_area(train, leaf_fraction, paper_value):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = train.label.to_numpy()
    for ax, values, title, xlabel in [
        (a1, paper_value, "Background (paper) brightness per class", "Median HSV value of background pixels"),
        (a2, leaf_fraction * 100, "Leaf area per class", "Leaf pixels (% of image)"),
    ]:
        data = [values[labels == c][~np.isnan(values[labels == c])] for c in range(NUM_CLASSES)]
        ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0], "linewidth": 1.2},
                   medianprops={"color": SERIES[0], "linewidth": 2}, whiskerprops={"color": AXIS},
                   capprops={"color": AXIS})
        ax.set_yticks(range(1, NUM_CLASSES + 1), SHORT_NAMES)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    save(fig, OUT / "background_and_leaf_area.png")


def main():
    apply_style()
    index = load_index()
    leakage = load_json(RESULTS_DIR / "leakage_official_split.json")
    splits = load_splits()
    train = splits[splits.split == "train"].reset_index(drop=True)
    images = load_images(train.path)

    fig_class_distribution(index)
    fig_augmentation_family(index)
    fig_image_properties(index, leakage)
    fig_leakage(index)
    fig_samples(train, images)
    fig_samples_compact(train, images)
    fig_mean_images(train, images)
    stats, leaf_fraction, paper_value, masks, mean, std = pixel_statistics(train, images)
    fig_rgb(stats)
    fig_intensity(stats)
    fig_hue(stats)
    fig_masks(train, images, masks)
    fig_background_and_area(train, leaf_fraction, paper_value)

    labels = train.label.to_numpy()
    per_class = []
    for c in CLASSES:
        sel = labels == c["index"]
        rgb = stats["rgb"][c["index"]]
        hue = stats["hue"][c["index"]]
        per_class.append({
            "key": c["key"], "train_photos": int(sel.sum()),
            "mean_leaf_area_pct": float(100 * leaf_fraction[sel].mean()),
            "median_paper_value": float(np.nanmedian(paper_value[sel])),
            "mean_leaf_rgb": [float((rgb[ch] * np.arange(256)).sum() / rgb[ch].sum()) for ch in range(3)],
            "median_leaf_hue_deg": float(2 * np.searchsorted(np.cumsum(hue) / hue.sum(), 0.5)),
        })
    eda = {"train_photos_used": int(len(train)), "train_rgb_mean": mean.tolist(), "train_rgb_std": std.tolist(),
           "imagenet_mean": [0.485, 0.456, 0.406], "imagenet_std": [0.229, 0.224, 0.225],
           "mean_leaf_area_pct": float(100 * leaf_fraction.mean()), "per_class": per_class}
    save_json(eda, RESULTS_DIR / "eda_stats.json")
    print(eda)


if __name__ == "__main__":
    main()
