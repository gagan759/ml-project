"""Phase 6 — interpretability.

ML : Random Forest impurity importances (HOG importance mapped back onto the image grid; colour-bin importances)
     and permutation importance per FEATURE GROUP for the best ML model on the validation photos.
DL : Grad-CAM on test photos for the custom CNN and fine-tuned MobileNetV2, misclassified examples, and a
     shortcut check on the authors' black-cornered augmented copies (model trained on originals vs official split).
Outputs: reports/figures/interpretability/*.png, experiments/results/interpretability.json
"""
import colorsys

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score

from tomatoleaf.config import CLASS_NAMES, DATA_DIR, FIGURES_DIR, MODELS_DIR, NUM_CLASSES, RESULTS_DIR, SEED
from tomatoleaf.data import load_images, load_index, load_splits
from tomatoleaf.features import FEATURE_SETS, H_BINS, HOG_DIM, HOG_PARAMS, S_BINS, V_BINS
from tomatoleaf.gradcam import GradCAM, overlay
from tomatoleaf.models import gradcam_layer, load_checkpoint
from tomatoleaf.plotting import INK, INK2, MUTED, SERIES, SHORT_NAMES, apply_style, plt, save
from tomatoleaf.preprocessing import dihedral_views, preprocess_for_cnn
from tomatoleaf.utils import load_json, save_json, set_seed

OUT = FIGURES_DIR / "interpretability"


def feature_groups(feature_set: str) -> dict:
    groups, start = {}, 0
    if "hog" in FEATURE_SETS[feature_set]:
        groups["HOG (shape/texture)"] = np.arange(0, HOG_DIM)
        start = HOG_DIM
    if "color" in FEATURE_SETS[feature_set]:
        for name, width in [("Hue histogram", H_BINS), ("Saturation histogram", S_BINS),
                            ("Brightness histogram", V_BINS), ("L*a*b* colour moments", 9)]:
            groups[name] = np.arange(start, start + width)
            start += width
    return groups


def hog_importance_map(importances: np.ndarray) -> np.ndarray:
    """Spread each HOG block's total importance over the 32x32-pixel patch it describes (128x128 grid)."""
    blocks = importances.reshape(7, 7, 2, 2, HOG_PARAMS["orientations"]).sum(axis=(2, 3, 4))
    heat = np.zeros((128, 128))
    for by in range(7):
        for bx in range(7):
            heat[by * 16:by * 16 + 32, bx * 16:bx * 16 + 32] += blocks[by, bx]
    return heat / heat.max()


def load_dl(key, protocol):
    model, ckpt = load_checkpoint(MODELS_DIR / "dl" / protocol / f"{key}.pt")
    return model, ckpt["arch"], int(ckpt["input_size"])


CAM_MODELS = [("custom_cnn", "Custom CNN"), ("mobilenetv2_ft", "MobileNetV2"), ("convnext_tiny_v2", "ConvNeXt-Tiny")]


def main():
    set_seed(SEED)
    apply_style()
    out = {}
    splits = load_splits()
    va, te = splits[splits.split == "val"], splits[splits.split == "test"]
    feats = np.load(DATA_DIR / "features.npz", allow_pickle=True)
    features = {"hog": feats["hog"], "color": feats["color"]}
    ml_val = load_json(RESULTS_DIR / "ml_validation.json")
    final = load_json(RESULTS_DIR / "final_results.json")

    # --- Random Forest impurity importances
    rf_sel = ml_val["selected"]["rf"]
    rf = joblib.load(MODELS_DIR / rf_sel["file"])["pipeline"].named_steps["clf"]
    imp = rf.feature_importances_
    groups = feature_groups(rf_sel["feature_set"])
    out["rf_feature_set"] = rf_sel["feature_set"]
    out["rf_group_importance"] = {g: float(imp[idx].sum()) for g, idx in groups.items()}
    panels = (["hog"] if "HOG (shape/texture)" in groups else []) + (["hue"] if "Hue histogram" in groups else [])
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(5.2 * (len(panels) + 1), 4.2))
    axes = np.atleast_1d(axes)
    axg = axes[-1]
    gnames = list(out["rf_group_importance"])
    gvals = [out["rf_group_importance"][g] for g in gnames]
    axg.barh(range(len(gnames)), gvals, color=SERIES[0], height=0.6)
    for i, v in enumerate(gvals):
        axg.text(v, i, f" {v:.2f}", va="center", fontsize=8.5, color=INK2)
    axg.set_yticks(range(len(gnames)), gnames)
    axg.invert_yaxis()
    axg.grid(axis="y", visible=False)
    axg.set_xlabel("Share of total impurity decrease")
    axg.set_title("Random Forest: importance by feature group")
    pi = 0
    if "hog" in panels:
        train_imgs = load_images(splits[splits.split == "train"].path[:300])
        heat = hog_importance_map(imp[groups["HOG (shape/texture)"]])
        axes[pi].imshow(train_imgs.mean(axis=0).astype(np.uint8), extent=(0, 128, 128, 0))
        axes[pi].imshow(heat, cmap="inferno", alpha=0.55, extent=(0, 128, 128, 0))
        axes[pi].set_axis_off()
        axes[pi].set_title("RF: where HOG importance lies\n(over the mean training image)")
        pi += 1
    if "hue" in panels:
        hue_imp = imp[groups["Hue histogram"]]
        centers = (np.arange(H_BINS) + 0.5) * (360 / H_BINS)
        bar_colors = [colorsys.hsv_to_rgb(c / 360, 0.75, 0.85) for c in centers]
        axes[pi].bar(centers, hue_imp, width=360 / H_BINS * 0.85, color=bar_colors)
        axes[pi].set_xlim(0, 200)
        axes[pi].set_xlabel("Hue of leaf pixels (degrees)")
        axes[pi].set_ylabel("Impurity importance")
        axes[pi].set_title("RF: importance of each hue bin\n(yellow ~ 50-60 deg, green ~ 80-140 deg)")
    fig.tight_layout()
    save(fig, OUT / "rf_feature_importance.png")

    # --- Permutation importance by feature group, best ML model, validation photos
    ml_results = [r for r in final["models"] if r["family"] == "Machine Learning"]
    best_ml = max(ml_results, key=lambda r: r["val"]["f1_macro"])
    sel = ml_val["selected"][best_ml["key"]]
    pipe = joblib.load(MODELS_DIR / sel["file"])["pipeline"]
    if "clf__n_jobs" in pipe.get_params():
        pipe.set_params(clf__n_jobs=-1)
    X_va = np.hstack([features[p][va.index_row.to_numpy()] for p in FEATURE_SETS[sel["feature_set"]]])
    y_va = va.label.to_numpy()
    base = f1_score(y_va, pipe.predict_proba(X_va).argmax(1), average="macro")
    rng = np.random.default_rng(SEED)
    perm = {}
    for g, idx in feature_groups(sel["feature_set"]).items():
        drops = []
        for _ in range(10):
            Xp = X_va.copy()
            Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
            drops.append(base - f1_score(y_va, pipe.predict_proba(Xp).argmax(1), average="macro"))
        perm[g] = {"mean_drop": float(np.mean(drops)), "std": float(np.std(drops))}
    out["permutation_importance"] = {"model": best_ml["name"], "feature_set": sel["feature_set"],
                                     "baseline_val_f1_macro": float(base), "groups": perm}
    fig, ax = plt.subplots(figsize=(8, 3.4))
    names = list(perm)
    ax.barh(range(len(names)), [perm[n]["mean_drop"] for n in names], xerr=[perm[n]["std"] for n in names],
            color=SERIES[0], height=0.6, ecolor=INK2, capsize=3)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in validation macro-F1 when the group is shuffled")
    ax.set_title(f"Permutation importance by feature group: {best_ml['name']} ({sel['feature_set']})")
    save(fig, OUT / "permutation_importance.png")

    # --- Grad-CAM on test photos
    te_imgs = load_images(te.path)
    y_te = te.label.to_numpy()
    cams = {}
    for key, _ in CAM_MODELS:
        if (MODELS_DIR / "dl" / "clean" / f"{key}.pt").exists():
            model, arch, size = load_dl(key, "clean")
            cams[key] = (model, GradCAM(model, gradcam_layer(model, arch)), size)
    cam_rows = [(k, label) for k, label in CAM_MODELS if k in cams]
    picks = [np.flatnonzero(y_te == c)[:2] for c in range(NUM_CLASSES)]
    fig, axes = plt.subplots(NUM_CLASSES, 6, figsize=(10.5, 15.5))
    for c in range(NUM_CLASSES):
        for e in range(2):
            col = e * 3
            if e >= len(picks[c]):
                for j in range(3):
                    axes[c, col + j].set_axis_off()
                continue
            k = picks[c][e]
            img = te_imgs[k]
            axes[c, col].imshow(img)
            axes[c, col].set_title(f"true: {SHORT_NAMES[c]}", fontsize=7.5, loc="center", fontweight="normal", color=INK)
            for j, key in enumerate(("custom_cnn", "mobilenetv2_ft"), start=1):
                cam, idx, probs = cams[key][1](preprocess_for_cnn(img, cams[key][2]))
                axes[c, col + j].imshow(overlay(img, cam))
                color = INK if idx == c else SERIES[7]
                label = "CNN" if key == "custom_cnn" else "MobileNetV2"
                axes[c, col + j].set_title(f"{label}: {SHORT_NAMES[idx]} {probs[idx]:.0%}", fontsize=7.5, loc="center",
                                           fontweight="normal", color=color)
        for j in range(6):
            axes[c, j].set_axis_off()
    fig.suptitle("Grad-CAM on test photos (bright = regions that raised the predicted class score; red title = wrong)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, OUT / "gradcam_test_examples_full.png")

    # Compact version for the report body: one test photo per class (columns) x [photo + one row per model]
    nrows = 1 + len(cam_rows)
    fig, axes = plt.subplots(nrows, NUM_CLASSES, figsize=(16, 2.2 * nrows))
    for c in range(NUM_CLASSES):
        img = te_imgs[picks[c][0]]
        axes[0, c].imshow(img)
        axes[0, c].set_title(SHORT_NAMES[c], fontsize=10, loc="center", fontweight="normal", color=INK)
        for row, (key, _) in enumerate(cam_rows, start=1):
            cam, idx, probs = cams[key][1](preprocess_for_cnn(img, cams[key][2]))
            axes[row, c].imshow(overlay(img, cam))
            axes[row, c].set_title(f"{SHORT_NAMES[idx]} {probs[idx]:.0%}", fontsize=8.5, loc="center", fontweight="normal",
                                   color=INK if idx == c else SERIES[7])
        for row in range(nrows):
            axes[row, c].set_axis_off()
    for row, label in enumerate(["Test photo"] + [label for _, label in cam_rows]):
        fig.text(0.004, 1 - (row + 0.55) / nrows, label, rotation=90, va="center", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.025, right=0.995, top=0.96, bottom=0.01, wspace=0.05, hspace=0.25)
    save(fig, OUT / "gradcam_test_examples.png")

    # Misclassified test photos of the SELECTED deep model (its own TTA predictions; Grad-CAM of the first member)
    sel = next(m for m in final["models"] if m["selected"])
    members = sel.get("members") or [{"key": sel["key"], "tta_views": 1, "temperature": sel.get("temperature") or 1.0}]
    member_models = []
    for m in members:
        model, arch, size = load_dl(m["key"], "clean")
        member_models.append((model, size, m["tta_views"], m.get("temperature") or 1.0))
    base_key = members[0]["key"]
    if base_key not in cams:
        model, arch, size = load_dl(base_key, "clean")
        cams[base_key] = (model, GradCAM(model, gradcam_layer(model, arch)), size)
    wrong = []
    for k, img in enumerate(te_imgs):
        probs = []
        with torch.no_grad():
            for model, size, views, temp in member_models:
                x = preprocess_for_cnn(img, size)
                logits = model(dihedral_views(x, views) if views > 1 else x).float().mean(0) / temp
                probs.append(torch.softmax(logits, 0).numpy())
        p = np.mean(probs, axis=0)
        idx = int(p.argmax())
        if idx != y_te[k]:
            cam, _, _ = cams[base_key][1](preprocess_for_cnn(img, cams[base_key][2]), class_index=idx)
            wrong.append((k, idx, float(p[idx]), cam))
    out["selected_model"] = sel["name"]
    out["selected_test_errors"] = [{"photo_id": te.photo_id.iloc[k], "true": CLASS_NAMES[y_te[k]],
                                    "predicted": CLASS_NAMES[i], "confidence": p} for k, i, p, _ in wrong]
    if wrong:
        n = min(len(wrong), 8)
        fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8), squeeze=False)
        for j, (k, idx, p, cam) in enumerate(wrong[:n]):
            axes[0, j].imshow(te_imgs[k])
            axes[0, j].set_title(f"true: {SHORT_NAMES[y_te[k]]}", fontsize=8, loc="center", fontweight="normal")
            axes[1, j].imshow(overlay(te_imgs[k], cam))
            axes[1, j].set_title(f"pred: {SHORT_NAMES[idx]} {p:.0%}", fontsize=8, loc="center", fontweight="normal", color=SERIES[7])
            axes[0, j].set_axis_off()
            axes[1, j].set_axis_off()
        fig.suptitle(f"{sel['name']}: {len(wrong)} misclassified test photos (first {n} shown) with Grad-CAM",
                     x=0.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout()
        save(fig, OUT / "gradcam_errors.png")

    # Shortcut check on black-cornered augmented copies
    official_ckpt = MODELS_DIR / "dl" / "official" / "mobilenetv2_ft.pt"
    if official_ckpt.exists():
        index = load_index()
        copies = index[~index.is_original & (index.official_split == "test")]
        imgs = load_images(copies.path)
        corners = np.stack([imgs[:, :12, :12], imgs[:, :12, -12:], imgs[:, -12:, :12], imgs[:, -12:, -12:]], axis=1)
        dark = corners.reshape(len(imgs), -1).mean(axis=1) < 25
        choose = np.flatnonzero(dark)[:4]
        out["black_corner_copies_in_official_test"] = int(dark.sum())
        off_model, off_arch, off_size = load_dl("mobilenetv2_ft", "official")
        off_gc = GradCAM(off_model, gradcam_layer(off_model, off_arch))
        fig, axes = plt.subplots(3, len(choose), figsize=(2.6 * len(choose), 8.2), squeeze=False)
        for j, k in enumerate(choose):
            img, true = imgs[k], copies.label.iloc[k]
            axes[0, j].imshow(img)
            axes[0, j].set_title(f"copy, true: {SHORT_NAMES[true]}", fontsize=8, loc="center", fontweight="normal")
            for row, (gcam, label) in enumerate([(cams["mobilenetv2_ft"][1], "ours (originals)"), (off_gc, "official split")], start=1):
                cam, idx, probs = gcam(preprocess_for_cnn(img, off_size))
                axes[row, j].imshow(overlay(img, cam))
                axes[row, j].set_title(f"{label}: {SHORT_NAMES[idx]} {probs[idx]:.0%}", fontsize=7.5, loc="center",
                                       fontweight="normal", color=INK if idx == true else SERIES[7])
            for row in range(3):
                axes[row, j].set_axis_off()
        fig.suptitle("Shortcut check: Grad-CAM on the authors' rotated copies (black corners)", x=0.01, ha="left",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        save(fig, OUT / "gradcam_shortcut_check.png")

    save_json(out, RESULTS_DIR / "interpretability.json")
    print(out)


if __name__ == "__main__":
    main()
