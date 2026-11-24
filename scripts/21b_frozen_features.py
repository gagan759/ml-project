"""Upgrade round, step 2b — frozen DINOv2 ViT-B/14 features + class-balanced logistic regression (same 5 folds).

The backbone is NOT trained. Each photo is embedded once, as in the DINOv2 paper's linear evaluation: the final
CLS token concatenated with the mean of the patch tokens (1536 numbers). A logistic regression
(standardised features, L2, class_weight='balanced') is then fitted per fold; its C is chosen by grouped 4-fold CV
INSIDE the fold's training part (nested), so the out-of-fold predictions stay unbiased.
Writes the same files as 21_train_cv.py (fold<k>.npz/.json, summary.json) under
experiments/cv/<task>/dinov2_b_linear/, plus heads.pt (the 5 heads as tensors) for deployment.

Usage: python scripts/21b_frozen_features.py --task 8class [--device cpu]
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tomatoleaf.config import CLASS_KEYS, DATA_DIR, EXPERIMENTS_DIR, SEED
from tomatoleaf.data import load_images
from tomatoleaf.evaluation import classification_metrics, expected_calibration_error, fit_temperature, softmax
from tomatoleaf.ood import FeatureDensity, energy_score
from tomatoleaf.preprocessing import dihedral_views, eval_transform
from tomatoleaf.probes import far_ood_probes
from tomatoleaf.utils import save_json, set_seed

NAME = "dinov2_b_linear"
ARCH = "timm:vit_base_patch14_dinov2.lvd142m"
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
TASKS = {"8class": list(CLASS_KEYS),
         "5class": ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]}
CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def build_backbone(device):
    import timm
    return timm.create_model(ARCH[5:], pretrained=True, num_classes=0, img_size=SIZE).to(device).eval()


@torch.no_grad()
def embed(model, images, device, views=1, batch=16):
    """CLS token ++ mean patch token of the last layer (after the final norm); averaged over dihedral views."""
    tf = eval_transform(SIZE, MEAN, STD, "resize")
    out = []
    for i in range(0, len(images), batch):
        from torchvision import tv_tensors
        x = tf(tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(images[i:i + batch])).permute(0, 3, 1, 2)))
        x = x.as_subclass(torch.Tensor).to(device)
        b = len(x)
        if views > 1:
            x = dihedral_views(x, views)
        tokens = model.forward_features(x)
        f = torch.cat([tokens[:, 0], tokens[:, model.num_prefix_tokens:].mean(1)], dim=1).float()
        out.append(f.view(views, b, -1).mean(0).cpu() if views > 1 else f.cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS), default="8class")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--views", default="1,8")
    args = parser.parse_args()
    set_seed(SEED)
    torch.set_num_threads(8)
    device = torch.device(args.device)
    keys = TASKS[args.task]
    out_dir = EXPERIMENTS_DIR / "cv" / args.task / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(DATA_DIR / "cv_folds.csv")
    local = {CLASS_KEYS.index(k): i for i, k in enumerate(keys)}
    inside = pool[pool.class_key.isin(keys)].reset_index(drop=True)
    outside = pool[~pool.class_key.isin(keys)].reset_index(drop=True)

    cache = EXPERIMENTS_DIR / "cv" / f"features_{NAME}.npz"  # embeddings of every train+val photo (task-independent)
    views = [int(v) for v in args.views.split(",")]
    model = build_backbone(device)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feats = {v: dict(zip(z["photo_id"], z[f"f{v}"])) for v in views if f"f{v}" in z}
    else:
        feats = {}
    missing = [v for v in views if v not in feats]
    if missing:
        t0 = time.perf_counter()
        X_pool = load_images(pool.path)
        arrays = {}
        for v in views:
            arrays[f"f{v}"] = embed(model, X_pool, device, views=v)
            print(f"embedded {len(pool)} photos with {v} view(s) in {time.perf_counter() - t0:.0f}s", flush=True)
        np.savez_compressed(cache, photo_id=pool.photo_id.to_numpy(), **arrays)
        feats = {v: dict(zip(pool.photo_id, arrays[f"f{v}"])) for v in views}
    _, probes, _ = far_ood_probes()
    probe_f = {v: embed(model, probes, device, views=v) for v in views}

    heads = []
    for k in range(5):
        tr, he = inside[inside.fold != k], inside[inside.fold == k]
        y_tr, y_he = tr.label.map(local).to_numpy(), he.label.map(local).to_numpy()
        F_tr = np.stack([feats[1][p] for p in tr.photo_id])
        pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
        search = GridSearchCV(pipe, {"logisticregression__C": CS}, scoring="f1_macro", cv=inner, n_jobs=-1)
        search.fit(F_tr, y_tr, groups=tr.group_id.to_numpy())
        clf = search.best_estimator_
        density = FeatureDensity(F_tr, y_tr)
        arrays = {"held_y": y_he, "held_photo_id": he.photo_id.to_numpy()}
        for v in views:
            F_he = np.stack([feats[v][p] for p in he.photo_id])
            lg = clf.decision_function(F_he)
            arrays.update({f"held_logits{v}": lg, f"held_feat{v}": F_he.astype(np.float16),
                           f"held_maha{v}": density.mahalanobis(F_he), f"held_knn{v}": density.knn(F_he),
                           f"held_energy{v}": energy_score(lg),
                           f"probe_logits{v}": clf.decision_function(probe_f[v]),
                           f"probe_maha{v}": density.mahalanobis(probe_f[v]), f"probe_knn{v}": density.knn(probe_f[v])})
            oos = outside[outside.fold == k]
            if len(oos):
                F_o = np.stack([feats[v][p] for p in oos.photo_id])
                arrays.update({f"oos_logits{v}": clf.decision_function(F_o), f"oos_maha{v}": density.mahalanobis(F_o),
                               f"oos_knn{v}": density.knn(F_o)})
        if len(outside[outside.fold == k]):
            arrays.update({"oos_photo_id": outside[outside.fold == k].photo_id.to_numpy(),
                           "oos_label": outside[outside.fold == k].label.to_numpy()})
        np.savez_compressed(out_dir / f"fold{k}.npz", **arrays)
        scaler, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
        heads.append({"fold": k, "C": float(search.best_params_["logisticregression__C"]),
                      "mean": torch.from_numpy(scaler.mean_.astype(np.float32)),
                      "scale": torch.from_numpy(scaler.scale_.astype(np.float32)),
                      "coef": torch.from_numpy(lr.coef_.astype(np.float32)),
                      "intercept": torch.from_numpy(lr.intercept_.astype(np.float32)),
                      "density": density.to_dict()})
        met = {v: classification_metrics(y_he, arrays[f"held_logits{v}"].argmax(1), range(len(keys))) for v in views}
        save_json({"fold": k, "train_photos": len(tr), "held_photos": len(he), "train_seconds": 0.0,
                   "best_C": heads[-1]["C"], "held_1view": met[1], "held_tta8": met.get(8, met[1]), "history": []},
                  out_dir / f"fold{k}.json")
        print(f"fold {k}: C={heads[-1]['C']} | held-out macro-F1 {met[1]['f1_macro']:.3f}"
              + (f" (TTA {met[8]['f1_macro']:.3f})" if 8 in met else "")
              + f", acc {met[1]['accuracy']:.3f}" + (f" (TTA {met[8]['accuracy']:.3f})" if 8 in met else ""), flush=True)
    torch.save({"arch": ARCH, "model_kwargs": {"img_size": SIZE, "num_classes": 0}, "input_size": SIZE,
                "mean": MEAN, "std": STD, "class_keys": keys, "heads": heads}, out_dir / "heads.pt")

    # summary in the same format as 21_train_cv.py
    folds = [np.load(out_dir / f"fold{k}.npz", allow_pickle=True) for k in range(5)]
    y = np.concatenate([f["held_y"] for f in folds])
    summary = {"recipe": {"name": NAME, "arch": ARCH, "label": "ViT-B/14 (DINOv2, frozen) + logistic regression",
                          "input_size": SIZE, "aug": "none (frozen features)"},
               "task": args.task, "classes": keys, "photos": int(len(y)), "views": {}}
    for v in views:
        logits = np.concatenate([f[f"held_logits{v}"] for f in folds])
        t = fit_temperature(logits, y)
        probs = softmax(logits, t)
        summary["views"][str(v)] = {**classification_metrics(y, logits.argmax(1), range(len(keys))), "temperature": t,
                                    "ece_raw": expected_calibration_error(softmax(logits), y),
                                    "ece_calibrated": expected_calibration_error(probs, y),
                                    "nll_calibrated": float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())}
    if "8" not in summary["views"]:
        summary["views"]["8"] = summary["views"]["1"]
    summary["per_fold"] = [{"fold": k, "held_photos": int(len(f["held_y"])), "train_seconds": 0.0,
                            "held_1view": classification_metrics(f["held_y"], f["held_logits1"].argmax(1), range(len(keys))),
                            "held_tta8": classification_metrics(f["held_y"], f[f"held_logits{max(views)}"].argmax(1), range(len(keys)))}
                           for k, f in enumerate(folds)]
    summary["train_minutes_total"] = 0.0
    save_json(summary, out_dir / "summary.json")
    v1, v8 = summary["views"]["1"], summary["views"]["8"]
    print(f"OOF {args.task} {NAME}: macro-F1 {v1['f1_macro']:.4f} (TTA {v8['f1_macro']:.4f}) | "
          f"acc {v1['accuracy']:.4f} (TTA {v8['accuracy']:.4f})")


if __name__ == "__main__":
    main()
