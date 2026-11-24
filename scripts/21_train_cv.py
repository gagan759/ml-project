"""Upgrade round, step 2 — train one recipe with grouped 5-fold cross-validation on the train+val photos.

For every fold k: train on the other 4 folds for a fixed number of epochs (no early stopping), then record for the
held-out fold the logits (1 view and 8-view TTA), the penultimate features and OOD scores (Mahalanobis / kNN fitted
on the fold's own training features). Out-of-scope photos (5-class task) and far-OOD probe images are scored too.
The test photos are never loaded.

--full trains the same recipe once on ALL train+val photos (the deployable model) and stores its feature density.

Usage:  python scripts/21_train_cv.py --recipe convnext_t_in22k_robust --task 8class
        python scripts/21_train_cv.py --recipe convnext_t_in22k_robust --task 5class --full
Outputs: experiments/cv/<task>/<recipe>/fold<k>.npz|.pt|.json, summary.json;
         experiments/models/final/<task>__<recipe>.pt (with --full)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from tomatoleaf.config import CLASS_KEYS, DATA_DIR, EXPERIMENTS_DIR, MODELS_DIR, SEED
from tomatoleaf.crop import leaf_crop_batch
from tomatoleaf.data import load_images as _load_images
from tomatoleaf.evaluation import classification_metrics, expected_calibration_error, fit_temperature, softmax
from tomatoleaf.models import build_model, count_parameters, normalisation
from tomatoleaf.ood import FeatureDensity, energy_score
from tomatoleaf.probes import far_ood_probes
from tomatoleaf.recipes import RECIPES
from tomatoleaf.training import fit_recipe, infer
from tomatoleaf.utils import save_json, set_seed

TASKS = {
    "8class": list(CLASS_KEYS),
    # Scope fixed in the original study on validation F1 only (scripts/13_five_class.py); kept for comparability.
    "5class": ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"],
}
CV_DIR = EXPERIMENTS_DIR / "cv"
FINAL_DIR = MODELS_DIR / "final"


def task_frame(task: str) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """(in-scope photos with local labels, out-of-scope photos, class keys) from the CV pool."""
    pool = pd.read_csv(DATA_DIR / "cv_folds.csv")
    keys = TASKS[task]
    local = {CLASS_KEYS.index(k): i for i, k in enumerate(keys)}
    inside = pool[pool.class_key.isin(keys)].copy()
    inside["y"] = inside.label.map(local)
    return inside.reset_index(drop=True), pool[~pool.class_key.isin(keys)].reset_index(drop=True), keys


CROP = {"on": False}  # set per run from the recipe: the same leaf crop for every image the networks see


def load_images(paths):
    images = _load_images(paths)
    return leaf_crop_batch(images) if CROP["on"] else images


def save_weights(model, recipe, keys, path, extra=None):
    mean, std = normalisation(recipe.arch)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"arch": recipe.arch, "model_kwargs": recipe.model_kwargs, "recipe": recipe.to_dict(),
                "state_dict": state, "class_keys": keys, "input_size": recipe.input_size,
                "eval_resize": recipe.eval_resize, "mean": mean, "std": std, "weights_dtype": "float16",
                **(extra or {})}, path)


def scores_for(model, images, recipe, device, density=None):
    """Logits / features (1 view and 8 views) and OOD scores for a batch of uint8 images."""
    mean, std = normalisation(recipe.arch)
    kw = dict(size=recipe.input_size, mean=mean, std=std, resize=recipe.eval_resize, features=True)
    l1, f1 = infer(model, images, device, views=1, **kw)
    l8, f8 = infer(model, images, device, views=8, batch_size=16, **kw)
    out = {"logits1": l1, "logits8": l8, "feat1": f1.astype(np.float16), "feat8": f8.astype(np.float16)}
    if density is not None:
        out.update({"maha1": density.mahalanobis(f1), "maha8": density.mahalanobis(f8),
                    "knn1": density.knn(f1), "knn8": density.knn(f8),
                    "energy1": energy_score(l1), "energy8": energy_score(l8)})
    return out


def run_fold(recipe, task, fold, device, inside, outside, keys, probes, log):
    out_dir = CV_DIR / task / recipe.name
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"fold{fold}.npz").exists():
        log(f"fold {fold}: already done, skipping")
        return
    set_seed(SEED + fold)
    tr, he = inside[inside.fold != fold], inside[inside.fold == fold]
    X_tr, X_he = load_images(tr.path), load_images(he.path)
    y_tr, y_he = tr.y.to_numpy(), he.y.to_numpy()
    model = build_model(recipe.arch, pretrained=True, num_classes=len(keys), **recipe.model_kwargs).to(device)
    mean, std = normalisation(recipe.arch)

    def monitor(m):
        lg = infer(m, X_he, device, size=recipe.input_size, mean=mean, std=std, resize=recipe.eval_resize)
        met = classification_metrics(y_he, lg.argmax(1), range(len(keys)))
        m.train()
        return {"val_f1": met["f1_macro"], "val_acc": met["accuracy"]}

    X_oe = None
    if recipe.outlier_weight > 0:  # outlier exposure: the TRAINING folds' out-of-scope photos only
        oe = outside[outside.fold != fold]
        if not len(oe):
            raise ValueError(f"{recipe.name}: outlier exposure needs out-of-scope photos (focused task only)")
        X_oe = load_images(oe.path)
        log(f"fold {fold}: outlier exposure with {len(oe)} out-of-scope training photos")
    log(f"fold {fold}: {len(tr)} train / {len(he)} held-out photos | {recipe.label}")
    model, info = fit_recipe(model, recipe, X_tr, y_tr, device=device, seed=SEED + fold, monitor=monitor, log=log,
                             outliers=X_oe)

    _, f_tr = infer(model, X_tr, device, size=recipe.input_size, mean=mean, std=std, resize=recipe.eval_resize,
                    features=True)
    density = FeatureDensity(f_tr, y_tr)
    held = scores_for(model, X_he, recipe, device, density)
    arrays = {f"held_{k}": v for k, v in held.items()}
    arrays.update({"held_photo_id": he.photo_id.to_numpy(), "held_y": y_he})
    oos = outside[outside.fold == fold]
    if len(oos):
        s = scores_for(model, load_images(oos.path), recipe, device, density)
        arrays.update({f"oos_{k}": v for k, v in s.items()})
        arrays.update({"oos_photo_id": oos.photo_id.to_numpy(), "oos_label": oos.label.to_numpy()})
    s = scores_for(model, probes, recipe, device, density)
    arrays.update({f"probe_{k}": v for k, v in s.items()})
    np.savez_compressed(out_dir / f"fold{fold}.npz", **arrays)
    save_weights(model, recipe, keys, out_dir / f"fold{fold}.pt", {"fold": fold, "task": task})
    met1 = classification_metrics(y_he, held["logits1"].argmax(1), range(len(keys)))
    met8 = classification_metrics(y_he, held["logits8"].argmax(1), range(len(keys)))
    save_json({"fold": fold, "train_photos": len(tr), "held_photos": len(he),
               "outlier_photos": 0 if X_oe is None else len(X_oe), "train_seconds": info["train_seconds"],
               "held_1view": met1, "held_tta8": met8, "history": info["history"],
               "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None},
              out_dir / f"fold{fold}.json")
    log(f"fold {fold}: held-out macro-F1 {met1['f1_macro']:.3f} (TTA {met8['f1_macro']:.3f}), "
        f"acc {met1['accuracy']:.3f} (TTA {met8['accuracy']:.3f}), {info['train_seconds'] / 60:.1f} min")


def summarise(recipe, task, keys, log):
    out_dir = CV_DIR / task / recipe.name
    folds = [np.load(out_dir / f"fold{k}.npz", allow_pickle=True) for k in range(5)]
    y = np.concatenate([f["held_y"] for f in folds])
    summary = {"recipe": recipe.to_dict(), "task": task, "classes": keys, "photos": int(len(y)), "views": {}}
    for v in ("1", "8"):
        logits = np.concatenate([f[f"held_logits{v}"] for f in folds])
        t = fit_temperature(logits, y)
        probs = softmax(logits, t)
        summary["views"][v] = {**classification_metrics(y, logits.argmax(1), range(len(keys))),
                               "temperature": t, "ece_raw": expected_calibration_error(softmax(logits), y),
                               "ece_calibrated": expected_calibration_error(probs, y),
                               "nll_calibrated": float(-np.log(probs[np.arange(len(y)), y] + 1e-12).mean())}
    per_fold = [json.loads((out_dir / f"fold{k}.json").read_text()) for k in range(5)]
    summary["per_fold"] = [{k: p[k] for k in ("fold", "held_photos", "train_seconds", "held_1view", "held_tta8")}
                           for p in per_fold]
    summary["train_minutes_total"] = sum(p["train_seconds"] for p in per_fold) / 60
    save_json(summary, out_dir / "summary.json")
    v1, v8 = summary["views"]["1"], summary["views"]["8"]
    log(f"OOF {task} {recipe.name}: macro-F1 {v1['f1_macro']:.4f} (TTA {v8['f1_macro']:.4f}) | "
        f"acc {v1['accuracy']:.4f} (TTA {v8['accuracy']:.4f})")


def train_full(recipe, task, device, inside, keys, log):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    path = FINAL_DIR / f"{task}__{recipe.name}.pt"
    if path.exists():
        log(f"{path.name} exists, skipping")
        return
    set_seed(SEED + 100)
    X, y = load_images(inside.path), inside.y.to_numpy()
    model = build_model(recipe.arch, pretrained=True, num_classes=len(keys), **recipe.model_kwargs).to(device)
    log(f"full training: {len(X)} photos | {recipe.label}")
    model, info = fit_recipe(model, recipe, X, y, device=device, seed=SEED + 100, log=log)
    mean, std = normalisation(recipe.arch)
    _, feats = infer(model, X, device, size=recipe.input_size, mean=mean, std=std, resize=recipe.eval_resize,
                     features=True)
    density = FeatureDensity(feats, y).to_dict()
    save_weights(model, recipe, keys, path, {"task": task, "density": density, "train_photos": int(len(X)),
                                             "train_seconds": info["train_seconds"],
                                             "params": count_parameters(model)})
    save_json({"recipe": recipe.to_dict(), "task": task, "train_photos": int(len(X)), "history": info["history"],
               "train_seconds": info["train_seconds"]}, FINAL_DIR / f"{task}__{recipe.name}.json")
    log(f"saved {path} ({info['train_seconds'] / 60:.1f} min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=list(RECIPES), required=True)
    parser.add_argument("--task", choices=list(TASKS), default="8class")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--full", action="store_true", help="train once on all train+val photos instead of CV")
    args = parser.parse_args()
    recipe = RECIPES[args.recipe]
    CROP["on"] = recipe.crop == "leaf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    inside, outside, keys = task_frame(args.task)
    log_path = EXPERIMENTS_DIR / "logs" / f"cv_{args.task}_{recipe.name}{'_full' if args.full else ''}.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    if args.full:
        train_full(recipe, args.task, device, inside, keys, log)
        return
    _, probes, _ = far_ood_probes()
    if CROP["on"]:
        probes = leaf_crop_batch(probes)
    for fold in [int(f) for f in args.folds.split(",")]:
        run_fold(recipe, args.task, fold, device, inside, outside, keys, probes, log)
    if all((CV_DIR / args.task / recipe.name / f"fold{k}.npz").exists() for k in range(5)):
        summarise(recipe, args.task, keys, log)


if __name__ == "__main__":
    main()
