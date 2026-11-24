"""Upgrade round, step 3 — choose the deployed model for each task from OUT-OF-FOLD predictions only.

Candidates: every cross-validated recipe (with and without 8-view test-time augmentation) and every probability
ensemble of 2 or 3 recipes (each member with the view setting that was better for it on its own).
Deployed form: a "fold ensemble" — the 5 fold models of each member recipe, averaging calibrated probabilities. Each
fold model has truly held-out data (its own fold), so temperatures and out-of-distribution thresholds are fitted on
photos the model never saw. The 290 test photos are not loaded here.

Selection rule (fixed before scoring the test set, written into the output JSON):
  maximise pooled out-of-fold macro-F1 among candidates whose estimated CPU time per photo (5 fold models per
  member) is within the latency budget; ties (< 0.001) go to higher out-of-fold accuracy, then lower latency.
Also chosen on out-of-fold data: the "Uncertain" threshold (lowest calibrated confidence whose out-of-fold accuracy
on answered photos reaches the target, coverage >= 50%) and the out-of-distribution score (kNN or Mahalanobis,
best AUROC on out-of-scope photos + probe half A).

Usage: python scripts/22_select_models.py --task 8class
Output: experiments/results/selection_<task>.json
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tomatoleaf.config import CLASS_KEYS, DATA_DIR, EXPERIMENTS_DIR, MODELS_DIR, RESULTS_DIR, SEED
from tomatoleaf.data import load_images
from tomatoleaf.evaluation import (
    classification_metrics,
    expected_calibration_error,
    fit_temperature,
    group_bootstrap_ci,
    per_class_metrics,
    reliability_bins,
    selective_accuracy,
    softmax,
)
from tomatoleaf.models import load_checkpoint
from tomatoleaf.ood import FeatureDensity, auroc, fpr_at_tpr
from tomatoleaf.preprocessing import dihedral_views, preprocess_for_cnn
from tomatoleaf.probes import far_ood_probes
from tomatoleaf.recipes import RECIPES
from tomatoleaf.training import infer
from tomatoleaf.utils import save_json, set_seed

CV_DIR = EXPERIMENTS_DIR / "cv"
DEPLOY_DIR = MODELS_DIR / "deploy"
LATENCY_BUDGET_MS = 1000.0  # per photo, CPU, 8 threads, whole fold ensemble
TARGET = {"5class": 0.95, "8class": 0.90}  # accuracy on answered photos for the "Uncertain" threshold
OOD_KEEP = 0.95  # share of in-distribution photos NOT flagged as unusual


def load_run(task, recipe):
    d = CV_DIR / task / recipe
    folds = [np.load(d / f"fold{k}.npz", allow_pickle=True) for k in range(5)]
    return {"dir": d, "folds": folds,
            "y": np.concatenate([f["held_y"] for f in folds]),
            "photo_id": np.concatenate([f["held_photo_id"] for f in folds]),
            "fold": np.concatenate([np.full(len(f["held_y"]), k) for k, f in enumerate(folds)]),
            **{f"logits{v}": np.concatenate([f[f"held_logits{v}"] for f in folds]) for v in ("1", "8")}}


@torch.no_grad()
def cpu_latency(recipe, ckpt_path, views, n=12):
    """Median CPU ms for one photo through one fold model, with the runtime the API uses (ONNX Runtime if available)."""
    import sys
    sys.path.insert(0, str(EXPERIMENTS_DIR.parent))
    from backend.services import onnx_runtime

    torch.set_num_threads(8)
    model, ckpt = load_checkpoint(ckpt_path, "cpu")
    run = model
    if onnx_runtime.enabled():
        run = onnx_runtime.OnnxModel(model, Path(ckpt_path), ckpt["input_size"], threads=8)
    rgb = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    times = []
    for i in range(n + 2):
        t0 = time.perf_counter()
        x = preprocess_for_cnn(rgb, ckpt["input_size"], ckpt["mean"], ckpt["std"], ckpt.get("eval_resize", "resize"))
        run(dihedral_views(x, views) if views > 1 else x)
        if i >= 2:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["5class", "8class"], required=True)
    parser.add_argument("--max-members", type=int, default=3)
    parser.add_argument("--device", default=None, help="cuda (default when available) or cpu")
    args = parser.parse_args()
    task = args.task
    set_seed(SEED)
    pool = pd.read_csv(DATA_DIR / "cv_folds.csv")
    group_of = dict(zip(pool.photo_id, pool.group_id))

    # fine-tuned recipes with all 5 folds (a frozen-feature baseline without fold checkpoints is reported, not deployed)
    recipes = [r for r in RECIPES if all((CV_DIR / task / r / f"fold{k}.npz").exists() for k in range(5))
               and (CV_DIR / task / r / "fold0.pt").exists()]
    print(f"{task}: complete CV runs: {recipes}")
    runs = {r: load_run(task, r) for r in recipes}
    ref = runs[recipes[0]]
    for r in recipes[1:]:  # every run must hold out the same photos in the same order
        assert (runs[r]["photo_id"] == ref["photo_id"]).all(), r
    y, groups = ref["y"], np.array([group_of[p] for p in ref["photo_id"]])
    labels = range(int(y.max()) + 1)
    keys = json.loads((CV_DIR / task / recipes[0] / "summary.json").read_text())["classes"]
    to_global = np.array([CLASS_KEYS.index(k) for k in keys])  # local label -> global class index (for names)

    # ---- per recipe and view: pooled out-of-fold temperature and calibrated probabilities
    calib = {}
    for r in recipes:
        for v in ("1", "8"):
            t = fit_temperature(runs[r][f"logits{v}"], y)
            calib[(r, v)] = {"T": t, "p": softmax(runs[r][f"logits{v}"], t), "p_raw": softmax(runs[r][f"logits{v}"])}

    # ---- CPU latency per recipe and view (one fold model; the fold ensemble costs 5x)
    lat = {}
    for r in recipes:
        for v in (1, 8):
            lat[(r, str(v))] = cpu_latency(RECIPES[r], CV_DIR / task / r / "fold0.pt", v)
        print(f"  CPU latency {r}: 1 view {lat[(r, '1')]:.0f} ms, 8 views {lat[(r, '8')]:.0f} ms (one model)")

    def f1_of(p):
        return classification_metrics(y, p.argmax(1), labels)["f1_macro"]

    best_view = {r: max(("1", "8"), key=lambda v: (f1_of(calib[(r, v)]["p"]), v == "1")) for r in recipes}

    candidates = []
    for r in recipes:
        for v in ("1", "8"):
            candidates.append({"members": [(r, v)]})
    for k in range(2, args.max_members + 1):
        for combo in itertools.combinations(recipes, k):
            candidates.append({"members": [(r, best_view[r]) for r in combo]})
    for c in candidates:
        p = np.mean([calib[m]["p"] for m in c["members"]], axis=0)
        p_raw = np.mean([calib[m]["p_raw"] for m in c["members"]], axis=0)
        pred = p.argmax(1)
        c.update({"key": " + ".join(f"{r}{'@tta' if v == '8' else ''}" for r, v in c["members"]),
                  "oof": {**classification_metrics(y, pred, labels), **group_bootstrap_ci(y, pred, groups, seed=SEED, labels=labels),
                          "ece": expected_calibration_error(p, y), "ece_raw": expected_calibration_error(p_raw, y),
                          "nll": float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())},
                  "latency_ms": 5 * sum(lat[m] for m in c["members"]),
                  "_p": p, "_p_raw": p_raw})
    within = [c for c in candidates if c["latency_ms"] <= LATENCY_BUDGET_MS]
    best_f1 = max(c["oof"]["f1_macro"] for c in within)
    tied = [c for c in within if c["oof"]["f1_macro"] >= best_f1 - 0.001]
    chosen = sorted(tied, key=lambda c: (-c["oof"]["accuracy"], c["latency_ms"]))[0]
    p = chosen["_p"]
    print(f"chosen: {chosen['key']} | OOF macro-F1 {chosen['oof']['f1_macro']:.4f} acc {chosen['oof']['accuracy']:.4f} | "
          f"~{chosen['latency_ms']:.0f} ms CPU")

    # ---- "Uncertain" threshold on out-of-fold calibrated probabilities
    grid = [round(x, 2) for x in np.arange(0.40, 0.991, 0.01)]
    curve = selective_accuracy(p, y, grid)
    ok = [r for r in curve if r["accuracy_kept"] is not None and r["accuracy_kept"] >= TARGET[task] and r["coverage"] >= 0.5]
    if ok:
        thr_row, thr_rule = ok[0], (f"lowest calibrated confidence with out-of-fold accuracy on answered photos >= "
                                    f"{TARGET[task]:.0%} (coverage >= 50%)")
    else:
        thr_row = max((r for r in curve if r["coverage"] >= 0.5), key=lambda r: r["accuracy_kept"])
        thr_rule = f"target {TARGET[task]:.0%} not reachable at >= 50% coverage; threshold with the best accuracy at >= 50% coverage"

    # ---- out-of-distribution scores of the chosen fold ensemble (each member: its own density and threshold)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, probes, halves = far_ood_probes()
    outside = pool[~pool.class_key.isin(keys)]
    X_oos = load_images(outside.path) if len(outside) else None
    member_cfg, in_scores, oos_scores, probe_scores = [], {}, {}, {}
    for r, v in chosen["members"]:
        recipe = RECIPES[r]
        for k in range(5):
            f = runs[r]["folds"][k]
            model, ckpt = load_checkpoint(CV_DIR / task / r / f"fold{k}.pt", device)
            kw = dict(size=recipe.input_size, mean=ckpt["mean"], std=ckpt["std"], resize=recipe.eval_resize)
            # density from this fold model's own training photos (re-computed: not stored in the npz)
            tr = pool[(pool.class_key.isin(keys)) & (pool.fold != k)]
            local = {CLASS_KEYS.index(c): i for i, c in enumerate(keys)}
            # density of single-view TRAINING features (as in 21_train_cv.py); photos are scored with the deployed views
            _, f_tr = infer(model, load_images(tr.path), device, features=True, **kw)
            density = FeatureDensity(f_tr, tr.label.map(local).to_numpy())
            views = int(v)
            deploy = DEPLOY_DIR / task / r / f"fold{k}.pt"  # deployable copy: weights + this fold's feature density
            deploy.parent.mkdir(parents=True, exist_ok=True)
            raw = torch.load(CV_DIR / task / r / f"fold{k}.pt", map_location="cpu", weights_only=True)
            torch.save({**raw, "density": density.to_dict()}, deploy)
            _, f_p = infer(model, probes, device, views=views, batch_size=16, features=True, **kw)
            f_o = infer(model, X_oos, device, views=views, batch_size=16, features=True, **kw)[1] if X_oos is not None else None
            for method in ("knn", "maha"):
                score = density.knn if method == "knn" else density.mahalanobis
                held = f[f"held_{method}{v}"]
                thr = float(np.quantile(held, OOD_KEEP))
                in_scores.setdefault(method, []).append((k, held / thr))
                probe_scores.setdefault(method, []).append(score(f_p) / thr)
                if f_o is not None:
                    oos_scores.setdefault(method, []).append(score(f_o) / thr)
                member_cfg.append({"recipe": r, "fold": k, "views": views, "method": method, "threshold": thr})
            del model
            torch.cuda.empty_cache()

    ood_eval = {}
    for method in ("knn", "maha"):
        # in-distribution: each held-out photo has ONE unseen member per recipe -> mean ratio over recipes only
        per_recipe = {}
        idx = 0
        for (r, _v) in chosen["members"]:
            ratios = np.concatenate([x for k, x in in_scores[method][idx: idx + 5]])
            per_recipe[r] = ratios
            idx += 5
        in_ratio = np.mean(list(per_recipe.values()), axis=0)
        probe_ratio = np.mean(probe_scores[method], axis=0)  # all fold models of all members (as deployed)
        oos_ratio = np.mean(oos_scores[method], axis=0) if oos_scores else None
        scale = float(np.quantile(in_ratio, OOD_KEEP))  # re-scale so OOD_KEEP of in-distribution photos stay below 1
        a = probe_ratio[halves == "A"]
        out_a = np.r_[a, oos_ratio] if oos_ratio is not None else a
        ood_eval[method] = {
            "scale": scale, "auroc_select": auroc(in_ratio, out_a),
            "auroc_far_A": auroc(in_ratio, a), "auroc_near": auroc(in_ratio, oos_ratio) if oos_ratio is not None else None,
            "fpr95_select": fpr_at_tpr(in_ratio, out_a),
            "flagged_in": float((in_ratio / scale > 1).mean()),
            "flagged_far_A": float((a / scale > 1).mean()),
            "flagged_near": float((oos_ratio / scale > 1).mean()) if oos_ratio is not None else None,
            "probe_ratios": {n: float(x / scale) for n, x in zip(far_ood_probes()[0], probe_ratio)}}
    method = max(ood_eval, key=lambda m: ood_eval[m]["auroc_select"])
    print(f"OOD method: {method} | " + " | ".join(f"{m}: AUROC {e['auroc_select']:.3f}" for m, e in ood_eval.items()))

    candidates_out = sorted(({k: v for k, v in c.items() if not k.startswith("_")} | {"members": [list(m) for m in c["members"]]}
                             for c in candidates), key=lambda c: -c["oof"]["f1_macro"])
    out = {
        "task": task, "classes": keys, "photos": int(len(y)), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_rule": (f"Highest pooled out-of-fold macro-F1 among candidates (single recipes with/without 8-view "
                           f"TTA, and probability ensembles of 2-3 recipes) whose estimated CPU time for the whole fold "
                           f"ensemble is <= {LATENCY_BUDGET_MS:.0f} ms per photo; ties within 0.001 go to higher "
                           f"out-of-fold accuracy, then lower latency. Decided on out-of-fold data only; the test set "
                           f"is scored once afterwards."),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "recipes": {r: {"best_view": best_view[r], "temperature_1": calib[(r, "1")]["T"], "temperature_8": calib[(r, "8")]["T"],
                        "f1_1": f1_of(calib[(r, "1")]["p"]), "f1_8": f1_of(calib[(r, "8")]["p"]),
                        "latency_1": lat[(r, "1")], "latency_8": lat[(r, "8")]} for r in recipes},
        "candidates": candidates_out,
        "chosen": {"key": chosen["key"], "members": [{"recipe": r, "views": int(v), "temperature": calib[(r, v)]["T"]}
                                                      for r, v in chosen["members"]],
                   "oof": chosen["oof"], "latency_ms_estimate": chosen["latency_ms"],
                   "per_class_oof": per_class_metrics(to_global[y], to_global[p.argmax(1)], list(to_global)),
                   "reliability_oof": reliability_bins(p, y),
                   "reliability_oof_raw": reliability_bins(chosen["_p_raw"], y)},
        "selective": {"target": TARGET[task], "rule": thr_rule, "threshold": thr_row["threshold"], "oof": thr_row,
                      "curve_oof": curve},
        "ood": {"method": method, "keep": OOD_KEEP, "evaluation_oof": ood_eval,
                "thresholds": [m for m in member_cfg if m["method"] == method],
                "scale": ood_eval[method]["scale"]},
    }
    save_json(out, RESULTS_DIR / f"selection_{task}.json")
    print(f"wrote {RESULTS_DIR / f'selection_{task}.json'}")


if __name__ == "__main__":
    main()
