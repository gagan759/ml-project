"""Phase 4c — focused 5-class model (Healthy + 4 conditions) on the same leakage-free split.

Scope rule (decided on VALIDATION data only, never the test set): keep Healthy plus the four conditions that the
selected 8-class model recognised best on the validation photographs (highest validation F1). Dropped: Early Blight
(62 photographs), Late Blight (113; half of its validation photographs were predicted as Spotted Wilt) and
Potassium Deficiency (9 photographs in the whole dataset).

Candidates, all trained on the same 5-class training photographs:
  classical: Logistic Regression / SVM / Random Forest on the feature set each used in the 8-class study, re-tuned
             with grouped 5-fold cross-validation;
  deep:      ConvNeXt-Tiny and EfficientNet-B0 (scripts/07b_train_dl_v2.py --classes ...), each with and without
             8-view test-time augmentation (TTA), plus their probability-averaged ensemble.
The final model is the candidate with the best validation macro-F1 (ties: validation accuracy). The test photographs
are scored once, after selection, exactly as in the 8-class study.
Outputs: experiments/results/five_class_results.json, reports/figures/five_class/*.png, experiments/models/ml/*_5class.joblib
"""
import sys
import time
from importlib import import_module

import joblib
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from tomatoleaf.config import CLASS_KEYS, CLASS_NAMES, DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, ROOT, SEED
from tomatoleaf.data import load_images, load_splits
from tomatoleaf.evaluation import (classification_metrics, confusion, expected_calibration_error, fit_temperature,
                                   group_bootstrap_ci, paired_bootstrap_diff, per_class_metrics, selective_accuracy,
                                   softmax)
from tomatoleaf.features import extract_features, feature_vector
from tomatoleaf.models import load_checkpoint
from tomatoleaf.plotting import (AXIS, DL_COLOR, INK2, ML_COLOR, MUTED, SERIES, SHORT_NAMES, apply_style,
                                 plot_confusion, plt, save)
from tomatoleaf.preprocessing import dihedral_views, eval_transform, preprocess_for_cnn
from tomatoleaf.training import make_loader
from tomatoleaf.utils import load_json, save_json, set_seed

sys.path.insert(0, str(ROOT / "scripts"))
ml06 = import_module("06_train_ml")

KEEP = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
KEEP_IDX = [CLASS_KEYS.index(k) for k in KEEP]
TO_GLOBAL = np.array(KEEP_IDX)
DL_ARCHS = [("convnext_tiny_5class", "ConvNeXt-Tiny"), ("efficientnet_b0_5class", "EfficientNet-B0")]
OUT = FIGURES_DIR / "five_class"


@torch.no_grad()
def logits_for(model, images, device, size, views):
    """Logits with the same CPU preprocessing as the API; views > 1 averages the 8 flips/rotations (TTA)."""
    loader = make_loader(images, np.zeros(len(images), dtype=int), eval_transform(size), 16, shuffle=False)
    out = []
    for x, _ in loader:
        x, b = x.to(device), len(x)
        if views > 1:
            x = dihedral_views(x, views)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(x).float()
        out.append((lg.view(views, b, -1).mean(0) if views > 1 else lg).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def cpu_latency_ms(model, images, size, views, n=20):
    model = model.to("cpu").eval()
    times = []
    for i, img in enumerate(images[: n + 3]):
        t0 = time.perf_counter()
        x = preprocess_for_cnn(img, size)
        model(dihedral_views(x, views) if views > 1 else x)
        if i >= 3:  # skip warm-up calls
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def val_f1_by_class(cm):
    cm = np.asarray(cm)
    out = {}
    for i, key in enumerate(CLASS_KEYS):
        tp = cm[i, i]
        prec, rec = tp / max(cm[:, i].sum(), 1), tp / max(cm[i].sum(), 1)
        out[key] = float(2 * prec * rec / max(prec + rec, 1e-9))
    return out


def main():
    set_seed(SEED)
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final8 = load_json(RESULTS_DIR / "final_results.json")
    models8 = {m["key"]: m for m in final8["models"]}
    sel8 = next(m for m in final8["models"] if m["selected"])
    val_f1_8 = val_f1_by_class(sel8["val_confusion_matrix"])
    others = sorted((k for k in CLASS_KEYS if k != "healthy"), key=lambda k: -val_f1_8[k])
    assert sorted(KEEP) == sorted(["healthy"] + others[:4]), f"scope rule mismatch: {others[:4]}"

    splits = load_splits()
    s5 = splits[splits.class_key.isin(KEEP)]
    tr, va, te = (s5[s5.split == x] for x in ("train", "val", "test"))
    local = {g: i for i, g in enumerate(KEEP_IDX)}
    y_tr = tr.label.map(local).to_numpy()
    yg_va, yg_te = va.label.to_numpy(), te.label.to_numpy()
    yl_va, yl_te = np.vectorize(local.get)(yg_va), np.vectorize(local.get)(yg_te)
    g_tr, g_te = tr.group_id.to_numpy(), te.group_id.to_numpy()
    X_va_img, X_te_img = load_images(va.path), load_images(te.path)

    cands = {}
    # ---- classical ML (same recipe as scripts/06_train_ml.py) ----
    feats = np.load(DATA_DIR / "features.npz", allow_pickle=True)
    features = {"hog": feats["hog"], "color": feats["color"]}
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for key, name in ml06.ML_MODELS.items():
        fs = models8[key]["feature_set"]
        X_tr, X_va, X_te = (ml06.stack(features, d.index_row.to_numpy(), fs) for d in (tr, va, te))
        pipe, grid = ml06.build(key, fs)
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=False).fit(X_tr, y_tr, groups=g_tr)
        model = ml06.finalize(pipe.set_params(**search.best_params_), key)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        if "clf__n_jobs" in model.get_params():
            model.set_params(clf__n_jobs=1)
        joblib.dump({"pipeline": model, "model_key": f"{key}_5class", "feature_set": fs, "class_keys": KEEP},
                    MODELS_DIR / "ml" / f"{key}_5class.joblib")
        times = []
        for i, img in enumerate(X_te_img[:23]):
            t0 = time.perf_counter()
            model.predict_proba(feature_vector(extract_features(img), fs)[None, :])
            if i >= 3:
                times.append(time.perf_counter() - t0)
        cands[f"{key}_5class"] = {
            "name": f"{name} (5-class)", "family": "Machine Learning", "type": "single", "feature_set": fs,
            "best_params": ml06.readable_params(search.best_params_), "cv_f1_macro": float(search.best_score_),
            "train_time_s": fit_s, "inference_ms_per_image": 1000 * float(np.median(times)),
            "path": f"ml/{key}_5class.joblib",
            "p_va": model.predict_proba(X_va), "p_te": model.predict_proba(X_te)}
        print(f"{name}: grouped-CV macro-F1 {search.best_score_:.3f}", flush=True)

    # ---- deep learning: each network with / without TTA, then the ensemble ----
    nets, best_view = {}, {}
    for key, label in DL_ARCHS:
        path = MODELS_DIR / "dl" / "clean" / f"{key}.pt"
        if not path.exists():
            print(f"skip {key}: checkpoint missing")
            continue
        model, ckpt = load_checkpoint(path, device)
        assert list(ckpt["class_keys"]) == KEEP, ckpt["class_keys"]
        size = int(ckpt["input_size"])
        run = load_json(RESULTS_DIR / f"dl_clean_{key}.json")
        nets[key] = (model, size)
        for views in (1, 8):
            lv, lt = logits_for(model, X_va_img, device, size, views), logits_for(model, X_te_img, device, size, views)
            t = fit_temperature(lv, yl_va)
            ck = key + ("_tta" if views > 1 else "")
            cands[ck] = {
                "name": f"{label} (5-class)" + (" + TTA" if views > 1 else ""), "family": "Deep Learning",
                "type": "tta" if views > 1 else "single", "arch": ckpt["arch"], "input_size": size,
                "feature_set": f"Learned features (ImageNet-pretrained {label}, {size}px)"
                               + ("; averaged over 8 test-time views" if views > 1 else ""),
                "best_params": {"best_epoch": run["models"][key]["best_epoch"],
                                "epochs_run": run["models"][key]["epochs_run"], "tta_views": views},
                "train_time_s": run["models"][key]["train_time_s"], "params": run["total_params"],
                "members": [{"key": key, "path": f"dl/clean/{key}.pt", "temperature": t, "tta_views": views}],
                "temperature": t, "p_va": softmax(lv, t), "p_te": softmax(lt, t),
                "p_va_raw": softmax(lv), "p_te_raw": softmax(lt)}
        f1 = {v: classification_metrics(yl_va, cands[key + ("_tta" if v > 1 else "")]["p_va"].argmax(1),
                                        range(len(KEEP)))["f1_macro"] for v in (1, 8)}
        best_view[key] = 8 if f1[8] > f1[1] else 1
    if len(nets) == 2:
        parts = [cands[k + ("_tta" if best_view[k] > 1 else "")] for k in nets]
        desc = " + ".join(f"{p['name'].split(' (')[0]} ({'8-view TTA' if p['type'] == 'tta' else 'single view'})"
                          for p in parts)
        cands["ensemble_5class"] = {
            "name": "Ensemble (5-class): " + " + ".join(p["name"].split(" (")[0] for p in parts),
            "family": "Deep Learning", "type": "ensemble",
            "feature_set": f"Average of calibrated probabilities: {desc}",
            "best_params": {"members": [m["key"] for p in parts for m in p["members"]]},
            "train_time_s": sum(p["train_time_s"] for p in parts), "params": sum(p["params"] for p in parts),
            "members": [m for p in parts for m in p["members"]], "temperature": None,
            "p_va": np.mean([p["p_va"] for p in parts], 0), "p_te": np.mean([p["p_te"] for p in parts], 0),
            "p_va_raw": np.mean([p["p_va_raw"] for p in parts], 0), "p_te_raw": np.mean([p["p_te_raw"] for p in parts], 0)}

    # CPU latency (deployment hardware) per network and view setting, summed over ensemble members
    lat = {}
    for key, (model, size) in nets.items():
        for views in (1, 8):
            lat[(key, views)] = cpu_latency_ms(model, X_te_img, size, views)
    for c in cands.values():
        if c["family"] == "Deep Learning":
            c["inference_ms_per_image"] = sum(lat[(m["key"], m["tta_views"])] for m in c["members"])

    # ---- metrics; selection on validation ----
    results = []
    for key, c in cands.items():
        pv, pt = TO_GLOBAL[c["p_va"].argmax(1)], TO_GLOBAL[c["p_te"].argmax(1)]
        row = {k: v for k, v in c.items() if not k.startswith("p_")}
        row.update({"key": key, "val": classification_metrics(yg_va, pv, KEEP_IDX),
                    "val_confusion_matrix": confusion(yg_va, pv, KEEP_IDX),
                    "test": {**classification_metrics(yg_te, pt, KEEP_IDX),
                             **group_bootstrap_ci(yg_te, pt, g_te, seed=SEED, labels=KEEP_IDX)},
                    "per_class": per_class_metrics(yg_te, pt, KEEP_IDX), "confusion_matrix": confusion(yg_te, pt, KEEP_IDX)})
        c["pred_te"] = pt
        results.append(row)
    ranking = sorted(results, key=lambda r: (r["val"]["f1_macro"], r["val"]["accuracy"]), reverse=True)
    best = ranking[0]
    for r in results:
        r["selected"] = r is best
    best_ml = max((r for r in results if r["family"] == "Machine Learning"),
                  key=lambda r: (r["val"]["f1_macro"], r["val"]["accuracy"]))
    best_dl = max((r for r in results if r["family"] == "Deep Learning"),
                  key=lambda r: (r["val"]["f1_macro"], r["val"]["accuracy"]))
    ml_vs_dl = paired_bootstrap_diff(yg_te, cands[best_dl["key"]]["pred_te"], cands[best_ml["key"]]["pred_te"], g_te,
                                     seed=SEED, labels=KEEP_IDX)
    ml_vs_dl.update({"dl_model": best_dl["name"], "ml_model": best_ml["name"],
                     "accuracy_gain": best_dl["test"]["accuracy"] - best_ml["test"]["accuracy"]})

    cal = None
    if best["family"] == "Deep Learning":
        c = cands[best["key"]]
        thr = float(final8["calibration"]["confidence_threshold"])
        sel_t = selective_accuracy(c["p_te"], yl_te, [thr])[0]
        cal = {"temperature": c["temperature"], "member_temperatures": [m["temperature"] for m in c["members"]],
               "ece_before": expected_calibration_error(c["p_te_raw"], yl_te),
               "ece_after": expected_calibration_error(c["p_te"], yl_te), "confidence_threshold": thr,
               "coverage": sel_t["coverage"], "accuracy_kept": sel_t["accuracy_kept"],
               "accuracy_rejected": sel_t["accuracy_rejected"]}

    # selective prediction: threshold chosen on VALIDATION as the lowest calibrated confidence whose validation accuracy
    # on the kept photographs reaches the target; the test set only reports coverage and accuracy at that threshold
    selective = None
    if best["family"] == "Deep Learning":
        c = cands[best["key"]]
        grid = [round(x, 2) for x in np.arange(0.50, 0.991, 0.01)]
        val_rows = selective_accuracy(c["p_va"], yl_va, grid)
        target = 0.95
        ok = [r for r in val_rows if r["accuracy_kept"] is not None and r["accuracy_kept"] >= target and r["coverage"] >= 0.5]
        if ok:
            thr = ok[0]["threshold"]
            test_row = selective_accuracy(c["p_te"], yl_te, [thr])[0]
            selective = {"target_accuracy": target, "threshold": thr,
                         "rule": "lowest calibrated-confidence threshold with validation accuracy on kept photos >= 95% "
                                 "(coverage >= 50%); chosen on validation, reported once on test",
                         "val": ok[0], "test": test_row}

    # how the 8-class model does on exactly these in-scope test photographs (its own confusion-matrix rows)
    cm8 = np.asarray(sel8["confusion_matrix"])
    rows8 = cm8[KEEP_IDX]
    eight = {"model": sel8["name"], "photos": int(rows8.sum()),
             "accuracy": float(sum(cm8[i, i] for i in KEEP_IDX) / rows8.sum()),
             "predicted_out_of_scope": int(rows8[:, [i for i in range(len(CLASS_KEYS)) if i not in KEEP_IDX]].sum())}

    per_class_split = [{"key": k, "name": CLASS_NAMES[g], "train": int((tr.label == g).sum()),
                        "val": int((va.label == g).sum()), "test": int((te.label == g).sum())}
                       for k, g in zip(KEEP, KEEP_IDX)]
    out = {"classes": KEEP, "class_names": [CLASS_NAMES[i] for i in KEEP_IDX],
           "rule": "Healthy plus the four conditions with the highest validation F1 under the selected 8-class model "
                   "(validation data only; the test set played no part)",
           "val_f1_8class_model": val_f1_8, "dropped": [k for k in CLASS_KEYS if k not in KEEP],
           "split": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                     "test_leaf_groups": int(te.group_id.nunique()), "per_class": per_class_split},
           "models": results, "selected": best["key"],
           "validation_ranking": [{"key": r["key"], "name": r["name"], "type": r["type"], "family": r["family"],
                                   "val_f1_macro": r["val"]["f1_macro"], "val_accuracy": r["val"]["accuracy"]}
                                  for r in ranking],
           "ml_vs_dl": ml_vs_dl, "calibration": cal, "selective": selective, "eight_class_model_on_same_photos": eight}
    save_json(out, RESULTS_DIR / "five_class_results.json")

    print(f"\n5-class scope {KEEP}: {len(tr)} train / {len(va)} val / {len(te)} test photographs")
    for r in ranking:
        t = r["test"]
        print(f"{r['name']:58s} val F1 {r['val']['f1_macro']:.3f} acc {r['val']['accuracy']:.3f} | TEST acc "
              f"{t['accuracy']:.3f} [{t['accuracy_ci'][0]:.3f}-{t['accuracy_ci'][1]:.3f}] macro-F1 {t['f1_macro']:.3f} "
              f"w-F1 {t['f1_weighted']:.3f}{'   <- selected' if r['selected'] else ''}")
    print("best DL vs best ML (test macro-F1):",
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ml_vs_dl.items()})
    print("calibration:", cal)
    print("selective prediction:", selective)
    print("8-class model on the same in-scope test photos:", eight)

    # ---- figures ----
    short = [SHORT_NAMES[i] for i in KEEP_IDX]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    plot_confusion(ax, best["confusion_matrix"], f"{best['name']}: test set", labels=short)
    fig.tight_layout()
    save(fig, OUT / "confusion_selected.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.42 * len(ranking) + 1.2), sharey=True)
    y = np.arange(len(ranking))
    colors = [ML_COLOR if r["family"] == "Machine Learning" else DL_COLOR for r in ranking]
    for ax, metric, ci, title in [(axes[0], "accuracy", "accuracy_ci", "Test accuracy"),
                                  (axes[1], "f1_macro", "f1_macro_ci", "Test macro-F1")]:
        vals = np.array([r["test"][metric] for r in ranking])
        lo = vals - np.array([r["test"][ci][0] for r in ranking])
        hi = np.array([r["test"][ci][1] for r in ranking]) - vals
        ax.barh(y, vals, color=colors, height=0.62)
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for yi, v in enumerate(vals):
            ax.text(0.015, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color="#ffffff", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_title(f"{title} with 95% CI (5-class scope)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [r["name"] + ("  *" if r["selected"] else "") for r in ranking])
    axes[0].invert_yaxis()
    save(fig, OUT / "comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    frozen = 0
    for n, (key, label) in enumerate([a for a in DL_ARCHS if a[0] in nets]):
        hist = load_json(RESULTS_DIR / f"dl_clean_{key}.json")["history"]
        frozen = max(h["epoch"] for h in hist if h["stage"] == "frozen")
        ep = [h["epoch"] for h in hist]
        for ax, (a, b) in zip(axes, [("train_loss", "val_loss"), ("train_acc", "val_acc")]):
            ax.plot(ep, [h[a] for h in hist], color=SERIES[n], lw=1.4, ls="--", label=f"{label} training")
            ax.plot(ep, [h[b] for h in hist], color=SERIES[n], lw=2.0, label=f"{label} validation")
    for ax, lab in zip(axes, ["loss (weighted, label-smoothed)", "accuracy"]):
        ax.axvline(frozen + 0.5, color=AXIS, lw=1)
        ax.text(frozen + 0.7, ax.get_ylim()[1], "fine-tuning", va="top", fontsize=8, color=MUTED)
        ax.set_title(f"5-class training: {lab}")
        ax.set_xlabel("Epoch")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    save(fig, OUT / "training_curves.png")


if __name__ == "__main__":
    main()
