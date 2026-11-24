"""Phase 5 — final evaluation of every model on the held-out TEST photos.

- Base models: 3 classical (LR, SVM, RF) and the deep models (custom CNN, MobileNetV2 frozen/fine-tuned, and the
  improvement-round EfficientNet-B0 / ConvNeXt-Tiny when trained).
- Improvement-round candidates: test-time augmentation (TTA, average logits over the 8 flips/rotations) for every
  deep model, and ensembles (average calibrated probabilities) of the top-2 / top-3 deep models.
- Selection: highest VALIDATION macro-F1 among all candidates (ties: validation accuracy). The test set is only
  reported; it never chooses anything.
- 95% CIs: bootstrap over leaf groups (2,000 resamples). Calibration: temperature scaling fitted on validation.
- Leakage benchmark: the same recipes trained/evaluated on the official (leaky) folders.
Outputs: experiments/results/final_results.json, reports/figures/results/*.png
"""
import time

import joblib
import numpy as np
import torch

from tomatoleaf.config import CLASS_KEYS, DATA_DIR, FIGURES_DIR, MODEL_INPUT_SIZE, MODELS_DIR, NUM_CLASSES, RESULTS_DIR, SEED
from tomatoleaf.data import load_images, load_index, load_splits
from tomatoleaf.evaluation import (classification_metrics, confusion, expected_calibration_error, fit_temperature,
                                   group_bootstrap_ci, paired_bootstrap_diff, per_class_metrics, reliability_bins,
                                   selective_accuracy, softmax)
from tomatoleaf.features import FEATURE_SETS, extract_features, feature_vector
from tomatoleaf.models import count_parameters, load_checkpoint
from tomatoleaf.plotting import (AXIS, DL_COLOR, INK, INK2, ML_COLOR, MUTED, SEQ_CMAP, SERIES, SHORT_NAMES, apply_style,
                                 plot_confusion, plt, save)
from tomatoleaf.preprocessing import dihedral_views, eval_transform, preprocess_for_cnn
from tomatoleaf.training import make_loader
from tomatoleaf.utils import load_json, save_json, set_seed

ML = {"logreg": "Logistic Regression", "svm": "SVM (RBF kernel)", "rf": "Random Forest"}
DL = {"custom_cnn": "Custom CNN", "mobilenetv2_fe": "MobileNetV2 (feature extraction)",
      "mobilenetv2_ft": "MobileNetV2 (fine-tuned)", "efficientnet_b0_v2": "EfficientNet-B0 (fine-tuned)",
      "convnext_tiny_v2": "ConvNeXt-Tiny (fine-tuned)"}
SHORT = {"custom_cnn": "CNN", "mobilenetv2_fe": "MobileNetV2-FE", "mobilenetv2_ft": "MobileNetV2",
         "efficientnet_b0_v2": "EfficientNet-B0", "convnext_tiny_v2": "ConvNeXt-T"}
ARCH_NAME = {"mobilenetv2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0", "convnext_tiny": "ConvNeXt-Tiny"}
TTA_VIEWS = 8
CONFIDENCE_THRESHOLD = 0.60
OUT = FIGURES_DIR / "results"


def load_dl(key: str, protocol: str, device):
    return load_checkpoint(MODELS_DIR / "dl" / protocol / f"{key}.pt", device)


def run_info(key: str) -> dict:
    if key == "custom_cnn":
        return load_json(RESULTS_DIR / "dl_clean_custom_cnn.json")["models"][key]
    if key.startswith("mobilenetv2"):
        return load_json(RESULTS_DIR / "dl_clean_mobilenetv2.json")["models"][key]
    return load_json(RESULTS_DIR / f"dl_clean_{key[:-3]}_v2.json")["models"][key]


@torch.no_grad()
def dl_logits(model, images, device, size=MODEL_INPUT_SIZE, tta_views=1):
    """Logits for every image; with TTA the logits of the flipped/rotated views are averaged."""
    loader = make_loader(images, np.zeros(len(images), dtype=int), eval_transform(size), 16 if tta_views > 1 else 64,
                         shuffle=False)
    model.eval()
    out = []
    for x, _ in loader:
        x, b = x.to(device), len(x)
        if tta_views > 1:
            x = dihedral_views(x, tta_views)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x).float()
        out.append((logits.view(tta_views, b, -1).mean(0) if tta_views > 1 else logits).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def cpu_latency_dl(members, images) -> float:
    """members: list of (model, input_size, tta_views); median CPU time for one image through all members."""
    members = [(m.to("cpu").eval(), size, tta) for m, size, tta in members]
    times = []
    for i, img in enumerate(images):
        t0 = time.perf_counter()
        for model, size, tta in members:
            x = preprocess_for_cnn(img, size)
            model(dihedral_views(x, tta) if tta > 1 else x)
        if i >= 3:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def cpu_latency_ml(pipeline, feature_set, images) -> float:
    if "clf__n_jobs" in pipeline.get_params():
        pipeline.set_params(clf__n_jobs=1)
    times = []
    for i, img in enumerate(images):
        t0 = time.perf_counter()
        pipeline.predict_proba(feature_vector(extract_features(img), feature_set)[None, :])
        if i >= 5:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times))


def f1_of(p, y):
    return classification_metrics(y, p.argmax(1))["f1_macro"]


def main():
    set_seed(SEED)
    apply_style()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    va, te = splits[splits.split == "val"], splits[splits.split == "test"]
    y_va, y_te, g_te = va.label.to_numpy(), te.label.to_numpy(), te.group_id.to_numpy()
    X_va_img, X_te_img = load_images(va.path), load_images(te.path)
    latency_imgs = X_te_img[:25]
    feats = np.load(DATA_DIR / "features.npz", allow_pickle=True)
    features = {"hog": feats["hog"], "color": feats["color"]}
    ml_val = load_json(RESULTS_DIR / "ml_validation.json")
    dl_keys = [k for k in DL if (MODELS_DIR / "dl" / "clean" / f"{k}.pt").exists()]

    results, scores, tta_scores, dl_meta = [], {}, {}, {}
    for key, name in ML.items():
        sel = ml_val["selected"][key]
        pipe, fs = joblib.load(MODELS_DIR / sel["file"])["pipeline"], sel["feature_set"]
        p_va = pipe.predict_proba(np.hstack([features[p][va.index_row.to_numpy()] for p in FEATURE_SETS[fs]]))
        p_te = pipe.predict_proba(np.hstack([features[p][te.index_row.to_numpy()] for p in FEATURE_SETS[fs]]))
        grid_row = next(r for r in ml_val["grid"] if r["model_key"] == key and r["feature_set"] == fs)
        scores[key] = {"p_va": p_va, "p_te": p_te}
        results.append({"key": key, "name": name, "family": "Machine Learning", "kind": "classical", "feature_set": fs,
                        "best_params": grid_row["best_params"], "params": None, "train_time_s": grid_row["train_time_s"],
                        "tuning_time_s": grid_row["tuning_time_s"],
                        "inference_ms_per_image": cpu_latency_ml(pipe, fs, latency_imgs), "score_type": "predict_proba"})
        print(f"evaluated {name}")

    for key in dl_keys:
        model, ckpt = load_dl(key, "clean", device)
        arch, size = ckpt["arch"], int(ckpt["input_size"])
        run = run_info(key)
        for views, store in ((1, scores), (TTA_VIEWS, tta_scores)):
            lv, lt = dl_logits(model, X_va_img, device, size, views), dl_logits(model, X_te_img, device, size, views)
            t = fit_temperature(lv, y_va)
            store[key] = {"p_va": softmax(lv, t), "p_te": softmax(lt, t), "p_va_raw": softmax(lv), "p_te_raw": softmax(lt),
                          "temperature": t, "tta_views": views}
        dl_meta[key] = {"model": model, "size": size, "arch": arch, "params": count_parameters(model), "run": run}
        results.append({"key": key, "name": DL[key], "family": "Deep Learning",
                        "kind": "cnn" if arch == "custom_cnn" else "transfer",
                        "feature_set": "Learned features (end-to-end CNN)" if arch == "custom_cnn"
                        else f"Learned features (ImageNet-pretrained {ARCH_NAME[arch]}, {size}px)",
                        "best_params": {"best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
                                        "trainable_params": run["trainable_params"], "input_size": size},
                        "params": dl_meta[key]["params"], "train_time_s": run["train_time_s"], "tuning_time_s": None,
                        "inference_ms_per_image": cpu_latency_dl([(model, size, 1)], latency_imgs),
                        "score_type": "calibrated_softmax", "temperature": scores[key]["temperature"], "input_size": size})
        model.to(device)
        print(f"evaluated {DL[key]} (T={scores[key]['temperature']:.3f}; val macro-F1 {f1_of(scores[key]['p_va'], y_va):.3f}, "
              f"with TTA {f1_of(tta_scores[key]['p_va'], y_va):.3f})")

    def fill_metrics(r, s):
        pred_va, pred_te = s["p_va"].argmax(1), s["p_te"].argmax(1)
        r["val"] = classification_metrics(y_va, pred_va)
        r["test"] = {**classification_metrics(y_te, pred_te), **group_bootstrap_ci(y_te, pred_te, g_te, seed=SEED)}
        r["per_class"] = per_class_metrics(y_te, pred_te)
        r["confusion_matrix"] = confusion(y_te, pred_te)
        r["val_confusion_matrix"] = confusion(y_va, pred_va)
        r["test_ece"] = expected_calibration_error(s["p_te"], y_te)
        r["test_ece_uncalibrated"] = expected_calibration_error(s.get("p_te_raw", s["p_te"]), y_te)

    for r in results:
        fill_metrics(r, scores[r["key"]])

    # ---- Improvement-round candidates (chosen on validation only)
    candidates = [{"key": r["key"], "name": r["name"], "type": "single", "s": scores[r["key"]], "val_f1": r["val"]["f1_macro"],
                   "val_acc": r["val"]["accuracy"]} for r in results]
    for key in dl_keys:
        s = tta_scores[key]
        candidates.append({"key": f"{key}_tta", "name": f"{DL[key]} + TTA", "type": "tta", "s": s,
                           "members": [{"key": key, "tta_views": TTA_VIEWS, "temperature": s["temperature"]}],
                           "val_f1": f1_of(s["p_va"], y_va), "val_acc": classification_metrics(y_va, s["p_va"].argmax(1))["accuracy"]})
    dl_ranked = sorted((r for r in results if r["family"] == "Deep Learning"), key=lambda r: -r["val"]["f1_macro"])
    for k in (2, 3):
        if len(dl_ranked) < k:
            continue
        specs, pv, pt, pvr, ptr = [], [], [], [], []
        for m in dl_ranked[:k]:
            src = tta_scores[m["key"]] if f1_of(tta_scores[m["key"]]["p_va"], y_va) > m["val"]["f1_macro"] else scores[m["key"]]
            specs.append({"key": m["key"], "tta_views": src["tta_views"], "temperature": src["temperature"]})
            pv.append(src["p_va"]), pt.append(src["p_te"]), pvr.append(src["p_va_raw"]), ptr.append(src["p_te_raw"])
        s = {"p_va": np.mean(pv, 0), "p_te": np.mean(pt, 0), "p_va_raw": np.mean(pvr, 0), "p_te_raw": np.mean(ptr, 0)}
        label = " + ".join(SHORT[x["key"]] + (" (TTA)" if x["tta_views"] > 1 else "") for x in specs)
        candidates.append({"key": f"ensemble_top{k}", "name": f"Ensemble: {label}", "type": "ensemble", "s": s,
                           "members": specs, "val_f1": f1_of(s["p_va"], y_va),
                           "val_acc": classification_metrics(y_va, s["p_va"].argmax(1))["accuracy"]})

    best = max(candidates, key=lambda c: (c["val_f1"], c["val_acc"]))
    if best["type"] != "single":
        members = best["members"]
        row = {"key": best["key"], "name": best["name"], "family": "Deep Learning",
               "kind": "ensemble" if best["type"] == "ensemble" else "transfer",
               "feature_set": "Learned features; " + ("averaged over models" if best["type"] == "ensemble" else
                                                      f"averaged over {TTA_VIEWS} test-time views"),
               "best_params": {"members": [m["key"] for m in members], "tta_views": [m["tta_views"] for m in members]},
               "params": sum(dl_meta[m["key"]]["params"] for m in members),
               "train_time_s": sum(dl_meta[m["key"]]["run"]["train_time_s"] for m in members), "tuning_time_s": None,
               "inference_ms_per_image": cpu_latency_dl([(dl_meta[m["key"]]["model"], dl_meta[m["key"]]["size"], m["tta_views"])
                                                         for m in members], latency_imgs),
               "score_type": "calibrated_softmax", "temperature": members[0]["temperature"] if len(members) == 1 else None,
               "members": members}
        fill_metrics(row, best["s"])
        results.append(row)
        scores[best["key"]] = best["s"]
    for r in results:
        r["selected"] = r["key"] == best["key"]
    sel_row = next(r for r in results if r["selected"])
    selection = {"model_key": best["key"],
                 "criterion": "highest validation macro-F1 among all candidates incl. TTA and ensembles (ties: validation accuracy)",
                 "validation_ranking": [{"key": c["key"], "name": c["name"], "type": c["type"], "val_f1_macro": c["val_f1"],
                                         "val_accuracy": c["val_acc"]} for c in sorted(candidates, key=lambda c: -c["val_f1"])]}

    best_ml = max((r for r in results if r["family"] == "Machine Learning"), key=lambda r: r["val"]["f1_macro"])
    ml_vs_dl = {"dl_model": sel_row["key"], "ml_model": best_ml["key"],
                **paired_bootstrap_diff(y_te, scores[sel_row["key"]]["p_te"].argmax(1), scores[best_ml["key"]]["p_te"].argmax(1),
                                        g_te, seed=SEED)}
    first_round = next(r for r in results if r["key"] == "mobilenetv2_ft")
    improvement = {"baseline_key": "mobilenetv2_ft", "selected_key": sel_row["key"],
                   **paired_bootstrap_diff(y_te, scores[sel_row["key"]]["p_te"].argmax(1),
                                           scores["mobilenetv2_ft"]["p_te"].argmax(1), g_te, seed=SEED),
                   "accuracy_gain": sel_row["test"]["accuracy"] - first_round["test"]["accuracy"]}

    s = scores[sel_row["key"]]
    calibration = {"model_key": sel_row["key"], "method": "temperature_scaling" if sel_row["family"] == "Deep Learning" else "none",
                   "temperature": sel_row.get("temperature"),
                   "member_temperatures": {m["key"]: m["temperature"] for m in sel_row.get("members", [])},
                   "ece_before": expected_calibration_error(s.get("p_te_raw", s["p_te"]), y_te),
                   "ece_after": expected_calibration_error(s["p_te"], y_te),
                   "confidence_threshold": CONFIDENCE_THRESHOLD,
                   "threshold_rule": f"Predictions whose top probability is below {CONFIDENCE_THRESHOLD:.2f} are shown as 'Uncertain'.",
                   "reliability_before": reliability_bins(s.get("p_te_raw", s["p_te"]), y_te),
                   "reliability_after": reliability_bins(s["p_te"], y_te),
                   "selective_val": selective_accuracy(s["p_va"], y_va, np.arange(0.3, 0.96, 0.05)),
                   "selective_test": selective_accuracy(s["p_te"], y_te, np.arange(0.3, 0.96, 0.05)),
                   "temperatures": {r["key"]: r.get("temperature") for r in results}}

    # ---- Leakage benchmark (official folders)
    index = load_index()
    off_te = index[index.official_split == "test"]
    X_off = load_images(off_te.path)
    y_off = off_te.label.to_numpy()
    bench = []
    for b in load_json(RESULTS_DIR / "ml_official_benchmark.json"):
        clean = next(r for r in results if r["key"] == b["model_key"])
        bench.append({"model_key": b["model_key"], "model": b["model"], "official": b["test"], "clean": clean["test"],
                      "official_per_class": b["per_class"]})
    for key in ("mobilenetv2_ft", "efficientnet_b0_v2", "convnext_tiny_v2"):
        if (MODELS_DIR / "dl" / "official" / f"{key}.pt").exists() and key in dl_keys:
            model, ckpt = load_dl(key, "official", device)
            pred = dl_logits(model, X_off, device, int(ckpt["input_size"])).argmax(1)
            clean = next(r for r in results if r["key"] == key)
            bench.append({"model_key": key, "model": DL[key], "official": classification_metrics(y_off, pred),
                          "clean": clean["test"], "official_per_class": per_class_metrics(y_off, pred)})

    test_info = {"photos": int(len(te)), "leaf_groups": int(te.group_id.nunique()),
                 "per_class_support": np.bincount(y_te, minlength=NUM_CLASSES).tolist()}
    save_json({"models": results, "selection": selection, "calibration": calibration, "leakage_benchmark": bench,
               "ml_vs_dl": ml_vs_dl, "improvement_round": improvement, "test_set": test_info, "class_keys": CLASS_KEYS},
              RESULTS_DIR / "final_results.json")

    print("\nValidation ranking of all candidates:")
    for c in selection["validation_ranking"]:
        print(f"  {c['name']:58s} val macro-F1 {c['val_f1_macro']:.3f}  val acc {c['val_accuracy']:.3f}")
    print(f"\n{'Model':58s} {'val F1':>7s} {'test acc':>9s} {'test F1':>8s} {'95% CI F1':>15s} {'w-F1':>6s} {'train s':>8s} {'ms/img':>7s}")
    for r in results:
        t = r["test"]
        print(f"{r['name']:58s} {r['val']['f1_macro']:7.3f} {t['accuracy']:9.3f} {t['f1_macro']:8.3f} "
              f"  [{t['f1_macro_ci'][0]:.3f},{t['f1_macro_ci'][1]:.3f}] {t['f1_weighted']:6.3f} {r['train_time_s']:8.1f} "
              f"{r['inference_ms_per_image']:7.1f}{'  <- selected' if r['selected'] else ''}")
    print("Leakage benchmark:", [(b["model"], round(b["official"]["accuracy"], 3), round(b["official"]["f1_macro"], 3),
                                  round(b["clean"]["f1_macro"], 3)) for b in bench])
    print("Selected vs best ML:", ml_vs_dl)
    print("Selected vs first-round MobileNetV2:", improvement)
    print("Calibration:", {k: calibration[k] for k in ("model_key", "temperature", "ece_before", "ece_after")})
    make_figures(results, calibration, bench)


def make_figures(results, calibration, bench):
    names = [r["name"] for r in results]
    colors = [ML_COLOR if r["family"] == "Machine Learning" else DL_COLOR for r in results]
    y = np.arange(len(results))
    h = 0.45 + 0.42 * len(results)

    fig, axes = plt.subplots(1, 2, figsize=(13, h), sharey=True)
    for ax, metric, ci_key, title in [(axes[0], "accuracy", "accuracy_ci", "Test accuracy"),
                                      (axes[1], "f1_macro", "f1_macro_ci", "Test macro-F1")]:
        vals = np.array([r["test"][metric] for r in results])
        lo = vals - np.array([r["test"][ci_key][0] for r in results])
        hi = np.array([r["test"][ci_key][1] for r in results]) - vals
        ax.barh(y, vals, color=colors, height=0.6)
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
        for yi, v in enumerate(vals):
            ax.text(0.01, yi, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color="#ffffff", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_title(f"{title} (95% CI, leaf-group bootstrap)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [n + ("  *" if r["selected"] else "") for n, r in zip(names, results)])
    axes[0].invert_yaxis()
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=ML_COLOR, label="Machine Learning"), Patch(color=DL_COLOR, label="Deep Learning")],
                   loc="lower right")
    save(fig, OUT / "model_comparison.png")

    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(17, 5.3 * rows))
    for ax in axes.flat[n:]:
        ax.set_axis_off()
    for ax, r in zip(axes.flat, results):
        plot_confusion(ax, r["confusion_matrix"], f"{r['name']}" + ("  [selected]" if r["selected"] else ""))
    fig.tight_layout()
    save(fig, OUT / "confusion_matrices_test.png")

    chosen = next(r for r in results if r["selected"])
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    plot_confusion(ax, chosen["confusion_matrix"], f"{chosen['name'][:48]}: test confusion matrix")
    fig.tight_layout()
    save(fig, OUT / "confusion_matrix_selected.png")

    fig, ax = plt.subplots(figsize=(11, 0.8 + 0.42 * n))
    mat = np.array([[c["f1"] for c in r["per_class"]] for r in results])
    ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if mat[i, j] > 0.55 else INK)
    support = results[0]["per_class"]
    ax.set_xticks(range(NUM_CLASSES), [f"{SHORT_NAMES[j]}\n(n={support[j]['support']})" for j in range(NUM_CLASSES)], fontsize=8.5)
    ax.set_yticks(range(n), names)
    ax.set_title("Per-class F1 on the test set (n = test photos per class)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, OUT / "per_class_f1.png")

    fig, ax = plt.subplots(figsize=(10, 0.8 + 0.42 * n))
    times = np.array([r["train_time_s"] for r in results])
    ax.barh(y, times, color=colors, height=0.6)
    for yi, t in enumerate(times):
        ax.text(t, yi, f"  {t:.1f} s", va="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Training time of the final model (seconds, log scale)")
    ax.set_title("Training time (ML: CPU, all cores; DL: RTX 4050 GPU)")
    save(fig, OUT / "training_time.png")

    curves = {"custom_cnn": ("Custom CNN", "dl_clean_custom_cnn.json"), "mobilenetv2": ("MobileNetV2", "dl_clean_mobilenetv2.json"),
              "efficientnet_b0": ("EfficientNet-B0", "dl_clean_efficientnet_b0_v2.json"),
              "convnext_tiny": ("ConvNeXt-Tiny", "dl_clean_convnext_tiny_v2.json")}
    curves = {k: v for k, v in curves.items() if (RESULTS_DIR / v[1]).exists()}
    fig, axes = plt.subplots(len(curves), 2, figsize=(12, 3.6 * len(curves)), squeeze=False)
    for row, (arch, (label, fname)) in enumerate(curves.items()):
        hist = load_json(RESULTS_DIR / fname)["history"]
        ep = [hh["epoch"] for hh in hist]
        for col, (a, b, ylab) in enumerate([("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy")]):
            ax = axes[row, col]
            ax.plot(ep, [hh[a] for hh in hist], color=SERIES[0], label="Training")
            ax.plot(ep, [hh[b] for hh in hist], color=SERIES[1], label="Validation")
            frozen = [hh["epoch"] for hh in hist if hh["stage"] == "frozen"]
            if frozen:
                ax.axvline(max(frozen) + 0.5, color=AXIS, lw=1)
                ax.text(max(frozen) + 0.5, ax.get_ylim()[1], " fine-tuning starts", va="top", fontsize=8, color=MUTED)
            ax.set_title(f"{label}: {'loss' if col == 0 else 'accuracy'}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylab)
            ax.legend(loc="best")
    fig.tight_layout()
    save(fig, OUT / "training_curves.png")

    if bench:
        fig, ax = plt.subplots(figsize=(10, 0.9 + 0.75 * len(bench)))
        yb = np.arange(len(bench))
        hb = 0.36
        off = [b["official"]["f1_macro"] for b in bench]
        cln = [b["clean"]["f1_macro"] for b in bench]
        ax.barh(yb - hb / 2, off, height=hb, color=SERIES[7], label="Official split (leaky)")
        ax.barh(yb + hb / 2, cln, height=hb, color=SERIES[0], label="Our group-aware split")
        for yi, (a, c) in enumerate(zip(off, cln)):
            ax.text(a, yi - hb / 2, f" {a:.3f}", va="center", fontsize=8.5, color=INK2)
            ax.text(c, yi + hb / 2, f" {c:.3f}", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(yb, [b["model"] for b in bench])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.12)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Test macro-F1")
        ax.set_title("Leakage benchmark: the official split inflates every model's score")
        ax.legend(loc="lower right")
        save(fig, OUT / "leakage_benchmark.png")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a1.plot([0, 1], [0, 1], color=AXIS, lw=1)
    for rows_, color, label in [(calibration["reliability_before"], SERIES[1], f"Before (ECE {calibration['ece_before']:.3f})"),
                                (calibration["reliability_after"], SERIES[0], f"After temperature scaling (ECE {calibration['ece_after']:.3f})")]:
        a1.plot([r["confidence"] for r in rows_], [r["accuracy"] for r in rows_], marker="o", ms=5, color=color, label=label)
    a1.set_xlabel("Mean predicted confidence")
    a1.set_ylabel("Observed accuracy")
    a1.set_title("Reliability diagram (test)")
    a1.legend(loc="upper left")
    sel = calibration["selective_val"]
    a2.plot([r["threshold"] for r in sel], [r["coverage"] for r in sel], color=SERIES[0], label="Coverage (share answered)")
    a2.plot([r["threshold"] for r in sel], [r["accuracy_kept"] for r in sel], color=SERIES[1], label="Accuracy on answered")
    a2.axvline(calibration["confidence_threshold"], color=AXIS, lw=1)
    a2.text(calibration["confidence_threshold"], 0.02, " app threshold", fontsize=8, color=MUTED)
    a2.set_xlabel("Confidence threshold")
    a2.set_title("Validation: confidence threshold trade-off")
    a2.legend(loc="lower left")
    a2.set_ylim(0, 1.02)
    save(fig, OUT / "calibration.png")


if __name__ == "__main__":
    main()
