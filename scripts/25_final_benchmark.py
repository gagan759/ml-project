"""Upgrade round, step 6 — the final benchmark: every model, the held-out TEST photos, scored once.

Runs every entry of experiments/benchmark_models.json (original study + upgrade) through the API's own predictor
code (backend.services.predictor) on identical test photos. All selection decisions were fixed earlier on
out-of-fold data (experiments/results/selection_*.json); nothing here feeds back into a choice.

Reported per model: accuracy, macro precision/recall/F1, weighted F1 with 95% CIs (bootstrap over leaf groups),
per-class metrics, confusion matrix, calibration (ECE, NLL, Brier), selective prediction at the model's threshold,
and the share of photos flagged "Unusual image". Plus: paired bootstrap comparisons, robustness to harmless changes
and to simulated phone uploads (old vs new upload path), out-of-distribution detection on out-of-scope test photos
and the held-out half of the probe images, CPU latency, and a CPU-vs-GPU parity check.
Inference runs on the GPU in float32 (identical arithmetic to the CPU path up to ~1e-6; checked below).
Output: experiments/results/final_benchmark.json; test metrics and latency are written into deployment.json.
"""
import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, DATASET_DIR, EXPERIMENTS_DIR, RESULTS_DIR, SEED  # noqa: E402
from tomatoleaf.data import load_images, load_splits  # noqa: E402
from tomatoleaf.evaluation import (  # noqa: E402
    brier_score,
    classification_metrics,
    confusion,
    expected_calibration_error,
    group_bootstrap_ci,
    paired_bootstrap,
    per_class_metrics,
    reliability_bins,
    selective_accuracy,
)
from tomatoleaf.preprocessing import prepare_upload  # noqa: E402
from tomatoleaf.probes import far_ood_probes  # noqa: E402
from tomatoleaf.utils import load_json, save_json, set_seed  # noqa: E402

audit = import_module("19_baseline_audit")
FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]


def run_model(predictor, images):
    probs, ratios = [], []
    for img in images:
        p, r = predictor.analyse(img)
        probs.append(p)
        ratios.append(np.nan if r is None else r)
    return np.stack(probs), np.asarray(ratios, dtype=float)


@torch.no_grad()
def cpu_latency(reg_cpu, entry, images, n=20):
    """Median / p90 milliseconds per photo on the CPU (8 threads), through the API's predictor (default runtime)."""
    torch.set_num_threads(8)
    pred = reg_cpu.load(entry["key"])
    times = []
    for i, img in enumerate(images[: n + 3]):
        t0 = time.perf_counter()
        pred.analyse(img)
        if i >= 3:
            times.append(time.perf_counter() - t0)
    return 1000 * float(np.median(times)), 1000 * float(np.percentile(times, 90)), pred


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="pipeline test on 48 train+val photos (NOT the test set); writes final_benchmark_dryrun.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    set_seed(SEED)
    bench = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    reg = ModelRegistry(EXPERIMENTS_DIR / "benchmark_models.json", device=device, lazy=True)
    splits = load_splits()
    te = splits[splits.split == "test"].reset_index(drop=True)
    if args.dry_run:  # never the test set: a few photos from the CV pool, only to exercise the code paths
        import pandas as pd
        pool = pd.read_csv(ROOT / "data" / "cv_folds.csv")
        te = pool[pool.fold == 0].groupby("class_key", group_keys=False).head(6).reset_index(drop=True)
    X = load_images(te.path)
    y8, groups = te.label.to_numpy(), te.group_id.to_numpy()
    in5 = te.class_key.isin(FIVE).to_numpy()
    print(f"test photos: {len(te)} (5-class scope: {in5.sum()})")

    results, store = [], {}
    t_start = time.perf_counter()
    for entry in bench["models"]:
        pred = reg.load(entry["key"])
        if pred is None:
            print(f"!! {entry['key']} failed to load: {reg.errors.get(entry['key'])}")
            continue
        task = entry["task"]
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        local = np.array([CLASS_KEYS.index(k) for k in pred.class_keys])
        probs, ratios = run_model(pred, X)
        reg.models.pop(entry["key"], None)  # free memory; reloaded later if needed
        y_local = np.array([list(local).index(v) if v in local else -1 for v in y8])
        p_in, yl_in, g_in = probs[mask], y_local[mask], groups[mask]
        yg_in, pg_in = y8[mask], local[p_in.argmax(1)]  # global class indices (names/keys stay correct)
        labels = list(local)
        thr = reg.threshold_for(pred)
        conf = p_in.max(1)
        unusual = ratios[mask] > 1  # NaN (no OOD score) compares as False
        answered = (conf >= thr) & ~unusual
        row = {"key": entry["key"], "name": entry["name"], "task": task, "generation": entry["generation"],
               "family": entry["family"], "photos": int(mask.sum()),
               "test": {**classification_metrics(yg_in, pg_in, labels),
                        **group_bootstrap_ci(yg_in, pg_in, g_in, seed=SEED, labels=labels),
                        "ece": expected_calibration_error(p_in, yl_in),
                        "nll": float(-np.log(p_in[np.arange(len(yl_in)), yl_in] + 1e-12).mean()),
                        "brier": brier_score(p_in, yl_in)},
               "per_class": per_class_metrics(yg_in, pg_in, labels),
               "confusion": confusion(yg_in, pg_in, labels),
               "reliability": reliability_bins(p_in, yl_in),
               "threshold": thr,
               "selective": {"threshold": thr, "coverage": float(answered.mean()),
                             "accuracy_kept": float((pg_in[answered] == yg_in[answered]).mean()) if answered.any() else None,
                             "accuracy_rejected": float((pg_in[~answered] == yg_in[~answered]).mean()) if (~answered).any() else None},
               "flagged_unusual": None if np.isnan(ratios).all() else float(np.mean(ratios[mask] > 1))}
        if task == "5class":
            oos = ~in5
            row["out_of_scope"] = {"photos": int(oos.sum()),
                                   "confidence_mean": float(probs[oos].max(1).mean()),
                                   "share_uncertain_or_unusual": float(np.mean((probs[oos].max(1) < thr) | (ratios[oos] > 1))),
                                   "share_flagged_unusual": None if np.isnan(ratios).all() else float(np.mean(ratios[oos] > 1))}
        # consistency across images of the SAME leaf: test leaf groups with several photos (re-shots)
        pairs_same, pairs_all, groups_agree, multi = 0, 0, 0, 0
        for g in np.unique(g_in):
            idx = np.flatnonzero(g_in == g)
            if len(idx) < 2:
                continue
            multi += 1
            preds_g = pg_in[idx]
            groups_agree += int(len(set(preds_g.tolist())) == 1)
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    pairs_all += 1
                    pairs_same += int(preds_g[i] == preds_g[j])
        row["same_leaf_consistency"] = {"groups": multi, "pairs": pairs_all,
                                        "pairwise_agreement": pairs_same / pairs_all if pairs_all else None,
                                        "groups_all_agree": groups_agree / multi if multi else None}
        store[entry["key"]] = {"probs": probs, "ratios": ratios, "pred_global": local[probs.argmax(1)]}
        results.append(row)
        print(f"{entry['key']:34s} [{task}] acc {row['test']['accuracy']:.3f} F1 {row['test']['f1_macro']:.3f} "
              f"(CI {row['test']['f1_macro_ci'][0]:.3f}-{row['test']['f1_macro_ci'][1]:.3f}) ECE {row['test']['ece']:.3f} "
              f"| answered {row['selective']['coverage']:.2f} at {thr:.2f} -> {row['selective']['accuracy_kept']}", flush=True)
    print(f"scored {len(results)} models in {(time.perf_counter() - t_start) / 60:.1f} min")

    # ---- risk-coverage curves of the defaults (test)
    defaults, previous = bench["defaults_by_task"], bench["previous_defaults"]
    curves = {}
    for task, key in defaults.items():
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        keys_t = FIVE if task == "5class" else CLASS_KEYS
        y_in = np.array([keys_t.index(CLASS_KEYS[v]) for v in y8[mask]])
        curves[task] = {"model": key, "curve": selective_accuracy(store[key]["probs"][mask], y_in, np.arange(0.4, 0.991, 0.01))}

    # ---- paired comparisons on the same photos
    by_key = {r["key"]: r for r in results}
    pairs = []
    for task in ("5class", "8class"):
        new, old = defaults[task], previous[task]
        others = [r for r in results if r["task"] == task]
        best_classical = max((r for r in others if r["family"] == "Classical ML"), key=lambda r: r["test"]["f1_macro"])
        pairs += [(task, new, old, True), (task, new, best_classical["key"], False)]
        cv = {r["key"]: r for r in others if r["key"].startswith(f"cv{task[0]}_")}
        p = f"cv{task[0]}_"
        if f"{p}convnext_t_in22k_base" in cv and f"{p}convnext_t_in1k_base" in cv:
            pairs.append((task, f"{p}convnext_t_in22k_base", f"{p}convnext_t_in1k_base", False))
        if f"{p}convnext_t_in22k_robust" in cv and f"{p}convnext_t_in22k_base" in cv:
            pairs.append((task, f"{p}convnext_t_in22k_robust", f"{p}convnext_t_in22k_base", False))
    comparisons = []
    for task, a, b, headline in pairs:
        if a not in store or b not in store:
            continue
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        labels = [CLASS_KEYS.index(k) for k in (FIVE if task == "5class" else CLASS_KEYS)]
        c = paired_bootstrap(y8[mask], store[a]["pred_global"][mask], store[b]["pred_global"][mask], groups[mask],
                             seed=SEED, labels=labels)
        comparisons.append({"task": task, "a": a, "b": b, "a_name": by_key[a]["name"], "b_name": by_key[b]["name"],
                            "headline": headline, **c})
        print(f"[{task}] {a} vs {b}: dF1 {c['f1_diff']:+.3f} [{c['f1_ci'][0]:+.3f}, {c['f1_ci'][1]:+.3f}] "
              f"dAcc {c['accuracy_diff']:+.3f} [{c['accuracy_ci'][0]:+.3f}, {c['accuracy_ci'][1]:+.3f}]")

    # ---- out-of-distribution: probe half B (never used to choose the OOD method), out-of-scope test photos
    names, probes, halves = far_ood_probes()
    ood = {}
    for task, key in defaults.items():
        pred = reg.load(key)
        pp, pr = run_model(pred, probes)
        thr = reg.threshold_for(pred)
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        ood[task] = {"model": key, "flagged_in": float(np.mean(store[key]["ratios"][mask] > 1)),
                     "flagged_far_B": float(np.mean(pr[halves == "B"] > 1)),
                     "flagged_far_A": float(np.mean(pr[halves == "A"] > 1)),
                     "no_diagnosis_far_B": float(np.mean((pr[halves == "B"] > 1) | (pp[halves == "B"].max(1) < thr))),
                     "probe_ratios": {n: float(r) for n, r in zip(names, pr)}}
        if task == "5class":
            oos = ~in5
            ood[task]["flagged_near"] = float(np.mean(store[key]["ratios"][oos] > 1))
            ood[task]["no_diagnosis_near"] = float(np.mean((store[key]["ratios"][oos] > 1) | (store[key]["probs"][oos].max(1) < thr)))
        reg.models.pop(key, None)
        print(f"OOD [{task}]: {ood[task]}")

    # ---- robustness on test: old vs new default, harmless changes + simulated phone uploads
    from PIL import Image
    icc = Image.open(DATASET_DIR / te.path.iloc[0]).info.get("icc_profile")
    portrait = []
    for rel in te.path:
        ex = Image.open(DATASET_DIR / rel).getexif()
        portrait.append(ex.get(257, 1) >= ex.get(256, 0))
    variants = {name: np.stack([f(a) for a in X]) for name, f in audit.perturbations().items()}
    for profile in ("p3", "srgb"):
        files = [audit.phone_photo(a, por, profile, icc) for a, por in zip(X, portrait)]
        variants[f"upload_{profile}_original_path"] = np.stack([audit.legacy_upload(f) for f in files])
        variants[f"upload_{profile}_new_path"] = np.stack([prepare_upload(f)[0] for f in files])
    robustness = {}
    for task in ("5class", "8class"):
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        robustness[task] = {}
        for key in (defaults[task], previous[task]):
            pred = reg.load(key)
            local = np.array([CLASS_KEYS.index(k) for k in pred.class_keys])
            clean = store[key]["pred_global"]
            rows = {}
            for name, imgs in variants.items():
                probs, _ = run_model(pred, imgs[mask])
                pg = local[probs.argmax(1)]
                rows[name] = {"flip_rate": float((pg != clean[mask]).mean()), "accuracy": float((pg == y8[mask]).mean())}
            robustness[task][key] = rows
            reg.models.pop(key, None)
            print(f"robustness [{task}] {key}: " + ", ".join(f"{n} {v['flip_rate']:.3f}" for n, v in rows.items()), flush=True)

    # ---- CPU latency of the API models + CPU/GPU parity for the defaults
    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    latency, parity = {}, {}
    reg_cpu = ModelRegistry(EXPERIMENTS_DIR / "deployment.json", device="cpu", lazy=True)
    for entry in deployment["models"]:
        med, p90, cpu_pred = cpu_latency(reg_cpu, entry, X)
        latency[entry["key"]] = {"median_ms": med, "p90_ms": p90, "threads": 8,
                                 "runtime": getattr(cpu_pred, "runtime", "scikit-learn")}
        if entry["key"] in defaults.values():
            gpu_probs = store[entry["key"]]["probs"][:10]
            cpu_probs = np.stack([cpu_pred.analyse(img)[0] for img in X[:10]])
            parity[entry["key"]] = float(np.abs(cpu_probs - gpu_probs).max())
        print(f"latency {entry['key']}: {med:.0f} ms (p90 {p90:.0f})", flush=True)

    out = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "test_photos": int(len(te)), "test_photos_5class": int(in5.sum()),
           "note": "Test set scored once, after every selection decision was fixed on out-of-fold data.",
           "models": results, "comparisons": comparisons, "risk_coverage": curves, "ood": ood, "robustness": robustness,
           "latency": latency, "cpu_gpu_max_abs_prob_diff": parity, "defaults": defaults, "previous_defaults": previous}
    if args.dry_run:
        out["note"] = "DRY RUN on train+val photos (not the test set) - pipeline check only, numbers meaningless"
        save_json(out, RESULTS_DIR / "final_benchmark_dryrun.json")
        print("wrote final_benchmark_dryrun.json (deployment.json not updated)")
        return
    save_json(out, RESULTS_DIR / "final_benchmark.json")

    # write test metrics + latency into the API deployment (shown by /api/models)
    for entry in deployment["models"]:
        r = by_key.get(entry["key"])
        if r:
            entry["test_metrics"] = {k: r["test"][k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                              "f1_weighted", "accuracy_ci", "f1_macro_ci", "ece")}
            entry["latency_ms"] = latency.get(entry["key"], {}).get("median_ms")
            if entry.get("operating_point") is not None:
                entry["operating_point"]["test"] = {"coverage": r["selective"]["coverage"],
                                                    "accuracy_kept": r["selective"]["accuracy_kept"],
                                                    "photos": r["photos"]}
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    print("wrote final_benchmark.json and updated deployment.json")


if __name__ == "__main__":
    main()
