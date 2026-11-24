"""Upgrade round, step 10 — the calibrated / bias-corrected variants (scripts/31) on the held-out TEST photos, once.

The variants were fixed on out-of-fold data before this script ran. Same protocol as scripts/25: the API's own
predictor code, identical test photos, CIs over leaf groups, paired bootstrap against the uncorrected version-2
model and the original study's model. Decisions use the variant's rule (Predictor.decide); confidence is the
calibrated probability of the decided class. Results are appended to final_benchmark.json (models, comparisons,
risk_coverage_variants) and written into deployment.json (test_metrics, latency, operating point).
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf import balance  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, EXPERIMENTS_DIR, RESULTS_DIR, SEED  # noqa: E402
from tomatoleaf.data import load_images, load_splits  # noqa: E402
from tomatoleaf.evaluation import (  # noqa: E402
    classification_metrics,
    confusion,
    group_bootstrap_ci,
    paired_bootstrap,
    per_class_metrics,
)
from tomatoleaf.utils import load_json, save_json, set_seed  # noqa: E402

FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
VARIANTS = ["calibrated_5class", "balanced_5class", "balanced_8class"]
REFERENCE = {"5class": ["upgrade_5class", "ensemble_5class"], "8class": ["upgrade_8class", "convnext_tiny_v2_tta"]}


def reliability(conf, ok, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "count": int(m.sum()), "confidence": float(conf[m].mean()),
                        "accuracy": float(ok[m].mean())})
    return out


def ece(conf, ok, bins=15):
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum(abs(conf[idx == b].mean() - ok[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg = ModelRegistry(EXPERIMENTS_DIR / "benchmark_models.json", device=device, lazy=True)
    splits = load_splits()
    te = splits[splits.split == "test"].reset_index(drop=True)
    X = load_images(te.path)
    y8, groups = te.label.to_numpy(), te.group_id.to_numpy()
    in5 = te.class_key.isin(FIVE).to_numpy()
    bench = load_json(RESULTS_DIR / "final_benchmark.json")
    names = {m["key"]: m["name"] for m in load_json(EXPERIMENTS_DIR / "benchmark_models.json")["models"]}
    preds, rows, curves = {}, [], {}
    for key in VARIANTS + [k for t in REFERENCE.values() for k in t]:
        pred = reg.load(key)
        task = pred.task
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        local = np.array([CLASS_KEYS.index(k) for k in pred.class_keys])
        out = [pred.analyse(img) for img in X]
        probs = np.stack([p for p, _ in out])
        ratios = np.array([np.nan if r is None else r for _, r in out], dtype=float)
        decision = np.array([pred.decide(p) for p in probs])
        ratios = np.array([np.nan if np.isnan(r) else pred.unusualness(r, int(d)) for r, d in zip(ratios, decision)])
        preds[key] = local[decision]
        if key not in VARIANTS:
            continue
        p_in, d_in, g_in = probs[mask], decision[mask], groups[mask]
        y_local = np.array([list(local).index(v) for v in y8[mask]])
        yg, pg = y8[mask], local[d_in]
        conf, ok = balance.confidence_of(p_in, d_in), d_in == y_local
        thr = reg.threshold_for(pred)
        unusual = ratios[mask] > 1
        answered = (conf >= thr) & ~unusual
        labels = list(local)
        row = {"key": key, "name": names[key], "task": task, "generation": "upgrade", "family": "Deep learning",
               "photos": int(mask.sum()),
               "test": {**classification_metrics(yg, pg, labels), **group_bootstrap_ci(yg, pg, g_in, seed=SEED, labels=labels),
                        "balanced_accuracy": balance.balanced_accuracy(y_local, d_in, len(labels)),
                        "ece": ece(conf, ok), "classwise_ece": balance.classwise_ece(p_in, y_local, d_in),
                        "nll": float(-np.log(p_in[np.arange(len(y_local)), y_local] + 1e-12).mean()),
                        "brier": float(np.mean(np.sum((p_in - np.eye(len(labels))[y_local]) ** 2, axis=1)))},
               "per_class": per_class_metrics(yg, pg, labels), "confusion": confusion(yg, pg, labels),
               "reliability": reliability(conf, ok), "threshold": thr,
               "selective": {"threshold": thr, "coverage": float(answered.mean()),
                             "accuracy_kept": float(ok[answered].mean()) if answered.any() else None,
                             "accuracy_rejected": float(ok[~answered].mean()) if (~answered).any() else None},
               "flagged_unusual": float(np.mean(unusual))}
        if task == "5class":
            oos = ~in5
            oos_conf = np.array([balance.confidence_of(p, pred.decide(p)) for p in probs[oos]])  # ratios: per answer
            row["out_of_scope"] = {"photos": int(oos.sum()), "confidence_mean": float(oos_conf.mean()),
                                   "share_uncertain_or_unusual": float(np.mean((oos_conf < thr) | (ratios[oos] > 1))),
                                   "share_flagged_unusual": float(np.mean(ratios[oos] > 1))}
        pairs_same = pairs_all = multi = agree = 0
        for g in np.unique(g_in):
            idx = np.flatnonzero(g_in == g)
            if len(idx) < 2:
                continue
            multi += 1
            agree += int(len(set(pg[idx].tolist())) == 1)
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    pairs_all += 1
                    pairs_same += int(pg[idx[i]] == pg[idx[j]])
        row["same_leaf_consistency"] = {"groups": multi, "pairs": pairs_all, "pairwise_agreement": pairs_same / pairs_all,
                                        "groups_all_agree": agree / multi}
        curves[key] = {"model": key, "curve": [
            {"threshold": float(t), "coverage": float((conf >= t).mean()),
             "accuracy_kept": float(ok[conf >= t].mean()) if (conf >= t).any() else None}
            for t in np.round(np.arange(0.40, 0.991, 0.01), 2)]}
        rows.append(row)
        t = row["test"]
        print(f"{key:18s} acc {t['accuracy']:.3f} [{t['accuracy_ci'][0]:.3f}-{t['accuracy_ci'][1]:.3f}] F1 {t['f1_macro']:.3f} "
              f"balAcc {t['balanced_accuracy']:.3f} ECE {t['ece']:.3f} classwise {t['classwise_ece']:.3f} | answers "
              f"{row['selective']['coverage']:.2f} at {thr:.2f} -> {row['selective']['accuracy_kept']}", flush=True)
        reg.models.pop(key, None)

    comparisons = []
    for key in VARIANTS:
        task = next(r["task"] for r in rows if r["key"] == key)
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        labels = [CLASS_KEYS.index(k) for k in (FIVE if task == "5class" else CLASS_KEYS)]
        for ref in REFERENCE[task]:
            c = paired_bootstrap(y8[mask], preds[key][mask], preds[ref][mask], groups[mask], seed=SEED, labels=labels)
            comparisons.append({"task": task, "a": key, "b": ref, "a_name": names[key], "b_name": names[ref],
                                "headline": False, "variant": True, **c})
            print(f"[{task}] {key} vs {ref}: dAcc {c['accuracy_diff']:+.3f} [{c['accuracy_ci'][0]:+.3f}, "
                  f"{c['accuracy_ci'][1]:+.3f}] dF1 {c['f1_diff']:+.3f} [{c['f1_ci'][0]:+.3f}, {c['f1_ci'][1]:+.3f}]")

    bench["models"] = [m for m in bench["models"] if m["key"] not in VARIANTS] + rows
    bench["comparisons"] = [c for c in bench["comparisons"] if not c.get("variant")] + comparisons
    bench["risk_coverage_variants"] = curves
    bench["variants_note"] = (f"Calibrated / bias-corrected variants scored on {time.strftime('%Y-%m-%d %H:%M')}, after "
                              "they had been fixed on out-of-fold data (scripts/31). Re-scored once after the full "
                              "model received per-condition unusual-image limits (scripts/33, fixed on out-of-fold data); "
                              "the answers themselves did not change.")
    for key in VARIANTS:  # same networks as the uncorrected model: same CPU time (+ microseconds of arithmetic)
        base = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
        base_key = next(m for m in base["models"] if m["key"] == key)["base_model"]
        bench["latency"][key] = bench["latency"].get(base_key)
    save_json(bench, RESULTS_DIR / "final_benchmark.json")

    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    by_key = {r["key"]: r for r in rows}
    for entry in deployment["models"]:
        r = by_key.get(entry["key"])
        if r:
            entry["test_metrics"] = {k: r["test"][k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                                                              "f1_weighted", "accuracy_ci", "f1_macro_ci", "ece",
                                                              "balanced_accuracy", "classwise_ece")}
            entry["latency_ms"] = (bench["latency"].get(entry["key"]) or {}).get("median_ms")
            entry["operating_point"]["test"] = {"coverage": r["selective"]["coverage"],
                                                "accuracy_kept": r["selective"]["accuracy_kept"], "photos": r["photos"]}
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    print("updated final_benchmark.json and deployment.json")


if __name__ == "__main__":
    main()
