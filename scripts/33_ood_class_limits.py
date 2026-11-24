"""Upgrade round, step 11 — equal "unusual image" false-flag rates for every condition (out-of-fold only).

With one limit for all classes, photos of rarer conditions were flagged as "unusual" far more often than Leaf Miner
(out-of-fold: Magnesium 14.6% vs Leaf Miner 4.0% in the focused task; Early/Late Blight ~15% vs 1.7% in the full
task) — a bias against exactly the conditions that matter. Here each served model gets one multiplier per answered
condition: the 95th percentile of that condition's out-of-fold unusualness ratios (shrunk towards 1 for conditions
with few photos), so about 5% of genuine photos of EVERY condition are flagged. Cross-fitted (fit on 4 CV folds,
check on the 5th); far-OOD probe images (all fold models, as deployed) and out-of-scope leaves are re-checked.
Adopted only if it does not cost safety: the share of probe images of half A (half B stays for reporting) that are
flagged may drop by at most 0.02 — otherwise the model keeps one limit for all conditions.
No test photo is used. Writes experiments/results/ood_class_limits.json and adjust.ood_class_scale in
deployment.json and benchmark_models.json.
"""
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from tomatoleaf import balance  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, EXPERIMENTS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.probes import far_ood_probes  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402

KEEP, SHRINK_N, CLAMP = 0.95, 30, (0.5, 2.0)
step31 = import_module("31_balance_models")


def class_scales(ratios, decisions, k):
    out = np.ones(k)
    for c in range(k):
        r = ratios[decisions == c]
        if len(r):
            q = float(np.quantile(r, KEEP))
            w = len(r) / (len(r) + SHRINK_N)
            out[c] = float(np.clip(w * q + (1 - w) * 1.0, *CLAMP))
    return out


def network_data(task, sel):
    """Own-fold (held-out) ratios and probs of the chosen networks, plus deployed-style probe ratios/probs."""
    method, scale = sel["ood"]["method"], sel["ood"]["scale"]
    held_r, probe_r, probe_p, oos_r, oos_p = [], [], [], [], []
    for m in sel["chosen"]["members"]:
        v = m["views"]
        f = [np.load(EXPERIMENTS_DIR / "cv" / task / m["recipe"] / f"fold{k}.npz", allow_pickle=True) for k in range(5)]
        thr = [np.quantile(x[f"held_{method}{v}"], 0.95) * scale for x in f]
        held_r.append(np.concatenate([x[f"held_{method}{v}"] / t for x, t in zip(f, thr)]))
        probe_r += [x[f"probe_{method}{v}"] / t for x, t in zip(f, thr)]
        probe_p += [step31_softmax(x[f"probe_logits{v}"] / m["temperature"]) for x in f]
        if "oos_logits1" in f[0].files:
            oos_r.append(np.concatenate([x[f"oos_{method}{v}"] / t for x, t in zip(f, thr)]))
            oos_p.append(np.concatenate([step31_softmax(x[f"oos_logits{v}"] / m["temperature"]) for x in f]))
    return (np.mean(held_r, 0), np.mean(probe_r, 0), np.mean(probe_p, 0),
            np.mean(oos_r, 0) if oos_r else None, np.mean(oos_p, 0) if oos_p else None)


def step31_softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    bench = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
    report = {}
    halves = far_ood_probes()[2]
    for entry in deployment["models"]:
        task, adj = entry["task"], entry["adjust"]
        keys = step31.FIVE if task == "5class" else list(CLASS_KEYS)
        sel = load_json(RESULTS_DIR / f"selection_{task}.json")
        p, y, folds = step31.oof_probs(task, sel["chosen"]["members"])
        cal = balance.cross_fit_calibration(p, y, folds)
        dec = balance.decide(cal, adj["prior"], adj["tau"])
        ratios, probe_r, probe_p, oos_r, oos_p = network_data(task, sel)
        k = len(keys)
        flagged_cf = np.zeros(len(y), bool)  # cross-fitted: limits from 4 folds, flags on the 5th
        for f in np.unique(folds):
            s = class_scales(ratios[folds != f], dec[folds != f], k)
            flagged_cf[folds == f] = ratios[folds == f] / s[dec[folds == f]] > 1
        scales = class_scales(ratios, dec, k)
        probe_dec = balance.decide(balance.apply(probe_p, adj["calibration"]), adj["prior"], adj["tau"])
        row = {"scales": dict(zip(keys, scales.round(4).tolist())),
               "flagged_in_before": float(np.mean(ratios > 1)), "flagged_in_after": float(np.mean(flagged_cf)),
               "per_class": [{"key": c, "photos": int(np.sum(y == i)), "before": float(np.mean(ratios[y == i] > 1)),
                              "after": float(np.mean(flagged_cf[y == i]))} for i, c in enumerate(keys)],
               "probes_flagged_before": float(np.mean(probe_r > 1)),
               "probes_flagged_after": float(np.mean(probe_r / scales[probe_dec] > 1)),
               "probes_A_before": float(np.mean(probe_r[halves == "A"] > 1)),
               "probes_A_after": float(np.mean((probe_r / scales[probe_dec])[halves == "A"] > 1))}
        row["adopted"] = row["probes_A_after"] >= row["probes_A_before"] - 0.02
        if oos_r is not None:
            oos_dec = balance.decide(balance.apply(oos_p, adj["calibration"]), adj["prior"], adj["tau"])
            row.update({"oos_flagged_before": float(np.mean(oos_r > 1)),
                        "oos_flagged_after": float(np.mean(oos_r / scales[oos_dec] > 1))})
        report[entry["key"]] = row
        big = [r for r in row["per_class"] if r["photos"] >= 30]

        def spread(key, big=big):
            return [r[key] for r in big]
        print(f"{entry['key']:18s} flagged {row['flagged_in_before']:.3f} -> {row['flagged_in_after']:.3f} | per-class "
              f"range {min(spread('before')):.3f}-{max(spread('before')):.3f} -> {min(spread('after')):.3f}-"
              f"{max(spread('after')):.3f} | probes {row['probes_flagged_before']:.2f} -> {row['probes_flagged_after']:.2f}"
              + (f" | out-of-scope {row['oos_flagged_before']:.2f} -> {row['oos_flagged_after']:.2f}" if oos_r is not None else ""))
        print(f"   probes half A {row['probes_A_before']:.2f} -> {row['probes_A_after']:.2f} | adopted: {row['adopted']}")
        for target in [entry] + [b for b in bench["models"] if b["key"] == entry["key"]]:
            if row["adopted"]:
                target["adjust"]["ood_class_scale"] = scales.tolist()
            else:
                target["adjust"].pop("ood_class_scale", None)
    save_json(report, RESULTS_DIR / "ood_class_limits.json")
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    save_json(bench, EXPERIMENTS_DIR / "benchmark_models.json")


if __name__ == "__main__":
    main()
