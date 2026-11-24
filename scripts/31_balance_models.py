"""Upgrade round, step 9 — per-class calibration and class-bias correction of the version-2 models (out-of-fold only).

The networks are unchanged; their averaged probabilities are post-processed (tomatoleaf.balance):
  * per-class calibration (vector scaling, penalised NLL): each condition's confidence matches its own accuracy;
  * prior correction p / prior^tau decides WHICH condition is answered (rare conditions are not out-voted by frequent
    ones); the confidence shown stays the calibrated probability of that answer.
Fitted and chosen on out-of-fold predictions of the 1,813 train+val photos, cross-fitted (fit on 4 CV folds, apply to
the 5th) so the reported numbers are honest. No test photo is used here. Rules fixed in code:
  * tau: highest balanced accuracy with accuracy at most 2 points below the uncorrected model (balance.choose_tau);
  * "not sure" threshold: lowest threshold whose accuracy on answered photos reaches the target (95% focused,
    90% full) with coverage >= 50% (as in scripts/22).
Served variants (user request: two focused models, one full model, bias removed):
  balanced_5class (default), calibrated_5class (calibration only), balanced_8class.
Writes experiments/results/balance_<task>.json; adds the variants to benchmark_models.json and deployment.json.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tomatoleaf import balance  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, DATA_DIR, EXPERIMENTS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402

FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
TARGET = {"5class": 0.95, "8class": 0.90}
VARIANTS = [  # (key, task, use prior correction, display name); served in this order, the first per task = default
    ("balanced_5class", "5class", True, "Tomato Care v2 · focused · balanced"),
    ("calibrated_5class", "5class", False, "Tomato Care v2 · focused · calibrated"),
    ("balanced_8class", "8class", True, "Tomato Care v2 · full · balanced"),
]
SERVED_DEFAULTS = {"5class": "balanced_5class", "8class": "balanced_8class"}  # the user asked for bias removed


def oof_probs(task: str, members: list[dict]):
    probs, y, folds, pid = [], None, None, None
    for m in members:
        d = EXPERIMENTS_DIR / "cv" / task / m["recipe"]
        f = [np.load(d / f"fold{k}.npz", allow_pickle=True) for k in range(5)]
        ids = np.concatenate([x["held_photo_id"] for x in f])
        if pid is None:
            pid, y = ids, np.concatenate([x["held_y"] for x in f])
            folds = np.concatenate([np.full(len(x["held_y"]), k) for k, x in enumerate(f)])
        assert (ids == pid).all()
        z = np.concatenate([x[f"held_logits{m['views']}"] for x in f]) / m["temperature"]
        e = np.exp(z - z.max(1, keepdims=True))
        probs.append(e / e.sum(1, keepdims=True))
    return np.mean(probs, 0), y, folds


def ece(conf: np.ndarray, ok: np.ndarray, bins: int = 15) -> float:
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum(abs(conf[idx == b].mean() - ok[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(p: np.ndarray, y: np.ndarray, keys: list[str], pred=None) -> dict:
    pred = p.argmax(1) if pred is None else np.asarray(pred)
    conf, ok = balance.confidence_of(p, pred), pred == y
    return {"accuracy": float(ok.mean()), "f1_macro": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": balance.balanced_accuracy(y, pred, len(keys)), "ece": ece(conf, ok),
            "classwise_ece": balance.classwise_ece(p, y, pred),
            "per_class": [{"key": c, "recall": float(np.mean(pred[y == i] == i)),
                           "confidence_when_answered": float(conf[pred == i].mean()) if np.any(pred == i) else None,
                           "accuracy_when_answered": float(np.mean(y[pred == i] == i)) if np.any(pred == i) else None}
                          for i, c in enumerate(keys)]}


def selective_threshold(conf: np.ndarray, ok: np.ndarray, target: float):
    for t in np.round(np.arange(0.40, 0.991, 0.01), 2):
        keep = conf >= t
        if keep.mean() < 0.5:
            break
        if ok[keep].mean() >= target:
            return float(t), {"threshold": float(t), "coverage": float(keep.mean()), "accuracy_kept": float(ok[keep].mean())}
    return None, None


def main():
    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    bench = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
    pool = pd.read_csv(DATA_DIR / "cv_folds.csv")
    by_key = {m["key"]: m for m in bench["models"]}
    fitted, new_entries = {}, []
    for task in ("5class", "8class"):
        keys = FIVE if task == "5class" else list(CLASS_KEYS)
        sel = load_json(RESULTS_DIR / f"selection_{task}.json")
        p, y, folds = oof_probs(task, sel["chosen"]["members"])
        prior = np.array([(pool.class_key == c).sum() for c in keys], dtype=float)
        prior /= prior.sum()
        base = metrics(p, y, keys)
        cal = balance.cross_fit_calibration(p, y, folds)
        tau = balance.choose_tau(cal, y, prior, base["accuracy"])
        fitted[task] = {"keys": keys, "prior": prior, "tau": tau, "calibration": balance.fit_vector_scaling(p, y),
                        "cal": cal, "y": y, "base": base}
        print(f"[{task}] uncorrected: acc {base['accuracy']:.3f} F1 {base['f1_macro']:.3f} "
              f"balAcc {base['balanced_accuracy']:.3f} ECE {base['ece']:.3f} classwise {base['classwise_ece']:.3f}; tau={tau}")

    results = {}
    for key, task, use_prior, name in VARIANTS:
        f = fitted[task]
        tau = f["tau"] if use_prior else 0.0
        pred = balance.decide(f["cal"], f["prior"], tau)
        m = metrics(f["cal"], f["y"], f["keys"], pred)
        thr, op = selective_threshold(balance.confidence_of(f["cal"], pred), pred == f["y"], TARGET[task])
        results[key] = {"task": task, "tau": tau, "threshold": thr, "operating_point_oof": op, "oof": m}
        rec = " ".join(f"{c['key'][:6]}={c['recall']:.2f}" for c in m["per_class"])
        print(f"  {key:18s} tau {tau:.2f} acc {m['accuracy']:.3f} F1 {m['f1_macro']:.3f} balAcc {m['balanced_accuracy']:.3f} "
              f"ECE {m['ece']:.3f} classwise {m['classwise_ece']:.3f} thr {thr} {op and round(op['coverage'], 3)} | {rec}")
        src = by_key[f"upgrade_{task}"]
        entry = copy.deepcopy(src)
        for k in ("test_metrics", "latency_ms"):
            entry.pop(k, None)
        entry.update({
            "key": key, "name": name, "variant": "balanced" if use_prior else "calibrated", "base_model": src["key"],
            "adjust": {"calibration": f["calibration"], "prior": f["prior"].tolist(), "tau": tau,
                       "fitted_on": "out-of-fold predictions of the train+val photos (cross-fitted)"},
            "confidence_threshold": thr,
            "operating_point": {"threshold": thr, "chosen_on": "out-of-fold cross-validation predictions (cross-fitted)",
                                "rule": f"lowest calibrated confidence with out-of-fold accuracy on answered photos >= "
                                        f"{TARGET[task]:.0%} (coverage >= 50%)", "oof": op},
            "description": ("Same networks; per-class calibration, and a class-prior correction so rare conditions "
                            "are not out-voted by frequent ones (confidence stays calibrated)." if use_prior else
                            "Same networks; per-class calibration so every condition's confidence matches its own "
                            "accuracy."),
        })
        new_entries.append(entry)
    for task in ("5class", "8class"):
        f = fitted[task]
        save_json({"task": task, "classes": f["keys"], "prior": f["prior"].tolist(), "tau": f["tau"],
                   "rule_tau": f"highest cross-fitted balanced accuracy with accuracy >= uncorrected - {balance.MAX_ACC_DROP}",
                   "calibration": f["calibration"], "oof_uncorrected": f["base"],
                   "variants": {k: v for k, v in results.items() if v["task"] == task},
                   "tau_curve": [{"tau": t, **{k: v for k, v in metrics(f["cal"], f["y"], f["keys"],
                                                                         balance.decide(f["cal"], f["prior"], t)).items()
                                               if k != "per_class"}} for t in balance.TAUS]},
                  RESULTS_DIR / f"balance_{task}.json")

    variant_keys = {e["key"] for e in new_entries}
    bench["models"] = [m for m in bench["models"] if m["key"] not in variant_keys] + new_entries
    save_json(bench, EXPERIMENTS_DIR / "benchmark_models.json")
    deployment["models"] = new_entries
    deployment["defaults_by_task"] = dict(SERVED_DEFAULTS)
    deployment["default_model"] = SERVED_DEFAULTS["5class"]
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    print("served:", [e["key"] for e in new_entries], "defaults:", SERVED_DEFAULTS)


if __name__ == "__main__":
    main()
