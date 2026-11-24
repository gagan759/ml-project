"""Final round, step 2 — one validated pipeline per model, chosen on out-of-fold data only (docs/AUDIT.md §6).

For each task (focused 5-class, full 8-class), every imbalance arm trained by scripts/21 is turned into a complete
pipeline on its cross-fitted out-of-fold (OOF) predictions, and one pipeline is selected. The rules were written down
before training:

P1  Arms: A class-weighted loss, B class-balanced sampling (one compensation mechanism each). The focused model uses
    arm A with outlier exposure; the other planned arms (C: plain + post-hoc logit adjustment, and B/C with outlier
    exposure) were dropped for time before any of their results existed. Last round adds leaf-crop arms (D focused,
    E full). Selection: highest OOF macro-F1 (argmax of the final calibrated probabilities); within 0.01 -> lower OOF
    NLL.
P2  Calibration: temperature scaling vs per-class vector scaling, both cross-fitted; lower OOF NLL wins, within 0.005
    -> temperature scaling. The answer is the argmax of the final calibrated probabilities (= the tallest bar).
P3  One threshold: the lowest calibrated confidence whose OOF accuracy on answered photos is >= 90% (the user's target,
    both models) with >= 50% of in-scope photos answered. Focused: an answered out-of-scope photo counts as an error.
    (An extra ">= 80% of out-of-scope photos refused" constraint forced 0.95 and left the focused model answering
    almost only Leaf Miner; it was dropped at the user's request to lower the threshold. Refusal is still reported.)
P4  Unusual-image gate of the selected networks: kNN vs Mahalanobis feature distance (each fold network's own
    training photos), chosen by OOF AUROC against synthetic non-leaf probes (half A), leaf-free paper made from
    held-out photos and, for the focused model, OOF out-of-scope photos. Limits: 95% of OOF genuine photos pass, per
    answered condition (shrunk towards the global limit for conditions with few photos).

No test photo is loaded. Writes experiments/results/final_pipeline_<task>.json, the deployable fold checkpoints
(experiments/models/deploy/<task>/<recipe>/fold<k>.pt, with feature density) and the registry entries
experiments/final_models.json (test metrics are added by scripts/37).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]
from tomatoleaf import balance  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, DATA_DIR, EXPERIMENTS_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.crop import leaf_crop_batch  # noqa: E402
from tomatoleaf.data import load_images  # noqa: E402
from tomatoleaf.evaluation import (  # noqa: E402
    brier_score,
    classification_metrics,
    confusion,
    expected_calibration_error,
    fit_temperature,
    per_class_metrics,
    reliability_bins,
    softmax,
)
from tomatoleaf.features import leaf_mask  # noqa: E402
from tomatoleaf.models import load_checkpoint  # noqa: E402
from tomatoleaf.ood import FeatureDensity, auroc, fpr_at_tpr  # noqa: E402
from tomatoleaf.probes import far_ood_probes  # noqa: E402
from tomatoleaf.recipes import RECIPES  # noqa: E402
from tomatoleaf.training import infer  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402

CV_DIR = EXPERIMENTS_DIR / "cv"
DEPLOY_DIR = MODELS_DIR / "deploy"
FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
KEYS = {"5class": FIVE, "8class": list(CLASS_KEYS)}
ARMS = {
    # focused: arms B/C (balanced sampling / logit adjustment with outlier exposure) were dropped for time before any
    # result was seen; the weighting-vs-sampling comparison is made on the full model
    "5class": {"A": "convnext_t_in1k_robust_oe", "D": "convnext_t_in1k_robust_oe_crop"},
    "8class": {"A": "convnext_t_in1k_robust", "B": "convnext_t_in1k_sampler", "E": "convnext_t_in1k_sampler_crop"},
}
ARM_LABEL = {"A": "class-weighted loss", "B": "class-balanced sampling",
             "D": "leaf crop + class-weighted loss (outlier exposure 1.0)", "E": "leaf crop + class-balanced sampling"}
REFERENCE = {"5class": "convnext_t_in1k_robust"}  # the v2 focused networks (no outlier exposure), for comparison only
TARGET = {"5class": 0.90, "8class": 0.90}
F1_TIE, NLL_TIE = 0.01, 0.005
GRID = np.round(np.arange(0.30, 0.991, 0.01), 2)
OOD_KEEP, SHRINK_N, CLAMP = 0.95, 30, (0.5, 2.0)
PAPER_PER_FOLD = 30
NAMES = {"5class": ("focused_5class", "Tomato Care · Deep learning (5 conditions)"),
         "8class": ("full_8class", "Tomato Care · Full (8 conditions)")}


# ---------------------------------------------------------------------------------------------------- OOF data
def load_arm(task: str, recipe: str) -> dict:
    parts = {}
    for k in range(5):
        f = np.load(CV_DIR / task / recipe / f"fold{k}.npz", allow_pickle=True)
        rows = {"z": f["held_logits1"], "y": f["held_y"], "fold": np.full(len(f["held_y"]), k),
                "pid": f["held_photo_id"], "knn": f["held_knn1"], "maha": f["held_maha1"]}
        if "oos_logits1" in f.files:
            rows.update({"oz": f["oos_logits1"], "ofold": np.full(len(f["oos_label"]), k), "opid": f["oos_photo_id"],
                         "olabel": f["oos_label"], "oknn": f["oos_knn1"], "omaha": f["oos_maha1"]})
        rows.update({"probe_z": f["probe_logits1"][None], "probe_knn": f["probe_knn1"][None],
                     "probe_maha": f["probe_maha1"][None]})
        for key, v in rows.items():
            parts.setdefault(key, []).append(v)
    return {key: np.concatenate(v) for key, v in parts.items()}


def macro_f1(y, pred, k):
    return f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0)



def fit_cal(kind, z, y):
    if kind == "temperature":
        return {"T": float(fit_temperature(z, y))}
    return balance.fit_vector_scaling(softmax(z), y)


def apply_cal(kind, cal, z):
    if kind == "temperature":
        return softmax(z, cal["T"])
    return balance.apply(softmax(z), cal)


def calibrate(z, y, folds, oz=None, ofold=None):
    """Cross-fitted calibrated probabilities for both calibrators (in-scope and out-of-scope OOF photos)."""
    out = {}
    for kind in ("temperature", "vector"):
        p = np.zeros_like(z, dtype=np.float64)
        po = None if oz is None else np.zeros_like(oz, dtype=np.float64)
        for f in range(5):
            cal = fit_cal(kind, z[folds != f], y[folds != f])
            p[folds == f] = apply_cal(kind, cal, z[folds == f])
            if oz is not None:
                po[ofold == f] = apply_cal(kind, cal, oz[ofold == f])
        nll = float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())
        out[kind] = {"p": p, "po": po, "nll": nll, "full": fit_cal(kind, z, y)}
    return out


def operating_point(p, y, po, target):
    conf, pred = p.max(1), p.argmax(1)
    oconf = po.max(1) if po is not None else np.zeros(0)
    curve = []
    for t in GRID:
        keep, okeep = conf >= t, oconf >= t
        answered = int(keep.sum() + okeep.sum())
        curve.append({"threshold": float(t), "coverage": float(keep.mean()),
                      "accuracy_kept": float((pred[keep] == y[keep]).sum() / answered) if answered else None,
                      "out_of_scope_answered": float(okeep.mean()) if len(oconf) else None})
    ok = [r for r in curve if r["accuracy_kept"] is not None and r["accuracy_kept"] >= target and r["coverage"] >= 0.5]
    if ok:
        return ok[0], (f"lowest calibrated confidence with out-of-fold accuracy on answered photos >= {target:.0%} "
                       f"(coverage of in-scope photos >= 50%"
                       + ("; answered out-of-scope photos count as errors" if po is not None else "") + ")"), curve
    row = max((r for r in curve if r["coverage"] >= 0.5), key=lambda r: r["accuracy_kept"])
    return row, f"target {target:.0%} not reachable at >= 50% coverage: best accuracy at >= 50% coverage", curve


def full_metrics(p, y, k):
    pred = p.argmax(1)
    m = classification_metrics(y, pred, list(range(k)))
    m.update({"balanced_accuracy": balance.balanced_accuracy(y, pred, k),
              "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
              "brier": brier_score(p, y), "ece": expected_calibration_error(p, y),
              "classwise_ece": balance.classwise_ece(p, y)})
    return m


def evaluate_arm(task, arm, recipe, prior):
    k = len(KEYS[task])
    d = load_arm(task, recipe)
    z, y, folds = d["z"].astype(np.float64), d["y"], d["fold"]
    oz, ofold = (d["oz"].astype(np.float64), d["ofold"]) if "oz" in d else (None, None)
    raw_f1 = macro_f1(y, z.argmax(1), k)
    cals = calibrate(z, y, folds, oz, ofold)
    nll_t, nll_v = cals["temperature"]["nll"], cals["vector"]["nll"]
    kind = "vector" if nll_v < nll_t - NLL_TIE else "temperature"
    c = cals[kind]
    op, rule, curve = operating_point(c["p"], y, c["po"], TARGET[task])
    refusal = None if c["po"] is None else 1.0 - op["out_of_scope_answered"]
    met = full_metrics(c["p"], y, k)
    p_raw = softmax(z)
    return {"arm": arm, "recipe": recipe, "label": ARM_LABEL.get(arm, arm), "photos": int(len(y)),
            "macro_f1_before_calibration": raw_f1,
            "uncalibrated": {"ece": expected_calibration_error(p_raw, y),
                             "nll": float(-np.log(np.clip(p_raw[np.arange(len(y)), y], 1e-12, 1)).mean()),
                             "brier": brier_score(p_raw, y), "classwise_ece": balance.classwise_ece(p_raw, y)},
            "_p_raw": p_raw,
            "calibration": {"chosen": kind, "nll_temperature": nll_t, "nll_vector": nll_v,
                            "fitted_on_all_folds": c["full"]},
            "oof": met, "threshold": op["threshold"], "threshold_rule": rule, "operating_point": op,
            "out_of_scope_refusal": refusal, "curve": curve,
            "_p": c["p"], "_po": c["po"], "_y": y, "_folds": folds, "_d": d}


def select(task, arms):
    pool = list(arms.values())
    top = max(a["oof"]["f1_macro"] for a in pool)
    tied = [a for a in pool if a["oof"]["f1_macro"] >= top - F1_TIE]
    chosen = min(tied, key=lambda a: a["oof"]["nll"])
    why = (f"highest out-of-fold macro-F1 ({top:.4f})" if len(tied) == 1 else
           f"macro-F1 within {F1_TIE} of the best ({', '.join(a['arm'] for a in tied)}): lowest out-of-fold NLL")
    return chosen, why


# ---------------------------------------------------------------------------------------------- unusual-image gate
def paper_images(rgb_batch):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    out = []
    for rgb in rgb_batch:
        m = cv2.dilate(leaf_mask(rgb).astype(np.uint8), kernel)
        out.append(cv2.inpaint(rgb, m * 255, 9, cv2.INPAINT_TELEA))
    return np.stack(out)


def class_scales(ratios, decisions, k):
    out = np.ones(k)
    for c in range(k):
        r = ratios[decisions == c]
        if len(r):
            w = len(r) / (len(r) + SHRINK_N)
            out[c] = float(np.clip(w * float(np.quantile(r, OOD_KEEP)) + (1 - w), *CLAMP))
    return out


def build_gate(task, chosen, device):
    """Densities of the chosen fold networks, deployable checkpoints, method choice and limits (OOF only)."""
    recipe_name, keys = chosen["recipe"], KEYS[task]
    recipe, d = RECIPES[recipe_name], chosen["_d"]
    pool = pd.read_csv(DATA_DIR / "cv_folds.csv")
    inside = pool[pool.class_key.isin(keys)]
    local = {CLASS_KEYS.index(c): i for i, c in enumerate(keys)}
    paper = {"knn": [], "maha": []}
    for k in range(5):
        src = CV_DIR / task / recipe_name / f"fold{k}.pt"
        model, ckpt = load_checkpoint(src, device)
        kw = dict(size=recipe.input_size, mean=ckpt["mean"], std=ckpt["std"], resize=recipe.eval_resize)
        tr = inside[inside.fold != k]
        prep = leaf_crop_batch if recipe.crop == "leaf" else (lambda x: x)  # the networks' own preprocessing
        _, f_tr = infer(model, prep(load_images(tr.path)), device, features=True, **kw)
        density = FeatureDensity(f_tr, tr.label.map(local).to_numpy())
        deploy = DEPLOY_DIR / task / recipe_name / f"fold{k}.pt"
        deploy.parent.mkdir(parents=True, exist_ok=True)
        raw = torch.load(src, map_location="cpu", weights_only=True)
        torch.save({**raw, "density": density.to_dict()}, deploy)
        held = inside[inside.fold == k].sort_values("photo_id").head(PAPER_PER_FOLD)
        _, f_pa = infer(model, prep(paper_images(load_images(held.path))), device, features=True, **kw)
        paper["knn"].append(density.knn(f_pa))
        paper["maha"].append(density.mahalanobis(f_pa))
        del model
        torch.cuda.empty_cache()
    _, _, halves = far_ood_probes()
    p_oof = chosen["_p"]
    decisions = p_oof.argmax(1)
    evals = {}
    for method in ("knn", "maha"):
        thr = np.array([np.quantile(d[method][d["fold"] == k], OOD_KEEP) for k in range(5)])
        in_r = d[method] / thr[d["fold"]]
        probe_r = (d[f"probe_{method}"] / thr[:, None]).mean(0)  # every fold network, as deployed
        paper_r = np.concatenate([paper[method][k] / thr[k] for k in range(5)])
        oos_r = d[f"o{method}"] / thr[d["ofold"]] if "oz" in d else None
        a = probe_r[halves == "A"]
        out_sel = np.concatenate([a, paper_r] + ([oos_r] if oos_r is not None else []))
        scale = float(np.quantile(in_r, OOD_KEEP))
        cs = class_scales(in_r / scale, decisions, len(keys))
        lim_in = in_r / (scale * cs[decisions])
        ev = {"auroc_select": auroc(in_r, out_sel), "fpr95_select": fpr_at_tpr(in_r, out_sel),
              "auroc_probes_A": auroc(in_r, a), "auroc_paper": auroc(in_r, paper_r),
              "auroc_out_of_scope": auroc(in_r, oos_r) if oos_r is not None else None,
              "scale": scale, "class_scales": cs.tolist(), "member_thresholds": thr.tolist(),
              "flagged_in_scope": float((lim_in > 1).mean()),
              "flagged_in_scope_by_condition": {keys[c]: float((lim_in[chosen["_y"] == c] > 1).mean())
                                                for c in range(len(keys))},
              "flagged_probes_A": float((a / scale > 1).mean()),
              "flagged_probes_B": float((probe_r[halves == "B"] / scale > 1).mean()),
              "flagged_paper": float((paper_r / scale > 1).mean())}
        if oos_r is not None:
            po = chosen["_po"]
            od = po.argmax(1)
            ev["flagged_out_of_scope"] = float((oos_r / (scale * cs[od]) > 1).mean())
            refused = (po.max(1) < chosen["threshold"]) | (oos_r / (scale * cs[od]) > 1)
            ev["out_of_scope_refused_with_gate"] = float(refused.mean())
        evals[method] = ev
    method = max(evals, key=lambda m: evals[m]["auroc_select"])
    return method, evals


# ----------------------------------------------------------------------------------------------------- registry
def registry_entry(task, chosen, method, gate):
    key, name = NAMES[task]
    recipe = chosen["recipe"]
    kind = chosen["calibration"]["chosen"]
    cal = chosen["calibration"]["fitted_on_all_folds"]
    ev = gate[method]
    members = []
    for k in range(5):
        spec = {"key": f"{recipe}_fold{k}", "name": f"{RECIPES[recipe].label} · fold {k}",
                "path": f"deploy/{task}/{recipe}/fold{k}.pt",
                "temperature": cal["T"] if kind == "temperature" else 1.0, "tta_views": 1,
                "ood_method": method, "ood_threshold": ev["member_thresholds"][k] * ev["scale"]}
        members.append(spec)
    adjust = {"ood_class_scale": ev["class_scales"],
              "fitted_on": "out-of-fold predictions of the train+val photos (cross-fitted)"}
    if kind == "vector":
        adjust["calibration"] = cal
    op = chosen["operating_point"]
    return {
        "key": key, "name": name, "model_name": "ConvNeXt-Tiny ensemble", "task": task, "generation": "upgrade",
        "round": "final", "family": "Deep learning",
        "kind": "ensemble", "type": "ensemble", "class_keys": KEYS[task], "members": members, "recipes": [recipe],
        "tta_views": 1, "feature_set": f"5-fold ensemble of {RECIPES[recipe].label}",
        "description": (f"Imbalance: {chosen['label']}. Calibration: {kind} scaling. One threshold "
                        f"({op['threshold']:.2f}) and a per-condition unusual-image gate, all fixed on out-of-fold data."),
        "confidence_threshold": op["threshold"],
        "operating_point": {"threshold": op["threshold"], "chosen_on": "out-of-fold cross-validation predictions "
                            "(cross-fitted)", "rule": chosen["threshold_rule"],
                            "oof": {k: op[k] for k in ("coverage", "accuracy_kept", "out_of_scope_answered")}},
        "ood": {"method": f"{method} feature distance (fold models' own training photos)", "keep": OOD_KEEP,
                "per_condition": True},
        "adjust": adjust,
        "pipeline": {"imbalance": chosen["label"], "arm": chosen["arm"],
                     "calibration": kind, "threshold": op["threshold"], "unusual_gate": method,
                     "outlier_exposure": RECIPES[recipe].outlier_weight > 0},
    }


def strip(a):
    return {k: v for k, v in a.items() if not k.startswith("_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="5class,8class")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    entries = load_json(EXPERIMENTS_DIR / "final_models.json") if (EXPERIMENTS_DIR / "final_models.json").exists() \
        else {"models": []}
    for task in args.tasks.split(","):
        keys = KEYS[task]
        pool = pd.read_csv(DATA_DIR / "cv_folds.csv")
        counts = pool[pool.class_key.isin(keys)].class_key.value_counts()
        prior = np.array([counts[c] for c in keys], dtype=np.float64)
        prior /= prior.sum()
        arms = {arm: evaluate_arm(task, arm, recipe, prior) for arm, recipe in ARMS[task].items()}
        ref = {}
        if task in REFERENCE:
            r = evaluate_arm(task, "reference", REFERENCE[task], prior)
            ref = {"recipe": REFERENCE[task], "label": "v2 networks without outlier exposure (reference only)",
                   **{k: v for k, v in strip(r).items() if k in ("oof", "threshold", "out_of_scope_refusal",
                                                                 "calibration", "operating_point")}}
        for a in arms.values():
            a["_prior"] = prior
            print(f"{task} arm {a['arm']} ({a['recipe']}): OOF macro-F1 {a['oof']['f1_macro']:.4f} acc "
                  f"{a['oof']['accuracy']:.4f} NLL {a['oof']['nll']:.4f} | cal {a['calibration']['chosen']} | thr "
                  f"{a['threshold']:.2f} cov {a['operating_point']['coverage']:.3f} acc_kept "
                  f"{a['operating_point']['accuracy_kept']:.3f}"
                  + (f" | out-of-scope refused {a['out_of_scope_refusal']:.3f}" if a["out_of_scope_refusal"] is not None
                     else ""))
        chosen, why = select(task, arms)
        print(f"{task}: chosen arm {chosen['arm']} ({chosen['recipe']}) — {why}")
        method, gate = build_gate(task, chosen, device)
        print(f"{task}: unusual-image gate {method} | " + " | ".join(
            f"{m}: AUROC {e['auroc_select']:.3f}, in-scope flagged {e['flagged_in_scope']:.3f}" for m, e in gate.items()))
        y, p = chosen["_y"], chosen["_p"]
        to_global = np.array([CLASS_KEYS.index(c) for c in keys])
        result = {
            "task": task, "classes": keys, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rules": "docs/AUDIT.md §6 (fixed before training); see this script's docstring",
            "prior": prior.tolist(), "arms": {k: strip(v) for k, v in arms.items()}, "reference": ref,
            "chosen": {"arm": chosen["arm"], "recipe": chosen["recipe"], "why": why,
                       "per_class_oof": per_class_metrics(to_global[y], to_global[p.argmax(1)], list(to_global)),
                       "confusion_oof": confusion(y, p.argmax(1), list(range(len(keys)))),
                       "reliability_oof": reliability_bins(p, y),
                       "reliability_oof_raw": reliability_bins(chosen["_p_raw"], y)},
            "unusual_gate": {"method": method, "keep": OOD_KEEP, "evaluation_oof": gate},
        }
        save_json(result, RESULTS_DIR / f"final_pipeline_{task}.json")
        entry = registry_entry(task, chosen, method, gate)
        entries["models"] = [e for e in entries["models"] if e["key"] != entry["key"]] + [entry]
        save_json(entries, EXPERIMENTS_DIR / "final_models.json")
        print(f"wrote final_pipeline_{task}.json and the {entry['key']} registry entry")


if __name__ == "__main__":
    main()
