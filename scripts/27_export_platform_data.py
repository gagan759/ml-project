"""Upgrade round, step 8 — everything the web platform shows, in one JSON document, plus the Grad-CAM gallery.

Reads only files written by earlier steps (no new evaluation of models, except Grad-CAM pictures for the gallery):
  experiments/results/final_benchmark.json, selection_*.json, baseline_audit.json, cv summaries, the original
  study's results (dataset statistics, leakage benchmark).
Writes:
  experiments/platform.json            served by the API at /api/analytics
  frontend/public/data/platform.json   bundled snapshot (the site works read-only without the API)
  frontend/public/data/disease_info.json
  frontend/public/gradcam/*.png        Grad-CAM gallery for the held-out sample photos
"""
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf.config import EXPERIMENTS_DIR, RESULTS_DIR  # noqa: E402
from tomatoleaf.data import load_splits  # noqa: E402
from tomatoleaf.features import leaf_mask  # noqa: E402
from tomatoleaf.gradcam import heatmap_png, overlay  # noqa: E402
from tomatoleaf.preprocessing import load_canonical_array  # noqa: E402
from tomatoleaf.recipes import RECIPES  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402

FRONT = ROOT / "frontend" / "public"
FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
TASKS = ("5class",)  # the apps diagnose five conditions; eight-condition results are no longer exported
TASK_LABEL = {"5class": "5 conditions"}
CLASSICAL = "classical_5class"  # the served classical ML model (scripts/43-44)
SHORT_RECIPE = {
    "convnext_t_in1k_base": "ConvNeXt-T 1k · orig.",
    "convnext_t_in1k_robust": "ConvNeXt-T 1k · robust",
    "convnext_t_in22k_base": "ConvNeXt-T 22k · orig.",
    "convnext_t_in22k_robust": "ConvNeXt-T 22k · robust",
    "convnextv2_t_robust": "ConvNeXt-V2-T · robust",
    "dinov2_s_robust": "ViT-S DINOv2 · robust",
    "effnetv2_s_robust": "EffNetV2-S · robust",
}
PERTURBATION_LABEL = {
    "jpeg_q75": "JPEG quality 75", "jpeg_q50": "JPEG quality 50", "blur_sigma1": "Mild blur (σ = 1)",
    "brightness_+15%": "Brightness +15%", "brightness_-15%": "Brightness −15%", "half_resolution": "Half resolution",
    "upload_p3_original_path": "Phone photo (P3) · old upload path", "upload_p3_new_path": "Phone photo (P3) · new upload path",
    "upload_srgb_original_path": "sRGB photo · old upload path", "upload_srgb_new_path": "sRGB photo · new upload path",
}


def png(arr: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, optimize=True)
    return "/" + path.relative_to(FRONT).as_posix()


def gradcam_for(predictor, rgb, class_index):
    """Grad-CAM of the first network, in the frame of `rgb` (leaf-crop models: placed back where the crop was)."""
    return getattr(predictor, "members", [predictor])[0].gradcam(rgb, class_index)


def build_gallery(defaults, bench_models):
    manifest = load_json(FRONT / "test-samples" / "manifest.json")
    reg = ModelRegistry(EXPERIMENTS_DIR / "deployment.json", device="cuda" if torch.cuda.is_available() else "cpu", lazy=True)
    models = [defaults["5class"]]  # Grad-CAM needs a network: the served deep-learning model
    names = {m["key"]: m["name"] for m in bench_models}
    items, hero = [], None
    for s in manifest["samples"]:
        rgb = load_canonical_array(FRONT / s["file"].lstrip("/"))
        entry = {"sample": s["key"], "true_key": s["key"], "photo_id": s.get("photo_id"),
                 "input": png(rgb, FRONT / "gradcam" / "input" / f"{s['key']}.png"), "overlays": []}
        mask = leaf_mask(rgb)
        for key in models:
            pred = reg.load(key)
            probs, _ = pred.analyse(rgb)
            top = pred.decide(probs)
            heat = gradcam_for(pred, rgb, top)
            heat256 = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR)) / 255.0
            pred_key = pred.class_keys[top]
            o = {"model": key, "model_name": names.get(key, key), "pred_key": pred_key, "confidence": float(probs[top]),
                 "correct": pred_key == s["key"], "out_of_scope": s["key"] not in pred.class_keys,
                 "overlay": png(overlay(rgb, heat), FRONT / "gradcam" / key / f"{s['key']}_overlay.png"),
                 "heatmap": png(heatmap_png(heat), FRONT / "gradcam" / key / f"{s['key']}_heat.png"),
                 "on_leaf_share": float((heat256 * mask).sum() / max(heat256.sum(), 1e-9))}
            entry["overlays"].append(o)
            if hero is None and key == defaults["5class"] and s["key"] == "leaf_miner":
                order = np.argsort(probs)[::-1]
                hero = {"input": entry["input"], "heatmap": o["heatmap"], "true_key": s["key"], "pred_key": pred_key,
                        "confidence": float(probs[top]), "model_name": names.get(key, key),
                        "probabilities": [{"key": pred.class_keys[i], "p": float(probs[i])} for i in order]}
        items.append(entry)
        print(f"gradcam {s['key']}: " + ", ".join(f"{o['model']}->{o['pred_key']} {o['confidence']:.2f} leaf {o['on_leaf_share']:.2f}"
                                                  for o in entry["overlays"]))
    shares = {m: float(np.mean([o["on_leaf_share"] for it in items for o in it["overlays"] if o["model"] == m])) for m in models}
    summary = ("Mean share of Grad-CAM mass on the leaf (leaf mask from saturation thresholding): "
               + "; ".join(f"{names.get(m, m)} {100 * v:.0f}%" for m, v in shares.items()) + ".")
    return {"items": items, "models": models, "summary": summary, "on_leaf_share": shares}, hero


def final_round_block(final_pipe, final_round, final_keys, classical, model_names):
    """What the apps show about the final round: the compared arms (out-of-fold), the chosen pipeline, and the
    before/after comparison on the test photos (with the served classical model) and stress / out-of-dataset checks."""
    if not final_pipe:
        return None
    out = {"rules": "docs/AUDIT.md §6 (fixed before training)", "tasks": {}}
    keep = ("accuracy", "f1_macro", "f1_weighted", "balanced_accuracy", "nll", "brier", "ece", "classwise_ece")
    for t, fp in final_pipe.items():
        arms = [{"arm": a["arm"], "recipe": a["recipe"], "label": a["label"], "oof": {k: a["oof"][k] for k in keep},
                 "calibration": a["calibration"]["chosen"], "nll_temperature": a["calibration"]["nll_temperature"],
                 "nll_vector": a["calibration"]["nll_vector"], "tau": (a.get("tau") or {}).get("deployed"),
                 "threshold": a["threshold"], "coverage": a["operating_point"]["coverage"],
                 "accuracy_kept": a["operating_point"]["accuracy_kept"],
                 "out_of_scope_refusal": a.get("out_of_scope_refusal"), "chosen": a["arm"] == fp["chosen"]["arm"]}
                for a in fp["arms"].values()]
        gate = fp["unusual_gate"]
        task = {"arms": arms, "chosen": fp["chosen"]["arm"], "why": fp["chosen"]["why"],
                "gate": {"method": gate["method"],
                         **{k: v for k, v in gate["evaluation_oof"][gate["method"]].items()
                            if k.startswith(("auroc", "flagged", "out_of_scope"))}},
                "reference": fp.get("reference") or None}
        if final_round:
            before = ["focused_5class_r1", "balanced_5class", "calibrated_5class", "ensemble_5class"]
            rows = []
            for key in [final_keys[t], classical] + before:
                r = final_round["models"].get(key)
                if r is None:
                    continue
                rows.append({"key": key, "name": r["name"], "model_name": model_names.get(key),
                             "final": key == final_keys[t], "classical": key == classical,
                             **{k: r["test"][k] for k in keep if k in r["test"]},
                             "accuracy_ci": r["test"].get("accuracy_ci"), "threshold": r["threshold"],
                             "coverage": r["selective"]["coverage"], "accuracy_kept": r["selective"]["accuracy_kept"],
                             "flagged_unusual": r["flagged_unusual"],
                             "label_differs_from_tallest_bar": r.get("label_differs_from_tallest_bar"),
                             "out_of_scope": r.get("out_of_scope"), "per_condition": r.get("per_condition")})
            task["test"] = rows
            task["comparisons"] = [c for c in final_round["comparisons"] if c["task"] == t]
            pair = [final_keys[t], before[0], classical]
            task["stress"] = {k: final_round["stress"].get(k) for k in pair}
            task["out_of_dataset"] = {k: final_round["out_of_dataset"].get(k) for k in pair}
            task["latency"] = {k: final_round["latency"].get(k) for k in pair}
        out["tasks"][t] = task
    if final_round:
        out["note"] = final_round["note"]
    return out


def curves_for(task, recipe):
    rows = []
    for k in range(5):
        p = EXPERIMENTS_DIR / "cv" / task / recipe / f"fold{k}.json"
        if p.exists():
            rows.append(load_json(p)["history"])
    if not rows:
        return []
    n = min(len(r) for r in rows)
    out = []
    for i in range(n):
        out.append({"epoch": rows[0][i]["epoch"],
                    **{k: float(np.mean([r[i][k] for r in rows if k in r[i]])) for k in ("train_loss", "val_f1", "val_acc")
                       if all(k in r[i] for r in rows)}})
    return out


def main():
    import sys as _sys
    bench_file = _sys.argv[_sys.argv.index("--benchmark") + 1] if "--benchmark" in _sys.argv else "final_benchmark.json"
    bench = load_json(RESULTS_DIR / bench_file)
    bench_models = load_json(EXPERIMENTS_DIR / "benchmark_models.json")["models"]
    meta = {m["key"]: m for m in bench_models}
    defaults, previous = bench["defaults"], bench["previous_defaults"]  # defaults: the version-2 networks
    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    served = deployment["defaults_by_task"]  # what the apps answer with (calibrated / bias-corrected variants)
    sel = {t: load_json(RESULTS_DIR / f"selection_{t}.json") for t in TASKS}
    variant_oof = {}
    for t in TASKS:
        bal = RESULTS_DIR / f"balance_{t}.json"
        if bal.exists():
            variant_oof.update({k: v["oof"] for k, v in load_json(bal)["variants"].items()})
    # final round (scripts/36 + 38): one validated pipeline per task
    FINAL_KEYS = {"5class": "focused_5class"}
    final_pipe = {t: load_json(RESULTS_DIR / f"final_pipeline_{t}.json") for t in TASKS
                  if (RESULTS_DIR / f"final_pipeline_{t}.json").exists()}
    classical = load_json(RESULTS_DIR / "classical_pipeline.json")
    ml_arm = classical["arms"][classical["chosen"]["arm"]]
    final_round = load_json(RESULTS_DIR / "final_round.json") if (RESULTS_DIR / "final_round.json").exists() else None
    final_oof = {FINAL_KEYS[t]: fp["arms"][fp["chosen"]["arm"]]["oof"] for t, fp in final_pipe.items()}
    final_oof[CLASSICAL] = ml_arm["oof"]
    old = load_json(EXPERIMENTS_DIR / "analytics.json")
    final8 = load_json(RESULTS_DIR / "final_results.json")
    five = load_json(RESULTS_DIR / "five_class_results.json")
    leak = load_json(RESULTS_DIR / "leakage_official_split.json")
    cvsum = load_json(RESULTS_DIR / "cv_folds_summary.json")

    # ---- dataset
    classes = []
    for c in old["dataset"]["class_distribution"]:
        cv_count = sum(f["per_class"][c["key"]] for f in cvsum["per_fold"])
        classes.append({"key": c["key"], "name": c["name"], "photos": c["photos"], "files": c["images"],
                        "train": c["clean"]["train"], "val": c["clean"]["val"], "test": c["clean"]["test"], "cv": cv_count})
    splits = load_splits()
    dataset = {"name": old["dataset"]["name"], "citation": old["dataset"]["citation"], "doi": old["dataset"]["doi"],
               "files": leak["total_images"], "photos": leak["unique_photos"], "groups": int(splits.group_id.nunique()),
               "classes": classes, "split": old["dataset"]["clean_split"], "official_split": old["dataset"]["official_split"],
               "cv": {"folds": cvsum["n_folds"], "photos": cvsum["photos"]},
               "source": {"phone": "realme GT Master Edition", "resolution": "3468×4624 / 4624×3468",
                          "stored": "256×256 (whole frame squashed)", "encoding": "Display P3, JPEG quality 90, 4:2:0"},
               "leakage": {"test_with_train_sibling": leak["test_images_with_train_sibling"] / leak["test_images"],
                           "val_with_train_sibling": leak["val_images_with_train_sibling"] / leak["val_images"],
                           "families_crossing": leak["families_crossing_splits"],
                           "benchmark": [{"model": b["model"], "official_accuracy": b["official"]["accuracy"],
                                          "clean_accuracy": b["clean"]["accuracy"], "official_f1": b["official"]["f1_macro"],
                                          "clean_f1": b["clean"]["f1_macro"]} for b in final8["leakage_benchmark"]]}}

    # ---- models
    legacy_val = {m["key"]: m["val"] for m in final8["models"]}
    legacy_val.update({m["key"]: m["val"] for m in five["models"]})
    rec_oof = {}
    for t in TASKS:
        for c in sel[t]["candidates"]:
            if len(c["members"]) == 1:
                r, v = c["members"][0]
                rec_oof[(t, r, v)] = c["oof"]
    models = []
    for r in bench["models"]:
        if r["task"] not in TASKS:
            continue
        m = meta[r["key"]]
        recipes = m.get("recipes")
        cv = None
        if r["key"].startswith("upgrade_"):
            cv = sel[r["task"]]["chosen"]["oof"]
        elif r["key"] in variant_oof:
            cv = variant_oof[r["key"]]
        elif r["key"] in final_oof:
            cv = final_oof[r["key"]]
        elif recipes and len(recipes) == 1:
            view = str(m.get("tta_views", 1))
            cv = rec_oof.get((r["task"], recipes[0], view))
        lat = bench["latency"].get(r["key"], {}).get("median_ms")
        temps = sorted({mm.get("temperature") for mm in m.get("members", []) if mm.get("temperature")}) or (
            [m["temperature"]] if m.get("temperature") else [])
        models.append({
            "key": r["key"], "name": r["name"], "model_name": m.get("model_name"), "task": r["task"], "family": r["family"],
            "generation": r["generation"],
            "params": m.get("params"), "tta_views": m.get("tta_views") or max([mm.get("tta_views", 1) for mm in m.get("members", [])] or [1]),
            "recipe": recipes[0] if recipes and len(recipes) == 1 else None, "recipes": recipes,
            "deployed": r["key"] in {e["key"] for e in deployment["models"]},
            "is_default": r["key"] in served.values(), "description": m.get("description") or m.get("feature_set"),
            "variant": m.get("variant"), "base_model": m.get("base_model"), "pipeline": m.get("pipeline"),
            "cv": {k: cv[k] for k in ("f1_macro", "accuracy")} if cv else None,
            "val": {k: legacy_val[r["key"]][k] for k in ("f1_macro", "accuracy")} if r["key"] in legacy_val else None,
            "test": r["test"], "per_class": r["per_class"], "confusion": r["confusion"],
            "selective": {"threshold": r["threshold"], "coverage": r["selective"]["coverage"],
                          "accuracy_kept": r["selective"]["accuracy_kept"],
                          "test": {"coverage": r["selective"]["coverage"], "accuracy_kept": r["selective"]["accuracy_kept"]}},
            "calibration": {"temperature": temps[0] if len(temps) == 1 else (float(np.mean(temps)) if temps else None),
                            "ece_after": r["test"]["ece"], "reliability_after": r["reliability"], "fitted_on":
                                "out-of-fold predictions" if r["generation"] == "upgrade" else "the validation split"},
            "latency_ms": lat, "flagged_unusual": r.get("flagged_unusual"), "out_of_scope": r.get("out_of_scope")})
    for t in TASKS:  # OOF reliability before/after for the upgrade defaults
        d = next(m for m in models if m["key"] == defaults[t])
        d["calibration"].update({"reliability_before": sel[t]["chosen"]["reliability_oof_raw"],
                                 "reliability_after": sel[t]["chosen"]["reliability_oof"],
                                 "ece_before": sel[t]["chosen"]["oof"]["ece_raw"], "ece_after": sel[t]["chosen"]["oof"]["ece"],
                                 "fitted_on": "out-of-fold predictions (diagram: out-of-fold, before vs after)"})

    for t, fp in final_pipe.items():  # OOF reliability before / after calibration for the final pipelines
        d = next((m for m in models if m["key"] == FINAL_KEYS[t]), None)
        if d is not None:
            arm = fp["arms"][fp["chosen"]["arm"]]
            d["calibration"].update({"reliability_before": fp["chosen"]["reliability_oof_raw"],
                                     "reliability_after": fp["chosen"]["reliability_oof"],
                                     "ece_before": arm["uncalibrated"]["ece"], "ece_after": arm["oof"]["ece"],
                                     "method": arm["calibration"]["chosen"] + " scaling",
                                     "fitted_on": "out-of-fold predictions, cross-fitted (diagram: out-of-fold, "
                                                  "before vs after)"})
    d = next((m for m in models if m["key"] == CLASSICAL), None)
    if d is not None:  # the classical model: same calibration rules (scripts/43)
        d["calibration"].update({"reliability_before": classical["chosen"]["reliability_oof_raw"],
                                 "reliability_after": classical["chosen"]["reliability_oof"],
                                 "ece_before": ml_arm["uncalibrated"]["ece"] if "uncalibrated" in ml_arm else None,
                                 "ece_after": ml_arm["oof"]["ece"], "method": ml_arm["calibration"]["chosen"] + " scaling",
                                 "fitted_on": "out-of-fold predictions, cross-fitted (diagram: out-of-fold, before vs "
                                              "after)"})
        d["feature_importance"] = classical["chosen"].get("feature_importance")

    # ---- cross-validation recipes
    recipes_out = []
    for t in TASKS:
        for rname, info in sel[t]["recipes"].items():
            summ = load_json(EXPERIMENTS_DIR / "cv" / t / rname / "summary.json")
            recipes_out.append({
                "recipe": rname, "label": RECIPES[rname].label, "short": SHORT_RECIPE.get(rname, rname), "task": t,
                "arch": RECIPES[rname].arch, "aug": RECIPES[rname].aug,
                "f1_1": summ["views"]["1"]["f1_macro"], "f1_8": summ["views"]["8"]["f1_macro"],
                "acc_1": summ["views"]["1"]["accuracy"], "acc_8": summ["views"]["8"]["accuracy"],
                "per_fold": [{"fold": f["fold"], "f1_1": f["held_1view"]["f1_macro"], "f1_8": f["held_tta8"]["f1_macro"],
                              "acc_1": f["held_1view"]["accuracy"], "acc_8": f["held_tta8"]["accuracy"]} for f in summ["per_fold"]],
                "train_minutes": summ["train_minutes_total"], "latency_1": info["latency_1"], "latency_8": info["latency_8"],
                "curves": curves_for(t, rname), "chosen": rname in [m["recipe"] for m in sel[t]["chosen"]["members"]]})
    recipes_out.sort(key=lambda r: (r["task"], -r["f1_8"]))
    screening = []  # fold-0 results of EVERY recipe tried (including ones stopped after fold 0)
    for t in TASKS:
        for rname, rec in RECIPES.items():
            p = EXPERIMENTS_DIR / "cv" / t / rname / "fold0.json"
            if p.exists():
                f0 = load_json(p)
                screening.append({"task": t, "recipe": rname, "label": rec.label,
                                  "f1_1": f0["held_1view"]["f1_macro"], "f1_8": f0["held_tta8"]["f1_macro"],
                                  "acc_1": f0["held_1view"]["accuracy"], "acc_8": f0["held_tta8"]["accuracy"],
                                  "full_cv": (EXPERIMENTS_DIR / "cv" / t / rname / "summary.json").exists()})

    # ---- audit: preprocessing (validation, original default) and robustness (test, old vs new defaults)
    audit = load_json(RESULTS_DIR / "baseline_audit.json")
    pre = []
    a = audit["models"]["ensemble_5class"]
    for k in ("clean", "upload_p3_original_path", "upload_p3_new_path", "upload_srgb_original_path", "upload_srgb_new_path"):
        if k in a:
            pre.append({"condition": k, "label": "Dataset file (reference)" if k == "clean" else PERTURBATION_LABEL[k],
                        "accuracy": a[k]["accuracy"], "f1_macro": a[k]["f1_macro"], "flip_rate": a[k].get("flip_rate", 0.0)})
    robustness = {}
    for t in TASKS:
        rb = bench["robustness"][t]
        names = list(PERTURBATION_LABEL)
        robustness[t] = {"rows": [{"label": PERTURBATION_LABEL[n], **{key: rb[key][n]["flip_rate"] for key in rb}} for n in names
                                  if all(n in rb[key] for key in rb)],
                         "series": [{"key": defaults[t], "label": "Version-2 networks", "color": "#15834e"},
                                    {"key": previous[t], "label": "Original default", "color": "#0891b2"}]}

    if "dryrun" in bench_file:  # pipeline check only: never score test photos (the gallery uses held-out samples)
        gallery, hero = {"items": [], "models": [], "summary": "dry run", "on_leaf_share": {}}, None
    else:
        gallery, hero = build_gallery(served, bench_models)

    selective = {}
    for t in TASKS:
        rc = bench.get("risk_coverage_variants", {}).get(served[t]) or bench["risk_coverage"][t]
        d = next(m for m in models if m["key"] == rc["model"])
        rule = (final_pipe[t]["arms"][final_pipe[t]["chosen"]["arm"]]["threshold_rule"]
                if served[t] == FINAL_KEYS[t] and t in final_pipe else sel[t]["selective"]["rule"])
        selective[t] = {"model": rc["model"], "curve": rc["curve"], "rule": rule,
                        "chosen": {"threshold": d["selective"]["threshold"], "coverage": d["selective"]["coverage"],
                                   "accuracy_kept": d["selective"]["accuracy_kept"]}}

    ood = {"method": f"{sel['5class']['ood']['method']} distance in the networks' feature space",
           "keep": sel["5class"]["ood"]["keep"],
           "oof": {t: {k: v for k, v in sel[t]["ood"]["evaluation_oof"][sel[t]["ood"]["method"]].items() if k != "probe_ratios"}
                   for t in TASKS},
           "test": {"flagged_in": bench["ood"]["5class"]["flagged_in"], "flagged_near": bench["ood"]["5class"].get("flagged_near"),
                    "flagged_far": bench["ood"]["5class"]["flagged_far_B"],
                    "by_task": {t: bench["ood"][t] for t in TASKS}}}

    if final_pipe:
        ood["final"] = {t: {"method": fp["unusual_gate"]["method"], "keep": fp["unusual_gate"]["keep"],
                            "oof": {k: v for k, v in
                                    fp["unusual_gate"]["evaluation_oof"][fp["unusual_gate"]["method"]].items()
                                    if k != "member_thresholds"}} for t, fp in final_pipe.items()}
        gate = classical["unusual_gate"]
        ood["final"][CLASSICAL] = {"method": gate["method"], "keep": gate["keep"],
                                   "oof": {k: v for k, v in gate["evaluation_oof"][gate["method"]].items()
                                           if k != "fold_thresholds"}}

    limitations = [
        "One phone model, two districts of Rajasthan, single leaves on paper: accuracy on field photos, other cameras or regions is unknown.",
        "Five conditions only: the dataset's Early Blight (62 photos), Late Blight and Potassium Deficiency (9 photos) were too hard to recognise reliably and are not diagnosed.",
        "Nitrogen Deficiency has 45 photographs (8 in the test set): its per-class scores are very uncertain.",
        f"The test set is small ({bench['test_photos_5class']} photos of the five conditions); differences of a few points are within noise — read the confidence intervals and paired tests.",
        "The same test photos were used to report the original study, so they are not perfectly fresh; no upgrade decision used them.",
        "Calibration and thresholds were fitted on out-of-fold predictions; coverage on new photos can still differ.",
        "The deep-learning model learned to refuse the dataset's other three conditions (outlier exposure); diseases it has never seen may still receive a diagnosis. The classical model had no such training.",
        "Out-of-dataset checks used PlantVillage leaves (another lab), everyday photos and synthetic stress tests; field photos with complex backgrounds were not available.",
        "Grad-CAM shows where the network looked, not why a leaf is diseased; the classical model has no Grad-CAM.",
    ]
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "timm": __import__("timm").__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    doc = {"version": 2, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "project": {"name": "Tomato Care"}, "dataset": dataset,
           "tasks": {t: {"label": TASK_LABEL[t], "classes": FIVE, "test_photos": bench["test_photos_5class"],
                         "default_model": served[t], "network_model": defaults[t], "previous_default": previous[t],
                         "classical_model": CLASSICAL,
                         "served_models": [e["key"] for e in deployment["models"] if e["task"] == t]}
                     for t in TASKS},
           "models": models, "comparisons": [c for c in bench["comparisons"] if c["task"] in TASKS],
           "cv": {"selection_rule": sel["5class"]["selection_rule"], "recipes": recipes_out, "screening": screening,
                  "chosen": {t: sel[t]["chosen"]["key"] for t in sel}},
           "audit": {"preprocessing": pre, "robustness": robustness, "dihedral": {k: v.get("dihedral_views") for k, v in audit["models"].items() if v.get("dihedral_views")}},
           "selective": selective, "ood": ood, "gradcam": gallery, "hero": hero, "limitations": limitations,
           "final_round": final_round_block(final_pipe, final_round, FINAL_KEYS, CLASSICAL,
                                            {m["key"]: m.get("model_name") for m in bench_models}),
           "classical": {"rules": "scripts/43_classical_model.py docstring (fixed before training)",
                         "arms": [{"arm": a["arm"], "classifier": a["classifier"], "feature_set": a["feature_set"],
                                   "oof": {k: a["oof"][k] for k in ("accuracy", "f1_macro", "balanced_accuracy", "nll",
                                                                    "ece")},
                                   "calibration": a["calibration"]["chosen"], "threshold": a["threshold"],
                                   "coverage": a["operating_point"]["coverage"],
                                   "accuracy_kept": a["operating_point"]["accuracy_kept"],
                                   "out_of_scope_refusal": a["out_of_scope_refusal"],
                                   "chosen": a["arm"] == classical["chosen"]["arm"]} for a in classical["arms"].values()],
                         "chosen": classical["chosen"]["arm"], "why": classical["chosen"]["why"],
                         "params": classical["chosen"]["final_params"],
                         "feature_importance": classical["chosen"].get("feature_importance")},
           "environment": env}
    save_json(doc, EXPERIMENTS_DIR / "platform.json")
    (FRONT / "data").mkdir(parents=True, exist_ok=True)
    save_json(doc, FRONT / "data" / "platform.json")
    shutil.copy(ROOT / "backend" / "data" / "disease_info.json", FRONT / "data" / "disease_info.json")
    size = (EXPERIMENTS_DIR / "platform.json").stat().st_size / 1024
    print(f"wrote platform.json ({size:.0f} KB): {len(models)} models, {len(recipes_out)} recipes")


if __name__ == "__main__":
    main()
