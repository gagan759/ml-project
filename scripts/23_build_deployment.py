"""Upgrade round, step 4 — model lists for the final benchmark and for the API.

experiments/benchmark_models.json  every model scored on the test photos:
    original study: 3 classical + 6 deep 8-class models, 3 classical + 5 deep 5-class models (incl. the old defaults)
    upgrade: the 5-fold ensemble of EVERY cross-validated recipe (best view setting), and the chosen default per task
experiments/deployment.json         what the API serves (a subset: new defaults + reference models for live comparison)
Both use the registry format read by backend/services/predictor.py. Test metrics and CPU latency are filled in later
by scripts/25_final_benchmark.py.
"""
import json

from tomatoleaf.config import CLASS_KEYS, EXPERIMENTS_DIR, RESULTS_DIR
from tomatoleaf.recipes import RECIPES
from tomatoleaf.utils import load_json, save_json

SHORT = {
    "convnext_t_in1k_base": "ConvNeXt-T IN-1k · original recipe",
    "convnext_t_in1k_robust": "ConvNeXt-T IN-1k · robust",
    "convnext_t_in22k_base": "ConvNeXt-T IN-22k · original recipe",
    "convnext_t_in22k_robust": "ConvNeXt-T IN-22k · robust",
    "convnextv2_t_robust": "ConvNeXt-V2-T · robust",
    "dinov2_s_robust": "ViT-S/14 DINOv2 · robust",
    "effnetv2_s_robust": "EfficientNetV2-S IN-21k · robust",
}
FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
FAMILY = {"Machine Learning": "Classical ML", "Deep Learning": "Deep learning"}


def legacy_entries():
    """The original study's models, in registry format, tagged generation='original'."""
    out = []
    old = load_json(EXPERIMENTS_DIR / "deployment_v1.json") if (EXPERIMENTS_DIR / "deployment_v1.json").exists() \
        else load_json(EXPERIMENTS_DIR / "deployment.json")
    for e in old["models"]:
        if e["key"] == "ensemble_5class":
            continue  # re-added below with the other 5-class models
        e = {**e, "task": "8class", "generation": "original", "family": FAMILY.get(e["family"], e["family"]),
             "class_keys": list(CLASS_KEYS)}
        e["name"] = e["name"].replace(" (8-class)", "")
        if e["key"] == "convnext_tiny_v2_tta":
            e["name"] = "ConvNeXt-Tiny + TTA (original 8-class default)"
        e.pop("test_metrics", None)
        out.append(e)
    five = load_json(RESULTS_DIR / "five_class_results.json")
    for m in five["models"]:
        e = {"key": m["key"] if m["key"] != "ensemble_5class" else "ensemble_5class", "task": "5class",
             "generation": "original", "family": FAMILY[m["family"]],
             "name": m["name"].replace(" (5-class)", "").replace("Ensemble: ", "Ensemble "),
             "kind": "classical" if m["family"] == "Machine Learning" else ("ensemble" if m["type"] == "ensemble" else "transfer"),
             "feature_set": m["feature_set"], "params": m.get("params"), "class_keys": FIVE}
        if m["family"] == "Machine Learning":
            e.update({"type": "sklearn", "path": m["path"]})
        else:
            e.update({"type": "ensemble", "members": [{k: v for k, v in mm.items()} for mm in m["members"]]})
        if m["key"] == "ensemble_5class":
            e["name"] = "Ensemble ConvNeXt-T + EfficientNet-B0 (original 5-class default)"
            e["confidence_threshold"] = five["selective"]["threshold"]
            e["operating_point"] = {"threshold": five["selective"]["threshold"], "chosen_on": "the original validation split",
                                    "rule": five["selective"]["rule"]}
        out.append(e)
    return out


def short(r: str) -> str:
    return SHORT.get(r, RECIPES[r].label)


def fold_ensemble(task, members, key, name, *, deploy_paths=False, ood=None, threshold=None, op=None, description=None):
    """Registry entry for the 5 fold models of each member recipe (averaged calibrated probabilities)."""
    specs = []
    ood_thr = {(t["recipe"], t["fold"]): t["threshold"] for t in (ood or {}).get("thresholds", [])}
    for m in members:
        r = m["recipe"]
        for k in range(5):
            path = f"deploy/{task}/{r}/fold{k}.pt" if deploy_paths else f"../cv/{task}/{r}/fold{k}.pt"
            spec = {"key": f"{r}_fold{k}", "name": f"{short(r)} · fold {k}", "path": path,
                    "temperature": m["temperature"], "tta_views": m["views"]}
            if ood and (r, k) in ood_thr:
                spec.update({"ood_method": ood["method"], "ood_threshold": ood_thr[(r, k)] * ood["scale"]})
            specs.append(spec)
    keys = FIVE if task == "5class" else list(CLASS_KEYS)
    entry = {"key": key, "name": name, "task": task, "generation": "upgrade", "family": "Deep learning",
             "kind": "ensemble", "type": "ensemble", "class_keys": keys, "members": specs,
             "recipes": [m["recipe"] for m in members], "tta_views": max(m["views"] for m in members),
             "feature_set": "5-fold ensemble of " + " + ".join(
                 f"{short(m['recipe'])}{' (8-view TTA)' if m['views'] > 1 else ''}" for m in members),
             "description": description}
    if threshold is not None:
        entry["confidence_threshold"] = threshold
    if op is not None:
        entry["operating_point"] = op
    if ood:
        entry["ood"] = {"method": f"{ood['method']} feature distance (fold models' own training photos)",
                        "keep": ood["keep"]}
    return entry


def main():
    if not (EXPERIMENTS_DIR / "deployment_v1.json").exists():  # keep the original study's deployment for reference
        (EXPERIMENTS_DIR / "deployment_v1.json").write_text((EXPERIMENTS_DIR / "deployment.json").read_text())
    legacy = legacy_entries()
    upgrade, defaults = [], {}
    for task in ("5class", "8class"):
        sel = load_json(RESULTS_DIR / f"selection_{task}.json")
        for r, info in sel["recipes"].items():  # every CV recipe as a fold ensemble (benchmark only)
            upgrade.append(fold_ensemble(task, [{"recipe": r, "views": int(info["best_view"]),
                                                 "temperature": info[f"temperature_{info['best_view']}"]}],
                                         f"cv{task[0]}_{r}", f"{short(r)} (5-fold)"))
        ch = sel["chosen"]
        op = {"threshold": sel["selective"]["threshold"], "chosen_on": "out-of-fold cross-validation predictions",
              "rule": sel["selective"]["rule"], "oof": sel["selective"]["oof"]}
        names = " + ".join(short(m["recipe"]) for m in ch["members"])
        key = f"upgrade_{task}"
        upgrade.append(fold_ensemble(
            task, ch["members"], key, f"{names} fold ensemble (upgrade default)", deploy_paths=True, ood=sel["ood"],
            threshold=sel["selective"]["threshold"], op=op,
            description=f"Chosen by cross-validation: {sel['selection_rule']}"))
        defaults[task] = key

    bench = {"version": 2, "default_model": defaults["5class"], "defaults_by_task": defaults,
             "previous_defaults": {"5class": "ensemble_5class", "8class": "convnext_tiny_v2_tta"},
             "confidence_threshold": 0.6, "models": upgrade + legacy}
    save_json(bench, EXPERIMENTS_DIR / "benchmark_models.json")

    # The apps and the API serve only the version-2 models; the original study's models stay in the benchmark (as the
    # baseline) and in deployment_v1.json (their weights remain in the release to reproduce the comparison).
    serve = set(defaults.values())
    deployment = {**{k: v for k, v in bench.items() if k != "models"},
                  "models": [m for m in bench["models"] if m["key"] in serve]}
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    print(f"benchmark: {len(bench['models'])} models; API: {[m['key'] for m in deployment['models']]}")
    print(json.dumps(defaults))


if __name__ == "__main__":
    main()
