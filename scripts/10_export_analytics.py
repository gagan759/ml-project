"""Phase 7a — compile every saved result into the files the web application reads.

experiments/analytics.json  : GET /api/analytics (all dashboard numbers come from here)
experiments/deployment.json : which trained models the API loads, the default (selected) model, temperatures,
                              thresholds and the default model's selective-prediction operating point
"""
from datetime import datetime, timezone

from tomatoleaf.config import CLASSES, EXPERIMENTS_DIR, RESULTS_DIR
from tomatoleaf.data import load_index, load_splits
from tomatoleaf.utils import load_json, save_json

DESCRIPTIONS = {
    "logreg": {
        "description": "Multinomial (softmax) logistic regression: a linear model that learns one weight vector per class "
                       "over standardised handcrafted features and turns the class scores into probabilities.",
        "strengths": ["Fast to train and to run", "Coefficients are directly interpretable", "Gives probabilities natively"],
        "limitations": ["Linear decision boundaries in feature space", "Only as good as the handcrafted features",
                        "Needs scaling and regularisation for high-dimensional HOG vectors"]},
    "svm": {
        "description": "Support Vector Machine with an RBF kernel: finds a maximum-margin boundary in an implicit "
                       "high-dimensional space (kernel trick); C trades margin width against training errors, gamma "
                       "sets how local each support vector's influence is.",
        "strengths": ["Strong on small, high-dimensional datasets", "Non-linear boundaries via the kernel trick",
                      "Only support vectors define the model"],
        "limitations": ["Sensitive to C and gamma", "Probabilities need extra Platt scaling",
                        "Training cost grows quickly with the number of samples"]},
    "rf": {
        "description": "Random Forest: an ensemble of decision trees, each trained on a bootstrap sample (bagging) and "
                       "a random subset of features at every split; the forest averages the trees' votes.",
        "strengths": ["No feature scaling needed", "Robust to noisy, irrelevant features",
                      "Built-in feature importances"],
        "limitations": ["Large model files", "Many weak HOG features dilute the useful colour features",
                        "Less effective than kernels on smooth high-dimensional data"]},
    "custom_cnn": {
        "description": "A convolutional network trained from scratch: 4 blocks of two 3x3 convolutions + BatchNorm + ReLU "
                       "and 2x2 max-pooling (32-64-128-256 filters), global average pooling, dropout 0.4 and a linear "
                       "softmax classifier.",
        "strengths": ["Learns its own features end to end", "Small (~1.2M parameters)", "No pretrained weights needed"],
        "limitations": ["Needs many labelled images; here only ~1,500 training photos",
                        "Slower to converge and more prone to overfitting than transfer learning"]},
    "mobilenetv2_fe": {
        "description": "MobileNetV2 pretrained on ImageNet used as a frozen feature extractor: only the new 8-class head "
                       "(1280 -> 8) is trained.",
        "strengths": ["Reuses general visual features learned from 1.2M ImageNet images", "Very fast to train",
                      "Hardly any parameters to overfit"],
        "limitations": ["ImageNet features are not specialised for leaf lesions", "Backbone cannot adapt to this domain"]},
    "mobilenetv2_ft": {
        "description": "MobileNetV2 (inverted residual blocks with depthwise-separable convolutions) pretrained on "
                       "ImageNet, then fine-tuned end to end with a 10x smaller learning rate after training the head.",
        "strengths": ["Adapts pretrained features to leaf symptoms", "Lightweight enough for CPU inference",
                      "Best use of a small dataset"],
        "limitations": ["More compute-intensive to train than the classical models", "Less interpretable than handcrafted features",
                        "Trained on leaves photographed on white paper only"]},
    "efficientnet_b0_v2": {
        "description": "EfficientNet-B0 pretrained on ImageNet (compound-scaled MBConv blocks with squeeze-and-excitation), "
                       "fine-tuned at the native 256x256 resolution with a cosine learning-rate schedule and label smoothing.",
        "strengths": ["Strong accuracy per parameter (5.3M)", "Uses the full 256x256 resolution",
                      "Label smoothing reduces over-confidence"],
        "limitations": ["Overfits the small training set quickly", "Slower than MobileNetV2 on a CPU"]},
    "convnext_tiny_v2": {
        "description": "ConvNeXt-Tiny pretrained on ImageNet: a modernised ResNet with 7x7 depthwise convolutions, "
                       "LayerNorm and GELU (28M parameters), fine-tuned at 256x256 with a cosine schedule and label smoothing.",
        "strengths": ["Strongest pretrained features of the models tried", "Large receptive field for lesion patterns"],
        "limitations": ["Largest model here (about 28M parameters)", "Slowest to train and to run on a CPU"]},
}


def describe(key: str, row: dict) -> dict:
    if key in DESCRIPTIONS:
        return DESCRIPTIONS[key]
    members = row.get("best_params", {}).get("members", [])
    return {"description": f"{row['name']}: the calibrated probabilities of {', '.join(members)} are averaged"
                           + (" over the 8 flips/rotations of the photo (test-time augmentation)" if any(
                               v > 1 for v in row.get("best_params", {}).get("tta_views", [])) else "") + ".",
            "strengths": ["Chosen on validation macro-F1 among all candidates", "Averaging reduces the variance of single models",
                          "A detached leaf has no 'up', so flips/rotations are valid views"],
            "limitations": ["Several forward passes per photo (slower inference)", "Less transparent than a single model"]}


def main():
    index = load_index()
    splits = load_splits()
    leakage = load_json(RESULTS_DIR / "leakage_official_split.json")
    split_summary = load_json(RESULTS_DIR / "split_summary.json")
    near = load_json(RESULTS_DIR / "near_duplicates.json")
    final = load_json(RESULTS_DIR / "final_results.json")
    ml_val = load_json(RESULTS_DIR / "ml_validation.json")

    class_distribution = []
    for c in CLASSES:
        sub = index[index.label == c["index"]]
        clean = splits[splits.label == c["index"]]
        class_distribution.append({
            "key": c["key"], "name": c["name"], "images": int(len(sub)), "photos": int(sub.is_original.sum()),
            "official": {s: int((sub.official_split == s).sum()) for s in ("train", "val", "test")},
            "clean": {s: int((clean.split == s).sum()) for s in ("train", "val", "test")},
        })

    dataset = {
        "name": "Tomato-Village (Variant-a, multiclass)",
        "citation": "Gehlot, M., Saxena, R.K. & Gandhi, G.C. (2023). \"Tomato-Village\": a dataset for end-to-end tomato "
                    "disease detection in a real-world environment. Multimedia Systems.",
        "doi": "10.1007/s00530-023-01158-y",
        "image_size": "256x256",
        "total_images": int(len(index)),
        "unique_photos": int(index.is_original.sum()),
        "augmented_copies": int((~index.is_original).sum()),
        "num_classes": len(CLASSES),
        "official_split": {s: int((index.official_split == s).sum()) for s in ("train", "val", "test")},
        "clean_split": split_summary["split_totals"],
        "class_distribution": class_distribution,
        "leakage": {
            "augmented_families": leakage["augmented_families"],
            "families_crossing_splits": leakage["families_crossing_splits"],
            "val_images": leakage["val_images"], "val_images_with_train_sibling": leakage["val_images_with_train_sibling"],
            "test_images": leakage["test_images"], "test_images_with_train_sibling": leakage["test_images_with_train_sibling"],
            "near_duplicate_pairs_merged": near["merged_pairs"], "leaf_groups": split_summary["leaf_groups"],
        },
    }

    models = []
    for r in final["models"]:
        models.append({k: r.get(k) for k in ("key", "name", "family", "kind", "feature_set", "best_params", "train_time_s",
                                             "tuning_time_s", "inference_ms_per_image", "params", "val", "test",
                                             "per_class", "confusion_matrix", "selected")} | describe(r["key"], r))

    sel_key = final["selection"]["model_key"]
    sel = next(r for r in final["models"] if r["key"] == sel_key)
    runner_up = final["selection"]["validation_ranking"][1]
    reasons = [
        f"Highest validation macro-F1 of all {len(final['selection']['validation_ranking'])} candidates, including "
        f"test-time augmentation and ensembles ({sel['val']['f1_macro']:.3f}; runner-up {runner_up['name']} "
        f"{runner_up['val_f1_macro']:.3f}). The test set was not used to choose.",
        f"Held-out test: accuracy {sel['test']['accuracy']:.3f}, macro-F1 {sel['test']['f1_macro']:.3f} "
        f"(95% CI {sel['test']['f1_macro_ci'][0]:.3f}-{sel['test']['f1_macro_ci'][1]:.3f}).",
        f"Practical: {sel['inference_ms_per_image']:.1f} ms per image on CPU"
        + (f", {sel['params'] / 1e6:.2f}M parameters." if sel.get("params") else "."),
    ]
    selection = {"model_key": sel_key, "criterion": final["selection"]["criterion"], "reasons": reasons}

    curves = {}
    for arch, fname in (("custom_cnn", "dl_clean_custom_cnn.json"), ("mobilenetv2", "dl_clean_mobilenetv2.json"),
                        ("efficientnet_b0", "dl_clean_efficientnet_b0_v2.json"), ("convnext_tiny", "dl_clean_convnext_tiny_v2.json")):
        if not (RESULTS_DIR / fname).exists():
            continue
        run = load_json(RESULTS_DIR / fname)
        curves[arch] = [{k: h[k] for k in ("epoch", "stage", "train_loss", "val_loss", "train_acc", "val_acc", "val_f1_macro", "lr")}
                        for h in run["history"]]

    grid = [{"model": r["model"], "model_key": r["model_key"], "feature_set": r["feature_set"],
             "cv_f1_macro": r["cv_f1_macro"], "val_accuracy": r["val"]["accuracy"], "val_f1_macro": r["val"]["f1_macro"]}
            for r in ml_val["grid"]]

    bench = []
    for b in final["leakage_benchmark"]:
        bench.append({"model": b["model"], "protocol": "Official split (leaky)",
                      "accuracy": b["official"]["accuracy"], "f1_macro": b["official"]["f1_macro"]})
        bench.append({"model": b["model"], "protocol": "Group-aware split (ours)",
                      "accuracy": b["clean"]["accuracy"], "f1_macro": b["clean"]["f1_macro"]})

    cal = final["calibration"]
    five = load_json(RESULTS_DIR / "five_class_results.json") if (RESULTS_DIR / "five_class_results.json").exists() else None
    five_block, deployed_key = None, sel_key
    if five:
        keep_fields = ("key", "name", "family", "type", "feature_set", "best_params", "train_time_s", "inference_ms_per_image",
                       "params", "val", "test", "per_class", "confusion_matrix", "selected")
        five_curves = {}
        for arch in ("convnext_tiny", "efficientnet_b0"):
            f = RESULTS_DIR / f"dl_clean_{arch}_5class.json"
            if f.exists():
                five_curves[arch] = [{k: h[k] for k in ("epoch", "stage", "train_loss", "val_loss", "train_acc", "val_acc",
                                                        "val_f1_macro", "lr")} for h in load_json(f)["history"]]
        five_block = {k: five[k] for k in ("classes", "class_names", "rule", "dropped", "split", "selected",
                                           "validation_ranking", "ml_vs_dl", "calibration",
                                           "eight_class_model_on_same_photos")}
        five_block["selective"] = five.get("selective")
        five_block["models"] = [{k: r.get(k) for k in keep_fields} for r in five["models"]]
        five_block["curves"] = five_curves
        deployed_key = five["selected"]
    # written by scripts/14_verify_deployment.py (all val/test photos sent through the running API); optional
    check_file = RESULTS_DIR / "deployment_check.json"
    deployment_check = load_json(check_file) if check_file.exists() else None
    if deployment_check and deployment_check.get("model_key") != deployed_key:
        deployment_check = None  # stale: it checked a different default model
    analytics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset, "models": models, "ml_validation_grid": grid, "training_curves": curves,
        "selection": selection, "leakage_benchmark": bench, "five_class": five_block,
        "deployment_scope": {"default_model": deployed_key,
                             "classes": five["classes"] if five else [c["key"] for c in CLASSES],
                             "note": ("The web app's default model is the focused 5-class model (Healthy plus the four "
                                      "conditions recognised most reliably on validation data). The 8-class models stay "
                                      "available in the model selector.") if five else "All 8 classes."},
        "deployment_check": deployment_check,
        "calibration": {k: cal[k] for k in ("model_key", "temperature", "ece_before", "ece_after",
                                            "confidence_threshold", "threshold_rule")},
    }
    save_json(analytics, EXPERIMENTS_DIR / "analytics.json")

    deployment = {"default_model": deployed_key, "confidence_threshold": cal["confidence_threshold"], "models": []}
    if five:  # focused 5-class model first: it is the default
        r = next(m for m in five["models"] if m["selected"])
        entry = {"key": r["key"], "name": r["name"], "family": r["family"], "kind": "transfer" if r["family"] == "Deep Learning" else "classical",
                 "feature_set": r["feature_set"], "params": r.get("params"), "class_keys": five["classes"],
                 "confidence_threshold": (five.get("selective") or {}).get("threshold"),
                 "test_metrics": {k: r["test"][k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")}}
        if five.get("selective"):
            # deployed as a selective classifier: a diagnosis only at/above the validation-chosen threshold.
            # Coverage and accuracy-when-answering were measured once on the held-out test photos (13_five_class.py).
            entry["operating_point"] = {**five["selective"], "test_photos": five["split"]["test"]}
        if r["family"] == "Machine Learning":
            entry |= {"type": "sklearn", "path": r["path"]}
        elif len(r["members"]) > 1:
            entry |= {"type": "ensemble", "members": r["members"]}
        else:
            m = r["members"][0]
            entry |= {"type": "torch", "path": m["path"], "temperature": m["temperature"], "tta_views": m["tta_views"]}
        deployment["models"].append(entry)
    for r in final["models"]:
        entry = {"key": r["key"], "name": r["name"] + (" (8-class)" if five else ""), "family": r["family"], "kind": r["kind"],
                 "feature_set": r["feature_set"], "params": r.get("params"), "class_keys": [c["key"] for c in CLASSES],
                 "test_metrics": {k: r["test"][k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted")}}
        if r["family"] == "Machine Learning":
            entry |= {"type": "sklearn", "path": ml_val["selected"][r["key"]]["file"]}
        elif r.get("members"):
            entry |= {"type": "ensemble", "members": [{"key": m["key"], "path": f"dl/clean/{m['key']}.pt",
                                                       "temperature": m["temperature"], "tta_views": m["tta_views"]}
                                                      for m in r["members"]]}
        else:
            entry |= {"type": "torch", "path": f"dl/clean/{r['key'].removesuffix('_tta')}.pt", "temperature": r["temperature"],
                      "tta_views": 8 if r["key"].endswith("_tta") else 1}
        deployment["models"].append(entry)
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    print(f"Wrote analytics.json and deployment.json (default model: {deployed_key}; 8-class selection: {sel_key})")


if __name__ == "__main__":
    main()
