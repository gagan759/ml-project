"""Phase 3b — classical ML: Logistic Regression, SVM (RBF) and Random Forest x {HOG, Colour, HOG + Colour}.

- Hyperparameters: GridSearchCV, scoring = macro-F1, 5-fold StratifiedGroupKFold on TRAIN photos (leaf groups intact).
- Feature set per algorithm: chosen on the VALIDATION photos. The test set is not touched in this script.
- Leakage benchmark: the chosen configuration is also refit on the OFFICIAL train folder and scored on the
  OFFICIAL test folder, to measure how much the leaky official split inflates the score.
Outputs: experiments/models/ml/*.joblib, experiments/results/ml_validation.json, ml_official_benchmark.json, figures
"""
import time

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tomatoleaf.config import CLASS_KEYS, DATA_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, SEED
from tomatoleaf.data import load_index, load_splits
from tomatoleaf.evaluation import classification_metrics, confusion, per_class_metrics
from tomatoleaf.features import FEATURE_SETS
from tomatoleaf.plotting import INK, SEQ_CMAP, apply_style, plot_confusion, plt, save
from tomatoleaf.utils import save_json, set_seed

ML_MODELS = {"logreg": "Logistic Regression", "svm": "SVM (RBF kernel)", "rf": "Random Forest"}


def build(model_key: str, feature_set: str):
    pca_options = ["passthrough", PCA(n_components=0.95, random_state=SEED)] if "HOG" in feature_set else ["passthrough"]
    if model_key == "logreg":
        pipe = Pipeline([("scale", StandardScaler()), ("pca", "passthrough"),
                         ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))])
        grid = {"pca": pca_options, "clf__C": [0.01, 0.1, 1.0, 10.0]}
    elif model_key == "svm":
        pipe = Pipeline([("scale", StandardScaler()), ("pca", "passthrough"),
                         ("clf", SVC(kernel="rbf", class_weight="balanced"))])
        grid = {"pca": pca_options, "clf__C": [1.0, 10.0, 100.0], "clf__gamma": ["scale", 1e-4, 1e-3, 1e-2]}
    else:
        pipe = Pipeline([("clf", RandomForestClassifier(class_weight="balanced_subsample", random_state=SEED, n_jobs=1))])
        grid = {"clf__n_estimators": [300, 600], "clf__max_depth": [None, 20],
                "clf__max_features": ["sqrt", "log2"], "clf__min_samples_leaf": [1, 2]}
    return pipe, grid


def finalize(pipe, model_key):
    """Unfitted copy with the best hyperparameters; SVM gets Platt-scaled probabilities for the web app."""
    final = clone(pipe)
    if model_key == "svm":
        final.set_params(clf__probability=True, clf__random_state=SEED)
    if model_key == "rf":
        final.set_params(clf__n_jobs=-1)
    return final


def readable_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        key = k.replace("clf__", "")
        if key == "pca":
            out[key] = "none" if v == "passthrough" else "PCA (95% variance)"
        else:
            out[key] = v if isinstance(v, (int, float, str)) or v is None else str(v)
    return out


def stack(features, rows, feature_set):
    return np.hstack([features[part][rows] for part in FEATURE_SETS[feature_set]])


def main():
    set_seed(SEED)
    apply_style()
    feats = np.load(DATA_DIR / "features.npz", allow_pickle=True)
    features = {"hog": feats["hog"], "color": feats["color"]}
    splits = load_splits()
    tr, va = splits[splits.split == "train"], splits[splits.split == "val"]
    y_tr, y_va, g_tr = tr.label.to_numpy(), va.label.to_numpy(), tr.group_id.to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    out_dir = MODELS_DIR / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_key, model_name in ML_MODELS.items():
        for feature_set in FEATURE_SETS:
            X_tr = stack(features, tr.index_row.to_numpy(), feature_set)
            X_va = stack(features, va.index_row.to_numpy(), feature_set)
            pipe, grid = build(model_key, feature_set)
            search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-5, refit=False)  # leave 4 threads for GPU job
            t0 = time.perf_counter()
            search.fit(X_tr, y_tr, groups=g_tr)
            tuning_s = time.perf_counter() - t0

            final = finalize(pipe.set_params(**search.best_params_), model_key)
            t0 = time.perf_counter()
            final.fit(X_tr, y_tr)
            fit_s = time.perf_counter() - t0
            val_pred = final.predict_proba(X_va).argmax(axis=1)
            metrics = classification_metrics(y_va, val_pred)
            pca = final.named_steps.get("pca")
            row = {
                "model_key": model_key, "model": model_name, "feature_set": feature_set,
                "best_params": readable_params(search.best_params_),
                "cv_f1_macro": float(search.best_score_),
                "cv_f1_macro_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "val": metrics, "train_time_s": fit_s, "tuning_time_s": tuning_s,
                "n_features": int(X_tr.shape[1]),
                "pca_components": int(pca.n_components_) if hasattr(pca, "n_components_") else None,
                "candidates_tried": int(len(search.cv_results_["params"])),
                "val_confusion_matrix": confusion(y_va, val_pred),
            }
            rows.append(row)
            slug = feature_set.lower().replace(" + ", "_").replace(" ", "_")
            joblib.dump({"pipeline": final, "model_key": model_key, "name": model_name, "feature_set": feature_set,
                         "best_params": row["best_params"], "class_keys": CLASS_KEYS},
                        out_dir / f"{model_key}__{slug}.joblib")
            print(f"{model_name:22s} {feature_set:12s} | CV macro-F1 {row['cv_f1_macro']:.3f} | "
                  f"val acc {metrics['accuracy']:.3f} macro-F1 {metrics['f1_macro']:.3f} | "
                  f"tune {tuning_s:6.1f}s fit {fit_s:5.1f}s | {row['best_params']}")

    selected = {}
    for model_key in ML_MODELS:
        cands = [r for r in rows if r["model_key"] == model_key]
        best = max(cands, key=lambda r: (r["val"]["f1_macro"], r["cv_f1_macro"]))
        slug = best["feature_set"].lower().replace(" + ", "_").replace(" ", "_")
        selected[model_key] = {"feature_set": best["feature_set"], "file": f"ml/{model_key}__{slug}.joblib"}
    save_json({"grid": rows, "selected": selected}, RESULTS_DIR / "ml_validation.json")
    print("Selected feature set per algorithm (by validation macro-F1):", selected)

    # Leakage benchmark: same configuration, official (leaky) folders
    index = load_index()
    off_tr = np.flatnonzero(index.official_split.to_numpy() == "train")
    off_te = np.flatnonzero(index.official_split.to_numpy() == "test")
    bench = []
    for model_key, sel in selected.items():
        bundle = joblib.load(MODELS_DIR / sel["file"])
        model = clone(bundle["pipeline"])
        model.fit(stack(features, off_tr, sel["feature_set"]), index.label.to_numpy()[off_tr])
        pred = model.predict_proba(stack(features, off_te, sel["feature_set"])).argmax(axis=1)
        y_off = index.label.to_numpy()[off_te]
        bench.append({"model_key": model_key, "model": ML_MODELS[model_key], "feature_set": sel["feature_set"],
                      "protocol": "Official split (leaky)", "test": classification_metrics(y_off, pred),
                      "per_class": per_class_metrics(y_off, pred), "confusion_matrix": confusion(y_off, pred)})
        print(f"[official split] {ML_MODELS[model_key]:22s} acc {bench[-1]['test']['accuracy']:.3f} "
              f"macro-F1 {bench[-1]['test']['f1_macro']:.3f}")
    save_json(bench, RESULTS_DIR / "ml_official_benchmark.json")

    # Figures: validation grid heat map + validation confusion matrices of the selected configurations
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    mat = np.array([[next(r["val"]["f1_macro"] for r in rows if r["model_key"] == m and r["feature_set"] == f)
                     for f in FEATURE_SETS] for m in ML_MODELS])
    ax.imshow(mat, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="#ffffff" if mat[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(FEATURE_SETS)), list(FEATURE_SETS))
    ax.set_yticks(range(len(ML_MODELS)), list(ML_MODELS.values()))
    ax.set_title("Validation macro-F1 by algorithm and feature set")
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, FIGURES_DIR / "ml" / "validation_grid.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, model_key in zip(axes, ML_MODELS):
        r = next(r for r in rows if r["model_key"] == model_key and r["feature_set"] == selected[model_key]["feature_set"])
        plot_confusion(ax, r["val_confusion_matrix"], f"{ML_MODELS[model_key]} ({r['feature_set']}), validation")
    fig.tight_layout()
    save(fig, FIGURES_DIR / "ml" / "validation_confusion_matrices.png")


if __name__ == "__main__":
    main()
