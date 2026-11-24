"""Classical ML model for the 5 conditions (no deep learning), chosen on out-of-fold data only, with the same rules
as the deep pipeline (scripts/36). No test photo is loaded here; scripts/44 scores the test set once.

C1 Features (tomatoleaf.features) of the canonical 256x256 image the deep model also sees: colour (HSV histograms +
   L*a*b* moments over leaf pixels, 73 values), texture (uniform LBP histograms + GLCM properties over leaf pixels,
   43) and HOG (1,764, reduced to 64 principal components fitted inside each training fold). Feature sets: Colour,
   Colour + Texture, HOG + Colour + Texture; every block standardised inside the training fold.
C2 Classifiers, all class-balanced: logistic regression, SVM (RBF kernel), random forest, histogram gradient boosting.
   Small fixed grids, tuned by grouped 4-fold cross-validation inside each outer training fold (nested), macro-F1.
C3 Outer cross-validation: the deep model's grouped 5 folds (data/cv_folds.csv) over the in-scope train+val photos.
   The held-out fold's photos of the three unsupported conditions are predicted too: answered ones count as errors.
C4 Selection, calibration and threshold: scripts/36 P1-P3 — highest OOF macro-F1, within 0.01 the lower OOF NLL;
   temperature vs per-class vector scaling (cross-fitted, lower NLL, ties -> temperature); the lowest threshold with
   >= 90% OOF accuracy on answered photos and >= 50% of in-scope photos answered.
C5 Unusual-image gate: kNN vs Mahalanobis distance to the training photos in the standardised Colour + Texture space,
   chosen by OOF AUROC against synthetic non-leaf probes (half A), empty paper and OOF out-of-scope photos; limits:
   95% of OOF genuine photos pass, per answered condition (scripts/36 P4).
C6 Final model: the chosen arm re-tuned over the same 5 folds and refitted on all in-scope train+val photos.

Writes data/features_classical.npz and experiments/cv/classical/*.npz (caches), experiments/models/ml/classical_5class.joblib,
experiments/results/classical_pipeline.json and the classical_5class entry of experiments/final_models.json.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread per grid-search job (24 jobs in parallel)
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, PredefinedSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tomatoleaf.config import (  # noqa: E402
    CLASS_KEYS,
    DATA_DIR,
    DATASET_DIR,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
)
from tomatoleaf.data import load_images  # noqa: E402
from tomatoleaf.evaluation import confusion, per_class_metrics, reliability_bins  # noqa: E402
from tomatoleaf.features import (  # noqa: E402
    COLOR_DIM,
    FEATURE_SETS,
    HOG_DIM,
    TEXTURE_DIM,
    extract_features,
)
from tomatoleaf.ood import FeatureDensity, auroc, fpr_at_tpr  # noqa: E402
from tomatoleaf.preprocessing import load_canonical_array  # noqa: E402
from tomatoleaf.probes import far_ood_probes  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402

spec = importlib.util.spec_from_file_location("fp", ROOT / "scripts" / "36_final_pipeline.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)

FIVE = fp.FIVE
KEY, NAME = "classical_5class", "Tomato Care · Classical ML (5 conditions)"
MODEL_PATH = "ml/classical_5class.joblib"
CACHE = DATA_DIR / "features_classical.npz"
OOF_DIR = EXPERIMENTS_DIR / "cv" / "classical"
SETS = ["Color", "Color + Texture", "HOG + Color + Texture"]
DISPLAY = {"Logistic regression": "Logistic Regression", "SVM (RBF)": "SVM (RBF kernel)", "Random forest": "Random Forest",
           "Gradient boosting": "Gradient Boosting"}  # model names shown in the apps
GATE_SET = "Color + Texture"
SCORE = {"knn": "knn", "maha": "mahalanobis"}  # gate method -> FeatureDensity score
HOG_COMPONENTS = 64
DIMS = {"hog": HOG_DIM, "color": COLOR_DIM, "texture": TEXTURE_DIM}
CLASSIFIERS = {
    "Logistic regression": (LogisticRegression(max_iter=5000, class_weight="balanced"),
                            {"C": [0.01, 0.1, 1.0, 10.0]}),
    "SVM (RBF)": (SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
                  {"C": [1.0, 10.0, 100.0]}),
    "Random forest": (RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=1,
                                             random_state=SEED),
                      {"max_features": ["sqrt", 0.2], "min_samples_leaf": [1, 3]}),
    "Gradient boosting": (HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=SEED),
                          {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
}


# ------------------------------------------------------------------------------------------------------ features
def _features(rgb):
    f = extract_features(rgb)
    return np.concatenate([f["hog"], f["color"], f["texture"]])


def _features_of(rel_path):
    return _features(load_canonical_array(DATASET_DIR / rel_path))


def all_features(images=None, paths=None):
    """HOG | colour | texture, one row per photo (dataset paths or image arrays)."""
    if paths is not None:
        rows = Parallel(n_jobs=-1, batch_size=16)(delayed(_features_of)(p) for p in paths)
    else:
        rows = Parallel(n_jobs=-1, batch_size=8)(delayed(_features)(x) for x in images)
    return np.stack(rows).astype(np.float32)


def columns_of(part):
    start = {"hog": 0, "color": HOG_DIM, "texture": HOG_DIM + COLOR_DIM}[part]
    return np.arange(start, start + DIMS[part])


def columns(feature_set):
    return np.concatenate([columns_of(p) for p in FEATURE_SETS[feature_set]])


def cached_features(pool):
    if CACHE.exists():
        c = np.load(CACHE, allow_pickle=True)
        if list(c["paths"]) == list(pool.path):
            return c["features"]
    feats = all_features(paths=list(pool.path))
    np.savez_compressed(CACHE, features=feats, paths=pool.path.to_numpy())
    return feats


# ------------------------------------------------------------------------------------------------------- models
def make_model(classifier, feature_set):
    """Pipeline on the full HOG | colour | texture row: keeps the set's blocks (the rest is dropped) and standardises
    each one inside the training data (HOG: then 64 principal components)."""
    blocks = []
    for part in FEATURE_SETS[feature_set]:
        step = (Pipeline([("scale", StandardScaler()), ("pca", PCA(HOG_COMPONENTS, random_state=SEED))])
                if part == "hog" else StandardScaler())
        blocks.append((part, step, list(columns_of(part))))
    est, grid = CLASSIFIERS[classifier]
    pipe = Pipeline([("features", ColumnTransformer(blocks)), ("clf", clone(est))])
    return pipe, {f"clf__{k}": v for k, v in grid.items()}


def oof_arm(classifier, feature_set, X, y, groups, folds, Xo, ofold):
    p, po, params = np.zeros((len(y), len(FIVE))), np.zeros((len(Xo), len(FIVE))), []
    for k in range(5):
        tr = folds != k
        pipe, grid = make_model(classifier, feature_set)
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=GroupKFold(n_splits=4), n_jobs=-1)
        gs.fit(X[tr], y[tr], groups=groups[tr])
        p[folds == k] = gs.predict_proba(X[folds == k])
        po[ofold == k] = gs.predict_proba(Xo[ofold == k])
        params.append({k2.replace("clf__", ""): v for k2, v in gs.best_params_.items()})
    return p, po, params


def evaluate_arm(name, classifier, feature_set, p, po, y, folds, ofold, params):
    """scripts/36 evaluate_arm on classical probabilities (their logs act as logits)."""
    z, oz = np.log(np.clip(p, 1e-6, 1.0)), np.log(np.clip(po, 1e-6, 1.0))
    cals = fp.calibrate(z, y, folds, oz, ofold)
    nll_t, nll_v = cals["temperature"]["nll"], cals["vector"]["nll"]
    kind = "vector" if nll_v < nll_t - fp.NLL_TIE else "temperature"
    c = cals[kind]
    op, rule, curve = fp.operating_point(c["p"], y, c["po"], fp.TARGET["5class"])
    return {"arm": name, "classifier": classifier, "feature_set": feature_set, "label": f"{classifier} · {feature_set}",
            "photos": int(len(y)), "best_params_per_fold": params,
            "macro_f1_before_calibration": fp.macro_f1(y, z.argmax(1), len(FIVE)),
            "uncalibrated": {"ece": fp.expected_calibration_error(p, y), "brier": fp.brier_score(p, y),
                             "nll": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())},
            "calibration": {"chosen": kind, "nll_temperature": nll_t, "nll_vector": nll_v,
                            "fitted_on_all_folds": c["full"]},
            "oof": fp.full_metrics(c["p"], y, len(FIVE)), "threshold": op["threshold"], "threshold_rule": rule,
            "operating_point": op, "out_of_scope_refusal": 1.0 - op["out_of_scope_answered"], "curve": curve,
            "_p": c["p"], "_po": c["po"], "_p_raw": p}


# ---------------------------------------------------------------------------------------- unusual-image gate
def gate(G, y, folds, Go, ofold, G_probes, G_paper, paper_fold, chosen):
    """OOF distances in the standardised gate space, then scripts/36's method choice and limits."""
    keys = {"knn": [], "maha": []}
    d = {m: np.zeros(len(y)) for m in keys} | {f"o{m}": np.zeros(len(Go)) for m in keys}
    probe = {m: [] for m in keys}
    paper = {m: [] for m in keys}
    for k in range(5):
        tr = folds != k
        scaler = StandardScaler().fit(G[tr])
        dens = FeatureDensity(scaler.transform(G[tr]), y[tr])
        for m in keys:
            score = getattr(dens, SCORE[m])
            d[m][folds == k] = score(scaler.transform(G[folds == k]))
            d[f"o{m}"][ofold == k] = score(scaler.transform(Go[ofold == k]))
            probe[m].append(score(scaler.transform(G_probes)))
            paper[m].append(score(scaler.transform(G_paper[paper_fold == k])))
    _, _, halves = far_ood_probes()
    decisions = chosen["_p"].argmax(1)
    evals = {}
    for m in keys:
        thr = np.array([np.quantile(d[m][folds == k], fp.OOD_KEEP) for k in range(5)])
        in_r = d[m] / thr[folds]
        probe_r = (np.stack(probe[m]) / thr[:, None]).mean(0)
        paper_r = np.concatenate([paper[m][k] / thr[k] for k in range(5)])
        oos_r = d[f"o{m}"] / thr[ofold]
        a = probe_r[halves == "A"]
        out_sel = np.concatenate([a, paper_r, oos_r])
        scale = float(np.quantile(in_r, fp.OOD_KEEP))
        cs = fp.class_scales(in_r / scale, decisions, len(FIVE))
        lim_in = in_r / (scale * cs[decisions])
        od = chosen["_po"].argmax(1)
        lim_oos = oos_r / (scale * cs[od])
        refused = (chosen["_po"].max(1) < chosen["threshold"]) | (lim_oos > 1)
        evals[m] = {"auroc_select": auroc(in_r, out_sel), "fpr95_select": fpr_at_tpr(in_r, out_sel),
                    "auroc_probes_A": auroc(in_r, a), "auroc_paper": auroc(in_r, paper_r),
                    "auroc_out_of_scope": auroc(in_r, oos_r), "scale": scale, "class_scales": cs.tolist(),
                    "fold_thresholds": thr.tolist(), "flagged_in_scope": float((lim_in > 1).mean()),
                    "flagged_in_scope_by_condition": {FIVE[c]: float((lim_in[y == c] > 1).mean())
                                                      for c in range(len(FIVE))},
                    "flagged_probes_A": float((a / scale > 1).mean()),
                    "flagged_probes_B": float((probe_r[halves == "B"] / scale > 1).mean()),
                    "flagged_paper": float((paper_r / scale > 1).mean()),
                    "flagged_out_of_scope": float((lim_oos > 1).mean()),
                    "out_of_scope_refused_with_gate": float(refused.mean())}
    method = max(evals, key=lambda m: evals[m]["auroc_select"])
    return method, evals


def main():
    t0 = time.time()
    pool = pd.read_csv(DATA_DIR / "cv_folds.csv")
    F = cached_features(pool)
    print(f"features {F.shape} ({time.time() - t0:.0f} s)", flush=True)
    inside = pool.class_key.isin(FIVE).to_numpy()
    local = {c: i for i, c in enumerate(FIVE)}
    X, Xo = F[inside], F[~inside]
    y = pool.class_key[inside].map(local).to_numpy()
    groups, folds = pool.group_id[inside].to_numpy(), pool.fold[inside].to_numpy()
    ofold = pool.fold[~inside].to_numpy()

    arms = {}
    for feature_set in SETS:
        for classifier in CLASSIFIERS:
            name = f"{classifier} | {feature_set}"
            cache = OOF_DIR / f"{classifier}__{feature_set}.npz".replace(" ", "_").replace("|", "")
            if cache.exists():  # out-of-fold predictions of an arm already run (same folds, same grid)
                c = np.load(cache, allow_pickle=True)
                p, po, params = c["p"], c["po"], list(c["params"])
            else:
                p, po, params = oof_arm(classifier, feature_set, X, y, groups, folds, Xo, ofold)
                OOF_DIR.mkdir(parents=True, exist_ok=True)
                np.savez(cache, p=p, po=po, params=np.array(params, dtype=object))
            a = arms[name] = evaluate_arm(name, classifier, feature_set, p, po, y, folds, ofold, params)
            print(f"{name:45s} OOF macro-F1 {a['oof']['f1_macro']:.4f} acc {a['oof']['accuracy']:.4f} NLL "
                  f"{a['oof']['nll']:.4f} | cal {a['calibration']['chosen']} | thr {a['threshold']:.2f} cov "
                  f"{a['operating_point']['coverage']:.3f} acc_kept {a['operating_point']['accuracy_kept']:.3f} | "
                  f"out-of-scope refused {a['out_of_scope_refusal']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    chosen, why = fp.select("5class", arms)
    print(f"chosen: {chosen['arm']} — {why}", flush=True)

    held_paper = pool[inside].assign(_i=np.arange(inside.sum()))
    paper_rows = pd.concat([held_paper[held_paper.fold == k].sort_values("photo_id").head(fp.PAPER_PER_FOLD)
                            for k in range(5)])
    G_paper = all_features(images=fp.paper_images(load_images(paper_rows.path)))
    G_probes = all_features(images=far_ood_probes()[1])
    gcols = columns(GATE_SET)
    method, gate_eval = gate(X[:, gcols], y, folds, Xo[:, gcols], ofold, G_probes[:, gcols], G_paper[:, gcols],
                             paper_rows.fold.to_numpy(), chosen)
    ev = gate_eval[method]
    print(f"gate: {method} | " + " | ".join(f"{m}: AUROC {e['auroc_select']:.3f}, in-scope flagged "
                                           f"{e['flagged_in_scope']:.3f}" for m, e in gate_eval.items()), flush=True)

    # final model: the chosen arm re-tuned over the 5 folds and refitted on every in-scope train+val photo
    pipe, grid = make_model(chosen["classifier"], chosen["feature_set"])
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=PredefinedSplit(folds), n_jobs=-1).fit(X, y)
    scaler = StandardScaler().fit(X[:, gcols])
    density = FeatureDensity(scaler.transform(X[:, gcols]), y)
    served_limit = float(np.mean(ev["fold_thresholds"]) * ev["scale"])
    cal = chosen["calibration"]
    bundle = {"pipeline": gs.best_estimator_, "class_keys": FIVE, "feature_set": chosen["feature_set"],
              "input": "hog|color|texture",  # the pipeline takes the full row and keeps its own blocks
              "gate": {"feature_set": GATE_SET, "columns": [int(c) for c in gcols], "scaler": scaler,
                       "density": density, "method": method, "score": SCORE[method], "threshold": served_limit},
              "created": time.strftime("%Y-%m-%d %H:%M")}
    (MODELS_DIR / "ml").mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODELS_DIR / MODEL_PATH, compress=3)
    size = (MODELS_DIR / MODEL_PATH).stat().st_size / 2**20
    print(f"final params {gs.best_params_} | wrote {MODEL_PATH} ({size:.1f} MB)", flush=True)

    op = chosen["operating_point"]
    best = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    kind = cal["chosen"]
    adjust = {"ood_class_scale": ev["class_scales"],
              "fitted_on": "out-of-fold predictions of the train+val photos (cross-fitted)"}
    if kind == "vector":
        adjust["calibration"] = cal["fitted_on_all_folds"]
    entry = {
        "key": KEY, "name": NAME, "model_name": DISPLAY[chosen["classifier"]], "task": "5class", "generation": "upgrade",
        "round": "final",
        "family": "Machine learning", "kind": "classical", "type": "sklearn", "path": MODEL_PATH,
        "class_keys": FIVE, "feature_set": chosen["feature_set"],
        "temperature": cal["fitted_on_all_folds"]["T"] if kind == "temperature" else 1.0,
        "description": (f"{chosen['classifier']} on handcrafted "
                        f"{chosen['feature_set'].replace('Color', 'colour').replace('Texture', 'texture')} features (no deep "
                        f"learning). Calibration: {kind} scaling. One threshold ({op['threshold']:.2f}) and a "
                        "per-condition unusual-image gate, all fixed on out-of-fold data."),
        "confidence_threshold": op["threshold"],
        "operating_point": {"threshold": op["threshold"], "chosen_on": "out-of-fold cross-validation predictions "
                            "(cross-fitted)", "rule": chosen["threshold_rule"],
                            "oof": {k: op[k] for k in ("coverage", "accuracy_kept", "out_of_scope_answered")}},
        "ood": {"method": f"{method} distance in the standardised colour + texture feature space (training photos)",
                "keep": fp.OOD_KEEP, "per_condition": True},
        "adjust": adjust,
        "pipeline": {"classifier": chosen["classifier"], "features": chosen["feature_set"], "params": best,
                     "calibration": kind, "threshold": op["threshold"], "unusual_gate": method},
    }
    entries = load_json(EXPERIMENTS_DIR / "final_models.json")
    entries["models"] = [e for e in entries["models"] if e["key"] != KEY] + [entry]
    save_json(entries, EXPERIMENTS_DIR / "final_models.json")

    p = chosen["_p"]
    to_global = np.array([CLASS_KEYS.index(c) for c in FIVE])
    importance = None
    clf = gs.best_estimator_.named_steps["clf"]
    if hasattr(clf, "feature_importances_") and "HOG" not in chosen["feature_set"]:
        from tomatoleaf.features import COLOR_FEATURE_NAMES, TEXTURE_FEATURE_NAMES
        names = {"color": COLOR_FEATURE_NAMES, "texture": TEXTURE_FEATURE_NAMES}
        feat_names = [n for part in FEATURE_SETS[chosen["feature_set"]] for n in names[part]]
        order = np.argsort(clf.feature_importances_)[::-1][:15]
        importance = [{"feature": feat_names[i], "importance": float(clf.feature_importances_[i])} for i in order]
    save_json({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "classes": FIVE, "rules": "this script's docstring",
               "photos": {"in_scope": int(len(y)), "out_of_scope": int(len(Xo))},
               "arms": {k: fp.strip(v) for k, v in arms.items()},
               "chosen": {"arm": chosen["arm"], "why": why, "final_params": best,
                          "per_class_oof": per_class_metrics(to_global[y], to_global[p.argmax(1)], list(to_global)),
                          "confusion_oof": confusion(y, p.argmax(1), list(range(len(FIVE)))),
                          "reliability_oof": reliability_bins(p, y),
                          "reliability_oof_raw": reliability_bins(chosen["_p_raw"], y),
                          "feature_importance": importance},
               "unusual_gate": {"method": method, "keep": fp.OOD_KEEP, "served_limit": served_limit,
                                "evaluation_oof": gate_eval},
               "seconds": round(time.time() - t0)}, RESULTS_DIR / "classical_pipeline.json")
    print(f"wrote classical_pipeline.json and the {KEY} registry entry ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
