"""Final round, step 4 — the two final pipelines on the held-out TEST photos (scored once), against the models they
replace, plus robustness and out-of-dataset checks. Nothing here is fitted: every threshold, calibrator and limit
was fixed on out-of-fold data by scripts/36.

1. Test metrics of the final focused / full pipelines and, re-run for paired comparisons, of the models they replace
   (v2.1 served: focused balanced / calibrated, full balanced; v1 defaults): accuracy and macro-F1 with 95% CIs over
   leaf groups, weighted F1, balanced accuracy, per-class precision / recall, confusion matrices, Brier, ECE,
   class-wise ECE, NLL, reliability bins, answers at the model's threshold, unusual-image flags, out-of-scope photos,
   per-condition outcomes, same-leaf consistency; paired bootstrap of final vs each original.
2. Stress tests built from the in-scope test photos: blur, dark, over-exposed, heavy JPEG, four leaves in one frame,
   leaf on black / wood / grass / fabric backgrounds.
3. Out-of-dataset photos (scripts/37): other crops, tomato diseases outside the dataset, tomato photos from another
   lab (domain shift), everyday photos; synthetic non-leaf probes; empty paper (leaf removed from test photos).
4. CPU latency (same machine, same conditions) for the served models.

Writes experiments/results/final_round.json, the final rows / comparisons in final_benchmark.json, the two final
entries (with test metrics) in benchmark_models.json and deployment.json (which then serves the final focused pipeline;
scripts/44 adds the classical model; the eight-condition model is evaluated here for the record but not served), and
figures in reports/figures/final/.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf import balance  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, EXPERIMENTS_DIR, FIGURES_DIR, RESULTS_DIR, SEED  # noqa: E402
from tomatoleaf.data import load_images, load_splits  # noqa: E402
from tomatoleaf.evaluation import (  # noqa: E402
    classification_metrics,
    confusion,
    group_bootstrap_ci,
    paired_bootstrap,
    per_class_metrics,
)
from tomatoleaf.features import leaf_mask  # noqa: E402
from tomatoleaf.preprocessing import canonicalize, jpeg_roundtrip, prepare_upload  # noqa: E402
from tomatoleaf.probes import far_ood_probes  # noqa: E402
from tomatoleaf.utils import load_json, save_json, set_seed  # noqa: E402

FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
FINAL = {"5class": "focused_5class", "8class": "full_8class"}
ORIGINAL = {"5class": ["focused_5class_r1", "balanced_5class", "calibrated_5class", "ensemble_5class"],
            "8class": ["full_8class_r1", "balanced_8class", "convnext_tiny_v2_tta"]}
SERVED_BEFORE = {"5class": "focused_5class_r1", "8class": "full_8class_r1"}  # live before the last round
OUT_FIG = FIGURES_DIR / "final"


# ------------------------------------------------------------------------------------------------- inference
def run(pred, images):
    """-> probs (N x K), decisions, confidence of the decision, unusualness ratio (nan if the model has none)."""
    probs, ratios = [], []
    for img in images:
        p, r = pred.analyse(img)
        probs.append(p)
        ratios.append(np.nan if r is None else r)
    probs = np.stack(probs)
    dec = np.array([pred.decide(p) for p in probs])
    ratios = np.array([np.nan if np.isnan(r) else pred.unusualness(r, int(d)) for r, d in zip(ratios, dec)])
    return probs, dec, probs[np.arange(len(dec)), dec], ratios


def outcomes(dec_keys, conf, ratios, thr, true_keys=None):
    """Per-photo status: answered / uncertain / unusual, and (if the truth is known) right or wrong."""
    unusual = np.nan_to_num(ratios, nan=0.0) > 1
    uncertain = (conf < thr) & ~unusual
    answered = ~unusual & ~uncertain
    out = {"photos": int(len(conf)), "answered": float(answered.mean()), "uncertain": float(uncertain.mean()),
           "unusual": float(unusual.mean()), "refused": float((~answered).mean())}
    if true_keys is not None:
        right = np.array([a == b for a, b in zip(dec_keys, true_keys)])
        out.update({"answered_correct": float((answered & right).mean()),
                    "answered_wrong": float((answered & ~right).mean()),
                    "accuracy_all": float(right.mean()),
                    "accuracy_answered": float(right[answered].mean()) if answered.any() else None})
    return out


def reliability(conf, ok, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    return [{"lo": float(lo), "hi": float(hi), "count": int(m.sum()), "confidence": float(conf[m].mean()),
             "accuracy": float(ok[m].mean())}
            for lo, hi in zip(edges[:-1], edges[1:]) if (m := (conf > lo) & (conf <= hi)).any()]


def ece(conf, ok, bins=15):
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)) - 1, 0, bins - 1)
    return float(sum(abs(conf[idx == b].mean() - ok[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


# --------------------------------------------------------------------------------------------- stress images
def _mask(rgb):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(leaf_mask(rgb).astype(np.uint8), k).astype(bool)


def _texture(kind, rng):
    h = w = 256
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    if kind == "black":
        return np.full((h, w, 3), 18, float)
    if kind == "wood":
        grain = np.sin(xx / 6 + 3 * np.sin(yy / 40)) * 18 + rng.normal(0, 6, (h, w))
        return np.clip(np.stack([150 + grain, 100 + grain * 0.8, 60 + grain * 0.5], -1), 0, 255)
    if kind == "grass":
        base = cv2.GaussianBlur(rng.normal(0, 1, (h, w)), (0, 0), 3)
        base = (base - base.min()) / (np.ptp(base) + 1e-9)
        return np.clip(np.stack([40 + 50 * base, 90 + 90 * base, 30 + 30 * base], -1), 0, 255)
    check = ((xx // 16 + yy // 16) % 2)[..., None]  # blue gingham fabric
    return np.clip(check * np.array([60, 90, 170]) + (1 - check) * np.array([225, 230, 240]) + rng.normal(0, 5, (h, w, 3)),
                   0, 255)


def stress_sets(images, keys, rng):
    """{name: (images, true keys or None)} built from in-scope test photos (fixed seed)."""
    out = {}
    out["blur_sigma2"] = np.stack([cv2.GaussianBlur(x, (0, 0), 2) for x in images])
    out["blur_sigma4"] = np.stack([cv2.GaussianBlur(x, (0, 0), 4) for x in images])
    out["dark_x0.35"] = (images.astype(float) * 0.35).round().astype(np.uint8)
    out["overexposed_x1.7"] = np.clip(images.astype(float) * 1.7, 0, 255).round().astype(np.uint8)
    out["jpeg_q15"] = np.stack([np.asarray(jpeg_roundtrip(Image.fromarray(x), quality=15))
                                for x in images])
    masks = [_mask(x) for x in images]
    for bg in ("black", "wood", "grass", "fabric"):
        imgs = []
        for x, m in zip(images, masks):
            t = _texture(bg, rng)
            y = x.astype(float).copy()
            y[~m] = t[~m]
            imgs.append(y.round().astype(np.uint8))
        out[f"background_{bg}"] = np.stack(imgs)
    sets = {k: (v, list(keys)) for k, v in out.items()}
    # four leaves in one frame: same condition (truth known) and four different conditions (no single truth)
    by_key = {}
    for i, k in enumerate(keys):
        by_key.setdefault(k, []).append(i)
    same, same_keys = [], []
    for k, idx in by_key.items():
        for j in range(0, len(idx) - 3, 4):
            q = [images[i] for i in idx[j:j + 4]]
            same.append(np.vstack([np.hstack(q[:2]), np.hstack(q[2:])]))
            same_keys.append(k)
    mixed = []
    pools = [v for v in by_key.values() if len(v) >= 2]
    for _ in range(40):
        ks = rng.choice(len(pools), size=min(4, len(pools)), replace=False)
        q = [images[rng.choice(pools[c])] for c in ks]
        while len(q) < 4:
            q.append(q[0])
        mixed.append(np.vstack([np.hstack(q[:2]), np.hstack(q[2:])]))

    def to256(arr):
        return np.stack([np.asarray(canonicalize(Image.fromarray(a))) for a in arr])

    if same:
        sets["four_leaves_same"] = (to256(same), same_keys)
    sets["four_leaves_mixed"] = (to256(mixed), None)
    return sets


def paper_only(images):
    return np.stack([cv2.inpaint(x, _mask(x).astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA) for x in images])


def external_sets():
    path = ROOT / "data" / "external_ood" / "manifest.csv"
    if not path.exists():
        return {}
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for r in rows:
        try:
            arr, _ = prepare_upload((ROOT / r["file"]).read_bytes())
        except Exception:  # noqa: BLE001 - unreadable download: skip
            continue
        out.setdefault(r["set"], ([], []))
        out[r["set"]][0].append(arr)
        out[r["set"]][1].append(r["true_key"] or None)
    return {k: (np.stack(v[0]), v[1]) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------- main
def test_row(key, name, pred, thr, probs, dec, conf, ratios, te, mask, groups, y8):
    local = np.array([CLASS_KEYS.index(k) for k in pred.class_keys])
    p_in, d_in, g_in = probs[mask], dec[mask], groups[mask]
    y_local = np.array([list(local).index(v) for v in y8[mask]])
    yg, pg, labels = y8[mask], local[d_in], list(local)
    ok = d_in == y_local
    c_in = conf[mask]
    unusual = np.nan_to_num(ratios[mask], nan=0.0) > 1
    answered = (c_in >= thr) & ~unusual
    row = {"key": key, "name": name, "task": pred.task, "photos": int(mask.sum()),
           "test": {**classification_metrics(yg, pg, labels), **group_bootstrap_ci(yg, pg, g_in, seed=SEED, labels=labels),
                    "balanced_accuracy": balance.balanced_accuracy(y_local, d_in, len(labels)),
                    "ece": ece(c_in, ok), "classwise_ece": balance.classwise_ece(p_in, y_local, d_in),
                    "nll": float(-np.log(p_in[np.arange(len(y_local)), y_local] + 1e-12).mean()),
                    "brier": float(np.mean(np.sum((p_in - np.eye(len(labels))[y_local]) ** 2, axis=1)))},
           "per_class": per_class_metrics(yg, pg, labels), "confusion": confusion(yg, pg, labels),
           "reliability": reliability(c_in, ok), "threshold": thr,
           "selective": {"threshold": thr, "coverage": float(answered.mean()),
                         "accuracy_kept": float(ok[answered].mean()) if answered.any() else None,
                         "accuracy_rejected": float(ok[~answered].mean()) if (~answered).any() else None},
           "flagged_unusual": float(unusual.mean()),
           "label_differs_from_tallest_bar": float(np.mean(dec != probs.argmax(1)))}
    keys_all = te.class_key.to_numpy()
    dec_keys = np.array([pred.class_keys[d] for d in dec])
    row["per_condition"] = {k: outcomes(dec_keys[keys_all == k], conf[keys_all == k], ratios[keys_all == k], thr,
                                        keys_all[keys_all == k]) for k in sorted(set(keys_all))}
    if pred.task == "5class":
        oos = ~mask
        row["out_of_scope"] = {**outcomes(dec_keys[oos], conf[oos], ratios[oos], thr, keys_all[oos]),
                               "confidence_mean": float(conf[oos].mean())}
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
    row["same_leaf_consistency"] = {"groups": multi, "pairs": pairs_all,
                                    "pairwise_agreement": pairs_same / max(pairs_all, 1),
                                    "groups_all_agree": agree / max(multi, 1)}
    row["risk_coverage"] = [{"threshold": float(t), "coverage": float((c_in >= t).mean()),
                             "accuracy_kept": float(ok[c_in >= t].mean()) if (c_in >= t).any() else None}
                            for t in np.round(np.arange(0.30, 0.991, 0.01), 2)]
    return row, local[dec]


def latency(keys, n=30):
    torch.set_num_threads(8)
    reg = ModelRegistry(EXPERIMENTS_DIR / "benchmark_models.json", device="cpu", lazy=True)
    te = load_splits()
    imgs = load_images(te[te.split == "test"].path.head(n + 3))
    out = {}
    for key in keys:
        pred = reg.load(key)
        for x in imgs[:3]:
            pred.analyse(x)
        times = []
        for x in imgs[3:]:
            t0 = time.perf_counter()
            pred.analyse(x)
            times.append(1000 * (time.perf_counter() - t0))
        out[key] = {"median_ms": float(np.median(times)), "p90_ms": float(np.quantile(times, 0.9)), "threads": 8,
                    "runtime": getattr(pred, "runtime", "pytorch"), "measured": time.strftime("%Y-%m-%d %H:%M")}
        reg.models.pop(key, None)
        reg._members.clear()  # free this model's networks before the next one (CPU memory)
        print(f"latency {key}: {out[key]['median_ms']:.0f} ms", flush=True)
    return out


def figures(rows, oof, stress, ood):
    from tomatoleaf.plotting import INK2, SERIES, apply_style, plot_confusion, plt, save

    apply_style()
    short = {"early_blight": "EB", "healthy": "H", "late_blight": "LB", "leaf_miner": "LM",
             "magnesium_deficiency": "Mg", "nitrogen_deficiency": "N", "potassium_deficiency": "K",
             "spotted_wilt_virus": "SWV"}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (title, bins, e) in zip(axes, [
            ("Focused · out-of-fold", oof["5class"]["reliability"], oof["5class"]["ece"]),
            ("Focused · test", rows[FINAL["5class"]]["reliability"], rows[FINAL["5class"]]["test"]["ece"]),
            ("Full · out-of-fold", oof["8class"]["reliability"], oof["8class"]["ece"]),
            ("Full · test", rows[FINAL["8class"]]["reliability"], rows[FINAL["8class"]]["test"]["ece"])]):
        mids = [(b["lo"] + b["hi"]) / 2 for b in bins]
        ax.bar(mids, [b["accuracy"] for b in bins], width=0.09, color=SERIES[0], alpha=0.85, label="accuracy")
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="perfect calibration")
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o", color=SERIES[1], ms=4,
                label="bin mean confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_title(f"{title} (ECE {e:.3f})")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="upper left", fontsize=8)
    save(fig, OUT_FIG / "reliability.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, key, title in [(axes[0, 0], SERVED_BEFORE["5class"], "Focused · before (v2.1 balanced)"),
                           (axes[0, 1], FINAL["5class"], "Focused · final"),
                           (axes[1, 0], SERVED_BEFORE["8class"], "Full · before (v2.1 balanced)"),
                           (axes[1, 1], FINAL["8class"], "Full · final")]:
        r = rows[key]
        labels = [short[p["key"]] for p in r["per_class"]]
        plot_confusion(ax, r["confusion"], f"{title} — test acc {r['test']['accuracy']:.3f}", labels=labels)
    fig.tight_layout()
    save(fig, OUT_FIG / "confusion_before_after.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, task in zip(axes, ("5class", "8class")):
        for key, color, lab in [(SERVED_BEFORE[task], SERIES[0], "before"), (FINAL[task], SERIES[1], "final")]:
            cur = [c for c in rows[key]["risk_coverage"] if c["accuracy_kept"] is not None]
            ax.plot([c["coverage"] for c in cur], [c["accuracy_kept"] for c in cur], color=color, label=lab)
            s = rows[key]["selective"]
            ax.plot([s["coverage"]], [s["accuracy_kept"]], "o", color=color, ms=6)
        ax.set_xlabel("Share of in-scope test photos answered")
        ax.set_ylabel("Accuracy on answered photos")
        ax.set_title(("Focused" if task == "5class" else "Full") + " · risk-coverage on test (dot = threshold)")
        ax.legend()
    save(fig, OUT_FIG / "risk_coverage.png")

    for task in ("5class", "8class"):
        names = list(stress[FINAL[task]].keys())
        fig, ax = plt.subplots(figsize=(13, 4))
        x = np.arange(len(names))
        for i, (key, lab) in enumerate([(SERVED_BEFORE[task], "before"), (FINAL[task], "final")]):
            s = stress[key]
            right = [s[n].get("answered_correct", 0) for n in names]
            wrong = [s[n].get("answered_wrong", s[n]["answered"]) for n in names]
            ref = [s[n]["refused"] for n in names]
            xx = x + (i - 0.5) * 0.38
            ax.bar(xx, right, 0.36, color=SERIES[2], label="answered correctly" if i == 0 else None)
            ax.bar(xx, wrong, 0.36, bottom=right, color=SERIES[7], label="answered wrongly" if i == 0 else None)
            ax.bar(xx, ref, 0.36, bottom=np.add(right, wrong), color="#c3c2b7",
                   label="refused (uncertain / unusual)" if i == 0 else None)
            for xi in xx:
                ax.text(xi, 1.01, lab[0].upper(), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(x, [n.replace("_", " ") for n in names], rotation=25, ha="right")
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Share of photos")
        ax.set_title(("Focused" if task == "5class" else "Full") + " · stress tests (B = before, F = final)")
        ax.legend(loc="lower left", fontsize=8, ncol=3)
        save(fig, OUT_FIG / f"stress_{task}.png")

    sets = [s for s in ("other_crop", "tomato_unsupported", "everyday", "probes", "empty_paper")
            if s in ood[FINAL["8class"]]]
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(sets))
    for i, key in enumerate([SERVED_BEFORE["5class"], FINAL["5class"], SERVED_BEFORE["8class"], FINAL["8class"]]):
        ax.bar(x + (i - 1.5) * 0.2, [ood[key][s]["refused"] for s in sets], 0.19, color=SERIES[i],
               label={0: "focused before", 1: "focused final", 2: "full before", 3: "full final"}[i])
    ax.set_xticks(x, [s.replace("_", " ") for s in sets])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Share refused (no diagnosis)")
    ax.set_title("Out-of-dataset images: share refused (higher is better)")
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    save(fig, OUT_FIG / "out_of_dataset.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()
    set_seed(SEED)

    final_entries = load_json(EXPERIMENTS_DIR / "final_models.json")["models"]
    bench_models = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
    bench_models["models"] = [m for m in bench_models["models"] if m["key"] not in {e["key"] for e in final_entries}] \
        + final_entries
    save_json(bench_models, EXPERIMENTS_DIR / "benchmark_models.json")
    names = {m["key"]: m["name"] for m in bench_models["models"]}

    reg = ModelRegistry(EXPERIMENTS_DIR / "benchmark_models.json", device=args.device, lazy=True)
    te = load_splits()
    te = te[te.split == "test"].reset_index(drop=True)
    X = load_images(te.path)
    y8, groups = te.label.to_numpy(), te.group_id.to_numpy()
    in5 = te.class_key.isin(FIVE).to_numpy()
    keys_all = te.class_key.to_numpy()

    probe_names, probes, _ = far_ood_probes()
    paper = paper_only(X[:60])
    external = external_sets()
    print("external sets:", {k: len(v[0]) for k, v in external.items()}, flush=True)

    rows, preds, stress, ood = {}, {}, {}, {}
    for task in ("5class", "8class"):
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        X_in, k_in = X[mask], keys_all[mask]
        sets = stress_sets(X_in, k_in, np.random.default_rng(SEED))
        for key in [FINAL[task]] + ORIGINAL[task]:
            pred = reg.load(key)
            thr = reg.threshold_for(pred)
            probs, dec, conf, ratios = run(pred, X)
            row, preds[key] = test_row(key, names[key], pred, thr, probs, dec, conf, ratios, te, mask, groups, y8)
            rows[key] = row
            t = row["test"]
            print(f"{key:22s} acc {t['accuracy']:.3f} [{t['accuracy_ci'][0]:.3f}-{t['accuracy_ci'][1]:.3f}] "
                  f"F1 {t['f1_macro']:.3f} wF1 {t['f1_weighted']:.3f} ECE {t['ece']:.3f} Brier {t['brier']:.3f} | "
                  f"answers {row['selective']['coverage']:.2f} at {thr:.2f} -> {row['selective']['accuracy_kept']}"
                  + (f" | out-of-scope refused {row['out_of_scope']['refused']:.2f}" if "out_of_scope" in row else ""),
                  flush=True)
            if key in (FINAL[task], SERVED_BEFORE[task]):
                stress[key] = {}
                for sname, (imgs, truth) in sets.items():
                    _, d, c, r = run(pred, imgs)
                    stress[key][sname] = outcomes([pred.class_keys[i] for i in d], c, r, thr, truth)
                ood[key] = {}
                for sname, imgs in [("probes", probes), ("empty_paper", paper)]:
                    _, d, c, r = run(pred, imgs)
                    ood[key][sname] = outcomes([pred.class_keys[i] for i in d], c, r, thr)
                for sname, (imgs, truth) in external.items():
                    _, d, c, r = run(pred, imgs)
                    dk = [pred.class_keys[i] for i in d]
                    if sname == "tomato_known_shift":
                        ood[key][sname] = {k: outcomes([x for x, t in zip(dk, truth) if t == k],
                                                       c[[t == k for t in truth]], r[[t == k for t in truth]], thr,
                                                       [k] * sum(t == k for t in truth))
                                           for k in sorted(set(truth))}
                    else:
                        ood[key][sname] = outcomes(dk, c, r, thr)
                print(f"   stress/OOD done for {key}", flush=True)
            reg.models.pop(key, None)
        reg._members.clear()  # fold networks shared between ensembles: free them before the next task
        torch.cuda.empty_cache()

    comparisons = []
    for task in ("5class", "8class"):
        mask = in5 if task == "5class" else np.ones(len(te), bool)
        labels = [CLASS_KEYS.index(k) for k in (FIVE if task == "5class" else CLASS_KEYS)]
        for ref in ORIGINAL[task]:
            c = paired_bootstrap(y8[mask], preds[FINAL[task]][mask], preds[ref][mask], groups[mask], seed=SEED,
                                 labels=labels)
            comparisons.append({"task": task, "a": FINAL[task], "b": ref, "a_name": names[FINAL[task]],
                                "b_name": names[ref], "final_round": True, **c})
            print(f"[{task}] final vs {ref}: dAcc {c['accuracy_diff']:+.3f} [{c['accuracy_ci'][0]:+.3f}, "
                  f"{c['accuracy_ci'][1]:+.3f}] dF1 {c['f1_diff']:+.3f} [{c['f1_ci'][0]:+.3f}, {c['f1_ci'][1]:+.3f}]")

    lat = {} if args.skip_latency else latency([FINAL["5class"], FINAL["8class"], SERVED_BEFORE["5class"],
                                                SERVED_BEFORE["8class"]])
    oof = {}
    for task in ("5class", "8class"):
        fp = load_json(RESULTS_DIR / f"final_pipeline_{task}.json")
        arm = fp["arms"][fp["chosen"]["arm"]]
        oof[task] = {"reliability": fp["chosen"]["reliability_oof"], "ece": arm["oof"]["ece"]}

    created = time.strftime("%Y-%m-%d %H:%M")
    result = {"created": created, "note": ("Final pipelines fixed on out-of-fold data (scripts/36, rules in "
              "docs/AUDIT.md §6) and scored here once on the test photos. The test photos had been used for "
              "reporting in earlier rounds (v1, v2, v2.1), never for a choice."),
              "models": rows, "comparisons": comparisons, "stress": stress, "out_of_dataset": ood,
              "latency": lat, "probe_names": probe_names}
    save_json(result, RESULTS_DIR / "final_round.json")

    bench = load_json(RESULTS_DIR / "final_benchmark.json")
    keep = {k: v for k, v in rows.items() if k in FINAL.values()}
    bench["models"] = [m for m in bench["models"] if m["key"] not in keep] + [
        {**{k: v for k, v in r.items() if k not in ("per_condition", "risk_coverage")},
         "generation": "upgrade", "family": "Deep learning", "round": "final"} for r in keep.values()]
    bench["comparisons"] = [c for c in bench["comparisons"] if not c.get("final_round")] + comparisons
    bench.setdefault("latency", {}).update(lat)
    bench["final_defaults"] = dict(FINAL)  # "defaults" stays the version-2 networks (robustness series of scripts/25)
    bench.setdefault("risk_coverage_variants", {}).update(
        {k: {"model": k, "curve": rows[k]["risk_coverage"]} for k in FINAL.values()})
    bench["final_round_note"] = result["note"] + f" Scored {created}."
    save_json(bench, RESULTS_DIR / "final_benchmark.json")

    bench_models = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
    for m in bench_models["models"]:
        if m["key"] in keep:
            r = keep[m["key"]]
            m["test_metrics"] = {k: r["test"][k] for k in ("accuracy", "f1_macro", "f1_weighted", "balanced_accuracy",
                                                           "ece", "brier", "accuracy_ci", "f1_macro_ci")}
            m["operating_point"]["test"] = {"coverage": r["selective"]["coverage"],
                                            "accuracy_kept": r["selective"]["accuracy_kept"], "photos": r["photos"]}
            if m["key"] in lat:
                m["latency_ms"] = lat[m["key"]]["median_ms"]
    save_json(bench_models, EXPERIMENTS_DIR / "benchmark_models.json")
    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    # the apps diagnose five conditions: the deep model here, the classical model added by scripts/44
    deployment.update({"version": 3, "default_model": FINAL["5class"], "defaults_by_task": {"5class": FINAL["5class"]},
                       "previous_defaults": {"5class": SERVED_BEFORE["5class"]},
                       "models": [m for m in bench_models["models"] if m["key"] == FINAL["5class"]]})
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    print("deployment.json now serves:", [m["key"] for m in deployment["models"]])
    figures(rows, oof, stress, ood)
    print("figures in", OUT_FIG)


if __name__ == "__main__":
    main()
