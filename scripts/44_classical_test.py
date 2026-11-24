"""The classical ML model on the held-out TEST photos (scored once), with the evaluation code of the deep model
(scripts/38: same metrics, stress tests, out-of-dataset photos, CPU latency), and a paired comparison with the deep
model on the same photos. Nothing is fitted here: every choice was made on out-of-fold data by scripts/43.

Then the app serves the two five-condition models: deployment.json lists the deep-learning model (default) and the
classical model; the eight-condition model is no longer served.

Updates experiments/results/final_round.json, final_benchmark.json, experiments/benchmark_models.json and
experiments/deployment.json.
"""
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, EXPERIMENTS_DIR, RESULTS_DIR, SEED  # noqa: E402
from tomatoleaf.data import load_images, load_splits  # noqa: E402
from tomatoleaf.probes import far_ood_probes  # noqa: E402
from tomatoleaf.utils import load_json, save_json  # noqa: E402

spec = importlib.util.spec_from_file_location("ft", ROOT / "scripts" / "38_final_test.py")
ft = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ft)

ML, DL = "classical_5class", "focused_5class"


def main():
    entry = next(e for e in load_json(EXPERIMENTS_DIR / "final_models.json")["models"] if e["key"] == ML)
    bench_models = load_json(EXPERIMENTS_DIR / "benchmark_models.json")
    bench_models["models"] = [m for m in bench_models["models"] if m["key"] != ML] + [entry]
    save_json(bench_models, EXPERIMENTS_DIR / "benchmark_models.json")

    reg = ModelRegistry(EXPERIMENTS_DIR / "benchmark_models.json", device="cpu", lazy=True)
    te = load_splits()
    te = te[te.split == "test"].reset_index(drop=True)
    X = load_images(te.path)
    y8, groups = te.label.to_numpy(), te.group_id.to_numpy()
    in5 = te.class_key.isin(ft.FIVE).to_numpy()
    keys_all = te.class_key.to_numpy()

    rows, preds = {}, {}
    for key in (ML, DL):
        pred = reg.load(key)
        thr = reg.threshold_for(pred)
        probs, dec, conf, ratios = ft.run(pred, X)
        rows[key], preds[key] = ft.test_row(key, entry["name"] if key == ML else pred.name, pred, thr, probs, dec, conf,
                                            ratios, te, in5, groups, y8)
        t, s = rows[key]["test"], rows[key]["selective"]
        print(f"{key:18s} acc {t['accuracy']:.3f} [{t['accuracy_ci'][0]:.3f}-{t['accuracy_ci'][1]:.3f}] F1 "
              f"{t['f1_macro']:.3f} | answers {s['coverage']:.2f} at {thr:.2f} -> {s['accuracy_kept']} | "
              f"out-of-scope refused {rows[key]['out_of_scope']['refused']:.2f}", flush=True)
        if key == DL:
            reg.models.pop(key, None)
            reg._members.clear()

    labels = [CLASS_KEYS.index(k) for k in ft.FIVE]
    comp = ft.paired_bootstrap(y8[in5], preds[ML][in5], preds[DL][in5], groups[in5], seed=SEED, labels=labels)
    comparison = {"task": "5class", "a": ML, "b": DL, "a_name": rows[ML]["name"], "b_name": rows[DL]["name"],
                  "ml_vs_dl": True, **comp}
    print(f"classical - deep: dAcc {comp['accuracy_diff']:+.3f} [{comp['accuracy_ci'][0]:+.3f}, "
          f"{comp['accuracy_ci'][1]:+.3f}] dF1 {comp['f1_diff']:+.3f} [{comp['f1_ci'][0]:+.3f}, {comp['f1_ci'][1]:+.3f}]")

    pred = reg.load(ML)
    thr = reg.threshold_for(pred)
    stress, ood = {}, {}
    for sname, (imgs, truth) in ft.stress_sets(X[in5], keys_all[in5], np.random.default_rng(SEED)).items():
        _, d, c, r = ft.run(pred, imgs)
        stress[sname] = ft.outcomes([pred.class_keys[i] for i in d], c, r, thr, truth)
    probe_names, probes, _ = far_ood_probes()
    for sname, imgs in [("probes", probes), ("empty_paper", ft.paper_only(X[:60]))]:
        _, d, c, r = ft.run(pred, imgs)
        ood[sname] = ft.outcomes([pred.class_keys[i] for i in d], c, r, thr)
    for sname, (imgs, truth) in ft.external_sets().items():
        _, d, c, r = ft.run(pred, imgs)
        dk = [pred.class_keys[i] for i in d]
        if sname == "tomato_known_shift":
            ood[sname] = {k: ft.outcomes([x for x, t in zip(dk, truth) if t == k], c[[t == k for t in truth]],
                                         r[[t == k for t in truth]], thr, [k] * sum(t == k for t in truth))
                          for k in sorted(set(truth))}
        else:
            ood[sname] = ft.outcomes(dk, c, r, thr)
    lat = ft.latency([ML])

    created = time.strftime("%Y-%m-%d %H:%M")
    fr = load_json(RESULTS_DIR / "final_round.json")
    fr["models"][ML] = rows[ML]
    fr["comparisons"] = [c for c in fr["comparisons"] if not c.get("ml_vs_dl")] + [comparison]
    fr["stress"][ML], fr["out_of_dataset"][ML] = stress, ood
    fr.setdefault("latency", {}).update(lat)
    fr["classical_note"] = f"Classical model (scripts/43) scored once on the test photos by scripts/44, {created}."
    save_json(fr, RESULTS_DIR / "final_round.json")

    bench = load_json(RESULTS_DIR / "final_benchmark.json")
    r = rows[ML]
    bench["models"] = [m for m in bench["models"] if m["key"] != ML] + [
        {**{k: v for k, v in r.items() if k not in ("per_condition", "risk_coverage")},
         "generation": "upgrade", "family": "Classical ML", "round": "final"}]
    bench["comparisons"] = [c for c in bench["comparisons"] if not c.get("ml_vs_dl")] + [comparison]
    bench.setdefault("latency", {}).update(lat)
    bench.setdefault("risk_coverage_variants", {})[ML] = {"model": ML, "curve": r["risk_coverage"]}
    save_json(bench, RESULTS_DIR / "final_benchmark.json")

    for m in bench_models["models"]:
        if m["key"] == ML:
            m["test_metrics"] = {k: r["test"][k] for k in ("accuracy", "f1_macro", "f1_weighted", "balanced_accuracy",
                                                           "ece", "brier", "accuracy_ci", "f1_macro_ci")}
            m["operating_point"]["test"] = {"coverage": r["selective"]["coverage"],
                                            "accuracy_kept": r["selective"]["accuracy_kept"], "photos": r["photos"]}
            m["latency_ms"] = lat[ML]["median_ms"]
    save_json(bench_models, EXPERIMENTS_DIR / "benchmark_models.json")

    deployment = load_json(EXPERIMENTS_DIR / "deployment.json")
    served = {m["key"]: m for m in bench_models["models"]}
    deployment.update({"version": 4, "default_model": DL, "defaults_by_task": {"5class": DL},
                       "previous_defaults": {"5class": "focused_5class_r1"}, "models": [served[DL], served[ML]]})
    save_json(deployment, EXPERIMENTS_DIR / "deployment.json")
    print("deployment.json now serves:", [m["key"] for m in deployment["models"]])


if __name__ == "__main__":
    main()
