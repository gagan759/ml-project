"""Model lab: every model on identical test photos — leaderboard, paired tests, per-class results, confusion matrices,
calibration and abstention, robustness, speed, and the cross-validation that chose the version-2 models."""
import pandas as pd
import streamlit as st

from .. import charts, core, theme


def _verdict(comp: dict, first: str = "Version 2") -> str:
    lo, hi = comp["accuracy_ci"]
    if lo > 0:
        return f"{first} is more accurate, and the 95% interval excludes zero."
    if hi < 0:
        return f"{first} is less accurate, and the 95% interval excludes zero."
    word = "higher" if comp["accuracy_diff"] > 0 else "lower" if comp["accuracy_diff"] < 0 else "equal"
    n = core.platform()["tasks"][core.TASK]["test_photos"]
    return (f"{first}'s accuracy is {word}, but the 95% interval includes zero: on {n} test photos the "
            "difference is within noise.")


def _leaderboard(models: list[dict]):
    rows = []
    for m in sorted(models, key=lambda m: -m["test"]["f1_macro"]):
        t, s = m["test"], m["selective"]
        rows.append({
            "Model": ("★ " if m.get("deployed") else "") + m["name"], "Version": charts.generation(m),
            "Accuracy": t["accuracy"], "Accuracy 95% CI": f"{100 * t['accuracy_ci'][0]:.1f}–{100 * t['accuracy_ci'][1]:.1f}%",
            "Macro-F1": t["f1_macro"], "F1 95% CI": f"{t['f1_macro_ci'][0]:.3f}–{t['f1_macro_ci'][1]:.3f}",
            "ECE": t.get("ece"), "Answers": s.get("coverage"), "Accuracy when answering": s.get("accuracy_kept"),
            "CPU ms": m.get("latency_ms"),
            "Selection score": (m["cv"] or {}).get("f1_macro") or (m["val"] or {}).get("f1_macro"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Macro-F1": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1,
                                                    help="Mean F1 over the conditions: every condition counts equally"),
        "ECE": st.column_config.NumberColumn(format="%.3f", help="Expected calibration error (lower is better)"),
        "Answers": st.column_config.NumberColumn(format="percent", help="Share of photos given a diagnosis (not "
                                                                         "'not sure' or 'unusual')"),
        "Accuracy when answering": st.column_config.NumberColumn(format="percent"),
        "CPU ms": st.column_config.NumberColumn(format="%.0f", help="Median time per photo on an 8-thread laptop CPU"),
        "Selection score": st.column_config.NumberColumn(
            format="%.3f", help="Macro-F1 on the data the model was chosen with: out-of-fold (v2) or the "
                                "validation split (baseline). Not comparable with test scores."),
    })
    st.caption("★ = served by the app (deep learning and classical ML). Confidence intervals: 2,000 bootstrap "
               "resamples of whole leaf groups.")


def _per_class(models: list[dict], task: str):
    new, old = core.task_defaults(task)
    picks = [new, old]
    ml = core.classical_result() or max((m for m in models if m["family"] == "Classical ML"),
                                        key=lambda m: m["test"]["f1_macro"], default=None)
    if ml:
        picks.append(ml)
    label = {new["key"]: f"{new['name']} (v2)", old["key"]: f"{old['name']} (baseline)"}
    rows = [{"Condition": c["name"], "Model": label.get(m["key"], m["name"]), "F1": c["f1"], "Support": c["support"]}
            for m in picks for c in m["per_class"]]
    st.altair_chart(charts.per_class(rows), width="stretch")
    table = pd.DataFrame([{"Condition": c["name"], "Test photos": c["support"], "Precision": c["precision"],
                           "Recall": c["recall"], "F1": c["f1"], "F1 (baseline)": o["f1"]}
                          for c, o in zip(new["per_class"], old["per_class"])])
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("Precision", "Recall", "F1", "F1 (baseline)")})
    st.caption("Conditions with few test photos (Nitrogen Deficiency: 8) have very uncertain per-class scores.")


def _confusion(models: list[dict], task: str):
    new, _ = core.task_defaults(task)
    keys = [m["key"] for m in models]
    key = st.selectbox("Model", keys, index=keys.index(new["key"]), key=f"cm_{task}",
                       format_func=lambda k: next(m["name"] for m in models if m["key"] == k))
    m = next(m for m in models if m["key"] == key)
    labels = [c["name"] for c in m["per_class"]]
    st.altair_chart(charts.confusion(m["confusion"], labels), width="stretch")
    st.caption("Rows: true condition; columns: prediction (before any 'not sure'). Colour: share of the row.")


def _calibration(models: list[dict], task: str):
    data = core.platform()
    new, old = core.task_defaults(task)
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Reliability of the served model** (out-of-fold photos)")
        cal = new["calibration"]
        series = {}
        if cal.get("reliability_before"):
            series[f"Before {cal.get('method', 'temperature scaling')}"] = cal["reliability_before"]
        series[f"After {cal.get('method', 'temperature scaling')}"] = cal["reliability_after"]
        st.altair_chart(charts.reliability(series), width="stretch")
        if cal.get("ece_before") is not None:
            st.caption(f"Expected calibration error {cal['ece_before']:.3f} → {cal['ece_after']:.3f} on out-of-fold "
                       f"photos; {new['test']['ece']:.3f} on the test photos (baseline: {old['test']['ece']:.3f}).")
    with b:
        st.markdown("**Saying “not sure”** (test photos)")
        sel = data["selective"][task]
        st.altair_chart(charts.risk_coverage(sel["curve"], sel["chosen"]), width="stretch")
        ch = sel["chosen"]
        st.caption(f"Rule (fixed on out-of-fold data): {sel['rule']}. Threshold {ch['threshold']:.2f} → answers "
                   f"{core.pct(ch['coverage'])} of test photos with {core.pct(ch['accuracy_kept'])} accuracy. The baseline at its "
                   f"own threshold: {core.pct(old['selective']['coverage'])} answered, "
                   f"{core.pct(old['selective']['accuracy_kept'])} accurate.")


def _bias(models: list[dict], task: str):
    rows = [{"Model": ("★ " if m.get("deployed") else "") + m["name"], "Accuracy": m["test"]["accuracy"],
             "Balanced accuracy": m["test"]["balanced_accuracy"], "Macro-F1": m["test"]["f1_macro"],
             "Per-class calibration error": m["test"]["classwise_ece"]}
            for m in models if m["test"].get("balanced_accuracy") is not None]
    if not rows:
        return
    st.markdown("**Class bias and per-class confidence** (test photos)")
    st.dataframe(pd.DataFrame(rows).sort_values("Balanced accuracy", ascending=False), hide_index=True, width="stretch",
                 column_config={"Accuracy": st.column_config.NumberColumn(format="percent"),
                                "Balanced accuracy": st.column_config.NumberColumn(
                                    format="percent", help="Mean of the per-condition recalls: rare conditions count "
                                                           "as much as common ones (higher = less bias)"),
                                "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                                "Per-class calibration error": st.column_config.NumberColumn(
                                    format="%.3f", help="How far each condition's average confidence is from its own "
                                                        "accuracy (lower = more even confidence)")})
    st.caption("Earlier balanced / calibrated variants (v2.1, no longer served): the same networks with per-class "
               "calibration (and, for balanced, a class-prior correction on top of a class-weighted loss), fitted on "
               "out-of-fold data. The final round replaced them with one validated pipeline per model (tab Final "
               "pipeline). A GAN was not used to synthesise rare-condition photos: with 45 Nitrogen photographs it "
               "would copy them rather than add information, and generated images would need their own validation.")


STRESS_LABEL = {"blur_sigma2": "Blur (σ 2)", "blur_sigma4": "Strong blur (σ 4)", "dark_x0.35": "Very dark",
                "overexposed_x1.7": "Over-exposed", "jpeg_q15": "Heavy JPEG (q15)", "background_black": "Black background",
                "background_wood": "Wood background", "background_grass": "Grass background",
                "background_fabric": "Fabric background", "four_leaves_same": "4 leaves, same condition",
                "four_leaves_mixed": "4 leaves, mixed conditions"}
OOD_LABEL = {"other_crop": "Other crops (PlantVillage)", "tomato_unsupported": "Tomato diseases not in the dataset",
             "everyday": "Everyday photos", "probes": "Synthetic non-leaf images", "empty_paper": "Empty paper"}
PCT = st.column_config.NumberColumn(format="percent")


def _final_round(task: str):
    fr = ((core.platform().get("final_round") or {}).get("tasks") or {}).get(task)
    if not fr:
        st.info("Final-round results are not available in this build.")
        return
    st.markdown("**1 · Imbalance arms** — out-of-fold, cross-fitted; the rules were fixed before training")
    rows = [{"Arm": ("★ " if a["chosen"] else "") + f"{a['arm']} · {a['label']}", "Macro-F1": a["oof"]["f1_macro"],
             "Accuracy": a["oof"]["accuracy"], "NLL": a["oof"]["nll"], "ECE": a["oof"]["ece"],
             "Calibration": a["calibration"], "Threshold": a["threshold"], "Answered": a["coverage"],
             "Accuracy when answering": a["accuracy_kept"],
             "Unsupported conditions refused": a["out_of_scope_refusal"]}
            for a in fr["arms"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Macro-F1": PCT, "Accuracy": PCT, "Answered": PCT, "Accuracy when answering": PCT,
                                "Unsupported conditions refused": PCT,
                                "NLL": st.column_config.NumberColumn(format="%.3f"),
                                "ECE": st.column_config.NumberColumn(format="%.3f"),
                                "Threshold": st.column_config.NumberColumn(format="%.2f")})
    st.caption(f"Chosen: arm {fr['chosen']} — {fr['why']}. Unusual-image check: {fr['gate']['method']} feature "
               f"distance, out-of-fold AUROC {fr['gate']['auroc_select']:.3f}.")
    if not fr.get("test"):
        return
    st.markdown("**2 · Served models and earlier versions** — held-out test photos")
    rows = [{"Model": ("★ " if r["final"] or r.get("classical") else "") + r["name"], "Accuracy": r["accuracy"], "Macro-F1": r["f1_macro"],
             "Weighted F1": r["f1_weighted"], "ECE": r["ece"], "Brier": r["brier"], "Threshold": r["threshold"],
             "Answered": r["coverage"], "Accuracy when answering": r["accuracy_kept"],
             "Unsupported: confident wrong answer": (r.get("out_of_scope") or {}).get("answered_wrong")}
            for r in fr["test"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"Accuracy": PCT, "Macro-F1": PCT, "Weighted F1": PCT, "Answered": PCT,
                                "Accuracy when answering": PCT, "Unsupported: confident wrong answer": PCT,
                                "ECE": st.column_config.NumberColumn(format="%.3f"),
                                "Brier": st.column_config.NumberColumn(format="%.3f"),
                                "Threshold": st.column_config.NumberColumn(format="%.2f")})
    final_key = next(r["key"] for r in fr["test"] if r["final"])
    before_key = next(r["key"] for r in fr["test"] if not r["final"] and not r.get("classical"))
    ml_key = next((r["key"] for r in fr["test"] if r.get("classical")), None)
    s_ml = fr["stress"].get(ml_key) or {}
    o_ml = fr["out_of_dataset"].get(ml_key) or {}
    dl_name = core.display_name(core.model_entry(final_key))
    ml_name = core.display_name(core.model_entry(ml_key)) if ml_key else "Classical ML"
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**3 · Stress tests** — test photos changed on purpose")
        s_f, s_b = fr["stress"][final_key], fr["stress"][before_key]
        rows = [{"Condition": STRESS_LABEL.get(n, n),
                 "Before: wrong answer": s_b[n].get("answered_wrong", s_b[n]["answered"]),
                 f"{dl_name}: right answer": s_f[n].get("answered_correct"),
                 f"{dl_name}: wrong answer": s_f[n].get("answered_wrong", s_f[n]["answered"]),
                 f"{dl_name}: refused": s_f[n]["refused"],
                 **({f"{ml_name}: wrong answer": s_ml[n].get("answered_wrong", s_ml[n]["answered"])}
                    if n in s_ml else {})} for n in s_f]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={k: PCT for k in rows[0] if k != "Condition"})
        st.caption("Mixed-condition collages have no single right answer: any answer counts as wrong.")
    with b:
        st.markdown("**4 · Out-of-dataset images** — share refused (higher is better)")
        o_f, o_b = fr["out_of_dataset"][final_key], fr["out_of_dataset"][before_key]
        rows = [{"Images": OOD_LABEL[n], "Photos": o_f[n]["photos"], "Before": o_b[n]["refused"],
                 dl_name: o_f[n]["refused"], **({ml_name: o_ml[n]["refused"]} if n in o_ml else {})}
                for n in OOD_LABEL if n in o_f]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"Before": PCT, dl_name: PCT, ml_name: PCT})
    if (core.platform().get("final_round") or {}).get("note"):
        st.caption(core.platform()["final_round"]["note"])


def _robustness(task: str):
    data = core.platform()
    rb = data["audit"]["robustness"][task]
    series = {s["key"]: s["label"] for s in rb["series"]}
    rows = [{"Change": r["label"], "Model": label, "Changed predictions": r[key]}
            for r in rb["rows"] for key, label in series.items() if key in r]
    st.altair_chart(charts.grouped_bars(rows, "Change", "Model", "Changed predictions",
                                        "Share of test predictions that change", colors=(charts.GREEN, charts.CYAN)),
                    width="stretch")
    st.caption("Each test photo is changed harmlessly (re-compressed, blurred, brightened, halved) or turned back into "
               "a full-size phone photo and uploaded through the old or the new upload path. Lower is better.")
    pre = data["audit"]["preprocessing"]
    if pre:
        st.markdown("**The upload fix, measured on validation photos** (original default model)")
        st.dataframe(pd.DataFrame([{"Condition": p["label"], "Accuracy": p["accuracy"], "Macro-F1": p["f1_macro"],
                                    "Changed predictions": p["flip_rate"]} for p in pre]),
                     hide_index=True, width="stretch", column_config={
                         "Accuracy": st.column_config.NumberColumn(format="percent"),
                         "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
                         "Changed predictions": st.column_config.NumberColumn(format="percent")})


def _cross_validation(task: str):
    data = core.platform()
    recipes = [r for r in data["cv"]["recipes"] if r["task"] == task]
    st.markdown(f"**Selection rule.** {data['cv']['selection_rule']}")
    st.dataframe(pd.DataFrame([{
        "Recipe": ("★ " if r["chosen"] else "") + r["label"], "Macro-F1 (1 view)": r["f1_1"], "Macro-F1 (8 views)": r["f1_8"],
        "Accuracy (1 view)": r["acc_1"], "Accuracy (8 views)": r["acc_8"], "Training (min, 5 folds)": r["train_minutes"]}
        for r in recipes]), hide_index=True, width="stretch", column_config={
            "Macro-F1 (1 view)": st.column_config.NumberColumn(format="%.3f"),
            "Macro-F1 (8 views)": st.column_config.NumberColumn(format="%.3f"),
            "Accuracy (1 view)": st.column_config.NumberColumn(format="percent"),
            "Accuracy (8 views)": st.column_config.NumberColumn(format="percent"),
            "Training (min, 5 folds)": st.column_config.NumberColumn(format="%.0f")})
    st.caption("Out-of-fold scores: every one of the 1,813 train+validation photos is predicted by the fold model "
               "that never saw its leaf. ★ = member of the deployed model.")
    if recipes:
        st.altair_chart(charts.fold_strip(recipes), width="stretch")
    screen = [s for s in data["cv"]["screening"] if s["task"] == task]
    if screen:
        with st.expander("Every recipe tried (fold 0), including the ones stopped early"):
            st.dataframe(pd.DataFrame([{"Recipe": s["label"], "Macro-F1 (1 view)": s["f1_1"], "Macro-F1 (8 views)": s["f1_8"],
                                        "Accuracy (1 view)": s["acc_1"], "All 5 folds run": s["full_cv"]} for s in screen]),
                         hide_index=True, width="stretch")


def render() -> None:
    theme.section("Model lab", "Every five-condition model — classical machine learning, the original study's "
                  "networks and the version-2 upgrade — scored by the same code on the same held-out test photos, once.",
                  kicker="Evidence")
    task = core.TASK
    data = core.platform()
    models = [m for m in data["models"] if m["task"] == task]
    new, old = core.task_defaults(task)
    comp = core.headline_comparison(task)
    n = data["tasks"][task]["test_photos"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Served model · accuracy", core.pct(new["test"]["accuracy"]),
              delta=f"{core.pts(new['test']['accuracy'] - old['test']['accuracy'])} vs baseline", border=True)
    c2.metric("Served model · macro-F1", f"{new['test']['f1_macro']:.3f}",
              delta=f"{new['test']['f1_macro'] - old['test']['f1_macro']:+.3f} vs baseline", border=True)
    ml = core.classical_result()
    if ml:
        c3.metric(f"{core.display_name(core.model_entry(ml['key']))} · accuracy", core.pct(ml["test"]["accuracy"]),
                  delta=f"{core.pts(ml['test']['accuracy'] - new['test']['accuracy'])} vs "
                        f"{core.display_name(core.model_entry(new['key']))}",
                  border=True, help=f"{ml['name']}: handcrafted colour and texture features, no neural network.")
    else:
        c3.metric("Baseline · accuracy", core.pct(old["test"]["accuracy"]), delta=old["name"].split(" (original")[0],
                  delta_color="off", border=True, help="The original study's model (not served by the app).")
    c4.metric("Test photos", f"{n}", delta="scored once", delta_color="off", border=True)
    if comp:
        st.info(f"**Paired comparison (served model − original study's model, bootstrap over leaf groups):** accuracy {core.pts(comp['accuracy_diff'])} "
                f"[95% CI {core.pts(comp['accuracy_ci'][0])}, {core.pts(comp['accuracy_ci'][1])}], macro-F1 "
                f"{comp['f1_diff']:+.3f} [{comp['f1_ci'][0]:+.3f}, {comp['f1_ci'][1]:+.3f}]. {_verdict(comp)}",
                icon=":material/balance:")
    mlc = next((c for c in data["comparisons"] if c.get("ml_vs_dl")), None)
    if mlc:
        st.info(f"**{core.display_name(core.model_entry(mlc['a']))} (classical ML) − "
                f"{core.display_name(core.model_entry(mlc['b']))} (deep learning), same photos, bootstrap over leaf "
                f"groups:** accuracy "
                f"{core.pts(mlc['accuracy_diff'])} [95% CI {core.pts(mlc['accuracy_ci'][0])}, "
                f"{core.pts(mlc['accuracy_ci'][1])}], macro-F1 {mlc['f1_diff']:+.3f} [{mlc['f1_ci'][0]:+.3f}, "
                f"{mlc['f1_ci'][1]:+.3f}]. {_verdict(mlc, 'The classical model')}", icon=":material/compare_arrows:")
    tabs = st.tabs(["Leaderboard", "Final pipeline", "Per condition", "Confusion", "Calibration", "Robustness", "Speed",
                    "Cross-validation"])
    with tabs[1]:
        _final_round(task)
    tabs = [tabs[0]] + list(tabs[2:])  # the remaining tabs keep their original indices below
    with tabs[0]:
        _leaderboard(models)
        st.altair_chart(charts.metric_with_ci(models, "f1_macro", "Macro-F1 (test, 95% CI)"), width="stretch")
    with tabs[1]:
        _per_class(models, task)
    with tabs[2]:
        _confusion(models, task)
    with tabs[3]:
        _calibration(models, task)
        _bias(models, task)
    with tabs[4]:
        _robustness(task)
    with tabs[5]:
        st.altair_chart(charts.latency_scatter(models), width="stretch")
        st.caption("Median CPU time per photo through the API's own predictor (ONNX Runtime, 8 threads, laptop CPU). "
                   "The online demo runs on a smaller shared server and is slower.")
    with tabs[6]:
        _cross_validation(task)
