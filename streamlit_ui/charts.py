"""Altair charts for the Streamlit app (rendered with Streamlit's theme, so they follow light/dark mode)."""
import altair as alt
import pandas as pd

GREEN, CYAN, AMBER, VIOLET, GRAY = "#22a262", "#0891b2", "#f29a16", "#7c3aed", "#94a3b8"


def generation(m: dict) -> str:
    if m.get("generation") == "upgrade":
        return "Tomato Care v2"
    return "Baseline · classical ML" if m.get("family") == "Classical ML" else "Baseline · original study"


GEN_SCALE = alt.Scale(domain=["Tomato Care v2", "Baseline · original study", "Baseline · classical ML"],
                      range=[GREEN, CYAN, GRAY])


def probabilities(rows: list[dict], top_key: str, threshold: float | None) -> alt.Chart:
    df = pd.DataFrame([{"Condition": r["name"], "key": r["key"], "p": r["probability"]} for r in rows])
    base = alt.Chart(df).encode(
        y=alt.Y("Condition:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220, labelFontSize=12.5)),
        x=alt.X("p:Q", title="Calibrated probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")))
    bars = base.mark_bar(cornerRadiusEnd=5, height=20).encode(
        color=alt.condition(alt.datum.key == top_key, alt.value(GREEN), alt.value(GRAY)),
        tooltip=[alt.Tooltip("Condition:N"), alt.Tooltip("p:Q", title="Probability", format=".1%")])
    text = base.mark_text(align="left", dx=5, fontSize=12).encode(text=alt.Text("p:Q", format=".1%"))
    layers = [bars, text]
    if threshold is not None:
        layers.append(alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=AMBER, strokeDash=[5, 4], size=2)
                      .encode(x="t:Q", tooltip=[alt.Tooltip("t:Q", title="Answer threshold", format=".0%")]))
    return alt.layer(*layers).properties(height=34 * len(df) + 24)


def metric_with_ci(models: list[dict], metric: str, title: str) -> alt.Chart:
    rows = []
    for m in models:
        lo, hi = m["test"].get(f"{metric}_ci", [None, None])
        rows.append({"Model": m["name"], "Generation": generation(m), "value": m["test"][metric], "lo": lo, "hi": hi,
                     "default": m.get("is_default", False)})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y("Model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(cornerRadiusEnd=4, height=16).encode(
        x=alt.X("value:Q", title=title, axis=alt.Axis(format="%" if metric == "accuracy" else ".2f")),
        color=alt.Color("Generation:N", scale=GEN_SCALE, legend=alt.Legend(orient="bottom", title=None)),
        opacity=alt.condition(alt.datum.default, alt.value(1), alt.value(0.72)),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("value:Q", title=title, format=".3f"),
                 alt.Tooltip("lo:Q", title="95% CI low", format=".3f"), alt.Tooltip("hi:Q", title="95% CI high", format=".3f")])
    err = base.mark_rule(size=2, opacity=0.65).encode(x="lo:Q", x2="hi:Q")
    return alt.layer(bars, err).properties(height=26 * len(df) + 40)


def per_class(rows: list[dict]) -> alt.Chart:
    """rows: {Condition, Model, F1, support}"""
    df = pd.DataFrame(rows)
    models = list(dict.fromkeys(df["Model"]))
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=10).encode(
        y=alt.Y("Condition:N", title=None, sort=list(dict.fromkeys(df["Condition"])), axis=alt.Axis(labelLimit=220)),
        yOffset=alt.YOffset("Model:N", sort=models),
        x=alt.X("F1:Q", title="F1 on the test photos", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", sort=models, scale=alt.Scale(range=[GREEN, CYAN, AMBER, VIOLET]),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=320)),
        tooltip=["Condition:N", "Model:N", alt.Tooltip("F1:Q", format=".3f"), alt.Tooltip("Support:Q", title="Test photos")],
    ).properties(height=max(220, 26 * len(df)))


def confusion(matrix: list[list[int]], labels: list[str]) -> alt.Chart:
    rows = []
    for i, t in enumerate(labels):
        total = sum(matrix[i]) or 1
        for j, p in enumerate(labels):
            rows.append({"True": t, "Predicted": p, "Photos": matrix[i][j], "Share": matrix[i][j] / total})
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(
        x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=-35, labelLimit=140), title="Predicted"),
        y=alt.Y("True:N", sort=labels, axis=alt.Axis(labelLimit=160), title="True condition"))
    cells = base.mark_rect(cornerRadius=4, stroke="#ffffff22").encode(
        color=alt.Color("Share:Q", scale=alt.Scale(scheme="greens", domain=[0, 1]), legend=None),
        tooltip=["True:N", "Predicted:N", "Photos:Q", alt.Tooltip("Share:Q", format=".0%", title="Share of the row")])
    text = base.mark_text(fontSize=12, fontWeight=600).encode(
        text=alt.condition(alt.datum.Photos > 0, alt.Text("Photos:Q"), alt.value("")),
        color=alt.condition(alt.datum.Share > 0.5, alt.value("#ffffff"), alt.value("#16301f")))
    side = 46 * len(labels) + 60
    return alt.layer(cells, text).properties(height=side)


def reliability(series: dict[str, list[dict]]) -> alt.Chart:
    rows = [{"Series": name, "Confidence": b["confidence"], "Accuracy": b["accuracy"], "Photos": b["count"]}
            for name, bins in series.items() for b in bins if b.get("count")]
    df = pd.DataFrame(rows)
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color=GRAY, strokeDash=[5, 4]).encode(x="x:Q", y="y:Q")
    names = list(series)
    lines = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(filled=True, size=70), strokeWidth=2.5).encode(
        x=alt.X("Confidence:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Stated confidence"),
        y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title="Actual accuracy"),
        color=alt.Color("Series:N", sort=names, scale=alt.Scale(range=[GRAY, GREEN, CYAN]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Series:N", alt.Tooltip("Confidence:Q", format=".1%"), alt.Tooltip("Accuracy:Q", format=".1%"), "Photos:Q"])
    return alt.layer(diag, lines).properties(height=320)


def risk_coverage(curve: list[dict], chosen: dict | None) -> alt.Chart:
    df = pd.DataFrame([c for c in curve if c.get("accuracy_kept") is not None])
    line = alt.Chart(df).mark_line(color=GREEN, strokeWidth=2.5).encode(
        x=alt.X("coverage:Q", title="Share of photos answered", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("accuracy_kept:Q", title="Accuracy on answered photos", axis=alt.Axis(format="%"), scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("threshold:Q", title="Threshold", format=".2f"), alt.Tooltip("coverage:Q", format=".1%"),
                 alt.Tooltip("accuracy_kept:Q", title="Accuracy", format=".1%")])
    layers = [line]
    if chosen and chosen.get("coverage") is not None:
        pt = pd.DataFrame([{"coverage": chosen["coverage"], "accuracy_kept": chosen["accuracy_kept"],
                            "threshold": chosen["threshold"]}])
        layers.append(alt.Chart(pt).mark_point(size=160, filled=True, color=AMBER, stroke="white", strokeWidth=1.5).encode(
            x="coverage:Q", y="accuracy_kept:Q",
            tooltip=[alt.Tooltip("threshold:Q", title="Deployed threshold", format=".2f"),
                     alt.Tooltip("coverage:Q", format=".1%"), alt.Tooltip("accuracy_kept:Q", title="Accuracy", format=".1%")]))
    return alt.layer(*layers).properties(height=300)


def grouped_bars(rows: list[dict], category: str, series: str, value: str, title: str, fmt: str = "%",
                 colors=(GREEN, CYAN, AMBER, VIOLET)) -> alt.Chart:
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df[series]))
    return alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=11).encode(
        y=alt.Y(f"{category}:N", title=None, sort=list(dict.fromkeys(df[category])), axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset(f"{series}:N", sort=order),
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(format=fmt)),
        color=alt.Color(f"{series}:N", sort=order, scale=alt.Scale(range=list(colors)),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=320)),
        tooltip=[f"{category}:N", f"{series}:N", alt.Tooltip(f"{value}:Q", format=".3f" if fmt != "%" else ".1%")],
    ).properties(height=max(200, 13 * len(df) + 50 * len(set(df[category])) // 2))


def latency_scatter(models: list[dict]) -> alt.Chart:
    rows = [{"Model": m["name"], "Generation": generation(m), "Latency (ms)": m["latency_ms"],
             "Macro-F1": m["test"]["f1_macro"], "Accuracy": m["test"]["accuracy"]}
            for m in models if m.get("latency_ms")]
    df = pd.DataFrame(rows)
    return alt.Chart(df).mark_circle(size=170, opacity=0.9, stroke="white", strokeWidth=1).encode(
        x=alt.X("Latency (ms):Q", scale=alt.Scale(type="log"), title="CPU time per photo (ms, log scale)"),
        y=alt.Y("Macro-F1:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Generation:N", scale=GEN_SCALE, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model:N", alt.Tooltip("Latency (ms):Q", format=".0f"), alt.Tooltip("Macro-F1:Q", format=".3f"),
                 alt.Tooltip("Accuracy:Q", format=".1%")]).properties(height=320)


def fold_strip(recipes: list[dict], metric: str = "f1_1") -> alt.Chart:
    rows = [{"Recipe": r["short"], "Fold": f["fold"] + 1, "Macro-F1": f[metric], "Chosen": r["chosen"]}
            for r in recipes for f in r["per_fold"]]
    df = pd.DataFrame(rows)
    order = [r["short"] for r in recipes]
    points = alt.Chart(df).mark_circle(size=90, opacity=0.85).encode(
        y=alt.Y("Recipe:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("Macro-F1:Q", scale=alt.Scale(zero=False), title="Held-out macro-F1 of each fold"),
        color=alt.condition(alt.datum.Chosen, alt.value(GREEN), alt.value(GRAY)),
        tooltip=["Recipe:N", "Fold:O", alt.Tooltip("Macro-F1:Q", format=".3f")])
    means = alt.Chart(df).mark_tick(thickness=3, size=22, color=AMBER).encode(
        y=alt.Y("Recipe:N", sort=order), x="mean(Macro-F1):Q",
        tooltip=["Recipe:N", alt.Tooltip("mean(Macro-F1):Q", title="Mean", format=".3f")])
    return alt.layer(points, means).properties(height=46 * len(order) + 40)
