"""Diagnose: photo in (upload, camera or example), calibrated diagnosis, Grad-CAM, condition guide, model comparison."""
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from .. import charts, core, nav, theme

# ------------------------------------------------------------------------------------------------------------ input
SHORT = {"healthy": "Healthy", "leaf_miner": "Leaf miner", "magnesium_deficiency": "Magnesium",
         "nitrogen_deficiency": "Nitrogen", "spotted_wilt_virus": "Spotted wilt"}


def _examples():
    items = core.samples()
    more = st.toggle("More examples", key="ex_more", help="Three more photos per condition.")
    shown = items if more else [s for s in items if s["held_out"]]
    current = st.session_state.get("example")
    with st.container(key="examples"):
        cols = st.columns(4)
    for i, s in enumerate(shown):
        with cols[i % 4]:
            st.image(str(s["path"]), width="stretch")
            chosen = current == str(s["path"])
            if st.button(SHORT[s["key"]], key=f"ex_{i}", width="stretch", type="primary" if chosen else "secondary",
                         help=s["name"]):
                st.session_state["example"] = str(s["path"])
                st.rerun()
    st.caption(f"Tomato-Village dataset (Gehlot et al., 2023). The first {sum(s['held_out'] for s in items)} are demo "
               "examples: held-out test photos (never used for training) on which the models give a clear result — "
               "they show how the app works, not how often it is right (see Model lab). The three more per condition "
               "are training photos.")
    if current:
        name = next((s["name"] for s in items if str(s["path"]) == current), "example")
        return core.sample_bytes(current), f"Example · {name}"
    return None, None


# How the training photos were taken; photos taken this way get the most reliable results.
PHOTO_TIPS = ("Photo tips: one leaf only · plain, light paper · daylight, no glare · leaf fills the frame · "
              "sharp, in focus.")


def _guidance(r: dict) -> None:
    """Why no diagnosis was given (with the top candidates), or photo-quality warnings on an answered photo."""
    refusal, quality = r.get("refusal"), r.get("quality") or {}
    if refusal:
        likely = " · ".join(f"{c['name']} {core.pct(c['probability'], 0)}" for c in r.get("candidates", []))
        hints = "\n".join(f"- {h}" for h in refusal.get("hints", []))
        st.warning(f"**Why no diagnosis.** {refusal['message']}\n\nMost likely: {likely}\n\n{hints}",
                   icon=":material/lightbulb:")
    elif not quality.get("ok", True):
        tips = "\n".join(f"- {i['tip']}" for i in quality["issues"])
        st.warning(f"**Photo quality: the result may be less reliable.**\n\n{tips}", icon=":material/photo_camera:")


def _input_panel():
    with st.container(border=True):
        st.markdown("**1 · Photo**")
        mode = st.segmented_control("Photo source", ["Upload", "Camera", "Examples"], default="Upload", key="src_mode",
                                    required=True, label_visibility="collapsed")
        if mode == "Upload":
            f = st.file_uploader("Leaf photo", type=core.UPLOAD_TYPES, key="upload", label_visibility="collapsed",
                                 help="JPEG, PNG, WebP, BMP or TIFF up to 10 MB. iPhone HEIC photos: share or export "
                                      "them as JPEG first.")
            st.caption(PHOTO_TIPS)
            return (f.getvalue(), f.name) if f else (None, None)
        if mode == "Camera":
            st.caption(PHOTO_TIPS)
            shot = st.camera_input("Take a photo", key="camera", label_visibility="collapsed")
            return (shot.getvalue(), "Camera photo") if shot else (None, None)
        return _examples()


def _model_panel():
    with st.container(border=True):
        st.markdown("**2 · Model**")
        entries = core.served_models()
        key = st.segmented_control("Model", [e["key"] for e in entries], default=entries[0]["key"], key="model",
                                   format_func=lambda k: core.display_name(core.model_entry(k)), required=True,
                                   label_visibility="collapsed")
        entry = core.model_entry(key)
        st.markdown(f"**{core.benefit(entry)}**")
        st.caption(core.scope_text())
        st.caption(core.model_help(entry))
        acc = (entry.get("test_metrics") or {}).get("accuracy")
        st.html(theme.chip(core.short_name(entry), "g")
                + (theme.chip(f"{100 * acc:.1f}% on held-out test photos", "c") if acc is not None else ""))
        explain = not core.is_classical(entry) and st.toggle("Show where it looked (Grad-CAM)", value=True,
                                                             key="explain")
    return key, explain


# ----------------------------------------------------------------------------------------------------------- result
def _empty_state():
    with st.container(border=True):
        st.markdown("#### Add a photo to begin")
        st.markdown("Upload a photo, use your camera, or pick an **example** on the left. You will get:")
        st.markdown("- the most likely condition and a **calibrated** confidence\n"
                    "- **“Not sure”** when the evidence is weak, and **“Unusual image”** for photos unlike the training data\n"
                    "- a **Grad-CAM** heat map of the regions that drove the answer\n"
                    "- symptoms and management of the condition")
        st.caption("Photos are processed in memory for this session only and are not stored.")


@st.fragment
def _gradcam(r: dict):
    heat = r.get("heatmap_rgba")
    if heat is None:
        st.info((r.get("explanation") or {}).get("note") or "Grad-CAM was switched off for this run.")
        return
    opacity = st.slider("Heat-map strength", 0.0, 1.0, 0.7, 0.05, key="cam_opacity")
    a, b = st.columns(2)
    a.image(r["canonical"], caption="What the model sees (256 × 256)", width="stretch")
    b.image(core.blend(r["canonical"], heat, opacity), caption=f"Evidence for “{r['prediction']['name']}”",
            width="stretch")
    st.caption("Grad-CAM highlights the regions that most increased the score of the predicted class "
               + (f"({(r.get('explanation') or {}).get('note')}) " if (r.get("explanation") or {}).get("note") else "")
               + "— it shows where the network looked, not why the leaf is diseased.")


def _condition(key: str):
    info = core.disease_info().get(key)
    if not info:
        return
    st.markdown(f"#### {info['name']}")
    st.html(theme.chip(info.get("category", ""), "g" if info.get("is_healthy") else "r")
            + (theme.chip(info["cause"]) if info.get("cause") else ""))
    st.markdown(info["summary"])
    a, b = st.columns(2)
    with a:
        st.markdown("**Symptoms**")
        st.markdown("\n".join(f"- {s}" for s in info.get("symptoms", [])[:5]))
    with b:
        st.markdown("**What to do**")
        st.markdown("\n".join(f"- {s}" for s in info.get("management", [])[:5]))
    st.caption(core.disease_info().get("_disclaimer", ""))
    nav.link("library", "Open the disease library", icon=":material/local_florist:")


def _details(r: dict, label: str):
    entry = core.model_entry(r["model"]["key"])
    inp = r.get("input") or {}
    ood = r.get("ood")
    a, b = st.columns(2)
    with a:
        st.markdown("**Photo**")
        st.markdown(f"- Source: {theme.esc(label)}\n"
                    f"- Original size: {inp.get('width')} × {inp.get('height')} {inp.get('format') or ''}\n"
                    f"- Colour: {inp.get('colour', '—')}\n- Geometry: {inp.get('geometry', '—')}")
    with b:
        st.markdown("**Model**")
        gen = "version 2 (upgrade)" if entry.get("generation") == "upgrade" else "version 1 (original study)"
        runs = (f"{entry['pipeline']['classifier']} on handcrafted features" if core.is_classical(entry)
                else f"{len(entry.get('members') or [1])} network(s)")
        st.markdown(f"- {entry['name']} — {gen}\n- Runs: {runs}, {r['model'].get('family', '')}\n"
                    f"- Answer threshold: {core.pct(r['confidence_threshold'], 0)}\n"
                    f"- Analysis time on this server: {r['total_ms']:.0f} ms")
    if ood:
        st.markdown("**Unusual-image check**")
        ratio = float(ood["ratio"])
        st.progress(min(ratio / 2, 1.0), text=f"Distance to the training photos: {ratio:.2f} × the limit "
                                              f"({'over' if ratio > 1 else 'within'} the limit)")
        space = "colour + texture feature space" if core.is_classical(entry) else "network's feature space"
        st.caption(f"Measured in the {space} and divided by a limit chosen so that 95% of held-out training-like "
                   "photos stay below it.")
    report = {k: v for k, v in r.items() if k not in ("canonical", "heatmap_rgba", "explanation")}
    st.download_button("Download the report (JSON)", json.dumps(report, indent=2, default=float), "tomato-care-report.json",
                       "application/json", icon=":material/download:")


def _compare(data: bytes):
    keys = tuple(e["key"] for e in core.deployment()["models"])
    with st.spinner(f"Asking all {len(keys)} models…"):
        rows = core.compare(data, keys)
    table = []
    for r in rows:
        e = core.model_entry(r["model"]["key"])
        p = r["prediction"]
        table.append({"Model": core.short_name(e),
                      "Answer": p["name"] if p["status"] in ("Healthy", "Diseased") else p["status"],
                      "Top class": p["name"], "Confidence": p["confidence"], "Status": p["status"],
                      "Time (ms)": round(r["total_ms"])})
    df = pd.DataFrame(table)
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Answer": st.column_config.TextColumn(help="What the app would show: the condition, or Uncertain / Unusual image"),
    })
    agree = df["Top class"].nunique() == 1
    st.caption("Both models agree on the condition." if agree else
               f"The models disagree — a hint that this photo is hard. {core.display_name(core.served_models()[0])} "
               "is the more accurate one on the test photos (see Model lab).")


def _history_add(r: dict, label: str):
    hist = st.session_state.setdefault("history", [])
    item = {"thumb": np.asarray(Image.fromarray(r["canonical"]).resize((96, 96))), "status": r["prediction"]["status"],
            "name": r["prediction"]["name"], "conf": r["prediction"]["confidence"], "model": r["model"]["name"],
            "label": label}
    if not hist or (hist[0]["label"], hist[0]["model"]) != (label, item["model"]):
        hist.insert(0, item)
        del hist[6:]


def _history():
    hist = st.session_state.get("history") or []
    if len(hist) < 2:
        return
    with st.container(border=True):
        st.markdown("**Recent in this session**")
        cols = st.columns(3)
        for i, h in enumerate(hist[:6]):
            with cols[i % 3]:
                st.image(h["thumb"], width="stretch")
                answer = h["name"] if h["status"] in ("Healthy", "Diseased") else h["status"]
                st.caption(f"{answer} · {100 * h['conf']:.0f}%")


def _result(data: bytes, label: str, key: str, explain: bool):
    try:
        with st.spinner("Analysing the photo…"):
            r = core.predict(data, key, explain)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"This file could not be read as an image ({type(exc).__name__}). Please upload a JPEG or PNG photo.")
        return
    except Exception as exc:  # a server problem: say so instead of crashing the page
        st.error(f"The diagnosis failed on the server ({type(exc).__name__}: {exc}). Please try again in a moment.")
        return
    theme.status_card(r)
    _guidance(r)
    p = r["prediction"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", core.pct(p["confidence"]), border=True,
              help="Calibrated: among photos given this confidence, about this share were right (out-of-fold data).")
    c2.metric("Threshold", core.pct(r["confidence_threshold"], 0), border=True,
              help="Chosen on out-of-fold data: " + (r.get("operating_point") or {}).get("rule", "see the Model lab") + ".")
    c3.metric("Time", f"{r['total_ms'] / 1000:.1f} s", border=True,
              help="Including Grad-CAM (deep learning), on this server's CPU.")
    tabs = st.tabs(["Probabilities", "Where it looked", "The condition", "Details"])
    with tabs[0]:
        st.altair_chart(charts.probabilities(r["probabilities"], p["key"], r["confidence_threshold"]), width="stretch")
        st.caption("Bars: calibrated probability of each condition. Dashed line: this model's answer threshold.")
    with tabs[1]:
        _gradcam(r)
    with tabs[2]:
        if p["status"] == "Unusual image":
            st.info("No condition is shown for unusual images. Make sure the photo shows a single tomato leaf on plain "
                    "paper; the models were not trained on other crops, whole plants or field scenes.")
        else:
            _condition(p["key"])
    with tabs[3]:
        _details(r, label)
    _history_add(r, label)
    st.space("small")
    if st.toggle("Ask the other model too", key="compare_on",
                 help="Runs both Tomato Care models — " + " and ".join(core.display_name(e) for e in
                                                                        core.served_models()) + " — on this photo."):
        _compare(data)


def render() -> None:
    theme.section("Diagnose a leaf", "Photograph a single tomato leaf on plain paper. The photo is converted exactly "
                  "like the training photos before any model sees it.", kicker="Diagnose")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        data, label = _input_panel()
        key, explain = _model_panel()
    with right:
        if data is None:
            _empty_state()
        else:
            _result(data, label, key, explain)
    with left:
        _history()
