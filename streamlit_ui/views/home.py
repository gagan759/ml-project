"""Home: what Tomato Care does, its headline test results and what is inside."""
import base64
import io

import numpy as np
import streamlit as st
from PIL import Image

from .. import core, nav, theme


def _hero_figure(hero: dict | None) -> str:
    if not hero:
        return ""
    rgb = np.asarray(Image.open(core.PUBLIC / hero["input"].lstrip("/")).convert("RGB"))
    heat = np.asarray(Image.open(core.PUBLIC / hero["heatmap"].lstrip("/")).convert("RGBA"))
    composite = Image.fromarray(core.blend(rgb, heat, 0.75)).resize((520, 520), Image.LANCZOS)
    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=88)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    name = core.CLASS_NAME.get(hero["pred_key"], hero["pred_key"])
    return (f'<div class="tc-hero-figure"><img src="{src}" alt="Grad-CAM heat map on a tomato leaf">'
            f'<div class="tc-hero-badge"><div><b>{theme.esc(name)}</b><br><span style="opacity:.75">Held-out test photo · '
            f'Grad-CAM of the deep-learning model</span></div><div class="p">{100 * hero["confidence"]:.0f}%</div></div></div>')


def render() -> None:
    data = core.platform()
    new5, _ = core.task_defaults()
    ml = core.classical_result()
    dl_name = core.display_name(core.served_models()[0])
    ml_name = core.display_name(next((e for e in core.served_models() if core.is_classical(e)), None))

    with st.container(key="hero"):
        left, right = st.columns([1.25, 1], gap="large", vertical_alignment="center")
        with left:
            st.html('<span class="tc-eyebrow">Tomato leaf diagnosis · version 2</span>'
                    '<h1 class="tc-hero-title">Healthy leaf or a problem?<br><span>Know from one photo.</span></h1>'
                    '<p class="tc-hero-sub">Tomato Care recognises diseases, a pest and nutrient deficiencies on tomato '
                    'leaves. It tells you how sure it is, says <b>“not sure”</b> instead of guessing, flags photos unlike '
                    'anything it was trained on, and shows where it looked.</p>')
            st.space("small")
            with st.container(horizontal=True):
                if st.button("Diagnose a leaf", type="primary", icon=":material/photo_camera:"):
                    nav.go("diagnose")
                if st.button("See the evidence", icon=":material/monitoring:"):
                    nav.go("lab")
        with right:
            st.html(_hero_figure(data.get("hero")))

    st.space("medium")
    theme.section("Measured, not claimed",
                  f"Every number below comes from the {data['tasks'][core.TASK]['test_photos']} held-out test photos of the "
                  "five conditions, "
                  "scored once after the models had been chosen on cross-validation data. Brackets: 95% confidence "
                  "intervals.",
                  kicker="Results")
    c1, c2, c3, c4 = st.columns(4)
    ci5 = new5["test"]["accuracy_ci"]
    c1.metric(f"{dl_name} · accuracy", core.pct(new5["test"]["accuracy"]),
              delta=f"[{core.pct(ci5[0])} – {core.pct(ci5[1])}]", delta_color="off", border=True,
              help=f"Deep learning. {core.scope_text()} Macro-F1 {new5['test']['f1_macro']:.3f}.")
    sel = new5["selective"]
    c2.metric(f"{dl_name} · when it answers", core.pct(sel["accuracy_kept"]),
              delta=f"answers {core.pct(sel['coverage'], 0)} of photos", delta_color="off", border=True,
              help="Below its confidence threshold, or for an unusual photo, the app says 'not sure' instead of "
                   "guessing; this is its accuracy on the photos it does answer.")
    if ml:
        ci = ml["test"]["accuracy_ci"]
        c3.metric(f"{ml_name} · accuracy", core.pct(ml["test"]["accuracy"]),
                  delta=f"[{core.pct(ci[0])} – {core.pct(ci[1])}]", delta_color="off", border=True,
                  help=f"Classical ML: handcrafted colour / texture features, no neural network; macro-F1 "
                       f"{ml['test']['f1_macro']:.3f}. Same photos, same rules.")
    fr5 = ((data.get("final_round") or {}).get("tasks") or {}).get("5class") or {}
    od = (fr5.get("out_of_dataset") or {}).get(new5["key"]) or {}
    if "everyday" in od:  # final round: real unrelated photos (scripts/37-38)
        c4.metric("Unrelated photos refused", core.pct(od["everyday"]["refused"], 0),
                  delta=f"{core.pct(new5.get('flagged_unusual') or 0, 0)} of genuine leaf photos flagged too",
                  delta_color="off", border=True,
                  help="Everyday photos (Lorem Picsum) answered with 'Unusual image' or 'not sure' instead of a "
                       "diagnosis — and the price: genuine test leaf photos sent there as well.")
    else:
        ood = data["ood"]["test"]
        c4.metric("Non-leaf photos caught", core.pct(ood["flagged_far"], 0),
                  delta=f"{core.pct(ood['flagged_in'], 0)} of normal photos flagged too", delta_color="off",
                  border=True,
                  help="Held-out non-leaf probe images (photos, textures, drawings) answered with 'Unusual image' "
                       "instead of a diagnosis — and the price: normal leaf photos sent there as well.")
    bal = new5["test"].get("balanced_accuracy")
    st.caption((f"{dl_name}: balanced accuracy {core.pct(bal)} — the mean of the per-condition "
                "recalls, so rare conditions count as much as common ones. " if bal else "")
               + "How these models were chosen, how they compare with the original study's models and where they fail: "
                 "see the Model lab and How it works.")

    st.space("medium")
    theme.section("What is inside", kicker="Tomato Care")
    rows = [
        [("tune", "Sees uploads the way it was trained",
          "The dataset photos are whole 3:4 phone frames squashed to 256×256 in Display P3 colour. Every upload goes "
          "through exactly the same steps before a model sees it."),
         ("account_tree", "Chosen by cross-validation",
          "Five-fold cross-validation on 1,813 photos, leaf groups never split, and a selection rule fixed before "
          "the test photos were scored."),
         ("groups", "Fair to rare conditions",
          "One imbalance correction per model (class-weighted loss for the network, class-balanced weights for the "
          "classical model), and calibrated confidence for every condition.")],
        [("help", "Says “not sure”",
          "Calibrated confidence and one threshold per model, chosen on cross-validation data: below it, no "
          "diagnosis. The deep-learning model also learned to be unsure about conditions outside the five."),
         ("image_not_supported", "Flags unusual photos",
          "Feature-space distance to the training photos, with a limit per condition so rare conditions are not "
          "flagged more often. Tested on other crops, everyday photos and stress tests (Model lab)."),
         ("visibility", "Shows where it looked",
          "Grad-CAM heat maps for every diagnosis, so you can check the model looked at the leaf.")],
    ]
    for row in rows:
        cols = st.columns(3)
        for col, (icon, title, body) in zip(cols, row):
            with col:
                theme.feature(icon, title, body)

    st.space("medium")
    theme.section("How it works", kicker="Three steps")
    s1, s2, s3 = st.columns(3)
    for col, (n, title, body) in zip((s1, s2, s3), [
        (1, "Photograph one leaf", "On plain paper, in daylight, leaf filling most of the frame — like the training photos."),
        (2, "Same steps as training", "EXIF rotation, colour into Display P3, the whole frame to 256×256, JPEG quality 90."),
        (3, "A calibrated answer", "Five networks vote; you get the condition and its confidence, or an honest “not sure”."),
    ]):
        with col:
            st.html(f'<span class="tc-step">{n}</span><b>{title}</b>')
            st.caption(body)
    st.space("small")
    with st.container(horizontal=True):
        nav.link("diagnose", "Try it on an example photo", icon=":material/arrow_forward:")
        nav.link("method", "Read the methodology", icon=":material/menu_book:")
