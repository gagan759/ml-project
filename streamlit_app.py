"""Tomato Care — Streamlit app: tomato leaf disease diagnosis with calibrated confidence, an unusual-image check and
Grad-CAM explanations, plus the evidence behind every number.

    streamlit run streamlit_app.py

On Streamlit Community Cloud this file is the entry point; requirements.txt installs CPU-only PyTorch and the model
weights are downloaded from the GitHub release on first start (SHA-256 checked).
"""
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Tomato Care — Tomato Leaf Disease Diagnosis",
    page_icon=Image.open(ROOT / "assets" / "brand" / "mark-small-64.png"),
    layout="wide",
    menu_items={"Get help": "https://github.com/CodewithBhaumik/Tomato-Care#readme",
                "Report a bug": "https://github.com/CodewithBhaumik/Tomato-Care/issues",
                "About": "**Tomato Care** — tomato leaf disease diagnosis from a single photo. Research prototype, "
                         "not a substitute for an agronomist."},
)

from streamlit_ui import core, nav, theme  # noqa: E402  (after set_page_config)
from streamlit_ui.views import diagnose, home, lab, library, method  # noqa: E402

theme.inject()
dark = getattr(st.context.theme, "type", None) == "dark"
st.logo(str(core.BRAND / ("logo-dark.png" if dark else "logo-light.png")), size="large")

core.ensure_weights()

nav.PAGES.update({
    "home": st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
    "diagnose": st.Page(diagnose.render, title="Diagnose", icon=":material/photo_camera:", url_path="diagnose"),
    "lab": st.Page(lab.render, title="Model lab", icon=":material/science:", url_path="lab"),
    "library": st.Page(library.render, title="Disease library", icon=":material/local_florist:", url_path="library"),
    "method": st.Page(method.render, title="How it works", icon=":material/menu_book:", url_path="how-it-works"),
})
page = st.navigation(list(nav.PAGES.values()), position="top")
page.run()
theme.footer()
