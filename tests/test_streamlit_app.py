"""The Streamlit app: every page renders without an exception; with the weights present, a diagnosis runs end to end.

Weights are never downloaded here (TOMATO_DOWNLOAD_WEIGHTS=0); the browser check (scripts/30_streamlit_check.py)
covers the real browser, both themes and phone width.
"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "experiments" / "platform.json").exists(),
                                reason="experiments/platform.json not exported")


def _run_view(view: str, example: str | None = None):
    """Registers the app's pages like streamlit_app.py, then renders one of them."""
    import streamlit as st

    from streamlit_ui import nav, theme
    from streamlit_ui.views import diagnose, home, lab, library, method

    views = {"home": home, "diagnose": diagnose, "lab": lab, "library": library, "method": method}
    nav.PAGES.clear()
    nav.PAGES.update({k: st.Page(v.render, title=k, url_path=k, default=k == "home") for k, v in views.items()})
    st.navigation(list(nav.PAGES.values()), position="hidden")
    if example and "example" not in st.session_state:
        st.session_state["example"] = example
    theme.inject()
    views[view].render()


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setenv("TOMATO_DOWNLOAD_WEIGHTS", "0")


@pytest.mark.parametrize("view, metric", [("home", "ConvNeXt-Tiny ensemble · accuracy"), ("lab", "Served model · accuracy"),
                                          ("library", None), ("method", "Non-leaf images flagged"),
                                          ("diagnose", None)])
def test_page_renders_without_exception(view, metric):
    at = AppTest.from_function(_run_view, kwargs={"view": view}, default_timeout=120).run()
    assert not at.exception, at.exception[0].value if at.exception else ""
    if metric:
        assert metric in [m.label for m in at.metric]


def test_numbers_on_the_home_page_come_from_the_benchmark():
    """Both served models' test accuracy, labelled with their model names."""
    platform = json.loads((ROOT / "experiments" / "platform.json").read_text(encoding="utf-8"))
    at = AppTest.from_function(_run_view, kwargs={"view": "home"}, default_timeout=120).run()
    shown = {m.label: m.value for m in at.metric}
    for m in (m for m in platform["models"] if m["deployed"]):
        assert shown[f"{m['model_name']} · accuracy"] == f"{100 * m['test']['accuracy']:.1f}%"


def _weights_present() -> bool:
    dep = json.loads((ROOT / "experiments" / "deployment.json").read_text(encoding="utf-8"))
    entry = next(e for e in dep["models"] if e["key"] == dep["default_model"])
    return all((ROOT / "experiments" / "models" / m["path"]).exists() for m in entry["members"])


@pytest.mark.skipif(not _weights_present(), reason="model weights not downloaded")
def test_example_photo_is_diagnosed_end_to_end():
    example = str(ROOT / "frontend" / "public" / "test-samples" / "leaf_miner.jpg")
    at = AppTest.from_function(_run_view, kwargs={"view": "diagnose", "example": example}, default_timeout=300)
    at.run()
    at.segmented_control(key="src_mode").set_value("Examples").run()
    assert not at.exception, at.exception[0].value if at.exception else ""
    labels = {m.label: m.value for m in at.metric}
    assert labels["Confidence"].endswith("%") and 0 < float(labels["Confidence"][:-1]) <= 100
    assert labels["Threshold"].endswith("%")


@pytest.mark.skipif(not _weights_present(), reason="model weights not downloaded")
def test_refusal_says_why_and_lists_candidates(tmp_path):
    import numpy as np
    from PIL import Image

    example = tmp_path / "noise.jpg"  # nothing like a leaf on paper: refused as an unusual image
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 300, 3), dtype=np.uint8)).save(example)
    at = AppTest.from_function(_run_view, kwargs={"view": "diagnose", "example": str(example)}, default_timeout=300)
    at.run()
    at.segmented_control(key="src_mode").set_value("Examples").run()
    assert not at.exception, at.exception[0].value if at.exception else ""
    why = [w.value for w in at.warning if "Why no diagnosis" in w.value]
    assert why and "Most likely:" in why[0] and "does not cover" in why[0]


def test_cached_deployment_follows_file_updates(tmp_path, monkeypatch):
    """A git push updates deployment.json under a running app: the cached model list must follow it (the live app
    kept offering a removed model until a full restart)."""
    import os

    from streamlit_ui import core

    dep = tmp_path / "deployment.json"
    dep.write_text(json.dumps({"models": [{"key": "a"}, {"key": "b"}]}), encoding="utf-8")
    monkeypatch.setattr(core, "DEPLOYMENT_FILE", dep)
    assert [m["key"] for m in core.deployment()["models"]] == ["a", "b"]
    dep.write_text(json.dumps({"models": [{"key": "a"}]}), encoding="utf-8")
    st_ = dep.stat()
    os.utime(dep, ns=(st_.st_atime_ns, st_.st_mtime_ns + 1_000_000_000))  # a later modification time, as after a pull
    assert [m["key"] for m in core.deployment()["models"]] == ["a"]
