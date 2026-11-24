"""Low-memory serving mode (TOMATO_LOWMEM=1, used by the Streamlit app): fold models share one float32 network and
copy their memory-mapped weights in before running. It must give exactly the predictions of the standard path."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "experiments" / "deployment.json"


def _default_entry():
    if not DEPLOYMENT.exists():
        return None
    dep = json.loads(DEPLOYMENT.read_text(encoding="utf-8"))
    entry = next(e for e in dep["models"] if e["key"] == dep["default_model"])
    present = all((ROOT / "experiments" / "models" / m["path"]).exists() for m in entry["members"])
    return entry if present else None


pytestmark = pytest.mark.skipif(_default_entry() is None, reason="model weights not downloaded")


def test_shared_network_mode_gives_identical_predictions_and_explanations(monkeypatch):
    from backend.services.predictor import EnsemblePredictor
    from tomatoleaf.preprocessing import prepare_upload

    entry = _default_entry()
    rgb = prepare_upload(ROOT / "frontend" / "public" / "samples" / "leaf_miner-1.jpg")[0]
    monkeypatch.setenv("TOMATO_RUNTIME", "torch")
    monkeypatch.setenv("TOMATO_LOWMEM", "0")
    standard = EnsemblePredictor(entry, torch.device("cpu"))
    monkeypatch.setenv("TOMATO_LOWMEM", "1")
    shared = EnsemblePredictor(entry, torch.device("cpu"))
    assert all(m.model is None and m.runtime == "pytorch-shared" for m in shared.members)
    assert len({id(m._shared) for m in shared.members}) == 1  # one network for all fold models

    p_std, r_std = standard.analyse(rgb)
    p_low, r_low = shared.analyse(rgb)
    np.testing.assert_array_equal(p_std, p_low)
    assert r_std == r_low
    top = int(np.argmax(p_low))
    assert shared.explain(rgb, top)["heatmap"] == standard.explain(rgb, top)["heatmap"]
    np.testing.assert_array_equal(shared.analyse(rgb)[0], p_low)  # Grad-CAM leaves the weights untouched
