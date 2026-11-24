"""Handcrafted features of the classical model: shapes, determinism, leaf-only texture, and the served pipeline."""
from pathlib import Path

import numpy as np
import pytest

from tomatoleaf.config import EXPERIMENTS_DIR, MODELS_DIR
from tomatoleaf.features import (
    COLOR_DIM,
    FEATURE_SETS,
    HOG_DIM,
    TEXTURE_DIM,
    TEXTURE_FEATURE_NAMES,
    extract_features,
    feature_vector,
    leaf_mask,
    texture_features,
)
from tomatoleaf.preprocessing import load_canonical_array
from tomatoleaf.utils import load_json

SAMPLE = Path(__file__).resolve().parents[1] / "frontend" / "public" / "test-samples" / "leaf_miner.jpg"


@pytest.fixture(scope="module")
def leaf():
    return load_canonical_array(SAMPLE)


def test_feature_blocks_have_their_documented_sizes(leaf):
    f = extract_features(leaf)
    assert (f["hog"].shape, f["color"].shape, f["texture"].shape) == ((HOG_DIM,), (COLOR_DIM,), (TEXTURE_DIM,))
    assert len(TEXTURE_FEATURE_NAMES) == TEXTURE_DIM == 43
    assert feature_vector(f, "Color + Texture").shape == (COLOR_DIM + TEXTURE_DIM,)
    assert all(np.isfinite(f[k]).all() for k in f)
    assert np.array_equal(extract_features(leaf)["texture"], f["texture"])  # deterministic


def test_texture_ignores_the_background(leaf):
    """Repainting the paper around the leaf must not change the leaf's texture (pairs touching it are dropped)."""
    mask = leaf_mask(leaf)
    other = leaf.copy()
    other[~mask] = 255 - other[~mask]
    lbp = sum(p + 2 for p in (8, 16))  # LBP bins; the GLCM part uses only leaf-leaf pixel pairs
    a, b = texture_features(leaf, mask), texture_features(other, mask)
    np.testing.assert_allclose(a[lbp:], b[lbp:], rtol=1e-6)


def test_lbp_histograms_are_distributions(leaf):
    t = texture_features(leaf, leaf_mask(leaf))
    assert t[:10].sum() == pytest.approx(1.0) and t[10:28].sum() == pytest.approx(1.0)


def test_feature_sets_reference_existing_blocks():
    assert set().union(*FEATURE_SETS.values()) == {"hog", "color", "texture"}


def test_served_classical_model_is_calibrated_and_gated(leaf):
    entry = next((m for m in load_json(EXPERIMENTS_DIR / "deployment.json")["models"] if m.get("type") == "sklearn"),
                 None)
    if entry is None or not (MODELS_DIR / entry["path"]).exists():
        pytest.skip("classical model not built (scripts/43_classical_model.py)")
    from backend.services.predictor import SklearnPredictor

    pred = SklearnPredictor(entry)
    probs, ratio = pred.analyse(leaf)
    assert probs.shape == (5,) and probs.sum() == pytest.approx(1.0) and ratio > 0
    noise = np.random.default_rng(0).integers(0, 255, leaf.shape, dtype=np.uint8)
    assert pred.analyse(noise)[1] > ratio  # random noise is further from the training photos than a dataset leaf
