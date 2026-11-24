"""Photo quality checks: dataset photos pass, damaged photos get the matching tip; refusal reasons."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.services.predictor import refusal_reason
from tomatoleaf.config import CLASS_KEYS
from tomatoleaf.preprocessing import load_canonical_array
from tomatoleaf.quality import TIPS, assess, limits

SAMPLES = Path(__file__).resolve().parents[1] / "frontend" / "public" / "test-samples"
FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]

pytestmark = pytest.mark.skipif(limits() is None, reason="run scripts/41_quality_limits.py first")


@pytest.fixture(scope="module")
def leaf():
    return load_canonical_array(SAMPLES / "leaf_miner.jpg")


def codes(rgb):
    return {i["code"] for i in assess(rgb)["issues"]}


def test_dataset_photo_passes(leaf):
    assert assess(leaf)["ok"]


def test_damaged_photos_get_the_matching_tip(leaf):
    assert "blurry" in codes(cv2.GaussianBlur(leaf, (0, 0), 4))
    assert "dark" in codes((leaf * 0.35).astype(np.uint8))
    assert "bright" in codes(np.clip(leaf.astype(np.float32) * 1.7, 0, 255).astype(np.uint8))
    assert "no_leaf" in codes(np.full_like(leaf, 235))
    small = cv2.resize(leaf, (128, 128))
    four = np.vstack([np.hstack([small, small]), np.hstack([small, small])])
    assert "several_leaves" in codes(four)


def test_every_issue_has_a_tip():
    from tomatoleaf.quality import CHECKS

    assert {c[2] for c in CHECKS} | {"no_leaf"} == set(TIPS)


def test_refusal_reason():
    ok = {"ok": True, "issues": []}
    assert refusal_reason("Diseased", 0.95, 0.81, 0.5, ok, FIVE) is None
    low = refusal_reason("Uncertain", 0.42, 0.81, 0.5, ok, FIVE)
    assert low["code"] == "low_confidence" and "42%" in low["message"] and "81%" in low["message"]
    assert any("does not cover" in h and "Early Blight" in h for h in low["hints"])  # unsupported, from data
    full = refusal_reason("Unusual image", 0.6, 0.76, 1.4, ok, CLASS_KEYS)
    assert full["code"] == "unusual_image" and not any("does not cover" in h for h in full["hints"])
    bad = {"ok": False, "issues": [{"code": "blurry", "tip": TIPS["blurry"]}]}
    assert TIPS["blurry"] in refusal_reason("Uncertain", 0.5, 0.81, 0.5, bad, CLASS_KEYS)["hints"]
