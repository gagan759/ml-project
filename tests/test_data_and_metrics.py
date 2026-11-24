"""Split integrity (no leaf group on two sides), metric helpers, calibration and OOD scores."""
import numpy as np
import pandas as pd
import pytest

from tomatoleaf.config import DATA_DIR
from tomatoleaf.evaluation import (
    brier_score,
    expected_calibration_error,
    fit_temperature,
    paired_bootstrap,
    selective_accuracy,
    softmax,
)
from tomatoleaf.ood import FeatureDensity, auroc, fpr_at_tpr


def test_clean_split_has_no_group_or_photo_leakage():
    s = pd.read_csv(DATA_DIR / "splits.csv")
    for col in ("group_id", "photo_id"):
        assert s.groupby(col).split.nunique().max() == 1, col
    assert set(s.split) == {"train", "val", "test"}


def test_cv_folds_cover_train_val_only_and_keep_groups_together():
    path = DATA_DIR / "cv_folds.csv"
    if not path.exists():
        pytest.skip("run scripts/20_make_cv_folds.py")
    folds = pd.read_csv(path)
    s = pd.read_csv(DATA_DIR / "splits.csv")
    assert set(folds.photo_id) == set(s[s.split.isin(["train", "val"])].photo_id)
    assert not set(folds.photo_id) & set(s[s.split == "test"].photo_id)
    assert folds.groupby("group_id").fold.nunique().max() == 1
    assert sorted(folds.fold.unique()) == [0, 1, 2, 3, 4]
    assert (folds.groupby("fold").label.nunique() == 8).all()  # every class held out in every fold


def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, (4000, 5))
    y = np.array([rng.choice(5, p=softmax(row[None])[0]) for row in true_logits])
    t = fit_temperature(true_logits * 2.0, y)  # logits twice too large -> T close to 2
    assert 1.8 < t < 2.2


def test_ece_is_zero_for_perfectly_calibrated_bins_and_brier_bounds():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]] * 50)
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(probs, y) == pytest.approx(0.0)
    assert brier_score(probs, y) == pytest.approx(0.0)
    assert brier_score(probs, 1 - y) == pytest.approx(2.0)


def test_selective_accuracy_coverage():
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.55, 0.45], [0.2, 0.8]])
    y = np.array([0, 1, 0, 1])
    row = selective_accuracy(probs, y, [0.7])[0]
    assert row["coverage"] == 0.5 and row["accuracy_kept"] == 1.0


def test_paired_bootstrap_identical_models_have_zero_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 200)
    pred = np.where(rng.random(200) < 0.8, y, (y + 1) % 3)
    groups = np.arange(200) // 2
    out = paired_bootstrap(y, pred, pred, groups, n_boot=200, labels=[0, 1, 2])
    assert out["f1_diff"] == 0 and out["f1_ci"] == [0.0, 0.0] and out["accuracy_diff"] == 0


def test_feature_density_scores_far_points_higher_and_roundtrips():
    rng = np.random.default_rng(2)
    feats = np.r_[rng.normal(0, 1, (200, 16)) + 3, rng.normal(0, 1, (200, 16)) - 3]
    labels = np.r_[np.zeros(200, int), np.ones(200, int)]
    d = FeatureDensity(feats, labels)
    near = rng.normal(0, 1, (50, 16)) + 3
    far = rng.normal(0, 1, (50, 16)) * 6  # random directions (cosine kNN ignores magnitude)
    assert auroc(d.mahalanobis(near), d.mahalanobis(far)) > 0.99
    assert auroc(d.knn(near), d.knn(far)) > 0.95
    assert fpr_at_tpr(d.mahalanobis(near), d.mahalanobis(far)) < 0.05
    d2 = FeatureDensity.from_dict(d.to_dict())
    np.testing.assert_allclose(d2.mahalanobis(near), d.mahalanobis(near), rtol=1e-4)
