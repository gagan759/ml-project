"""Per-class calibration and class-prior correction (tomatoleaf.balance)."""
import numpy as np

from tomatoleaf import balance


def _miscalibrated(n=3000, seed=0):
    """Three classes; the model is overconfident for class 0 and underconfident for class 2."""
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.6, 0.3, 0.1])
    logits = rng.normal(0, 1, (n, 3))
    logits[np.arange(n), y] += 1.5
    logits[:, 0] *= 1.8
    logits[:, 2] *= 0.6
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True), y


def _nll(p, y):
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def test_vector_scaling_lowers_nll_and_classwise_calibration_error():
    p, y = _miscalibrated()
    cal = balance.fit_vector_scaling(p[:2000], y[:2000])
    before, after = p[2000:], balance.apply(p[2000:], cal)
    assert _nll(after, y[2000:]) < _nll(before, y[2000:])
    assert balance.classwise_ece(after, y[2000:]) < balance.classwise_ece(before, y[2000:])
    np.testing.assert_allclose(after.sum(1), 1.0, atol=1e-9)


def test_identity_calibration_and_zero_tau_change_nothing():
    p, _ = _miscalibrated(200)
    same = balance.apply(p, {"a": [1, 1, 1], "b": [0, 0, 0]})
    np.testing.assert_allclose(same, p, atol=1e-9)
    np.testing.assert_array_equal(balance.decide(p, [0.6, 0.3, 0.1], 0.0), p.argmax(1))


def test_prior_correction_moves_close_calls_to_the_rare_class():
    probs = np.array([0.5, 0.1, 0.4])  # frequent class 0 narrowly ahead of rare class 2
    prior = [0.6, 0.3, 0.1]
    assert balance.decide(probs, prior, 0.0) == 0
    assert balance.decide(probs, prior, 1.0) == 2
    assert balance.confidence_of(probs, 2) == 0.4  # the confidence shown stays the calibrated probability


def test_prior_correction_raises_balanced_accuracy_on_imbalanced_data():
    p, y = _miscalibrated()
    prior = np.bincount(y, minlength=3) / len(y)
    plain = balance.balanced_accuracy(y, balance.decide(p, prior, 0.0), 3)
    tau = balance.choose_tau(p, y, prior, base_accuracy=float((p.argmax(1) == y).mean()))
    assert balance.balanced_accuracy(y, balance.decide(p, prior, tau), 3) >= plain
    assert (balance.decide(p, prior, tau) == y).mean() >= (p.argmax(1) == y).mean() - balance.MAX_ACC_DROP - 1e-9
