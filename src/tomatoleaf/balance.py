"""Class-bias correction and per-class calibration of a model's probabilities (post-hoc; the networks are unchanged).

Two steps, both fitted on out-of-fold predictions only, used in two different places:
1. Per-class calibration ("vector scaling", Guo et al. 2017): z'_c = a_c * log p_c + b_c, fitted by minimising the
   negative log-likelihood with an L2 penalty that keeps a near 1 and b near 0 (rare classes cannot pull it far).
   Each condition's stated confidence then matches how often that condition is actually right.
2. Prior (logit) adjustment (Menon et al. 2021, "Long-tail learning via logit adjustment"): divide by the class
   prior to the power tau. Frequent classes (Leaf Miner, Spotted Wilt) lose their head start, rare ones gain.
   It decides WHICH condition is answered; the confidence shown stays the calibrated probability of that answer
   (re-weighted probabilities would no longer match how often answers are right).
   tau is chosen by a rule fixed in code: the highest balanced accuracy (mean per-class recall) whose overall
   accuracy is at most MAX_ACC_DROP below the uncorrected model's, all on cross-fitted out-of-fold predictions.
"""
import numpy as np
from scipy.optimize import minimize

TAUS = np.round(np.arange(0.0, 1.51, 0.05), 2)
MAX_ACC_DROP = 0.02
L2 = 1e-2


def _log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-12, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def fit_vector_scaling(probs: np.ndarray, y: np.ndarray, l2: float = L2) -> dict:
    """Per-class scale a and bias b on log-probabilities, by penalised NLL. Returns {"a": [...], "b": [...]}."""
    k = probs.shape[1]
    z = _log(probs)

    def loss(theta):
        a, b = theta[:k], theta[k:]
        logits = z * a + b
        logits = logits - logits.max(axis=1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return -logp[np.arange(len(y)), y].mean() + l2 * (np.sum((a - 1) ** 2) + np.sum(b ** 2))

    res = minimize(loss, np.concatenate([np.ones(k), np.zeros(k)]), method="L-BFGS-B")
    return {"a": res.x[:k].tolist(), "b": res.x[k:].tolist()}


def apply(probs: np.ndarray, calibration: dict | None = None, prior=None, tau: float = 0.0) -> np.ndarray:
    """Calibrate (if given), then divide by prior**tau and renormalise. Works on one vector or a batch."""
    p = np.asarray(probs, dtype=np.float64)
    z = _log(p)
    if calibration:
        z = z * np.asarray(calibration["a"]) + np.asarray(calibration["b"])
    if prior is not None and tau:
        z = z - tau * np.log(np.asarray(prior, dtype=np.float64))
    return _softmax(z)


def decide(calibrated: np.ndarray, prior=None, tau: float = 0.0):
    """The answered class: argmax after the prior correction (1-D vector -> int, batch -> array)."""
    q = apply(calibrated, None, prior, tau) if (prior is not None and tau) else np.asarray(calibrated)
    return q.argmax(axis=-1)


def confidence_of(calibrated: np.ndarray, decision) -> np.ndarray:
    """Calibrated probability of the answered class."""
    calibrated = np.asarray(calibrated)
    if calibrated.ndim == 1:
        return calibrated[int(decision)]
    return calibrated[np.arange(len(calibrated)), np.asarray(decision)]


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    recalls = [np.mean(pred[y == c] == c) for c in range(k) if np.any(y == c)]
    return float(np.mean(recalls))


def classwise_ece(probs: np.ndarray, y: np.ndarray, pred=None) -> float:
    """Mean over answered classes of |mean confidence - accuracy|, weighted by how often each class is answered:
    0 means every condition's confidence matches its own accuracy. `pred` defaults to the argmax."""
    pred = probs.argmax(1) if pred is None else np.asarray(pred)
    conf = confidence_of(probs, pred)
    gaps, weights = [], []
    for c in np.unique(pred):
        m = pred == c
        gaps.append(abs(conf[m].mean() - (y[m] == c).mean()))
        weights.append(m.mean())
    return float(np.average(gaps, weights=weights))


def cross_fit_calibration(probs: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Fit the calibration on 4 CV folds and apply it to the 5th, for every fold: honest out-of-fold calibrated
    probabilities."""
    calibrated = np.zeros_like(probs, dtype=np.float64)
    for f in np.unique(folds):
        cal = fit_vector_scaling(probs[folds != f], y[folds != f])
        calibrated[folds == f] = apply(probs[folds == f], cal)
    return calibrated


def choose_tau(calibrated: np.ndarray, y: np.ndarray, prior, base_accuracy: float, taus=TAUS) -> float:
    """Highest balanced accuracy among taus whose accuracy is >= base_accuracy - MAX_ACC_DROP (ties: smaller tau)."""
    k = calibrated.shape[1]
    best_tau, best = 0.0, -1.0
    for tau in sorted(float(t) for t in taus):
        pred = decide(calibrated, prior, tau)
        if (pred == y).mean() < base_accuracy - MAX_ACC_DROP:
            continue
        bal = balanced_accuracy(y, pred, k)
        if bal > best + 1e-9:
            best_tau, best = tau, bal
    return best_tau
