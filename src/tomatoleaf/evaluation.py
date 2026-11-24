"""Metrics, confidence intervals and probability calibration."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import CLASSES, NUM_CLASSES

LABELS = list(range(NUM_CLASSES))


def classification_metrics(y_true, y_pred, labels=None) -> dict:
    labels = LABELS if labels is None else list(labels)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
    }


def per_class_metrics(y_true, y_pred, labels=None) -> list:
    labels = LABELS if labels is None else list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [{"key": CLASSES[c]["key"], "name": CLASSES[c]["name"], "precision": float(p[i]), "recall": float(r[i]),
             "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(labels)]


def confusion(y_true, y_pred, labels=None) -> list:
    return confusion_matrix(y_true, y_pred, labels=LABELS if labels is None else list(labels)).tolist()


def group_bootstrap_ci(y_true, y_pred, groups, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05, labels=None) -> dict:
    """95% percentile CI for accuracy and macro-F1, resampling whole leaf groups (not single images)."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=LABELS if labels is None else list(labels),
                            average="macro", zero_division=0))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"accuracy_ci": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
            "f1_macro_ci": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))]}


def paired_bootstrap_diff(y_true, pred_a, pred_b, groups, n_boot: int = 2000, seed: int = 42, labels=None) -> dict:
    """Macro-F1(a) - macro-F1(b) with a 95% CI, resampling the SAME leaf groups for both models (paired test)."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    lab = LABELS if labels is None else list(labels)
    f1 = lambda y, p: f1_score(y, p, labels=lab, average="macro", zero_division=0)  # noqa: E731
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        diffs.append(f1(y_true[idx], pred_a[idx]) - f1(y_true[idx], pred_b[idx]))
    diffs = np.asarray(diffs)
    return {"diff": float(f1(y_true, pred_a) - f1(y_true, pred_b)),
            "ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "share_of_resamples_a_not_better": float((diffs <= 0).mean())}


def paired_bootstrap(y_true, pred_a, pred_b, groups, n_boot: int = 2000, seed: int = 42, labels=None) -> dict:
    """Differences (a - b) in accuracy and macro-F1 with 95% CIs, resampling the SAME leaf groups for both models."""
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    lab = LABELS if labels is None else list(labels)
    f1 = lambda y, p: f1_score(y, p, labels=lab, average="macro", zero_division=0)  # noqa: E731
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    d_f1, d_acc = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        d_f1.append(f1(y_true[idx], pred_a[idx]) - f1(y_true[idx], pred_b[idx]))
        d_acc.append((pred_a[idx] == y_true[idx]).mean() - (pred_b[idx] == y_true[idx]).mean())
    d_f1, d_acc = np.asarray(d_f1), np.asarray(d_acc)
    return {"f1_diff": float(f1(y_true, pred_a) - f1(y_true, pred_b)),
            "f1_ci": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
            "accuracy_diff": float((pred_a == y_true).mean() - (pred_b == y_true).mean()),
            "accuracy_ci": [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
            "share_a_not_better_f1": float((d_f1 <= 0).mean())}


def brier_score(probs, y_true) -> float:
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def softmax(logits, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs, y_true, n_bins: int = 15) -> float:
    """Average |accuracy - confidence| over confidence bins, weighted by bin size."""
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs((pred[in_bin] == y_true[in_bin]).mean() - conf[in_bin].mean())
    return float(ece)


def reliability_bins(probs, y_true, n_bins: int = 10) -> list:
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            rows.append({"lo": float(lo), "hi": float(hi), "count": int(in_bin.sum()),
                         "confidence": float(conf[in_bin].mean()), "accuracy": float((pred[in_bin] == y_true[in_bin]).mean())})
    return rows


def fit_temperature(logits, y_true) -> float:
    """Temperature scaling: find T > 0 minimising validation negative log-likelihood of softmax(logits / T)."""
    logits_t = torch.tensor(np.asarray(logits), dtype=torch.float64)
    y_t = torch.tensor(np.asarray(y_true), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / log_t.exp(), y_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def selective_accuracy(probs, y_true, thresholds) -> list:
    """For each confidence threshold: share of images kept (coverage) and accuracy on the kept ones."""
    probs, y_true = np.asarray(probs), np.asarray(y_true)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    out = []
    for t in thresholds:
        keep = conf >= t
        out.append({"threshold": float(t), "coverage": float(keep.mean()),
                    "accuracy_kept": float((pred[keep] == y_true[keep]).mean()) if keep.any() else None,
                    "accuracy_rejected": float((pred[~keep] == y_true[~keep]).mean()) if (~keep).any() else None})
    return out
