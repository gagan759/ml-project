"""Uncertainty and out-of-distribution (OOD) scores. Higher score = more unusual input.

- energy: -logsumexp(logits) (Liu et al., 2020)
- maha:   minimum class-conditional Mahalanobis distance of the penultimate features, shared covariance with
          Ledoit-Wolf shrinkage (Lee et al., 2018)
- knn:    cosine distance to the k-th nearest training feature (Sun et al., 2022)
The densities are fitted on TRAINING features only; thresholds are chosen on out-of-fold / validation scores.
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score


class FeatureDensity:
    def __init__(self, features: np.ndarray, labels: np.ndarray, k: int = 5):
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels)
        self.classes = np.unique(labels)
        self.means = np.stack([features[labels == c].mean(0) for c in self.classes])
        centred = features - self.means[np.searchsorted(self.classes, labels)]
        self.precision = np.linalg.inv(LedoitWolf().fit(centred).covariance_)
        self.bank = features / np.linalg.norm(features, axis=1, keepdims=True)
        self.k = k

    def mahalanobis(self, f: np.ndarray) -> np.ndarray:
        f = np.atleast_2d(np.asarray(f, dtype=np.float64))
        d = f[:, None, :] - self.means[None, :, :]
        return np.einsum("nkd,de,nke->nk", d, self.precision, d).min(axis=1)

    def knn(self, f: np.ndarray) -> np.ndarray:
        f = np.atleast_2d(np.asarray(f, dtype=np.float64))
        f = f / np.linalg.norm(f, axis=1, keepdims=True)
        sims = f @ self.bank.T
        kth = np.sort(sims, axis=1)[:, -self.k]
        return 1.0 - kth

    def to_dict(self) -> dict:
        """Tensors and plain Python values only, so checkpoints still load with torch.load(weights_only=True)."""
        import torch

        return {"classes": [int(c) for c in self.classes], "k": int(self.k),
                "means": torch.from_numpy(self.means.astype(np.float32)),
                "precision": torch.from_numpy(self.precision.astype(np.float32)),
                "bank": torch.from_numpy(self.bank.astype(np.float16))}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureDensity":
        def arr(x):
            return (x.float().numpy() if hasattr(x, "numpy") else np.asarray(x)).astype(np.float64)

        obj = cls.__new__(cls)
        obj.classes = np.asarray(d["classes"])
        obj.means, obj.precision, obj.bank = arr(d["means"]), arr(d["precision"]), arr(d["bank"])
        obj.k = int(d["k"])
        return obj


def energy_score(logits: np.ndarray) -> np.ndarray:
    return -logsumexp(np.asarray(logits, dtype=np.float64), axis=1)


def auroc(in_scores, out_scores) -> float:
    """AUROC for separating OOD (positive) from in-distribution inputs; 0.5 = chance."""
    y = np.r_[np.zeros(len(in_scores)), np.ones(len(out_scores))]
    return float(roc_auc_score(y, np.r_[in_scores, out_scores]))


def fpr_at_tpr(in_scores, out_scores, tpr: float = 0.95) -> float:
    """Share of in-distribution inputs flagged when the threshold catches `tpr` of the OOD inputs."""
    thr = np.quantile(np.asarray(out_scores), 1 - tpr)
    return float((np.asarray(in_scores) >= thr).mean())

