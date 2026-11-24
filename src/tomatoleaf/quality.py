"""Photo quality checks shown with every diagnosis (they never change the model's answer).

Each upload is measured on the same 256x256 canonical image the models see:
- sharpness: variance of the Laplacian;
- brightness: mean grey level, and highlights (99th percentile grey level) for over-exposure;
- the leaf: the paper colour is the median colour of the image border; leaf pixels are those far from it in CIELAB
  (Otsu threshold on the distance), so plain paper of any colour works. Measured: contrast between leaf and paper
  (no leaf -> low), the largest region's share of the frame (leaf too small), the second-largest region's share
  (several leaves), and the grey-level spread of the paper (busy background).
The limits are percentiles of the same measures on the TRAINING + VALIDATION photos (scripts/41_quality_limits.py ->
experiments/quality_limits.json), so a typical dataset photo passes every check; a failed check becomes a specific tip
("the photo looks blurry", "move closer", ...).
"""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import EXPERIMENTS_DIR

LIMITS_FILE = EXPERIMENTS_DIR / "quality_limits.json"
LOW, HIGH = 0.5, 99.5  # percentiles of the training photos used as limits (about 1 in 100 genuine photos per check)
BAND = 8  # border band (pixels) whose median colour is taken as the paper colour
_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

# (measure, side, issue code): "low" = flagged below the low limit, "high" = flagged above the high limit
CHECKS = [
    ("leaf_contrast", "low", "no_leaf"),
    ("sharpness", "low", "blurry"),
    ("brightness", "low", "dark"),
    ("brightness", "high", "bright"),
    ("highlights", "high", "bright"),
    ("leaf_fraction", "low", "small_leaf"),
    ("second_leaf", "high", "several_leaves"),
    ("background_texture", "high", "busy_background"),
]
TIPS = {
    "no_leaf": "No leaf stands out from the background. Photograph one tomato leaf on plain, light paper.",
    "blurry": "The photo looks blurry. Hold the phone steady and tap the leaf to focus.",
    "dark": "The photo is too dark. Use daylight or a bright room.",
    "bright": "The photo is over-exposed. Avoid direct sun, flash and glare.",
    "small_leaf": "The leaf is small in the frame. Move closer so it fills most of the photo.",
    "several_leaves": "There seem to be several leaves. Photograph one leaf at a time.",
    "busy_background": "The background is not plain. Put the leaf on plain paper.",
}
NEEDS_LEAF = {"small_leaf", "several_leaves", "busy_background"}  # meaningless when no leaf stands out


def segment(rgb: np.ndarray):
    """-> (CIELAB distance of every pixel to the paper colour, uint8 mask of leaf-like regions)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    band = np.concatenate([lab[:BAND].reshape(-1, 3), lab[-BAND:].reshape(-1, 3),
                           lab[:, :BAND].reshape(-1, 3), lab[:, -BAND:].reshape(-1, 3)])
    dist = np.linalg.norm(lab - np.median(band, axis=0), axis=2)
    d8 = np.clip(dist, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(d8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN)
    return dist, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE)


def measures(rgb: np.ndarray) -> dict:
    """Quality measures of a canonical 256x256x3 uint8 RGB image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dist, mask = segment(rgb)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = np.sort(stats[1:, cv2.CC_STAT_AREA])[::-1] / gray.size
    leaf = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if n > 1 else np.zeros(gray.shape, bool)
    paper = mask == 0
    return {"sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "brightness": float(gray.mean()),
            "highlights": float(np.percentile(gray, 99)),
            "leaf_contrast": float(dist[leaf].mean() - dist[paper].mean()) if leaf.any() and paper.any() else 0.0,
            "leaf_fraction": float(areas[0]) if len(areas) else 0.0,
            "second_leaf": float(areas[1]) if len(areas) > 1 else 0.0,
            "background_texture": float(gray[paper].std()) if paper.any() else 0.0}


@lru_cache(maxsize=1)
def limits(path: Path = LIMITS_FILE) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8"))["limits"] if path.exists() else None


def assess(rgb: np.ndarray, lim: dict | None = None) -> dict:
    """{'ok', 'issues': [{'code', 'tip', 'measure', 'value', 'limit'}], 'measures'} for one canonical image."""
    m = measures(rgb)
    lim = lim if lim is not None else limits()
    issues = []
    for name, side, code in CHECKS if lim else []:
        limit = lim[name][side]
        failed = m[name] < limit if side == "low" else m[name] > limit
        no_leaf = any(i["code"] == "no_leaf" for i in issues)
        if failed and not (no_leaf and code in NEEDS_LEAF) and all(i["code"] != code for i in issues):
            issues.append({"code": code, "tip": TIPS[code], "measure": name, "value": m[name], "limit": limit})
    return {"ok": not issues, "issues": issues, "measures": m}
