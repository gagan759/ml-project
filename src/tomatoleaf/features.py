"""Handcrafted features for the classical ML models: leaf mask, HOG (shape), colour statistics and texture (LBP, GLCM)."""
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern

HOG_IMAGE_SIZE = 128
HOG_PARAMS = {"orientations": 9, "pixels_per_cell": (16, 16), "cells_per_block": (2, 2), "block_norm": "L2-Hys"}
HOG_BLOCKS = HOG_IMAGE_SIZE // 16 - 1  # 7 x 7 blocks
HOG_DIM = HOG_BLOCKS * HOG_BLOCKS * 2 * 2 * HOG_PARAMS["orientations"]  # 1764

H_BINS, S_BINS, V_BINS = 32, 16, 16
LAB_MOMENTS = [f"lab_{ch}_{stat}" for ch in ("L", "a", "b") for stat in ("mean", "std", "skew")]
COLOR_FEATURE_NAMES = ([f"hue_{i}" for i in range(H_BINS)] + [f"sat_{i}" for i in range(S_BINS)]
                       + [f"val_{i}" for i in range(V_BINS)] + LAB_MOMENTS)
COLOR_DIM = len(COLOR_FEATURE_NAMES)  # 73

LBP_SPECS = ((8, 1), (16, 2))  # (sampling points, radius): "uniform" codes -> P + 2 histogram bins each
GLCM_LEVELS, GLCM_DISTANCES = 32, (1, 2, 4)
GLCM_ANGLES = (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
TEXTURE_FEATURE_NAMES = ([f"lbp{p}r{r}_{i}" for p, r in LBP_SPECS for i in range(p + 2)]
                         + [f"glcm_{prop}_d{d}" for prop in GLCM_PROPS for d in GLCM_DISTANCES])
TEXTURE_DIM = len(TEXTURE_FEATURE_NAMES)  # 43

FEATURE_SETS = {"HOG": ("hog",), "Color": ("color",), "HOG + Color": ("hog", "color"),
                "Color + Texture": ("color", "texture"), "HOG + Color + Texture": ("hog", "color", "texture")}

_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


def leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of leaf pixels. The paper background has low saturation; leaves are saturated or dark."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    otsu, _ = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((sat > max(otsu, 40)) | (val < 60)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _CLOSE)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    mask = binary_fill_holes(mask.astype(bool))
    if mask.sum() < 0.02 * mask.size:
        return np.ones(mask.shape, dtype=bool)
    return mask


def hog_features(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (HOG_IMAGE_SIZE, HOG_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return hog(gray, feature_vector=True, **HOG_PARAMS).astype(np.float32)


def color_features(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """HSV histograms (normalised to sum 1 per channel) + L*a*b* mean/std/skew, all over leaf pixels only."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask]
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(np.float32)
    n = max(len(hsv), 1)
    h_hist = np.histogram(hsv[:, 0], bins=H_BINS, range=(0, 180))[0] / n
    s_hist = np.histogram(hsv[:, 1], bins=S_BINS, range=(0, 256))[0] / n
    v_hist = np.histogram(hsv[:, 2], bins=V_BINS, range=(0, 256))[0] / n
    moments = []
    for ch in range(3):
        values = lab[:, ch]
        moments += [values.mean(), values.std(), float(np.nan_to_num(skew(values)))]
    return np.concatenate([h_hist, s_hist, v_hist, np.asarray(moments)]).astype(np.float32)


def texture_features(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Texture of the leaf (grey levels, leaf pixels only): uniform LBP histograms (normalised to sum 1) and GLCM
    properties per distance, averaged over four angles. Pixel pairs that touch the background are left out."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    parts = []
    for p, r in LBP_SPECS:
        codes = local_binary_pattern(gray, p, r, method="uniform").astype(np.int64)[mask]
        hist = np.bincount(codes, minlength=p + 2)[: p + 2].astype(np.float64)
        parts.append(hist / max(hist.sum(), 1.0))
    ys, xs = np.nonzero(mask)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    levels = (gray[box].astype(np.int64) * GLCM_LEVELS // 256 + 1) * mask[box]  # 0 = background
    glcm = graycomatrix(levels.astype(np.uint8), GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS + 1,
                        symmetric=True)[1:, 1:]
    parts += [np.nan_to_num(graycoprops(glcm, prop).mean(axis=1)) for prop in GLCM_PROPS]
    return np.concatenate(parts).astype(np.float32)


def extract_features(rgb: np.ndarray) -> dict:
    """Canonical 256x256x3 uint8 RGB -> {'hog': (1764,), 'color': (73,), 'texture': (43,)}."""
    mask = leaf_mask(rgb)
    return {"hog": hog_features(rgb), "color": color_features(rgb, mask), "texture": texture_features(rgb, mask)}


def feature_vector(features: dict, feature_set: str) -> np.ndarray:
    return np.concatenate([features[part] for part in FEATURE_SETS[feature_set]])
