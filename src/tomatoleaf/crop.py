"""Leaf-centred crop — the final round's preprocessing step, shared by training, out-of-fold scoring, the API and apps.

A dataset photo is a whole 3:4 frame squashed to 256x256 in which the leaf covers about a quarter of the area. Cropping
a square around the leaf (its saturation mask plus a margin) and resizing it back to 256x256 gives the network about
twice as many pixels on the lesions, which is where Early Blight, Late Blight and Spotted Wilt differ.

Rules (deterministic): the box is the leaf mask's bounding box, enlarged by MARGIN of its longer side on every side,
made square, kept inside the frame, and never smaller than MIN_SIDE pixels (at most ~2.7x zoom). If no leaf is found
(the mask falls back to the whole frame) or the box would cover the whole frame, the image is used unchanged — so an
unusual photo is never cut down to an arbitrary patch.
"""
import cv2
import numpy as np

from .features import leaf_mask

MARGIN = 0.08
MIN_SIDE = 96


def leaf_box(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    """(left, top, width, height) of the square crop around the leaf, or None to keep the whole frame."""
    h, w = rgb.shape[:2]
    mask = leaf_mask(rgb)
    if mask.all():  # leaf_mask's fallback when no leaf is found
        return None
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    side = max(int(round(max(y1 - y0, x1 - x0) * (1 + 2 * MARGIN))), MIN_SIDE)
    if side >= min(h, w):
        return None
    top = int(round(min(max((y0 + y1) / 2 - side / 2, 0), h - side)))
    left = int(round(min(max((x0 + x1) / 2 - side / 2, 0), w - side)))
    return left, top, side, side


def leaf_crop(rgb: np.ndarray, size: int = 256) -> tuple[np.ndarray, tuple | None]:
    """-> (size x size x 3 uint8 image centred on the leaf, crop box or None)."""
    box = leaf_box(rgb)
    if box is None:
        return rgb, None
    x, y, bw, bh = box
    crop = rgb[y:y + bh, x:x + bw]
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC if bw < size else cv2.INTER_AREA)
    return np.ascontiguousarray(out), box


def leaf_crop_batch(images: np.ndarray) -> np.ndarray:
    return np.stack([leaf_crop(x)[0] for x in images])


def paste_back(heat: np.ndarray, box: tuple | None, shape: tuple) -> np.ndarray:
    """A heat map computed on the crop -> the same map placed in the original frame (zero outside the crop)."""
    if box is None:
        return heat
    x, y, bw, bh = box
    full = np.zeros(shape[:2], dtype=np.float32)
    full[y:y + bh, x:x + bw] = cv2.resize(np.asarray(heat, dtype=np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR)
    return full
