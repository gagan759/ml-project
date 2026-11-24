"""Leaf-centred crop (tomatoleaf.crop): the same preprocessing for training, scoring and serving."""
import numpy as np

from tomatoleaf.crop import MIN_SIDE, leaf_box, leaf_crop, paste_back


def _paper_with_leaf(cx=70, cy=80, rx=30, ry=45):
    img = np.full((256, 256, 3), (225, 226, 230), dtype=np.uint8)  # light paper
    yy, xx = np.mgrid[0:256, 0:256]
    leaf = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    img[leaf] = (60, 130, 50)  # saturated green leaf
    return img, leaf


def test_crop_is_a_square_around_the_leaf_inside_the_frame():
    img, leaf = _paper_with_leaf()
    box = leaf_box(img)
    assert box is not None
    x, y, w, h = box
    assert w == h >= MIN_SIDE and x >= 0 and y >= 0 and x + w <= 256 and y + h <= 256
    ys, xs = np.nonzero(leaf)
    assert x <= xs.min() and xs.max() < x + w and y <= ys.min() and ys.max() < y + h  # the whole leaf is kept
    out, same_box = leaf_crop(img)
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8 and same_box == box
    assert np.array_equal(leaf_crop(img)[0], out)  # deterministic


def test_no_leaf_keeps_the_whole_frame():
    paper = np.full((256, 256, 3), 228, dtype=np.uint8)
    out, box = leaf_crop(paper)
    assert box is None and out is paper


def test_heat_map_is_placed_back_where_the_crop_was():
    img, _ = _paper_with_leaf()
    _, box = leaf_crop(img)
    full = paste_back(np.ones((256, 256), np.float32), box, img.shape)
    x, y, w, h = box
    assert full.shape == (256, 256)
    assert full[y:y + h, x:x + w].min() == 1.0 and full.sum() == w * h
    assert paste_back(np.ones((4, 4)), None, img.shape).shape == (4, 4)  # no crop: unchanged
