"""Phase 3a — extract handcrafted features for all 4,525 images and cache them.

HOG: 128x128 greyscale, 9 orientations, 16x16-pixel cells, 2x2-cell blocks -> 1,764 values (shape/texture).
Colour: HSV histograms (32+16+16 bins) + L*a*b* mean/std/skew over leaf pixels only -> 73 values.
Output: data/features.npz (rows aligned with data/index.csv)
"""
import numpy as np
from joblib import Parallel, delayed

from tomatoleaf.config import DATA_DIR, DATASET_DIR
from tomatoleaf.data import load_index
from tomatoleaf.features import COLOR_DIM, HOG_DIM, extract_features
from tomatoleaf.preprocessing import load_canonical_array
from tomatoleaf.utils import save_json, timer


def extract_one(rel_path: str):
    feats = extract_features(load_canonical_array(DATASET_DIR / rel_path))
    return feats["hog"], feats["color"]


def main():
    index = load_index()
    with timer() as elapsed:
        out = Parallel(n_jobs=-1, batch_size=32)(delayed(extract_one)(p) for p in index.path)
    hog = np.stack([o[0] for o in out])
    color = np.stack([o[1] for o in out])
    assert hog.shape[1] == HOG_DIM and color.shape[1] == COLOR_DIM
    np.savez_compressed(DATA_DIR / "features.npz", hog=hog, color=color, paths=index.path.to_numpy())
    save_json({"images": len(index), "hog_dim": HOG_DIM, "color_dim": COLOR_DIM, "seconds": elapsed(),
               "ms_per_image": 1000 * elapsed() / len(index)}, DATA_DIR / "features_meta.json")
    print(f"Extracted HOG {hog.shape} and colour {color.shape} in {elapsed():.1f}s")


if __name__ == "__main__":
    main()
