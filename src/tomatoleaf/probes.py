"""Far out-of-distribution probe images: things that are not a tomato leaf on paper.

20 photographs bundled with scikit-image (people, animals, objects, textures incl. grass and gravel, microscopy,
documents) plus 10 synthetic images (blank paper, flat colours, gradients, noise, a checkerboard). Each goes through
the same upload preprocessing as a user photo. Split into two fixed halves: half "A" may be used to choose an OOD
method; half "B" is only used to report it.
"""
import numpy as np
from PIL import Image

from .preprocessing import canonicalize, to_dataset_colour

SKIMAGE_NAMES = ["astronaut", "camera", "coffee", "chelsea", "rocket", "coins", "moon", "page", "text", "horse",
                 "hubble_deep_field", "immunohistochemistry", "retina", "brick", "grass", "gravel", "colorwheel",
                 "logo", "clock", "checkerboard"]


def _synthetic(rng: np.random.Generator) -> list:
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w] / (h - 1)
    paper = np.clip(rng.normal(232, 6, (h, w, 3)), 0, 255)
    items = [
        ("blank_paper", paper),
        ("flat_green", np.full((h, w, 3), (70, 140, 60), float)),
        ("flat_brown", np.full((h, w, 3), (120, 85, 50), float)),
        ("flat_black", np.full((h, w, 3), 8, float)),
        ("gradient_green", np.stack([40 + 80 * xx, 90 + 120 * yy, 40 + 30 * xx], -1)),
        ("gradient_rainbow", np.stack([255 * xx, 255 * yy, 255 * (1 - xx)], -1)),
        ("noise_uniform", rng.uniform(0, 255, (h, w, 3))),
        ("noise_green", np.clip(rng.normal((70, 140, 60), 40, (h, w, 3)), 0, 255)),
        ("checker_colour", np.kron((np.indices((8, 8)).sum(0) % 2)[..., None] * np.array([200, 40, 40]) + 30,
                                   np.ones((64, 64, 1)))),
        ("stripes", np.repeat((np.sin(xx * 60) > 0)[..., None] * 200.0 + 20, 3, axis=2)),
    ]
    return [(name, np.clip(arr, 0, 255).astype(np.uint8)) for name, arr in items]


def far_ood_probes(seed: int = 0) -> tuple[list, np.ndarray, np.ndarray]:
    """Returns (names, images N x 256 x 256 x 3 uint8 in the dataset encoding, half labels 'A'/'B')."""
    import skimage.data as skd

    raw = []
    for name in SKIMAGE_NAMES:
        arr = getattr(skd, name)()
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[2] == 4:
            alpha = arr[..., 3:4] / 255.0
            arr = (arr[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        raw.append((name, arr.astype(np.uint8)))
    raw += _synthetic(np.random.default_rng(seed))
    names, images = [], []
    for name, arr in raw:
        img, _ = to_dataset_colour(Image.fromarray(arr))
        images.append(np.asarray(canonicalize(img), dtype=np.uint8))
        names.append(name)
    halves = np.array(["A" if i % 2 == 0 else "B" for i in range(len(names))])
    return names, np.stack(images), halves
