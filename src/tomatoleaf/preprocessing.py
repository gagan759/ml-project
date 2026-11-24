"""Image preprocessing shared by training scripts, the FastAPI backend and the audits.

Every image reaches a model through `open_image` -> `canonicalize` -> a model-specific tensor step, so an uploaded
photo is turned into exactly the kind of image the models were trained on.

How the dataset images were made (verified in scripts/20_baseline_audit.py, see docs/METHODOLOGY.md):
- Each photo was taken with a realme GT Master Edition phone at 3468x4624 (portrait, 72%) or 4624x3468 (landscape).
- The authors resized the WHOLE photo to 256x256, i.e. they squashed a 3:4 frame into a square. Registering their
  rotated copies against the originals shows the matching anisotropic scale (singular-value ratio (4/3)^2 at 90 deg).
- The pixels are stored in the Display P3 colour encoding (embedded ICC profile) as JPEG quality 90, 4:2:0 chroma.

The original upload path centre-cropped a square and ignored colour profiles, so a phone photo of a leaf reached the
model with a different shape (leaves 25% narrower), a smaller field of view and, for sRGB photos, different colour
values than every training image. `canonicalize` and `to_dataset_colour` now reproduce the dataset's own procedure.
Dataset images pass through both unchanged (pytest-verified), so training data and old results are unaffected.
"""
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageCms, ImageOps
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF

from .config import CANONICAL_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODEL_INPUT_SIZE

# Aspect ratios (width / height) of the dataset's source photos: 3:4 portrait and 4:3 landscape.
DATASET_ASPECT_RANGE = (3 / 4, 4 / 3)
DATASET_JPEG_QUALITY = 90
DATASET_JPEG_SUBSAMPLING = 2  # 4:2:0, as in every dataset file

# ---------------------------------------------------------------------------------------------------------------
# Colour: the dataset is Display P3 encoded. Uploads are converted into that encoding.
# ---------------------------------------------------------------------------------------------------------------
_SRGB_XY = np.array([[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]])
_P3_XY = np.array([[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]])
_D65 = np.array([0.3127, 0.3290])


def _rgb_to_xyz_matrix(primaries_xy: np.ndarray, white_xy: np.ndarray = _D65) -> np.ndarray:
    """Linear RGB -> XYZ matrix from chromaticity coordinates of the primaries and the white point."""
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries_xy]).T
    wx, wy = white_xy
    white = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    return xyz * np.linalg.solve(xyz, white)


# Display P3 shares the sRGB transfer curve and white point, so sRGB -> P3 is one 3x3 matrix in linear light.
SRGB_TO_P3 = np.linalg.inv(_rgb_to_xyz_matrix(_P3_XY)) @ _rgb_to_xyz_matrix(_SRGB_XY)
P3_TO_SRGB = np.linalg.inv(SRGB_TO_P3)


def _srgb_decode(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def convert_primaries(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """uint8 RGB in one sRGB-curve encoding -> uint8 RGB in another (e.g. sRGB -> Display P3)."""
    lin = _srgb_decode(rgb.astype(np.float64) / 255.0)
    out = _srgb_encode(lin @ matrix.T)
    return np.round(out * 255.0).astype(np.uint8)


def _profile_kind(icc: bytes) -> str:
    """'p3', 'srgb' or 'other' for an embedded ICC profile (by description, then by primaries)."""
    try:
        profile = ImageCms.ImageCmsProfile(BytesIO(icc))
        desc = (ImageCms.getProfileDescription(profile) or "").lower()
    except Exception:
        return "other"
    if "p3" in desc:
        return "p3"
    if "srgb" in desc or "iec61966" in desc:
        return "srgb"
    try:
        red = profile.profile.red_primary[1][:2]  # xyY of the red primary
        if abs(red[0] - 0.680) < 0.01 and abs(red[1] - 0.320) < 0.01:
            return "p3"
        if abs(red[0] - 0.640) < 0.01 and abs(red[1] - 0.330) < 0.01:
            return "srgb"
    except Exception:
        pass
    return "other"


def to_dataset_colour(img: Image.Image) -> tuple[Image.Image, str]:
    """Convert an RGB-convertible image into the dataset's colour encoding (Display P3).

    - Display P3 profile: already in the dataset encoding, pixels unchanged (dataset images take this path).
    - sRGB profile or no profile (the web default): sRGB -> Display P3.
    - Any other profile (Adobe RGB, ProPhoto, CMYK...): to sRGB with LittleCMS first, then to Display P3.
    Returns (RGB image, source description for the API response).
    """
    icc = img.info.get("icc_profile")
    kind = _profile_kind(icc) if icc else "none"
    if kind == "p3":
        return img.convert("RGB"), "Display P3 (unchanged)"
    if kind == "other":
        try:
            src = ImageCms.ImageCmsProfile(BytesIO(icc))
            img = ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
            kind = "embedded profile -> sRGB"
        except Exception:
            kind = "unreadable profile, assumed sRGB"
    rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    label = {"none": "untagged, assumed sRGB", "srgb": "sRGB"}.get(kind, kind)
    return Image.fromarray(convert_primaries(rgb, SRGB_TO_P3)), f"{label} -> Display P3"


# ---------------------------------------------------------------------------------------------------------------
# Decoding and geometry
# ---------------------------------------------------------------------------------------------------------------
def _flatten_alpha(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # transparent areas -> white paper
        flat = Image.alpha_composite(background, rgba).convert("RGB")
        if "icc_profile" in img.info:
            flat.info["icc_profile"] = img.info["icc_profile"]
        return flat
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):  # 16-bit greyscale PNG: scale to 8 bit before converting
        arr = np.asarray(img, dtype=np.float64)
        arr = 255.0 * arr / max(arr.max(), 1.0)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img


def open_image(source, manage_colour: bool = True) -> Image.Image:
    """Open a path, bytes or file object; apply EXIF orientation; convert to the dataset colour encoding; RGB."""
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)
    img = Image.open(source)
    img.load()
    icc = img.info.get("icc_profile")
    img = ImageOps.exif_transpose(img)
    if icc and "icc_profile" not in img.info:
        img.info["icc_profile"] = icc
    img = _flatten_alpha(img)
    if manage_colour:
        img, _ = to_dataset_colour(img)
    return img.convert("RGB")


def strip_jpeg_metadata(data: bytes, keep_app=(0xE0, 0xE2)) -> bytes:
    """Remove APPn / COM metadata segments from a JPEG without re-encoding it (lossless: the compressed pixel data,
    tables and, by default, the JFIF header and the ICC colour profile (APP2) are kept byte for byte).

    The dataset files carry ~240 KB of phone debug data (APP4); copying a photo for the website this way keeps its
    Display P3 profile, so the upload path treats it exactly like the dataset file (no re-encode, no colour shift)."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    out, i = bytearray(b"\xff\xd8"), 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError("corrupt JPEG segment structure")
        marker = data[i + 1]
        if marker == 0xDA:  # start of scan: copy the rest (entropy-coded data, EOI) unchanged
            out += data[i:]
            return bytes(out)
        length = int.from_bytes(data[i + 2:i + 4], "big")
        segment = data[i:i + 2 + length]
        if not ((0xE0 <= marker <= 0xEF or marker == 0xFE) and marker not in keep_app):
            out += segment
        i += 2 + length
    raise ValueError("no image data in JPEG")


def jpeg_roundtrip(img: Image.Image, quality: int = DATASET_JPEG_QUALITY,
                   subsampling: int = DATASET_JPEG_SUBSAMPLING) -> Image.Image:
    """Encode and decode as JPEG, like the dataset authors' final save (quality 90, 4:2:0)."""
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


def canonicalize(img: Image.Image, size: int = CANONICAL_SIZE, emulate_jpeg: bool = True) -> Image.Image:
    """Map any photo onto the dataset geometry, the way the dataset itself was made.

    The authors resized whole 3:4 / 4:3 phone photos to 256x256 (a non-uniform squash) and saved JPEG q90. So:
    1. if the aspect ratio is outside [3:4, 4:3], centre-crop it to the nearest of the two;
    2. resize the whole frame to 256x256 (non-uniform, exactly like the dataset);
    3. JPEG round-trip at quality 90 (only when step 2 changed the pixels).
    Dataset images (already 256x256) are returned unchanged.
    """
    w, h = img.size
    if (w, h) == (size, size):
        return img
    lo, hi = DATASET_ASPECT_RANGE
    aspect = w / h
    if aspect > hi:
        new_w = int(round(h * hi))
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif aspect < lo:
        new_h = int(round(w / lo))
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return jpeg_roundtrip(img) if emulate_jpeg else img


def canonicalize_legacy(img: Image.Image, size: int = CANONICAL_SIZE) -> Image.Image:
    """The ORIGINAL upload path (kept for the audit): centre-crop a square, resize, no JPEG step."""
    w, h = img.size
    if w != h:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def load_canonical_array(source, manage_colour: bool = False) -> np.ndarray:
    """Dataset file -> canonical 256x256x3 uint8 RGB array.

    Colour management is OFF by default here: every dataset file is Display P3 encoded, but the authors' augmented
    copies lost the ICC tag, so tag-based conversion would wrongly treat them as sRGB. Uploads use `prepare_upload`.
    """
    return np.asarray(canonicalize(open_image(source, manage_colour=manage_colour)), dtype=np.uint8)


def prepare_upload(source) -> tuple[np.ndarray, dict]:
    """Uploaded photo (path/bytes/file) -> (canonical 256x256x3 uint8 array, description of what was done).

    Orientation from EXIF, colour into the dataset's Display P3 encoding, geometry and JPEG like the dataset.
    """
    if isinstance(source, (bytes, bytearray)):
        source = BytesIO(source)
    img = Image.open(source)
    img.load()
    fmt = img.format
    icc = img.info.get("icc_profile")
    img = ImageOps.exif_transpose(img)
    if icc and "icc_profile" not in img.info:
        img.info["icc_profile"] = icc
    width, height = img.size
    img, colour = to_dataset_colour(_flatten_alpha(img))
    canonical = canonicalize(img.convert("RGB"))
    aspect = width / height
    lo, hi = DATASET_ASPECT_RANGE
    geometry = ("already 256x256" if (width, height) == (CANONICAL_SIZE, CANONICAL_SIZE) else
                ("centre-cropped to 4:3, " if aspect > hi + 1e-9 else "centre-cropped to 3:4, " if aspect < lo - 1e-9
                 else "whole frame, ") + "resized to 256x256 like the dataset, JPEG q90")
    return np.asarray(canonical, dtype=np.uint8), {"width": width, "height": height, "format": fmt,
                                                   "colour": colour, "geometry": geometry}


def to_image_tensor(rgb: np.ndarray) -> tv_tensors.Image:
    """HxWx3 uint8 array -> 3xHxW uint8 tensor (copies read-only arrays, e.g. from np.asarray(PIL image))."""
    rgb = np.ascontiguousarray(rgb)
    if not rgb.flags.writeable:
        rgb = rgb.copy()
    return tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1))


# ---------------------------------------------------------------------------------------------------------------
# Training augmentation
# ---------------------------------------------------------------------------------------------------------------
def _paper_fill(img: torch.Tensor) -> list:
    border = torch.cat([img[:, :4, :].reshape(3, -1), img[:, -4:, :].reshape(3, -1),
                        img[:, :, :4].reshape(3, -1), img[:, :, -4:].reshape(3, -1)], dim=1)
    return border.float().median(dim=1).values.round().int().tolist()


class RandomRotateWithPaperFill(torch.nn.Module):
    """Rotate by a random angle and fill the exposed corners with the image's own border (paper) colour.

    The authors' pre-augmented copies have black rotation corners in only 5 classes; filling with the
    paper colour avoids teaching the model that artifact.
    """

    def __init__(self, degrees: float = 30.0, p: float = 0.5):
        super().__init__()
        self.degrees, self.p = degrees, p

    def forward(self, img):
        if torch.rand(1).item() >= self.p:
            return img
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        return TF.rotate(img, angle, interpolation=T.InterpolationMode.BILINEAR, fill=_paper_fill(img))


class RandomDihedral(torch.nn.Module):
    """One of the 8 flips / 90-degree rotations, uniformly. A detached leaf has no 'up', and test-time augmentation
    averages exactly these 8 views, so training on them keeps every TTA view in-distribution."""

    def forward(self, img):
        k = int(torch.randint(4, (1,)).item())
        out = torch.rot90(img, k, dims=(-2, -1))
        if torch.rand(1).item() < 0.5:
            out = torch.flip(out, dims=(-1,))
        return tv_tensors.wrap(out, like=img) if isinstance(img, tv_tensors.Image) else out


class RandomJPEG(torch.nn.Module):
    """Re-compress as JPEG with a random quality (on the CPU; torchvision's JPEG op is CPU-only).

    Uploads are re-encoded at different qualities than the dataset (q90); this teaches the model to ignore
    compression artifacts (an earlier version flipped borderline predictions after a q95 re-encode)."""

    def __init__(self, quality=(50, 95), p: float = 0.4):
        super().__init__()
        self.jpeg, self.p = T.JPEG(quality=quality), p

    def forward(self, img):
        if torch.rand(1).item() >= self.p:
            return img
        device = img.device
        out = self.jpeg(tv_tensors.Image(img.as_subclass(torch.Tensor).cpu()))
        return tv_tensors.wrap(out.to(device), like=img) if isinstance(img, tv_tensors.Image) else out.to(device)


class RandomBlur(torch.nn.Module):
    def __init__(self, sigma=(0.1, 1.2), p: float = 0.25):
        super().__init__()
        self.blur, self.p = T.GaussianBlur(5, sigma=sigma), p

    def forward(self, img):
        return self.blur(img) if torch.rand(1).item() < self.p else img


def train_transform(size: int = MODEL_INPUT_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> T.Compose:
    """Original ('base') augmentation. No hue jitter: colour is the diagnostic signal for deficiencies."""
    return T.Compose([
        RandomRotateWithPaperFill(degrees=30, p=0.5),
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.18), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])


def train_transform_robust(size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD, strength: str = "mild") -> T.Compose:
    """Augmentation of this upgrade: the base policy plus
    - all 8 dihedral views (matches test-time augmentation),
    - aspect-ratio jitter 3:4..4:3 (the dataset mixes photos squashed along either axis),
    - JPEG re-compression and blur (uploads differ from the dataset's q90 files),
    still without hue jitter. strength="heavy" was the first version (more cropping, blur and JPEG); in
    cross-validation it under-fitted (scripts/21, recipe convnext_t_in22k_llrd), so "mild" is the default."""
    heavy = strength == "heavy"
    return T.Compose([
        RandomDihedral(),
        RandomRotateWithPaperFill(degrees=30, p=0.5),
        T.RandomResizedCrop(size, scale=(0.5, 1.0) if heavy else (0.6, 1.0), ratio=(0.75, 1.333), antialias=True),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        RandomBlur(sigma=(0.1, 1.2) if heavy else (0.1, 0.8), p=0.25 if heavy else 0.15),
        RandomJPEG(quality=(50, 95) if heavy else (60, 95), p=0.4 if heavy else 0.25),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------------------------------------------
# Evaluation / inference
# ---------------------------------------------------------------------------------------------------------------
def eval_transform(size: int = MODEL_INPUT_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD, resize: str = "resize") -> T.Compose:
    """Deterministic transform for validation, test and live inference.

    resize="resize": resize the 256x256 image to `size` (older 224px models); "crop": centre-crop `size` pixels
    (a 252px ViT input loses 2 border pixels instead of being resampled). For size 256 both are the identity.
    """
    ops = [T.Resize(size, antialias=True), T.CenterCrop(size)] if resize == "resize" else [T.CenterCrop(size)]
    return T.Compose(ops + [T.ToDtype(torch.float32, scale=True), T.Normalize(mean, std)])


_EVAL = {}


def preprocess_for_cnn(rgb: np.ndarray, size: int = MODEL_INPUT_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD,
                       resize: str = "resize") -> torch.Tensor:
    """Canonical 256x256x3 uint8 array -> normalised 1x3xSxS float tensor (inference path)."""
    key = (size, tuple(mean), tuple(std), resize)
    if key not in _EVAL:
        _EVAL[key] = eval_transform(size, mean, std, resize)
    return _EVAL[key](to_image_tensor(rgb)).unsqueeze(0).as_subclass(torch.Tensor)


def dihedral_views(x: torch.Tensor, n_views: int = 8) -> torch.Tensor:
    """Test-time augmentation: the 8 flips/90-degree rotations of a batch (a detached leaf has no 'up')."""
    views = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(-2, -1))
        views += [r, torch.flip(r, dims=(-1,))]
    return torch.cat(views[:n_views], dim=0)
