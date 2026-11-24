"""Upgrade round, step 0 — baseline audit of the ORIGINAL deployed models (before any change).

Runs on the VALIDATION photos only (the test photos stay untouched until the final benchmark), through the deployed
inference code on the CPU in float32.
1. Clean accuracy / macro-F1 of the original models (reproduces the original study's validation numbers).
2. Prediction consistency under harmless changes: the 8 flips/rotations, JPEG re-compression, mild blur, brightness,
   lower resolution. Reported as the share of photos whose predicted class flips.
3. Simulated phone uploads. Each validation photo is turned back into a full-frame phone photo (3:4 or 4:3, as its
   EXIF says), saved either as a Display P3 photo (like the dataset phone) or as an untagged sRGB photo (like most
   web images), and sent through
     (a) the ORIGINAL upload path: centre-crop to a square, resize, colour profile ignored;
     (b) the NEW upload path (tomatoleaf.preprocessing.prepare_upload): whole frame resized like the dataset,
         colour converted into the dataset's Display P3 encoding, JPEG q90 like the dataset.
   This measures how much of the original app's behaviour on real photos came from a preprocessing mismatch.
Output: experiments/results/baseline_audit.json
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services.predictor import ModelRegistry  # noqa: E402
from tomatoleaf.config import CLASS_KEYS, DATASET_DIR, EXPERIMENTS_DIR, RESULTS_DIR, SEED
from tomatoleaf.data import load_images, load_splits
from tomatoleaf.evaluation import classification_metrics
from tomatoleaf.preprocessing import P3_TO_SRGB, canonicalize_legacy, convert_primaries, jpeg_roundtrip, prepare_upload
from tomatoleaf.utils import save_json, set_seed

AUDIT_MODELS = ["ensemble_5class", "convnext_tiny_v2_tta", "convnext_tiny_v2", "efficientnet_b0_v2", "mobilenetv2_ft",
                "rf"]
SINGLE_VIEW = ["convnext_tiny_v2", "efficientnet_b0_v2"]


def perturbations():
    def jpeg(q):
        return lambda a: np.asarray(jpeg_roundtrip(Image.fromarray(a), quality=q, subsampling=2))

    def blur(a):
        import cv2
        return cv2.GaussianBlur(a, (5, 5), 1.0)

    def bright(f):
        return lambda a: np.clip(a.astype(np.float32) * f, 0, 255).astype(np.uint8)

    def lowres(a):
        small = Image.fromarray(a).resize((128, 128), Image.Resampling.LANCZOS)
        return np.asarray(small.resize((256, 256), Image.Resampling.BICUBIC))

    return {"jpeg_q75": jpeg(75), "jpeg_q50": jpeg(50), "blur_sigma1": blur, "brightness_+15%": bright(1.15),
            "brightness_-15%": bright(0.85), "half_resolution": lowres}


def phone_photo(rgb: np.ndarray, portrait: bool, profile: str, icc: bytes) -> bytes:
    """Re-create a full-frame phone photo (3:4 portrait or 4:3 landscape) from a squashed 256x256 dataset image."""
    size = (768, 1024) if portrait else (1024, 768)
    big = Image.fromarray(rgb).resize(size, Image.Resampling.BICUBIC)
    buf = io.BytesIO()
    if profile == "p3":
        big.save(buf, format="JPEG", quality=95, icc_profile=icc)
    else:  # sRGB values, no profile (the web default)
        Image.fromarray(convert_primaries(np.asarray(big), P3_TO_SRGB)).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def legacy_upload(data: bytes) -> np.ndarray:
    from PIL import ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    return np.asarray(canonicalize_legacy(img), dtype=np.uint8)


def main():
    set_seed(SEED)
    torch.set_num_threads(8)
    registry = ModelRegistry(EXPERIMENTS_DIR / "deployment.json", device="cpu", lazy=True)
    splits = load_splits()
    va = splits[splits.split == "val"].reset_index(drop=True)
    X = load_images(va.path)
    y = va.label.to_numpy()
    print(f"validation photos: {len(va)}")

    # EXIF orientation of each source photo (portrait 3468x4624 or landscape 4624x3468)
    portrait, icc = [], None
    for rel in va.path:
        im = Image.open(DATASET_DIR / rel)
        ex = im.getexif()
        portrait.append(ex.get(257, 1) >= ex.get(256, 0))
        icc = icc or im.info.get("icc_profile")

    preds = {}

    def probs_for(key, images):
        p = registry.load(key)
        return np.stack([p.predict_proba(img) for img in images]), p.class_keys

    def score(key, probs, keys, mask):
        local = np.array([CLASS_KEYS.index(k) for k in keys])
        pred = local[probs.argmax(1)]
        return pred, classification_metrics(y[mask], pred[mask], local)

    out = {"photos": int(len(va)), "models": {}, "note": "validation photos only; deployed CPU float32 code path"}
    variants = {"clean": X, **{name: np.stack([f(a) for a in X]) for name, f in perturbations().items()}}
    t0 = time.perf_counter()
    uploads = {}
    for profile in ("p3", "srgb"):
        files = [phone_photo(a, por, profile, icc) for a, por in zip(X, portrait)]
        uploads[f"upload_{profile}_original_path"] = np.stack([legacy_upload(f) for f in files])
        uploads[f"upload_{profile}_new_path"] = np.stack([prepare_upload(f)[0] for f in files])
    variants.update(uploads)
    print(f"built {len(variants)} input variants in {time.perf_counter() - t0:.0f}s")

    for key in AUDIT_MODELS:
        t0 = time.perf_counter()
        entry = {}
        clean_probs, keys = probs_for(key, X)
        mask = np.isin(y, [CLASS_KEYS.index(k) for k in keys])  # 5-class models: in-scope photos only
        clean_pred, clean_met = score(key, clean_probs, keys, mask)
        entry["clean"] = {**clean_met, "photos": int(mask.sum())}
        for name, imgs in variants.items():
            if name == "clean":
                continue
            probs, _ = probs_for(key, imgs)
            pred, met = score(key, probs, keys, mask)
            entry[name] = {**met, "flip_rate": float((pred[mask] != clean_pred[mask]).mean()),
                           "mean_abs_confidence_change": float(np.abs(probs.max(1) - clean_probs.max(1))[mask].mean())}
        if key in SINGLE_VIEW:  # agreement across the 8 dihedral views of the same photo
            view_preds = []
            for k in range(4):
                for flip in (False, True):
                    imgs = np.rot90(X, k, axes=(1, 2))
                    imgs = imgs[:, :, ::-1] if flip else imgs
                    p, _ = probs_for(key, np.ascontiguousarray(imgs))
                    view_preds.append(np.array([CLASS_KEYS.index(c) for c in keys])[p.argmax(1)])
            view_preds = np.stack(view_preds)
            entry["dihedral_views"] = {
                "all_8_views_agree": float((view_preds == view_preds[0]).all(0).mean()),
                "per_view_accuracy": [float((v == y).mean()) for v in view_preds],
                "photos_with_2plus_distinct_predictions": int((np.array([len(set(c)) for c in view_preds.T]) > 1).sum())}
        out["models"][key] = entry
        preds[key] = clean_pred
        print(f"{key}: clean acc {clean_met['accuracy']:.3f} F1 {clean_met['f1_macro']:.3f} | "
              + " | ".join(f"{n} flip {entry[n]['flip_rate']:.3f}" for n in entry if n.startswith("upload"))
              + f" ({time.perf_counter() - t0:.0f}s)", flush=True)
    save_json(out, RESULTS_DIR / "baseline_audit.json")


if __name__ == "__main__":
    main()
