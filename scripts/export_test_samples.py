"""Exports one held-out TEST photograph per diagnosed condition (the served models' classes) for the web apps'
example gallery.

Rules (deterministic):
  --rule demo (default): the test photo on which the served models give the clearest correct result (see
      demo_scores), ties -> the most typical one. Disclosed on the page: it shows how the app works, not how often
      it is right.
  --rule typical: the most typical test photo of each condition — the medoid of its test photos in a GENERIC
      image-feature space (ConvNeXt-Tiny with ImageNet weights only, never trained on this dataset). Images and labels
      only, no prediction.
  --rule first: the first test photo of each condition in splits.csv order (the gallery up to v2.1).
Files are byte-identical copies of the dataset photos (re-encoding could change pixels and flip borderline answers).
Output: frontend/public/test-samples/<key>.jpg and manifest.json (with the true label and the rule); photos of
conditions the app no longer diagnoses are removed.
"""
import argparse
import shutil

import numpy as np
import torch

from tomatoleaf.config import CLASSES, DATASET_DIR, ROOT
from tomatoleaf.data import load_images, load_splits
from tomatoleaf.utils import save_json

NOTES = {
    "demo": ("Demo examples: for each condition, the held-out test photo (never used for training or model selection) "
             "on which the served models give the clearest correct result — a confident correct diagnosis; ties go to "
             "the most typical photo. They show how the app works, NOT how often it is right: the unbiased test "
             "results are in the Model lab."),
    "typical": ("One held-out test photograph per class (never used for training or model selection): the most typical "
                "test photo of each class, i.e. the medoid in a generic ImageNet feature space (ConvNeXt-Tiny, not trained "
                "on this dataset). Chosen from images and labels only, without looking at any model's answer."),
    "first": ("One held-out test photograph per class (never used for training or model selection). Chosen as the first "
              "test photo of each class, not filtered by model correctness."),
}


@torch.no_grad()
def generic_features(images: np.ndarray) -> np.ndarray:
    """L2-normalised 768-d ImageNet ConvNeXt-Tiny features (global average pool), CPU, float32."""
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    from tomatoleaf.preprocessing import preprocess_for_cnn

    net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).eval()
    feats = []
    for rgb in images:
        x = preprocess_for_cnn(rgb, 224)
        f = net.classifier[:2](net.avgpool(net.features(x)))  # LayerNorm2d + Flatten: the penultimate features
        feats.append(torch.nn.functional.normalize(f, dim=1)[0].numpy())
    return np.stack(feats)


def served_registry():
    from backend.services.predictor import ModelRegistry
    from tomatoleaf.config import EXPERIMENTS_DIR

    return ModelRegistry(EXPERIMENTS_DIR / "deployment.json", device="cuda" if torch.cuda.is_available() else "cpu")


def demo_scores(reg, test, images) -> np.ndarray:
    """Per test photo: how clearly the SERVED models handle it (higher = clearer). Each model adds 2 for a confident
    correct answer (or, for a condition outside that model's classes, any refusal), 1 for 'not sure' with the right
    best guess, 0 for a wrong answer."""
    keys = [e["key"] for e in reg.entries]
    score = np.zeros(len(test))
    for i, (rgb, true) in enumerate(zip(images, test.class_key)):
        for key in keys:
            r = reg.predict(rgb, key)
            p = r["prediction"]
            answered = p["status"] in ("Healthy", "Diseased")
            if true not in r["classes"]:
                score[i] += 2 if not answered else 0
            elif answered:
                score[i] += 2 if p["key"] == true else 0
            else:
                score[i] += 1 if p["key"] == true else 0
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rule", choices=list(NOTES), default="demo")
    args = parser.parse_args()
    reg = served_registry()
    scope = set(reg.entry(reg.default_key)["class_keys"])
    splits = load_splits()
    splits = splits[splits.class_key.isin(scope)]
    test = splits[splits.split == "test"].sort_values("photo_id").reset_index(drop=True)
    out = ROOT / "frontend" / "public" / "test-samples"
    out.mkdir(parents=True, exist_ok=True)
    images = load_images(test.path) if args.rule != "first" else None
    feats = generic_features(images) if images is not None else None
    demo = demo_scores(reg, test, images) if args.rule == "demo" else None
    manifest = []
    for c in (c for c in CLASSES if c["key"] in scope):
        rows = test[test.label == c["index"]]
        if args.rule in ("typical", "demo"):
            idx = rows.index.to_numpy()
            f = feats[idx]
            sim = f @ f.T
            mean_other = (sim.sum(1) - 1.0) / max(len(rows) - 1, 1)
            order = np.lexsort((-mean_other, -(demo[idx] if demo is not None else np.zeros(len(idx)))))
            best = int(order[0])  # demo: clearest result first, then most typical; typical: most typical
            row = rows.iloc[best]
            extra = {"typicality": float(mean_other[best]), "class_test_photos": int(len(rows))}
            if demo is not None:
                extra["demo_score"] = float(demo[idx][best])
        else:
            row = splits[(splits.split == "test") & (splits.label == c["index"])].iloc[0]
            extra = {}
        shutil.copyfile(DATASET_DIR / row.path, out / f"{c['key']}.jpg")
        manifest.append({"key": c["key"], "name": c["name"], "file": f"/test-samples/{c['key']}.jpg",
                         "photo_id": row.photo_id, **extra})
    for stale in out.glob("*.jpg"):  # conditions the app no longer diagnoses
        if stale.stem not in scope:
            stale.unlink()
    save_json({"note": NOTES[args.rule], "rule": args.rule, "samples": manifest}, out / "manifest.json")
    print(f"Exported {len(manifest)} test samples ({args.rule}) to {out}: "
          + ", ".join(f"{m['key']}={m['photo_id']}" for m in manifest))


if __name__ == "__main__":
    main()
