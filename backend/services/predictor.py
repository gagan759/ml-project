"""Loads the models listed in experiments/deployment.json and runs inference.

Every predictor receives the same canonical 256x256 RGB array (tomatoleaf.preprocessing.prepare_upload) and applies
exactly the transformation used when the model was evaluated offline: its own input size, normalisation, number of
test-time views and validation-fitted temperature.

Decision for one photo (in this order):
1. "Unusual image": the photo's features are far from every training photo (out-of-distribution score above the
   threshold chosen on out-of-fold data) -> no diagnosis; probably not a tomato leaf on paper, or a condition the
   model was not trained on.
2. "Uncertain": calibrated confidence below the model's threshold -> no diagnosis.
3. Otherwise "Healthy" or "Diseased" with the predicted class.
"""
import base64
import itertools
import logging
import os
import threading
import time
from contextlib import contextmanager
from io import BytesIO

import joblib
import numpy as np
import torch
from PIL import Image

from tomatoleaf import balance
from tomatoleaf.config import CLASS_KEYS, CLASSES, EXPERIMENTS_DIR, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR
from tomatoleaf.crop import leaf_crop, paste_back
from tomatoleaf.features import extract_features, feature_vector
from tomatoleaf.gradcam import GradCAM, heatmap_png, overlay
from tomatoleaf.models import FeatureTap, build_model, gradcam_layer, gradcam_reshape, load_checkpoint
from tomatoleaf.ood import FeatureDensity
from tomatoleaf.preprocessing import dihedral_views, preprocess_for_cnn
from tomatoleaf.quality import assess as assess_quality
from tomatoleaf.utils import load_json

from . import onnx_runtime

log = logging.getLogger("tomato.predictor")


def png_data_url(arr: np.ndarray) -> str:
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def refusal_reason(status: str, confidence: float, threshold: float, ood_ratio, quality: dict,
                   class_keys: list) -> dict | None:
    """Why no diagnosis was given, and what the user can do (None when the model answered)."""
    if status not in ("Unusual image", "Uncertain"):
        return None
    if status == "Unusual image":
        reason = {"code": "unusual_image",
                  "message": "The photo does not look like the leaf photos the model was trained on "
                             f"(distance {ood_ratio:.2f}x the limit)."}
    else:
        reason = {"code": "low_confidence",
                  "message": f"The model is only {confidence:.0%} sure; it answers at {threshold:.0%} or more."}
    hints = [issue["tip"] for issue in quality["issues"]]
    missing = [CLASSES[CLASS_KEYS.index(k)]["name"] for k in CLASS_KEYS if k not in class_keys]
    if missing:
        hints.append(f"Tomato Care diagnoses {len(class_keys)} conditions; it does not cover {', '.join(missing[:-1])} or "
                     f"{missing[-1]}. If the leaf may have one of these, ask an agronomist.")
    if not quality["issues"]:
        hints.append("Try another photo: one leaf filling the frame, on plain light paper, in daylight.")
    return {**reason, "hints": hints}


def _feature_words(feature_set: str) -> str:
    """'Color + Texture' -> 'colour + texture' (display wording of a classical model's feature set)."""
    return feature_set.replace("Color", "colour").replace("Texture", "texture")


def lowmem_enabled() -> bool:
    """TOMATO_LOWMEM=1: checkpoints of one architecture share a single float32 network (small servers)."""
    return os.getenv("TOMATO_LOWMEM") == "1"


_TOKENS = itertools.count(1)


class _SharedNet:
    """One float32 network per architecture and output size, shared by every checkpoint of that shape.

    Low-memory mode: each checkpoint keeps only its stored weights, memory-mapped from disk (float16 for the upgrade
    models), and copies them into the shared network right before it runs. The copy is the same float16 -> float32
    conversion `load_checkpoint` does, so predictions are unchanged; it costs ~20-40 ms per model instead of
    ~110 MB of RAM per ConvNeXt-Tiny (the 10-model ensemble would otherwise hold 1.1 GB)."""

    _instances: dict = {}
    _guard = threading.Lock()

    def __init__(self, model: torch.nn.Module):
        self.model, self.owner, self.lock = model, None, threading.Lock()

    @classmethod
    def for_checkpoint(cls, ckpt: dict) -> "_SharedNet":
        kwargs = ckpt.get("model_kwargs", {})
        key = (ckpt["arch"], len(ckpt["class_keys"]), repr(sorted(kwargs.items())))
        with cls._guard:
            if key not in cls._instances:
                model = build_model(ckpt["arch"], pretrained=False, num_classes=len(ckpt["class_keys"]), **kwargs)
                cls._instances[key] = cls(model.eval())
            return cls._instances[key]

    def bind(self, owner: int, state: dict) -> torch.nn.Module:
        """Make `owner`'s weights the current ones (the caller holds self.lock)."""
        if self.owner != owner:
            self.model.load_state_dict(state)  # copy_ casts float16 -> float32
            self.owner = owner
        return self.model


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


class Predictor:
    score_type = "predict_proba"
    calibration = None

    def __init__(self, entry: dict):
        self.key, self.name, self.family = entry["key"], entry["name"], entry["family"]
        self.kind, self.feature_set = entry["kind"], entry["feature_set"]
        self.task = entry.get("task")
        self.params, self.test_metrics = entry.get("params"), entry.get("test_metrics")
        self.class_keys = list(entry.get("class_keys") or CLASS_KEYS)
        self.threshold = entry.get("confidence_threshold")  # per-model threshold chosen on validation / OOF data
        self.operating_point = entry.get("operating_point")
        self.ood_config = entry.get("ood")  # {"method": "knn8", "thresholds": [...]} for upgraded models
        # per-class calibration + class-prior correction fitted on out-of-fold data (tomatoleaf.balance, scripts/31)
        self.adjust = entry.get("adjust")
        self.variant = entry.get("variant")
        self.pipeline = entry.get("pipeline")  # final round: summary of the validated pipeline (scripts/36)

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Per-class calibration (if the entry has one): every condition's confidence matches its own accuracy."""
        if self.adjust and self.adjust.get("calibration"):
            return balance.apply(probs, self.adjust["calibration"])
        return probs

    def decide(self, probs: np.ndarray) -> int:
        """Index of the answered class: the most probable one or, for bias-corrected models, the most probable after
        dividing by the class prior ** tau. The confidence reported is probs[decision] (still calibrated)."""
        if self.adjust and self.adjust.get("tau"):
            return int(balance.decide(probs, self.adjust["prior"], self.adjust["tau"]))
        return int(np.argmax(probs))

    def unusualness(self, ratio: float | None, decision: int) -> float | None:
        """OOD ratio against the limit of the answered condition (scripts/33: equal false-flag rates per condition)."""
        if ratio is None or not (self.adjust and self.adjust.get("ood_class_scale")):
            return ratio
        return float(ratio / self.adjust["ood_class_scale"][decision])

    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        return self.analyse(rgb)[0]

    def analyse(self, rgb: np.ndarray):
        """-> (probabilities over self.class_keys, OOD ratio or None). Ratio > 1 means 'unusual image'."""
        raise NotImplementedError

    @property
    def info(self) -> dict:
        return {"key": self.key, "name": self.name, "family": self.family, "task": self.task}


class TorchPredictor(Predictor):
    score_type = "calibrated_softmax"

    def __init__(self, entry: dict, device: torch.device):
        super().__init__(entry)
        path = MODELS_DIR / entry["path"]
        self._shared = None
        if device.type == "cpu" and lowmem_enabled():
            ckpt = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            self._shared, self._state, self._token = _SharedNet.for_checkpoint(ckpt), ckpt["state_dict"], next(_TOKENS)
            self.model = None
        else:
            self.model, ckpt = load_checkpoint(path, device)
        self.class_keys = list(ckpt["class_keys"])
        self.device = device
        self.arch = ckpt["arch"]
        self.size = int(ckpt["input_size"])
        self.mean = tuple(ckpt.get("mean", IMAGENET_MEAN))
        self.std = tuple(ckpt.get("std", IMAGENET_STD))
        self.resize = ckpt.get("eval_resize", "resize")
        self.crop = (ckpt.get("recipe") or {}).get("crop") == "leaf"  # leaf-centred crop, as in training
        self.tta_views = int(entry.get("tta_views") or 1)
        self.temperature = float(entry.get("temperature") or 1.0)
        self.density = FeatureDensity.from_dict(ckpt["density"]) if ckpt.get("density") else None
        self.ood_method = (entry.get("ood_method") or "knn")  # "knn" or "maha" on this model's features
        self.ood_threshold = entry.get("ood_threshold")
        self.input = (f"{self.size}x{self.size} RGB, mean/std {self.mean}/{self.std}"
                      + (f", {self.tta_views}-view test-time augmentation" if self.tta_views > 1 else ""))
        self._lock = threading.Lock()  # one PyTorch forward pass at a time per model (the feature hook is shared state)
        self.ort = None
        if self._shared is None and device.type == "cpu" and onnx_runtime.enabled():
            try:
                ort_model = onnx_runtime.OnnxModel(self.model, MODELS_DIR / entry["path"], self.size)
                probe = torch.randn(2, 3, self.size, self.size)
                with torch.inference_mode():
                    ref = self.model(probe).float().numpy()
                if float(np.abs(ort_model(probe)[0] - ref).max()) < 1e-3:  # same answers, or stay on PyTorch
                    self.ort = ort_model
                else:
                    log.warning("ONNX Runtime output differs for %s; using PyTorch", self.key)
            except Exception:
                log.exception("ONNX export failed for %s; using PyTorch", self.key)
        self.runtime = "onnxruntime" if self.ort is not None else "pytorch-shared" if self._shared else "pytorch"

    @contextmanager
    def _net(self):
        """This model's network, locked for one call (low-memory mode: the shared network holding these weights)."""
        if self._shared is None:
            with self._lock:
                yield self.model
        else:
            with self._shared.lock:
                yield self._shared.bind(self._token, self._state)

    def tensor(self, rgb: np.ndarray, box_out: list | None = None) -> torch.Tensor:
        """Model input; for leaf-crop models the leaf-centred crop first (its box is appended to `box_out`)."""
        if self.crop:
            rgb, box = leaf_crop(rgb)
            if box_out is not None:
                box_out.append(box)
        return preprocess_for_cnn(rgb, self.size, self.mean, self.std, self.resize).to(self.device)

    def forward(self, rgb: np.ndarray):
        """-> (mean logits over the test-time views, mean penultimate features or None)."""
        x = self.tensor(rgb)
        if self.tta_views > 1:
            x = dihedral_views(x, self.tta_views)
        if self.ort is not None:  # ONNX Runtime sessions are thread-safe
            logits, feats = self.ort(x)
            return logits.mean(0), (feats.mean(0) if self.density is not None else None)
        with self._net() as model, torch.inference_mode():
            tap = FeatureTap(model) if self.density is not None else None
            try:
                logits = model(x).float()
                feats = tap.features.float().mean(0).cpu().numpy() if tap is not None else None
            finally:
                if tap is not None:
                    tap.close()
        return logits.mean(0).cpu().numpy(), feats

    def ood_ratio(self, feats) -> float | None:
        if feats is None or self.density is None or not self.ood_threshold:
            return None
        score = self.density.knn(feats)[0] if self.ood_method == "knn" else self.density.mahalanobis(feats)[0]
        return float(score / self.ood_threshold)

    def analyse(self, rgb: np.ndarray):
        logits, feats = self.forward(rgb)
        return self.calibrate(_softmax(logits, self.temperature)), self.ood_ratio(feats)

    def explain(self, rgb: np.ndarray, class_index: int) -> dict:
        """Grad-CAM for `class_index`: a blended overlay and the bare heat map (RGBA, alpha = strength), PNG data URLs.

        The hook is attached only for this call (and serialised with the model lock) so predictions are unaffected.
        """
        heat = self.gradcam(rgb, class_index)
        return {"method": "Grad-CAM", "class_index": class_index, "overlay": png_data_url(overlay(rgb, heat)),
                "heatmap": png_data_url(heatmap_png(heat, rgb.shape[0]))}

    def gradcam(self, rgb: np.ndarray, class_index: int) -> np.ndarray:
        """Grad-CAM heat map in the frame of `rgb` (a leaf-crop model's map is placed back where the crop was)."""
        boxes = []
        with self._net() as model:
            cam = GradCAM(model, gradcam_layer(model, self.arch), gradcam_reshape(model, self.arch))
            try:
                heat, _, _ = cam(self.tensor(rgb, boxes), class_index=class_index)
            finally:
                cam.close()
                model.zero_grad(set_to_none=True)
        return paste_back(heat, boxes[0] if boxes else None, rgb.shape)

    @property
    def calibration(self):
        return {"method": "temperature_scaling", "temperature": self.temperature}


class EnsemblePredictor(Predictor):
    """Average of calibrated member probabilities (each member: its own checkpoint, temperature and TTA setting).

    OOD: each member's score is divided by its own threshold; the ensemble flags a photo when the mean ratio > 1."""
    score_type = "calibrated_softmax"

    def __init__(self, entry: dict, device: torch.device, loaded: dict | None = None):
        """`loaded`: a cache shared by the registry, so ensembles built from the same fold networks (e.g. the
        calibrated and the balanced focused model) load each network once."""
        super().__init__(entry)
        self.members = []
        cache = {} if loaded is None else loaded
        for m in entry["members"]:
            spec = {**m, "key": m["key"], "name": m.get("name", m["key"]), "family": entry["family"],
                    "kind": "transfer", "feature_set": entry["feature_set"], "class_keys": entry.get("class_keys")}
            ident = (m["path"], int(m.get("tta_views") or 1), m.get("temperature"), m.get("ood_method"),
                     m.get("ood_threshold"))
            if ident not in cache:
                cache[ident] = TorchPredictor(spec, device)
            self.members.append(cache[ident])
        self.input = " + ".join(f"{m.arch.replace('timm:', '')} ({m.size}px{', TTA' if m.tta_views > 1 else ''})"
                                for m in self.members)
        self.class_keys = self.members[0].class_keys
        assert all(m.class_keys == self.class_keys for m in self.members), "ensemble members disagree on classes"
        for m in self.members[1:]:  # only the first member draws Grad-CAM: free the others' PyTorch weights
            if m.ort is not None:
                m.model = None
        self.runtime = "+".join(sorted({m.runtime for m in self.members}))

    def analyse(self, rgb: np.ndarray):
        outs = [m.analyse(rgb) for m in self.members]
        probs = np.mean([p for p, _ in outs], axis=0)
        ratios = [r for _, r in outs if r is not None]
        return self.calibrate(probs), (float(np.mean(ratios)) if ratios else None)

    def explain(self, rgb: np.ndarray, class_index: int) -> dict:
        out = self.members[0].explain(rgb, class_index)
        out["note"] = f"Grad-CAM of the first ensemble member ({self.members[0].arch.replace('timm:', '')})."
        return out

    @property
    def calibration(self):
        out = {"method": "temperature_scaling (per member)", "temperature": None,
               "member_temperatures": [m.temperature for m in self.members]}
        if self.pipeline:  # final round (scripts/36): one validated pipeline per model
            cal = "per-class vector scaling" if (self.adjust or {}).get("calibration") else "temperature scaling"
            out.update({"method": cal, "pipeline": self.pipeline})
            return out
        if self.adjust:
            out.update({"method": "temperature scaling per member + per-class calibration"
                        + (" + class-prior correction" if self.adjust.get("tau") else ""),
                        "prior_correction_tau": self.adjust.get("tau", 0.0)})
        return out


class SklearnPredictor(Predictor):
    """Classical model on handcrafted features. The served classical model (scripts/43) takes the full
    HOG | colour | texture row and adds its validated calibration (temperature in the entry, vector scaling in
    `adjust`) and an unusual-image gate (feature distance to its training photos in the colour + texture space)."""

    def __init__(self, entry: dict):
        super().__init__(entry)
        bundle = joblib.load(MODELS_DIR / entry["path"])
        self.estimator = bundle["pipeline"]
        self.class_keys = list(bundle.get("class_keys") or CLASS_KEYS)
        self.full_row = bundle.get("input") == "hog|color|texture"
        self.gate = bundle.get("gate")
        self.temperature = float(entry.get("temperature") or 1.0)
        if "clf__n_jobs" in self.estimator.get_params():
            self.estimator.set_params(clf__n_jobs=1)
        self.score_type = "calibrated_probability" if self.full_row else "predict_proba"
        self.input = f"256x256 RGB -> handcrafted features ({self.feature_set})"
        self.runtime = "scikit-learn"

    def analyse(self, rgb: np.ndarray):
        feats = extract_features(rgb)
        if not self.full_row:  # earlier classical models (benchmark only): raw probabilities, no gate
            return self.estimator.predict_proba(feature_vector(feats, self.feature_set)[None, :])[0], None
        row = np.concatenate([feats["hog"], feats["color"], feats["texture"]])[None, :]
        logits = np.log(np.clip(self.estimator.predict_proba(row)[0], 1e-6, 1.0))
        gate = self.gate
        distance = getattr(gate["density"], gate["score"])(gate["scaler"].transform(row[:, gate["columns"]]))[0]
        return self.calibrate(_softmax(logits, self.temperature)), float(distance / gate["threshold"])

    @property
    def calibration(self):
        if not self.full_row:
            return None
        kind = "per-class vector scaling" if (self.adjust or {}).get("calibration") else "temperature scaling"
        return {"method": kind, "temperature": self.temperature, "pipeline": self.pipeline}


class ModelRegistry:
    """The models listed in deployment.json. The API loads them all at startup (lazy=False) unless
    TOMATO_LAZY_MODELS=1; `lazy=True` loads each model the first time it is used (less memory on a small server)."""

    def __init__(self, deployment_path=EXPERIMENTS_DIR / "deployment.json", device: str = "cpu", lazy: bool = False):
        self.device = torch.device(device)
        self.models: dict[str, Predictor] = {}
        self.errors: dict[str, str] = {}
        self.default_key, self.threshold, self.entries, self.meta = None, 0.6, [], {}
        self._lock = threading.RLock()
        self._members: dict = {}  # fold networks shared between ensembles (EnsemblePredictor `loaded`)
        if not deployment_path.exists():
            self.errors["deployment"] = f"{deployment_path.name} not found - run the training pipeline first"
            return
        deployment = load_json(deployment_path)
        self.default_key = deployment["default_model"]
        self.threshold = float(deployment["confidence_threshold"])
        self.entries = deployment["models"]
        self.meta = {k: v for k, v in deployment.items() if k != "models"}
        lazy = lazy or os.getenv("TOMATO_LAZY_MODELS") == "1"
        if not lazy:
            for entry in self.entries:
                self.load(entry["key"])
        else:
            self.load(self.default_key)

    def entry(self, key: str) -> dict | None:
        return next((e for e in self.entries if e["key"] == key), None)

    def load(self, key: str) -> Predictor | None:
        """Load one model once. A broken model file is recorded in `errors` (and returns None), never raised:
        it must not take the whole app down."""
        with self._lock:
            if key not in self.models and key not in self.errors:
                entry = self.entry(key)
                if entry is None:
                    raise KeyError(f"unknown model {key!r}")
                builders = {"torch": lambda e: TorchPredictor(e, self.device), "sklearn": SklearnPredictor,
                            "ensemble": lambda e: EnsemblePredictor(e, self.device, loaded=self._members)}
                try:
                    self.models[key] = builders[entry["type"]](entry)
                except Exception as exc:
                    log.exception("Failed to load %s", key)
                    self.errors[key] = str(exc)
            return self.models.get(key)

    @property
    def ready(self) -> bool:
        return self.default_key in self.models

    def threshold_for(self, predictor: Predictor) -> float:
        """The model's own validation-chosen threshold if it has one, else the global one."""
        return float(predictor.threshold) if predictor.threshold is not None else self.threshold

    def predict(self, rgb: np.ndarray, key: str | None = None, explain: bool = False) -> dict:
        predictor = self.models.get(key or self.default_key) or self.load(key or self.default_key)
        if predictor is None:
            raise KeyError(f"model {key or self.default_key!r} is not available")
        t0 = time.perf_counter()
        probs, ood_ratio = predictor.analyse(rgb)
        elapsed = 1000 * (time.perf_counter() - t0)
        top = predictor.decide(probs)  # index into this model's own class list
        ood_ratio = predictor.unusualness(ood_ratio, top)
        top_cls = CLASSES[CLASS_KEYS.index(predictor.class_keys[top])]
        explanation = None
        if explain:
            if isinstance(predictor, (TorchPredictor, EnsemblePredictor)):
                explanation = predictor.explain(rgb, top)
                explanation["class_index"] = top_cls["index"]
            else:
                explanation = {"method": None, "overlay": None, "heatmap": None,
                               "note": "Grad-CAM needs a neural network; this classical model uses handcrafted "
                                       f"{_feature_words(predictor.feature_set)} features."}
        confidence = float(probs[top])
        threshold = self.threshold_for(predictor)
        unusual = ood_ratio is not None and ood_ratio > 1.0
        uncertain = confidence < threshold  # below the threshold the app gives no diagnosis ("Uncertain")
        is_healthy = top_cls["key"] == "healthy"
        status = "Unusual image" if unusual else "Uncertain" if uncertain else ("Healthy" if is_healthy else "Diseased")
        quality = assess_quality(rgb)
        order = np.argsort(probs)[::-1][:3]
        return {
            "model": predictor.info,
            "prediction": {"index": top_cls["index"], "key": top_cls["key"], "name": top_cls["name"],
                           "confidence": confidence, "is_healthy": is_healthy, "status": status},
            "uncertain": uncertain or unusual,
            "confidence_threshold": threshold,
            "operating_point": predictor.operating_point,
            "ood": None if ood_ratio is None else {
                "ratio": ood_ratio, "flag": unusual,
                "method": (predictor.ood_config or {}).get("method", "feature-space distance to training photos"),
                "note": "ratio = distance to the nearest training photos / threshold chosen on out-of-fold data; "
                        "above 1 the photo looks unlike every training photo"},
            "classes": predictor.class_keys,
            "probabilities": [{"index": CLASS_KEYS.index(k), "key": k, "name": CLASSES[CLASS_KEYS.index(k)]["name"],
                               "probability": float(probs[i])} for i, k in enumerate(predictor.class_keys)],
            "score_type": predictor.score_type,
            "calibration": predictor.calibration,
            "inference_ms": elapsed,
            "explanation": explanation,
            "quality": {"ok": quality["ok"], "issues": quality["issues"]},
            "candidates": [{"key": predictor.class_keys[i], "name": CLASSES[CLASS_KEYS.index(predictor.class_keys[i])]["name"],
                            "probability": float(probs[i])} for i in order],
            "refusal": refusal_reason(status, confidence, threshold, ood_ratio, quality, predictor.class_keys),
        }

    def compare(self, rgb: np.ndarray, keys=None) -> list:
        """Run several models (default: every model, deployment order) on the same image, without explanations."""
        keys = keys or [e["key"] for e in self.entries]
        return [self.predict(rgb, k) for k in keys if self.load(k) is not None]
