"""Shared state of the Streamlit app: model weights, the model registry, cached predictions and the published results.

Inference uses exactly the code the REST API uses (backend.services.predictor.ModelRegistry) on the same canonical
256x256 image (tomatoleaf.preprocessing.prepare_upload), so the app, the API and the offline benchmark agree.
Defaults suit a small server such as Streamlit Community Cloud (about 2.7 GB RAM, 2 CPUs):
  TOMATO_LOWMEM=1   the fold models of an ensemble share one float32 network; their float16 weights stay
                    memory-mapped on disk (identical predictions, ~1 GB less RAM)
  TOMATO_THREADS    CPU threads (default: the CPUs this container may use)
  TOMATO_DOWNLOAD_WEIGHTS=0  never download weights (tests, offline use)
"""
import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("TOMATO_LOWMEM", "1")
os.environ.setdefault("TOMATO_RUNTIME", "torch")

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from tomatoleaf import weights  # noqa: E402
from tomatoleaf.config import CLASSES  # noqa: E402

EXPERIMENTS = ROOT / "experiments"
PUBLIC = ROOT / "frontend" / "public"
BRAND = ROOT / "assets" / "brand"
REPO_URL = "https://github.com/CodewithBhaumik/Tomato-Care"
DOI_URL = "https://doi.org/10.1007/s00530-023-01158-y"
CLASS_NAME = {c["key"]: c["name"] for c in CLASSES}
TASK = "5class"  # the app diagnoses five conditions; both served models (deep learning, classical ML) cover them
UPLOAD_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]


# ---------------------------------------------------------------------------------------------------------- setup
def cpu_threads() -> int:
    """CPUs this process may use: TOMATO_THREADS, else the container's CPU quota (cgroup v2) or affinity, max 8."""
    if os.getenv("TOMATO_THREADS"):
        return max(1, int(os.environ["TOMATO_THREADS"]))
    n = os.cpu_count() or 2
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, round(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    return max(1, min(n, 8))


DEPLOYMENT_FILE = EXPERIMENTS / "deployment.json"
PLATFORM_FILE = EXPERIMENTS / "platform.json"


def data_stamp(path: Path | None = None) -> tuple:
    """(modification time, size) of a data file. Cached values are keyed on it, so they are rebuilt when a git push
    updates the file: the running app reloads changed code, but st.cache_* entries keyed only on their arguments
    would otherwise keep serving the old model list, results and models until a full restart."""
    s = (path or DEPLOYMENT_FILE).stat()
    return s.st_mtime_ns, s.st_size


@st.cache_data(show_spinner=False, max_entries=2)
def _deployment(stamp: tuple) -> dict:
    return json.loads(DEPLOYMENT_FILE.read_text(encoding="utf-8"))


def deployment() -> dict:
    return _deployment(data_stamp(DEPLOYMENT_FILE))


@st.cache_data(show_spinner=False, max_entries=2)
def _platform(stamp: tuple) -> dict:
    return json.loads(PLATFORM_FILE.read_text(encoding="utf-8"))


def platform() -> dict:
    """Every published result (written by scripts/27_export_platform_data.py from the final benchmark)."""
    return _platform(data_stamp(PLATFORM_FILE))


@st.cache_data(show_spinner=False)
def disease_info() -> dict:
    doc = json.loads((ROOT / "backend" / "data" / "disease_info.json").read_text(encoding="utf-8"))
    return {c["key"]: c for c in doc["classes"]} | {"_disclaimer": doc.get("disclaimer", "")}


def missing_weights(keys=None) -> list[dict]:
    try:
        return [f for f in weights.files_for(keys) if not weights.is_present(f)]
    except FileNotFoundError:  # no manifest: running from a training checkout, weights are local
        return []


_DOWNLOAD_LOCK = threading.Lock()  # one download at a time per server process, whatever the number of visitors
_READY = threading.Event()
_READY_FOR: dict = {"stamp": None}  # the deployment.json the "ready" flag was set for


def ensure_weights(keys=None) -> None:
    """Download every missing weight file (default: of all deployed models) with a progress bar. Called once per
    page run outside any cached function; after the first success it returns immediately. Files are verified
    with SHA-256 and kept on disk for the life of the server."""
    if os.getenv("TOMATO_DOWNLOAD_WEIGHTS", "1") == "0":  # tests / offline use: never touch the network
        return
    if _READY.is_set() and _READY_FOR["stamp"] != data_stamp():  # deployment changed since: check the files again
        _READY.clear()
    if _READY.is_set() or not missing_weights(keys):
        if keys is None:
            _READY_FOR["stamp"] = data_stamp()
            _READY.set()
        return
    with st.spinner("Preparing the models (first start only)…"), _DOWNLOAD_LOCK:
        todo = missing_weights(keys)  # another visitor may have fetched them while we waited
        total = sum(f["bytes"] for f in todo)
        bar = st.progress(0.0, text="Downloading model weights…") if todo else None
        done = 0
        for f in todo:
            base = done

            def progress(got, _size, base=base):
                bar.progress(min(1.0, (base + got) / total),
                             text=f"Downloading model weights (first start only) · {(base + got) / 2**20:,.0f} of "
                                  f"{total / 2**20:,.0f} MB")

            weights.fetch(f, on_progress=progress)
            done += f["bytes"]
        if bar is not None:
            bar.empty()
    if keys is None:
        _READY_FOR["stamp"] = data_stamp()
        _READY.set()


@st.cache_resource(show_spinner=False, max_entries=1)
def _registry(stamp: tuple):
    import torch

    from backend.services.predictor import ModelRegistry

    torch.set_num_threads(cpu_threads())
    return ModelRegistry(DEPLOYMENT_FILE, device="cpu", lazy=True)


def registry():
    """The model registry, shared by every visitor. Only the default model loads now; others on first use.
    Rebuilt when deployment.json changes (max_entries=1 drops the old models)."""
    return _registry(data_stamp())


def model_entry(key: str) -> dict:
    return next(e for e in deployment()["models"] if e["key"] == key)


def served_models() -> list[dict]:
    """The served models, default (deep learning) first."""
    default = deployment()["default_model"]
    return sorted(deployment()["models"], key=lambda e: e["key"] != default)


def scope() -> list[str]:
    """The conditions the app diagnoses (the default model's classes)."""
    return list(model_entry(deployment()["default_model"])["class_keys"])


def scope_text() -> str:
    names = [CLASS_NAME[k] for k in scope()]
    return f"{', '.join(names[:-1])} and {names[-1]}: the {len(names)} conditions the app diagnoses."


def is_classical(entry: dict) -> bool:
    return entry.get("type") == "sklearn"


def family_label(entry: dict) -> str:
    return "Classical ML" if is_classical(entry) else "Deep learning"


def display_name(entry: dict | None) -> str:
    """The model's name in the apps (e.g. ConvNeXt-Tiny ensemble, Gradient Boosting)."""
    if not entry:
        return "—"
    return entry.get("model_name") or family_label(entry)


def short_name(entry: dict) -> str:
    if is_classical(entry):
        return f"{display_name(entry)} · classical ML"
    return f"{display_name(entry)} · {len(entry.get('members') or [1])} networks"


def weights_mb(entry: dict) -> float | None:
    try:
        return sum(f["bytes"] for f in weights.files_for([entry["key"]])) / 2**20 or None
    except (OSError, KeyError, ValueError):
        return None


def benefit(entry: dict) -> str:
    """What this served model does better than the other one, from the measured test numbers."""
    other = next((e for e in served_models() if e["key"] != entry["key"]), None)
    family = family_label(entry)
    if other is None:
        return family
    acc, o_acc = (entry.get("test_metrics") or {}).get("accuracy"), (other.get("test_metrics") or {}).get("accuracy")
    cov = lambda e: ((e.get("operating_point") or {}).get("test") or {}).get("coverage")  # noqa: E731
    parts = []
    if acc and o_acc and acc > o_acc:
        parts.append(f"more accurate ({pct(acc)} vs {pct(o_acc)} on test photos)")
    if cov(entry) and cov(other) and cov(entry) > cov(other):
        parts.append(f"answers more photos ({pct(cov(entry), 0)} vs {pct(cov(other), 0)})")
    if not is_classical(entry):
        parts.append("shows where it looked (Grad-CAM)")
    lat, o_lat = entry.get("latency_ms"), other.get("latency_ms")
    if lat and o_lat and lat < o_lat:
        parts.append(f"{o_lat / lat:.1f}× faster ({lat:.0f} vs {o_lat:.0f} ms per photo on CPU)")
    mb, o_mb = weights_mb(entry), weights_mb(other)
    if mb and o_mb and mb < o_mb:
        parts.append(f"{o_mb / mb:.0f}× smaller ({mb:.1f} MB vs {o_mb:.0f} MB)")
    if is_classical(entry):
        parts.append("no neural network")
    return f"{family}: " + ", ".join(parts) + "." if parts else family


def model_help(entry: dict) -> str:
    """One line on how the served model was built (its validated pipeline)."""
    p = entry.get("pipeline")
    if not p:
        return entry.get("description") or ""
    tail = (f"; {p['calibration']} scaling; answers at ≥ {pct(p['threshold'], 0)} confidence. "
            "All chosen on cross-validation data before the test photos were scored.")
    if is_classical(entry):
        feats = p["features"].replace("Color", "colour").replace("Texture", "texture")
        return f"{p['classifier']} on handcrafted {feats} features of the leaf (no deep learning)" + tail
    return (f"{p['imbalance'][0].upper() + p['imbalance'][1:]}"
            + (" + outlier exposure (learned to be unsure on conditions outside the five)"
               if p.get("outlier_exposure") else "") + tail)


# ---------------------------------------------------------------------------------------------------- inference
def canonical(image_bytes: bytes):
    """(256x256 RGB array as the models see it, details of the conversion)."""
    from tomatoleaf.preprocessing import prepare_upload

    return prepare_upload(io.BytesIO(image_bytes))


def data_url_to_rgba(url: str | None) -> np.ndarray | None:
    if not url:
        return None
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA"))


def predict(image_bytes: bytes, model_key: str, explain: bool = True) -> dict:
    """One diagnosis (the API's /api/predict response) plus the canonical input and the Grad-CAM heat map."""
    return _predict(image_bytes, model_key, explain, data_stamp())


@st.cache_data(max_entries=64, show_spinner=False)
def _predict(image_bytes: bytes, model_key: str, explain: bool, stamp: tuple) -> dict:
    rgb, info = canonical(image_bytes)
    t0 = time.perf_counter()
    try:
        out = registry().predict(rgb, model_key, explain=explain)
    except Exception as exc:  # never lose the diagnosis because the heat map failed
        if not explain:
            raise
        out = registry().predict(rgb, model_key, explain=False)
        out["explanation"] = {"method": None, "overlay": None, "heatmap": None,
                              "note": f"Grad-CAM is unavailable on this server ({type(exc).__name__})."}
    out["total_ms"] = 1000 * (time.perf_counter() - t0)
    out["input"] = info
    out["canonical"] = rgb
    exp = out.get("explanation") or {}
    out["heatmap_rgba"] = data_url_to_rgba(exp.get("heatmap"))
    return out


def compare(image_bytes: bytes, keys: tuple) -> list[dict]:
    return _compare(image_bytes, keys, data_stamp())


@st.cache_data(max_entries=16, show_spinner=False)
def _compare(image_bytes: bytes, keys: tuple, stamp: tuple) -> list[dict]:
    rgb, _ = canonical(image_bytes)
    reg = registry()
    rows = []
    for key in keys:
        t0 = time.perf_counter()
        r = reg.predict(rgb, key)
        r["total_ms"] = 1000 * (time.perf_counter() - t0)
        rows.append(r)
    return rows


def blend(rgb: np.ndarray, heat_rgba: np.ndarray, opacity: float) -> np.ndarray:
    """Grad-CAM heat map over the photo; the map's own alpha (= strength) is scaled by `opacity`."""
    heat = np.asarray(Image.fromarray(heat_rgba).resize(rgb.shape[1::-1], Image.BILINEAR)).astype(np.float32)
    alpha = heat[..., 3:4] / 255.0 * opacity
    return (rgb.astype(np.float32) * (1 - alpha) + heat[..., :3] * alpha).clip(0, 255).astype(np.uint8)


# ------------------------------------------------------------------------------------------------------ samples
@st.cache_data(show_spinner=False)
def samples() -> list[dict]:
    """Demo photos shipped with the app (Tomato-Village dataset, cited): one held-out test photo per diagnosed
    condition first, then three more per condition."""
    out, keys = [], set(scope())
    manifest = json.loads((PUBLIC / "test-samples" / "manifest.json").read_text(encoding="utf-8"))
    for s in manifest["samples"]:
        if s["key"] in keys:
            out.append({"key": s["key"], "name": s["name"], "path": PUBLIC / s["file"].lstrip("/"), "held_out": True})
    for p in sorted((PUBLIC / "samples").glob("*.jpg")):
        key = p.stem.rsplit("-", 1)[0]
        if key in keys:
            out.append({"key": key, "name": CLASS_NAME[key], "path": p, "held_out": False})
    return out


def sample_bytes(path) -> bytes:
    return Path(path).read_bytes()


# ------------------------------------------------------------------------------------------------- result helpers
def model_result(key: str) -> dict | None:
    return next((m for m in platform()["models"] if m["key"] == key), None)


def task_defaults(task: str = TASK) -> tuple[dict, dict]:
    t = platform()["tasks"][task]
    return model_result(t["default_model"]), model_result(t["previous_default"])


def classical_result() -> dict | None:
    """Test results of the served classical model."""
    return next((model_result(e["key"]) for e in served_models() if is_classical(e)), None)


def headline_comparison(task: str = TASK) -> dict | None:
    t = platform()["tasks"][task]
    return next((c for c in platform()["comparisons"]
                 if c["task"] == task and c["a"] == t["default_model"] and c["b"] == t["previous_default"]), None)


def pct(x, digits=1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}%"


def pts(x, digits=1) -> str:
    return "—" if x is None else f"{100 * x:+.{digits}f} pts"

