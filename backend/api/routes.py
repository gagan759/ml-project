"""REST endpoints. Contract: docs/API.md"""
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from tomatoleaf.preprocessing import prepare_upload

router = APIRouter(prefix="/api")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "BMP"}  # MPO = multi-picture JPEG written by some phone cameras
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/pjpeg", "image/png", "image/webp", "image/bmp",
                         "image/x-ms-bmp", "application/octet-stream", ""}
Image.MAX_IMAGE_PIXELS = 60_000_000  # reject decompression bombs; 60 MP covers any phone photo


@router.get("/health")
def health(request: Request):
    registry, store = request.app.state.registry, request.app.state.store
    return {"status": "ok" if registry.ready else "degraded", "model_loaded": registry.ready,
            "default_model": registry.default_key, "loaded_models": list(registry.models),
            "device": str(registry.device), "analytics_available": store.analytics is not None,
            "version": request.app.version, "load_errors": registry.errors}


def _weights_mb(key: str) -> float | None:
    """Download size of a model's weight files (release manifest), or None if the manifest is missing."""
    from tomatoleaf import weights

    try:
        return sum(f["bytes"] for f in weights.files_for([key])) / 2**20 or None
    except (OSError, KeyError, ValueError):
        return None


@router.get("/models")
def models(request: Request):
    registry = request.app.state.registry
    out = []
    for entry in registry.entries:
        predictor = registry.models.get(entry["key"])
        out.append({"key": entry["key"], "name": entry["name"], "family": entry["family"], "kind": entry["kind"],
                    "task": entry.get("task"), "generation": entry.get("generation"),
                    "input": predictor.input if predictor else None, "features": entry["feature_set"],
                    "runtime": getattr(predictor, "runtime", "scikit-learn") if predictor else None,
                    "params": entry.get("params"), "loaded": predictor is not None,
                    "classes": predictor.class_keys if predictor else entry.get("class_keys"),
                    "confidence_threshold": registry.threshold_for(predictor) if predictor else entry.get(
                        "confidence_threshold", registry.threshold),
                    "operating_point": entry.get("operating_point"), "ood": entry.get("ood"),
                    "selected": entry["key"] == registry.default_key, "test_metrics": entry.get("test_metrics"),
                    "latency_ms": entry.get("latency_ms"), "description": entry.get("description"),
                    "model_name": entry.get("model_name"), "weights_mb": _weights_mb(entry["key"])})
    return {"default_model": registry.default_key, "defaults_by_task": registry.meta.get("defaults_by_task"),
            "models": out}


@router.get("/classes")
def classes(request: Request):
    store = request.app.state.store
    if store.diseases is None:
        raise HTTPException(503, "Disease library not available (backend/data/disease_info.json missing)")
    return store.diseases


@router.get("/analytics")
def analytics(request: Request):
    store = request.app.state.store
    if store.analytics is None:
        raise HTTPException(503, "Analytics not available - run scripts/27_export_platform_data.py after training")
    return store.analytics


def _validate(data: bytes) -> str:
    """Validate by content, not by file name: the bytes must decode as one of the allowed formats."""
    try:
        with Image.open(BytesIO(data)) as probe:
            fmt = probe.format
    except UnidentifiedImageError:  # not an image at all (must precede OSError: it is a subclass)
        raise HTTPException(415, "Unsupported file: please upload a JPEG, PNG, WEBP or BMP image") from None
    except Image.DecompressionBombError:
        raise HTTPException(413, "Image resolution too large (maximum 60 megapixels)") from None
    except Exception:  # recognised image type, but the header itself is truncated or corrupt
        raise HTTPException(400, "The image file is corrupt or truncated and could not be decoded") from None
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(415, f"Unsupported image format '{fmt}': please upload a JPEG, PNG, WEBP or BMP image")
    return fmt


async def _read_upload(request: Request, file: UploadFile):
    """Shared upload validation for /predict and /compare -> (canonical RGB array, image info)."""
    registry = request.app.state.registry
    if not registry.ready:
        raise HTTPException(503, "Model not loaded - download the weights (scripts/download_weights.py) and restart")
    if (file.content_type or "").lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(415, f"Unsupported content type '{file.content_type}': please upload an image")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large: the maximum size is 10 MB")
    if not data:
        raise HTTPException(400, "The uploaded file is empty")
    fmt = _validate(data)
    try:
        rgb, info = await run_in_threadpool(prepare_upload, data)
    except Image.DecompressionBombError:
        raise HTTPException(413, "Image resolution too large (maximum 60 megapixels)") from None
    except Exception:
        raise HTTPException(400, "The image file is corrupt or truncated and could not be decoded") from None
    info["format"] = "JPEG" if fmt == "MPO" else fmt
    return rgb, info


def _input_preview(rgb) -> str:
    from backend.services.predictor import png_data_url

    return png_data_url(rgb)


@router.post("/predict")
async def predict(request: Request, file: UploadFile = File(...), model: str | None = Query(default=None),
                  explain: bool = Query(default=False)):
    registry = request.app.state.registry
    if model is not None and registry.entry(model) is None:
        raise HTTPException(404, f"Unknown model '{model}'. Available: {', '.join(e['key'] for e in registry.entries)}")
    rgb, info = await _read_upload(request, file)
    try:
        result = await run_in_threadpool(registry.predict, rgb, model, explain)
    except KeyError as exc:
        raise HTTPException(503, str(exc)) from None
    result["image"] = info
    if explain:  # what the model actually saw, after the dataset-style preprocessing
        result["input_image"] = _input_preview(rgb)
    return result


@router.post("/compare")
async def compare(request: Request, file: UploadFile = File(...), models: str | None = Query(default=None),
                  task: str | None = Query(default=None)):
    """Run several models (all by default; or a comma-separated `models` list, or one `task`) on the same image."""
    registry = request.app.state.registry
    keys = None
    if models:
        keys = [k for k in models.split(",") if k]
        unknown = [k for k in keys if registry.entry(k) is None]
        if unknown:
            raise HTTPException(404, f"Unknown model(s): {', '.join(unknown)}")
    elif task:
        keys = [e["key"] for e in registry.entries if e.get("task") == task]
        if not keys:
            raise HTTPException(404, f"No models for task '{task}'")
    rgb, info = await _read_upload(request, file)
    results = await run_in_threadpool(registry.compare, rgb, keys)
    return {"default_model": registry.default_key, "image": info, "results": results}
