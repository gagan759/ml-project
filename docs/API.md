# Tomato Care — REST API

Base URL: `http://127.0.0.1:8000` locally, or the same origin as the site when served by the Docker image.
Interactive documentation (OpenAPI): `/api/docs` (schema at `/api/openapi.json`). All responses are JSON. Errors use HTTP status codes with a
`{"detail": "..."}` body.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Server status, loaded models, device, load errors |
| GET | `/api/models` | Served models: task, classes, thresholds, operating point, test metrics, CPU latency; default per task |
| POST | `/api/predict` | Diagnose one photo (multipart field `file`) |
| POST | `/api/compare` | Run several models on the same photo |
| GET | `/api/classes` | Disease library (symptoms, visual signs, management, references) |
| GET | `/api/analytics` | Every result shown on the website (`experiments/platform.json`) |

## POST /api/predict

Query parameters:

| Name | Default | Meaning |
|---|---|---|
| `model` | default of the focused task | any `key` from `/api/models` |
| `explain` | `false` | also return Grad-CAM (blended overlay and RGBA heat map) and the 256×256 image the model saw |

Upload rules (checked by content, not by file name): JPEG, PNG, WEBP or BMP; at most 10 MB and 60 megapixels.

Preprocessing (the same steps that produced the dataset images, see [METHODOLOGY.md](METHODOLOGY.md#3-preprocessing-audit)):
EXIF orientation → colour converted into Display P3 (sRGB / untagged / other ICC profiles) → aspect ratio clamped
to 3:4…4:3 by a centre crop → whole frame resized to 256×256 → JPEG quality 90. Dataset images pass unchanged.

Response (abridged):

```json
{
  "model": {"key": "upgrade_5class", "name": "...", "family": "Deep learning", "task": "5class"},
  "prediction": {"index": 3, "key": "leaf_miner", "name": "Leaf Miner", "confidence": 0.97,
                 "is_healthy": false, "status": "Diseased"},
  "uncertain": false,
  "confidence_threshold": 0.8,
  "operating_point": {"threshold": 0.8, "chosen_on": "out-of-fold cross-validation predictions",
                      "test": {"coverage": 0.9, "accuracy_kept": 0.97, "photos": 265}},
  "ood": {"ratio": 0.41, "flag": false, "method": "...", "note": "..."},
  "classes": ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"],
  "probabilities": [{"index": 1, "key": "healthy", "name": "Healthy", "probability": 0.01}, "..."],
  "score_type": "calibrated_softmax",
  "calibration": {"method": "temperature_scaling (per member)", "member_temperatures": ["..."]},
  "inference_ms": 380.2,
  "explanation": {"method": "Grad-CAM", "class_index": 3, "overlay": "data:image/png;base64,...",
                  "heatmap": "data:image/png;base64,...", "note": "Grad-CAM of the first ensemble member"},
  "input_image": "data:image/png;base64,...",
  "image": {"width": 3468, "height": 4624, "format": "JPEG", "colour": "Display P3 (unchanged)",
            "geometry": "whole frame, resized to 256x256 like the dataset, JPEG q90"}
}
```

`prediction.status` is one of:

| Status | Meaning |
|---|---|
| `Healthy` / `Diseased` | confident diagnosis (calibrated confidence ≥ the model's threshold, photo looks like training data) |
| `Uncertain` | confidence below the threshold chosen on out-of-fold data: no diagnosis |
| `Unusual image` | the photo's features are far from every training photo (`ood.ratio > 1`): no diagnosis |

`probabilities` are temperature-scaled; for ensembles they are the mean of the members' calibrated probabilities.
`explanation` and `input_image` are present only with `explain=true`; classical models return a note instead of a map.

## POST /api/compare

Multipart field `file`. Optional query: `task=5class|8class` (all models of a task) or `models=key1,key2`
(explicit list, in that order). Default: every served model. Returns
`{"default_model": "...", "image": {...}, "results": [<predict response without explanation>, ...]}`.

## Status codes

| Code | When |
|---|---|
| 400 | empty, truncated or corrupt image |
| 404 | unknown model key or task |
| 413 | file over 10 MB or image over 60 megapixels |
| 415 | not an image, or an unsupported format |
| 422 | the `file` field is missing |
| 503 | the default model is not loaded (weights missing) or analytics not exported |

## Examples

```bash
curl -s http://127.0.0.1:8000/api/health
curl -s -F "file=@leaf.jpg" "http://127.0.0.1:8000/api/predict"
curl -s -F "file=@leaf.jpg" "http://127.0.0.1:8000/api/predict?model=upgrade_8class&explain=true"
curl -s -F "file=@leaf.jpg" "http://127.0.0.1:8000/api/compare?task=5class"
```

```python
import requests
with open("leaf.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/api/predict", files={"file": f}, timeout=60)
r.raise_for_status()
print(r.json()["prediction"])
```
