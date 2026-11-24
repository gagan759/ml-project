"""API tests. Run from the project root:  python -m pytest -q

Needs the deployed model weights (python scripts/download_weights.py); without them the whole module is skipped.
"""
import base64
from io import BytesIO

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from tomatoleaf.config import CLASS_KEYS, DATASET_DIR, EXPERIMENTS_DIR, MODELS_DIR
from tomatoleaf.utils import load_json

FIVE = ["healthy", "leaf_miner", "magnesium_deficiency", "nitrogen_deficiency", "spotted_wilt_virus"]
STATUSES = {"Healthy", "Diseased", "Uncertain", "Unusual image"}


def _weights_present() -> bool:
    path = EXPERIMENTS_DIR / "deployment.json"
    if not path.exists():
        return False
    dep = load_json(path)
    default = next(e for e in dep["models"] if e["key"] == dep["default_model"])
    paths = [m["path"] for m in default.get("members", [])] or [default["path"]]
    return all((MODELS_DIR / p).exists() for p in paths)


pytestmark = pytest.mark.skipif(not _weights_present(), reason="model weights not downloaded")


@pytest.fixture(scope="module")
def client():
    from backend.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def test_photo():
    if not DATASET_DIR.exists():  # CI without the dataset: use a bundled held-out sample photo
        from pathlib import Path
        p = Path(__file__).resolve().parents[2] / "frontend" / "public" / "test-samples" / "leaf_miner.jpg"
        return {"key": "leaf_miner", "path": None}, p.read_bytes()
    from tomatoleaf.data import load_splits
    splits = load_splits()
    row = splits[(splits.split == "test") & (splits.class_key == "leaf_miner")].iloc[0]
    return {"key": row.class_key, "path": row.path}, (DATASET_DIR / row.path).read_bytes()


def post(client, data, query="", name="leaf.jpg", ctype="image/jpeg"):
    return client.post(f"/api/predict{query}", files={"file": (name, data, ctype)})


def image_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


# ------------------------------------------------------------------ metadata endpoints
def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["model_loaded"]
    assert body["default_model"] in body["loaded_models"]
    assert body["analytics_available"]


def test_two_five_condition_models_are_served(client):
    """Deep learning (default) and classical ML, both for the same five conditions; no eight-condition model."""
    body = client.get("/api/models").json()
    assert body["defaults_by_task"] == {"5class": body["default_model"]}
    by_key = {m["key"]: m for m in body["models"]}
    assert {m["kind"] for m in body["models"]} == {"ensemble", "classical"}
    assert by_key[body["default_model"]]["kind"] == "ensemble"
    assert all(m["task"] == "5class" and m["classes"] == FIVE for m in body["models"])
    assert all(m["loaded"] for m in body["models"]), [m["key"] for m in body["models"] if not m["loaded"]]
    assert {m["generation"] for m in body["models"]} == {"upgrade"}
    for m in body["models"]:
        assert m["operating_point"]["chosen_on"].startswith("out-of-fold")
        assert m["test_metrics"]["accuracy"] > 0.5 and m["latency_ms"] > 0


def test_library_lists_the_diagnosed_conditions(client):
    body = client.get("/api/classes").json()
    assert [c["key"] for c in body["classes"]] == FIVE
    for c in body["classes"]:
        assert c["symptoms"] and c["management"] and c["references"]


def test_analytics_numbers_are_internally_consistent(client):
    body = client.get("/api/analytics").json()
    for task, info in body["tasks"].items():
        k = len(info["classes"])
        models = [m for m in body["models"] if m["task"] == task]
        assert sum(m["key"] == info["default_model"] for m in models) == 1
        for m in models:
            cm = np.array(m["confusion"])
            assert cm.shape == (k, k) and cm.sum() == info["test_photos"], m["key"]
            assert abs(np.trace(cm) / cm.sum() - m["test"]["accuracy"]) < 1e-9, m["key"]
            assert [p["key"] for p in m["per_class"]] == info["classes"]
    assert sum(c["photos"] for c in body["dataset"]["classes"]) == body["dataset"]["photos"]


# ------------------------------------------------------------------ predictions
def test_default_prediction_is_a_calibrated_distribution(client, test_photo):
    _, data = test_photo
    r = post(client, data)
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [p["probability"] for p in body["probabilities"]]
    assert body["classes"] == FIVE and [p["key"] for p in body["probabilities"]] == FIVE
    assert abs(sum(probs) - 1) < 1e-5 and body["prediction"]["confidence"] == pytest.approx(max(probs))
    assert body["prediction"]["status"] in STATUSES
    assert body["prediction"]["index"] == CLASS_KEYS.index(body["prediction"]["key"])
    assert body["ood"] is not None and body["ood"]["ratio"] > 0
    assert body["image"]["format"] == "JPEG" and body["image"]["colour"] == "Display P3 (unchanged)"


def test_served_models_answer_the_tallest_bar(client):
    """Each served model's answer is always the most probable class (no separate decision rule that could disagree
    with the probability chart). Checked on every gallery photo."""
    from pathlib import Path

    info = client.get("/api/models").json()
    gallery = Path(__file__).resolve().parents[2] / "frontend" / "public" / "test-samples"
    for f in sorted(gallery.glob("*.jpg")):
        for key in (m["key"] for m in info["models"]):
            body = post(client, f.read_bytes(), f"?model={key}").json()
            top = max(body["probabilities"], key=lambda p: p["probability"])
            assert body["prediction"]["key"] == top["key"], (f.name, key)
            assert body["prediction"]["confidence"] == pytest.approx(top["probability"])


def test_classical_model_is_calibrated_and_gated(client, test_photo):
    """The classical model: five calibrated probabilities, its own unusual-image ratio, and no Grad-CAM."""
    _, data = test_photo
    key = next(m["key"] for m in client.get("/api/models").json()["models"] if m["kind"] == "classical")
    body = post(client, data, f"?model={key}&explain=true").json()
    probs = [p["probability"] for p in body["probabilities"]]
    assert [p["key"] for p in body["probabilities"]] == FIVE and abs(sum(probs) - 1) < 1e-6
    assert body["ood"] is not None and body["ood"]["ratio"] > 0
    assert body["explanation"]["overlay"] is None and "neural network" in body["explanation"]["note"]
    noise = Image.fromarray(np.random.default_rng(1).integers(0, 255, (400, 300, 3), dtype=np.uint8))
    assert post(client, image_bytes(noise), f"?model={key}").json()["ood"]["ratio"] > body["ood"]["ratio"]


def test_api_matches_offline_pipeline(client, test_photo):
    """Offline evaluation (tomatoleaf.training.infer, float32) and the API must agree to 1e-5 for the default."""
    meta, data = test_photo
    if meta["path"] is None:
        pytest.skip("dataset not available")
    from tomatoleaf.data import load_images
    from tomatoleaf.models import load_checkpoint
    from tomatoleaf.training import infer

    registry = client.app.state.registry
    predictor = registry.models[registry.default_key]
    specs = registry.entry(registry.default_key)["members"]
    images = load_images([meta["path"]])
    probs = []
    for spec, m in zip(specs, predictor.members, strict=True):
        model, _ = load_checkpoint(MODELS_DIR / spec["path"], "cpu")  # independent of the API's loaded objects
        logits = infer(model, images, torch.device("cpu"), size=m.size, mean=m.mean, std=m.std, resize=m.resize,
                       views=m.tta_views)[0] / m.temperature
        e = np.exp(logits - logits.max())
        probs.append(e / e.sum())
    expected = np.mean(probs, 0)
    adjust = registry.entry(registry.default_key).get("adjust")
    if adjust:  # per-class calibration fitted on out-of-fold data
        from tomatoleaf import balance
        expected = balance.apply(expected, adjust["calibration"])
    api = post(client, data).json()
    np.testing.assert_allclose([p["probability"] for p in api["probabilities"]], expected, atol=1e-5)


def test_threshold_decides_uncertain(client, test_photo):
    _, data = test_photo
    body = post(client, data).json()
    conf, thr = body["prediction"]["confidence"], body["confidence_threshold"]
    if body["prediction"]["status"] in ("Healthy", "Diseased"):
        assert conf >= thr and not body["ood"]["flag"]
    assert body["operating_point"]["threshold"] == pytest.approx(thr)


def test_noise_image_looks_more_unusual_than_a_leaf(client, test_photo):
    _, data = test_photo
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 300, 3), dtype=np.uint8))
    leaf = post(client, data).json()["ood"]["ratio"]
    other = post(client, image_bytes(noise)).json()
    assert other["ood"]["ratio"] > leaf


def test_response_explains_refusals_and_lists_top_candidates(client, test_photo):
    _, data = test_photo
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (400, 300, 3), dtype=np.uint8))
    for body in (post(client, data).json(), post(client, image_bytes(noise)).json()):
        answered = body["prediction"]["status"] in ("Healthy", "Diseased")
        probs = sorted((p["probability"] for p in body["probabilities"]), reverse=True)
        assert [c["probability"] for c in body["candidates"]] == pytest.approx(probs[:3])
        assert body["candidates"][0]["key"] == body["prediction"]["key"]
        assert (body["refusal"] is None) == answered
        assert set(body["quality"]) == {"ok", "issues"}
        if body["refusal"]:
            assert body["refusal"]["code"] in ("unusual_image", "low_confidence") and body["refusal"]["hints"]
    assert not body["quality"]["ok"]  # random noise fails the photo-quality checks


def test_explain_returns_heatmap_overlay_and_model_input(client, test_photo):
    _, data = test_photo
    body = post(client, data, "?explain=true").json()
    exp = body["explanation"]
    assert exp["method"] == "Grad-CAM" and exp["class_index"] == body["prediction"]["index"]
    for field in ("overlay", "heatmap"):
        img = Image.open(BytesIO(base64.b64decode(exp[field].split(",", 1)[1])))
        assert img.format == "PNG" and img.size == (256, 256)
    heat = Image.open(BytesIO(base64.b64decode(exp["heatmap"].split(",", 1)[1])))
    assert heat.mode == "RGBA"
    assert body["input_image"].startswith("data:image/png;base64,")


def test_explain_does_not_change_the_prediction(client, test_photo):
    _, data = test_photo
    plain = post(client, data).json()
    explained = post(client, data, "?explain=true").json()
    np.testing.assert_allclose([p["probability"] for p in plain["probabilities"]],
                               [p["probability"] for p in explained["probabilities"]], atol=1e-6)
    assert plain["explanation"] is None and "input_image" not in plain


def test_classical_model_explain_returns_note(test_photo, tmp_path):
    """No classical model is served any more, but the registry still handles one correctly (benchmark entry)."""
    import json

    from backend.services.predictor import ModelRegistry
    from tomatoleaf.preprocessing import prepare_upload

    rf = next(m for m in load_json(EXPERIMENTS_DIR / "benchmark_models.json")["models"] if m["key"] == "rf")
    if not (MODELS_DIR / rf["path"]).exists():
        pytest.skip("reference model weights not downloaded (scripts/download_weights.py --all)")
    dep = tmp_path / "deployment.json"
    dep.write_text(json.dumps({"default_model": "rf", "confidence_threshold": 0.6, "models": [rf]}), encoding="utf-8")
    body = ModelRegistry(dep, device="cpu").predict(prepare_upload(BytesIO(test_photo[1]))[0], "rf", explain=True)
    assert body["explanation"]["overlay"] is None and body["explanation"]["note"]
    assert body["ood"] is None


def test_every_served_model_predicts(client, test_photo):
    _, data = test_photo
    for m in client.get("/api/models").json()["models"]:
        r = post(client, data, f"?model={m['key']}")
        assert r.status_code == 200, (m["key"], r.text)
        assert r.json()["model"]["key"] == m["key"]


def test_compare_runs_models_on_the_same_image(client, test_photo):
    _, data = test_photo
    models = client.get("/api/models").json()["models"]
    body = client.post("/api/compare?task=5class", files={"file": ("leaf.jpg", data, "image/jpeg")}).json()
    assert [r["model"]["key"] for r in body["results"]] == [m["key"] for m in models if m["task"] == "5class"]
    pick = [models[0]["key"], models[-1]["key"]]
    body = client.post(f"/api/compare?models={','.join(pick)}", files={"file": ("leaf.jpg", data, "image/jpeg")}).json()
    assert [r["model"]["key"] for r in body["results"]] == pick
    body = client.post("/api/compare", files={"file": ("leaf.jpg", data, "image/jpeg")}).json()
    assert [r["model"]["key"] for r in body["results"]] == [m["key"] for m in models]  # both version-2 models
    r = client.post("/api/compare?models=nope", files={"file": ("leaf.jpg", data, "image/jpeg")})
    assert r.status_code == 404


# ------------------------------------------------------------------ preprocessing of real-world uploads
def test_wide_photo_is_cropped_to_four_by_three(client):
    img = Image.new("RGB", (3000, 2000), (120, 160, 80))
    body = post(client, image_bytes(img), name="big.jpg").json()
    assert body["image"]["width"] == 3000 and "centre-cropped to 4:3" in body["image"]["geometry"]


def test_portrait_phone_photo_keeps_the_whole_frame(client):
    img = Image.new("RGB", (1500, 2000), (120, 160, 80))
    body = post(client, image_bytes(img)).json()
    assert body["image"]["geometry"].startswith("whole frame")
    assert "sRGB" in body["image"]["colour"]


def test_png_with_transparency_is_accepted(client):
    img = Image.new("RGBA", (300, 200), (40, 140, 60, 255))
    r = post(client, image_bytes(img, "PNG"), name="leaf.png", ctype="image/png")
    assert r.status_code == 200, r.text
    assert r.json()["image"]["format"] == "PNG"


# ------------------------------------------------------------------ errors
def test_rejects_non_image(client):
    assert post(client, b"hello world", name="notes.txt", ctype="text/plain").status_code == 415


def test_rejects_non_image_disguised_as_jpeg(client):
    assert post(client, b"definitely not a jpeg", name="fake.jpg").status_code == 415


def test_rejects_truncated_jpeg(client, test_photo):
    _, data = test_photo
    assert post(client, data[: len(data) // 3], name="broken.jpg").status_code == 400


def test_rejects_file_over_10mb(client):
    assert post(client, b"\xff" * (10 * 1024 * 1024 + 1), name="huge.jpg").status_code == 413


def test_rejects_empty_file(client):
    assert post(client, b"", name="empty.jpg").status_code == 400


def test_unknown_model_is_404(client, test_photo):
    _, data = test_photo
    assert post(client, data, "?model=resnet9000").status_code == 404


def test_missing_file_is_422(client):
    assert client.post("/api/predict").status_code == 422


def test_cors_preflight_allows_vite_dev_server(client):
    r = client.options("/api/predict", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    r = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_built_frontend_is_served_with_page_fallback(client):
    from backend.main import FRONTEND_DIST
    if not (FRONTEND_DIST / "index.html").is_file():
        pytest.skip("frontend not built (npm run build)")
    for route in ("/", "/diagnose", "/lab", "/library/healthy"):
        r = client.get(route)
        assert r.status_code == 200 and '<div id="root">' in r.text, route
    script = next((FRONTEND_DIST / "assets").glob("*.js")).name
    r = client.get(f"/assets/{script}")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert "immutable" in r.headers["cache-control"]
    r = client.get("/api/no-such-endpoint")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")
    r = client.get("/..%2F..%2FREADME.md")  # nothing outside frontend/dist is ever served
    assert "Tomato Care" not in r.text or '<div id="root">' in r.text


def test_lazy_registry_loads_on_first_use(test_photo):
    from backend.services.predictor import ModelRegistry
    from tomatoleaf.preprocessing import prepare_upload

    _, data = test_photo
    lazy = ModelRegistry(lazy=True)
    assert list(lazy.models) == [lazy.default_key]  # only the default is loaded up front
    rgb, _ = prepare_upload(data)
    other = next(e["key"] for e in lazy.entries if e["key"] != lazy.default_key)
    out = lazy.predict(rgb, other)
    assert other in lazy.models and out["model"]["key"] == other
