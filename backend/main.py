"""FastAPI application. Run from the project root:  uvicorn backend.main:app --port 8000

One-server mode: once the React app has been built (`npm run build` -> frontend/dist), this server also serves it at
http://127.0.0.1:8000/, so the whole platform runs as one process (this is what the Docker image does). During
development the Vite dev server (port 5173) is used instead and calls this API across origins (CORS below).

Environment variables:
  TOMATO_DEVICE        cpu (default) or cuda
  TOMATO_THREADS       CPU threads for PyTorch (default: min(8, cores))
  TOMATO_LAZY_MODELS   1 = load only the default model at startup, others on first use (small servers)
  TOMATO_RUNTIME       onnx (default on CPU: ONNX Runtime, ~1.6x faster per image) or torch
  TOMATO_CORS_ORIGINS  comma-separated extra origins allowed to call the API
"""
import logging
import mimetypes
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

from backend.api.routes import router
from backend.services.predictor import ModelRegistry
from backend.services.store import DataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173"
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
# Windows can map .js to text/plain in the registry; browsers refuse ES modules served that way.
for _type, _ext in [("text/javascript", ".js"), ("text/javascript", ".mjs"), ("text/css", ".css"),
                    ("image/svg+xml", ".svg"), ("application/json", ".json"), ("image/webp", ".webp"),
                    ("image/x-icon", ".ico"), ("application/manifest+json", ".webmanifest")]:
    mimetypes.add_type(_type, _ext)


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.getenv("TOMATO_THREADS", min(8, os.cpu_count() or 4))))
    # Models and JSON content are loaded ONCE at startup, not on every request.
    app.state.registry = ModelRegistry(device=os.getenv("TOMATO_DEVICE", "cpu"))
    app.state.store = DataStore()
    logging.getLogger("tomato").info("Loaded models: %s (default: %s; errors: %s)", list(app.state.registry.models),
                                     app.state.registry.default_key, app.state.registry.errors)
    yield


app = FastAPI(title="Tomato Care API", version="2.0.0", lifespan=lifespan,
              docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json",
              description="Tomato leaf disease diagnosis with calibrated confidence, out-of-distribution detection "
                          "and Grad-CAM explanations. Contract: docs/API.md")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in os.getenv("TOMATO_CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.include_router(router)


if (FRONTEND_DIST / "index.html").is_file():
    _DIST = FRONTEND_DIST.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        """The built React app: real files (scripts, styles, demo photos) as they are; any page route -> index.html,
        where the React router takes over. Registered last, so /api/* (incl. /api/docs) keeps priority."""
        if path == "api" or path.startswith("api/"):  # an unknown API route gets a JSON 404, never the HTML page
            raise HTTPException(404, "Not Found")
        target = (_DIST / path).resolve()
        if path and target.is_file() and _DIST in target.parents:  # never serve anything outside frontend/dist
            immutable = path.startswith("assets/")  # content-hashed file names
            return FileResponse(target, headers={"Cache-Control": "public, max-age=31536000, immutable"
                                                 if immutable else "public, max-age=3600"})
        return FileResponse(_DIST / "index.html", headers={"Cache-Control": "no-cache"})
