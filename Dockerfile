# syntax=docker/dockerfile:1.7
# Tomato Care — one container: FastAPI + PyTorch/ONNX Runtime (CPU) serving the API and the built React site.
#   docker build -t tomato-care .                       # downloads the model weights from the GitHub release
#   docker build --build-arg DOWNLOAD_WEIGHTS=0 -t t .    # image without weights (mount them at /app/experiments/models)
#   docker run -p 8000:8000 tomato-care                  # http://localhost:8000

# ---------- stage 1: build the web app
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN echo "VITE_API_BASE_URL=/" > .env.production && npm run build

# ---------- stage 2: Python runtime
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src:/app \
    TOMATO_DEVICE=cpu \
    TOMATO_THREADS=4 \
    TOMATO_RUNTIME=onnx
WORKDIR /app

RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY backend ./backend
COPY scripts/download_weights.py ./scripts/
COPY experiments/deployment.json experiments/platform.json experiments/weights_manifest.json ./experiments/

# Weights from the GitHub release (SHA-256 verified). Loading every model once also exports and caches its ONNX graph,
# so the container starts quickly.
ARG DOWNLOAD_WEIGHTS=1
RUN if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then \
      python scripts/download_weights.py && \
      python -c "from backend.services.predictor import ModelRegistry; r = ModelRegistry(); print('models', list(r.models), r.errors)"; \
    else \
      mkdir -p experiments/models; \
    fi

COPY --from=web /web/dist ./frontend/dist

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
