import { useState } from "react";
import { Boxes, Check, Copy, ExternalLink, FolderTree, Play, Server, Settings2, TestTube2, Workflow } from "lucide-react";
import { useDocumentTitle } from "../components/layout/chrome.jsx";
import { Badge, ButtonLink, Card, CardHeader, PageHeader } from "../components/ui/primitives.jsx";
import { API_DOCS_URL } from "../lib/api.js";

function Code({ children, lang = "bash" }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="group relative">
      <pre className="scrollbar-thin overflow-x-auto rounded-xl border border-line bg-[#0b1210] p-4 font-mono text-[12.5px] leading-relaxed text-[#d5e2da]">
        <code data-lang={lang}>{children}</code>
      </pre>
      <button
        onClick={() => {
          navigator.clipboard?.writeText(children).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          });
        }}
        className="absolute right-2 top-2 grid size-8 place-items-center rounded-lg bg-white/10 text-white/70 opacity-0 transition hover:bg-white/20 group-hover:opacity-100 focus:opacity-100"
        aria-label="Copy"
      >
        {copied ? <Check className="size-4" aria-hidden /> : <Copy className="size-4" aria-hidden />}
      </button>
    </div>
  );
}

const ENDPOINTS = [
  { method: "GET", path: "/api/health", desc: "Server status, loaded models, device, load errors." },
  { method: "GET", path: "/api/models", desc: "The two served models — deep learning (default) and classical ML, both for the same five conditions — with kind, classes, thresholds, operating point and test metrics." },
  { method: "POST", path: "/api/predict?model=&explain=", desc: "Multipart `file`. Calibrated probabilities, status (Healthy / Diseased / Uncertain / Unusual image), OOD ratio, preprocessing details; with explain=true also the 256×256 model input and, for the deep-learning model, Grad-CAM (overlay + RGBA heat map)." },
  { method: "POST", path: "/api/compare?models=", desc: "Runs several models on the same photo: both served models by default, or a comma-separated list of model keys." },
  { method: "GET", path: "/api/classes", desc: "Disease library: symptoms, visual signs, management, references." },
  { method: "GET", path: "/api/analytics", desc: "All results shown on this site (experiments/platform.json)." },
];

const TREE = `tomato-leaf-ai/
├── src/tomatoleaf/        shared library: preprocessing, models, training, evaluation, OOD, Grad-CAM
├── scripts/               numbered pipeline (01–18 original study, 19–42 upgrade, 43–44 classical ML model)
├── backend/               FastAPI app (api/, services/, data/, tests/)
├── frontend/              React + Vite + Tailwind platform (this site)
├── tests/                 unit tests for the shared library
├── data/                  derived metadata: index, leaf groups, splits, CV folds (no images)
├── experiments/           results JSON, deployment.json, platform.json (weights downloaded separately)
├── docs/                  methodology, API, model card, viva notes
├── Dockerfile             one-container deployment (API + built frontend)
└── README.md`;

export default function DocsPage() {
  useDocumentTitle("Documentation");
  return (
    <div>
      <PageHeader
        eyebrow="Developer guide"
        title="Documentation"
        description="Architecture, setup, the REST API and deployment. The repository README has the same information in more detail."
        actions={
          <ButtonLink href={API_DOCS_URL} variant="secondary" icon={ExternalLink}>
            Interactive API docs
          </ButtonLink>
        }
      />
      <div className="flex flex-col gap-6">
        <Card>
          <CardHeader icon={Workflow} title="Architecture" />
          <div className="grid grid-cols-1 gap-4 p-5 md:grid-cols-3">
            {[
              { icon: Boxes, title: "Frontend", text: "React 19 + Vite + Tailwind 4 + Recharts. Reads results from /api/analytics (falls back to a bundled snapshot when the API is offline)." },
              { icon: Server, title: "API", text: "FastAPI. Validates uploads by content, applies the dataset-faithful preprocessing, runs the deep-learning network (PyTorch / ONNX Runtime) or the scikit-learn classical model on CPU, returns calibrated, explained predictions." },
              { icon: Settings2, title: "Models", text: "Two served models for the same five conditions, listed in experiments/deployment.json with their calibration, thresholds and OOD statistics; weights come from a GitHub release (SHA-256 verified)." },
            ].map(({ icon: Icon, title, text }) => (
              <div key={title} className="rounded-xl bg-surface-2 p-4">
                <Icon className="size-5 text-leaf-600" aria-hidden />
                <p className="mt-2 font-semibold text-ink">{title}</p>
                <p className="mt-1 text-sm leading-relaxed text-muted">{text}</p>
              </div>
            ))}
          </div>
          <div className="px-5 pb-5">
            <Code lang="text">{`photo ─▶ validate (type, size, decode) ─▶ EXIF orientation ─▶ colour → Display P3
      ─▶ aspect clamp 3:4..4:3 ─▶ resize whole frame to 256×256 ─▶ JPEG q90
      ─▶ deep learning: network (8-view TTA) ─▶ calibration ─▶ thresholds (Uncertain / Unusual) ─▶ Grad-CAM
      ─▶ classical ML: colour + texture + HOG features ─▶ classifier ─▶ calibration ─▶ thresholds (Uncertain / Unusual)`}</Code>
          </div>
        </Card>

        <Card>
          <CardHeader icon={Play} title="Quick start" subtitle="Python 3.10–3.12, Node 20+" />
          <div className="space-y-4 p-5">
            <p className="text-sm text-ink-2">Run the API and the site locally:</p>
            <Code>{`python -m venv .venv && .venv\\Scripts\\activate      # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt && pip install -e .
python scripts/download_weights.py                  # models from the GitHub release, SHA-256 checked
uvicorn backend.main:app --port 8000                # API on http://127.0.0.1:8000

cd frontend && npm ci && npm run dev                # site on http://localhost:5173`}</Code>
            <p className="text-sm text-ink-2">Or everything in one container:</p>
            <Code>{`docker build -t tomato-care .
docker run -p 8000:8000 tomato-care                # open http://localhost:8000`}</Code>
          </div>
        </Card>

        <Card>
          <CardHeader icon={Server} title="REST API" subtitle="All responses are JSON; errors use HTTP status codes with a `detail` message" />
          <ul className="divide-y divide-line">
            {ENDPOINTS.map((e) => (
              <li key={e.path} className="flex flex-col gap-1.5 px-5 py-3.5 sm:flex-row sm:items-start sm:gap-4">
                <Badge tone={e.method === "GET" ? "sky" : "leaf"} className="w-14 justify-center font-mono">
                  {e.method}
                </Badge>
                <div className="min-w-0">
                  <code className="font-mono text-sm text-ink">{e.path}</code>
                  <p className="mt-0.5 text-sm text-muted">{e.desc}</p>
                </div>
              </li>
            ))}
          </ul>
          <div className="space-y-3 border-t border-line p-5">
            <Code>{`curl -F "file=@leaf.jpg" "http://127.0.0.1:8000/api/predict?explain=false"`}</Code>
            <Code lang="json">{`{
  "model": {"key": "...", "name": "...", "task": "5class"},
  "prediction": {"key": "leaf_miner", "name": "Leaf Miner", "confidence": 0.97, "status": "Diseased"},
  "confidence_threshold": 0.8, "uncertain": false,
  "ood": {"ratio": 0.42, "flag": false, "method": "..."},
  "probabilities": [{"key": "healthy", "probability": 0.01}, ...],
  "image": {"width": 3468, "height": 4624, "format": "JPEG",
            "colour": "Display P3 (unchanged)", "geometry": "whole frame, resized to 256x256 like the dataset, JPEG q90"},
  "inference_ms": 410.5
}`}</Code>
            <p className="text-xs text-muted">Upload limits: 10 MB, 60 megapixels, JPEG / PNG / WEBP / BMP (validated by content, not file name). Errors: 400 corrupt, 404 unknown model, 413 too large, 415 unsupported type, 503 model not loaded.</p>
          </div>
        </Card>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <Card>
            <CardHeader icon={FolderTree} title="Project structure" />
            <div className="p-5">
              <Code lang="text">{TREE}</Code>
            </div>
          </Card>
          <Card>
            <CardHeader icon={TestTube2} title="Tests & configuration" />
            <div className="space-y-4 p-5">
              <Code>{`python -m pytest -q          # library + API tests
cd frontend && npm run build # production build of the site`}</Code>
              <dl className="space-y-2 text-sm">
                {[
                  ["TOMATO_DEVICE", "cpu (default) or cuda"],
                  ["TOMATO_THREADS", "PyTorch CPU threads (default min(8, cores))"],
                  ["TOMATO_LAZY_MODELS", "1 = load only the default model at startup"],
                  ["TOMATO_CORS_ORIGINS", "extra origins allowed to call the API"],
                  ["VITE_API_BASE_URL", "frontend: API origin (\"/\" when served by the API)"],
                ].map(([k, v]) => (
                  <div key={k} className="flex flex-col gap-0.5 sm:flex-row sm:gap-3">
                    <dt className="w-48 shrink-0 font-mono text-xs text-ink">{k}</dt>
                    <dd className="text-muted">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
