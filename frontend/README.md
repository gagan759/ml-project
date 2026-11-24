# Tomato Care — web platform

React 19 + Vite 7 + Tailwind CSS 4 + Recharts 3 + Framer Motion. Pages:

| Route | Page |
|---|---|
| `/` | landing page |
| `/dashboard` | KPIs, leaderboard, system status, recent diagnoses (stored in this browser) |
| `/diagnose` | upload / camera / paste / held-out samples → calibrated diagnosis, Grad-CAM, model comparison |
| `/lab` | model comparison laboratory: leaderboard, per-class F1, confusion matrices, paired tests, latency, robustness, CV recipes, live bench |
| `/performance` | one model in depth: confusion matrix, per-class table, reliability diagram, risk–coverage, training curves |
| `/explainability` | Grad-CAM explained + gallery of held-out photos |
| `/library`, `/library/:key` | disease library with management advice, references and per-class model reliability |
| `/methodology`, `/docs` | scientific protocol; developer documentation |

## Development

```bash
npm ci
npm run dev        # http://localhost:5173 — calls the API at VITE_API_BASE_URL (.env, default http://127.0.0.1:8000)
npm run build      # production build in dist/ (VITE_API_BASE_URL=/ from .env.production: same-origin API)
```

Data: every result comes from `/api/analytics` (`experiments/platform.json`); when the API is offline the site falls
back to the bundled snapshot `public/data/platform.json` (written by `scripts/27_export_platform_data.py`), so all
read-only pages keep working. Diagnosis and the live bench need the API.

Structure: `src/pages/` (routes), `src/components/{ui,layout,charts,diagnose}/`, `src/context/` (theme, data,
prediction state), `src/lib/` (API client, class metadata, formatting, analytics selectors, history).
