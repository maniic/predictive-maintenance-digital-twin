# Dashboard

Next.js app for the Predictive Maintenance Digital Twin: RUL predictions on real
C-MAPSS test engines, a degradation simulation, and a model comparison view.

It runs in two modes.

| | Full mode (local) | Demo mode (hosted) |
|---|---|---|
| Where | `npm run dev` on your machine | GitHub Pages |
| Predictions | Real PyTorch inference through a Python subprocess | Illustrative values from `public/demo/*.json` |
| Simulation | `DegradationSimulator` + trained models | An equivalent curve computed in the browser |
| Needs | Python env, trained checkpoints | Nothing |

Demo mode is what the deployed site runs, because GitHub Pages has no Python
backend and this repository does not commit trained checkpoints. The engine data
and ground-truth RUL in those payloads are real C-MAPSS; the *predictions* are
generated to match each dataset's reported test error. The nav carries a `DEMO`
badge and the prediction and simulation views carry a banner saying so.

Full mode also falls back to the demo payloads if an API route fails, so the UI
degrades gracefully rather than erroring — check the browser network tab if you
expect live inference and are not getting it.

## Setup

Node 18+ and, for full mode, the Python project installed and at least one
trained model.

```bash
# from the repository root, for full mode only
pip install -e .
python scripts/train.py --quick        # produces a checkpoint + preprocessor

cd web
npm install
npm run dev                            # http://localhost:3000
```

The API routes spawn a Python interpreter, resolved in this order:
`$PYTHON`, then the active virtualenv's `python`, then `python3`. If the
dashboard reports that the interpreter was not found, either activate the venv
before `npm run dev` or set it explicitly:

```bash
PYTHON=/path/to/.venv/bin/python npm run dev
```

## Static build

What GitHub Pages serves. `scripts/build-static.mjs` moves `app/api` aside for
the build — Next.js cannot statically export route handlers — and restores it
afterwards.

```bash
npm run build:static                   # output in web/out/
NEXT_PUBLIC_BASE_PATH=/predictive-maintenance-digital-twin npm run build:static
```

Regenerate the demo payloads it serves with
`python scripts/export_demo_data.py` from the repository root.

## API routes

All routes shell out to `src/api/predict.py` through `lib/python.js`. Every route
accepts `GET` with query parameters, which is what the client sends; `POST` with
a JSON body is kept for scripted callers.

| Route | Methods | Parameters | Returns |
|---|---|---|---|
| `/api/predict` | GET, POST | `dataset`, `engine`, `model`, `action=trajectory` for the per-cycle series | prediction snapshot |
| `/api/simulate` | GET, POST | `initial_rul`, `rate`, `mode` | degradation trajectory |
| `/api/engines` | GET | `dataset` | engine ids available for that dataset |
| `/api/comparison` | GET | — | published training results from `models/*.json` |

### Prediction response

```json
{
  "engine_id": 24,
  "dataset": "FD001",
  "model": "ensemble",
  "rul": 20.44,
  "uncertainty": 3.1,
  "health_score": 0.163,
  "true_rul": 20.0,
  "error": 0.44,
  "individual_predictions": { "lstm": 20.4, "cnn": 22.1, "transformer": 18.8 },
  "total_cycles": 186
}
```

`uncertainty` is the spread across ensemble members, so it is zero when only one
model is loaded. Errors come back as `{ "error": "..." }` with a 4xx/5xx status.

## Structure

```
web/
├── app/
│   ├── api/              # GET/POST routes bridging to Python
│   ├── prediction/       # engine picker, RUL + uncertainty, trajectory chart
│   ├── simulation/       # degradation playback
│   ├── comparison/       # cross-model charts and sortable table
│   ├── layout.jsx        # root layout and metadata
│   ├── icon.svg          # favicon
│   └── page.jsx          # landing page
├── components/           # Navigation, Sidebar, MetricCard, PlotlyChart, DemoNotice
├── lib/
│   ├── api.js            # data access + demo fallback + client-side simulation
│   └── python.js         # subprocess bridge shared by every API route
├── public/demo/          # precomputed demo payloads (generated)
└── scripts/build-static.mjs
```

## Design system

CSS variables and utility classes live in `app/globals.css`: a dark slate
palette (`--bg-base` through `--bg-overlay`), an amber accent (`--amber`),
status colours (`--green`, `--amber`, `--red`), a five-step text ramp
(`--text-bright` to `--text-faint`), and a monospace type scale for anything
numeric. Layout is mobile-first — sidebars collapse below 768px and tables
scroll horizontally.
