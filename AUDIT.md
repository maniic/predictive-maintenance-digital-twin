# AUDIT — predictive-maintenance-digital-twin

Temporary working document (Phase 0). Deleted before the final commit.
Every claim below was executed, not inferred. Commands and outputs are reproducible
from the repo root.

Environment used: Linux, Python 3.11.15, Node v22.22.2, npm 10.9.7.

---

## 1. Fresh-clone reproducibility

Cloned into a temp dir and followed **only** the README "Quick start".

```
git clone <repo> /tmp/fresh   →  58 MB working tree, ~13 MB downloaded
python -m venv .venv          →  OK
pip install -e .              →  FAILS
```

### BLOCKER A — `pip install -e .` fails on a clean clone

The one documented install command does not work:

```
ValueError: Unable to determine which files to ship inside the wheel using the
following heuristics: https://hatch.pypa.io/latest/plugins/builder/wheel/...

The most likely cause of this is that there is no directory that matches the name
of your project (predictive_maintenance_digital_twin).

At least one file selection option must be defined in the
`tool.hatch.build.targets.wheel` table.
```

`pyproject.toml` declares the hatchling backend but never tells it that the
importable package lives in `src/`. Nobody following the README can install the
project. This is the single most damaging defect in the repo.

### BLOCKER B — the documented download step contradicts the repo

README says:

> `# Download C-MAPSS from the NASA Prognostics Data Repository into data/raw/`

but all 13 C-MAPSS files are already committed and present after clone. The reader
is told to do work that is already done, with no script and no verification step.
`.gitignore` also lists `data/raw/*.txt` while those exact files are tracked —
so the ignore rule is inert and actively misleading.

### BLOCKER C — no documented path reaches a prediction

There is no "run one prediction" command anywhere in the README. The only
documented entry point is `scripts/train_all_models.py`, which trains from scratch
(tens of minutes to hours on CPU). No checkpoints are committed (`models/`
contains only three JSON files), so **live inference is impossible on a fresh
clone** until the user trains. The README's claim that "locally the dashboard
calls the real PyTorch models" is only true after a full training run — this is
never stated.

### Undocumented requirements discovered by execution

| Missing from docs | Consequence |
|---|---|
| `matplotlib` | `import src.evaluation.metrics` fails (see §2) |
| `shap` | same |
| `mlflow` | training crashes at logger creation (see §3) |
| `pytest` | test suite cannot be run |
| Node 18+ / npm | dashboard section assumes it silently |

**Verdict: a fresh clone cannot install dependencies, and cannot run a prediction.**

---

## 2. Test suite

```
$ python -m pytest -q
27 passed in 5.14s      # only after manually installing matplotlib + shap
```

### Collection fails on a documented install

With only the declared dependencies installed:

```
ImportError while importing test module 'tests/test_evaluation/test_metrics.py'
  src/evaluation/__init__.py:12: in <module>
      from src.evaluation.explainability import (
  src/evaluation/explainability.py:14: in <module>
      import matplotlib.pyplot as plt
  ModuleNotFoundError: No module named 'matplotlib'
```

`src/evaluation/__init__.py` eagerly re-exports the SHAP/matplotlib explainability
module, so importing the *metrics* — three pure-numpy functions — drags in two heavy
optional dependencies. Neither is in `requirements.txt`.

### What exists

258 lines of tests against 5,774 lines of `src/`.

| Test file | Covers |
|---|---|
| `tests/test_data/test_ingestion.py` | `compute_train_rul`, `compute_test_rul` |
| `tests/test_evaluation/test_metrics.py` | `rmse`, `mae`, `cmapss_score` |
| `tests/test_models/test_forward_pass.py` | forward-pass output shapes for LSTM/CNN/Transformer |

### What has zero coverage

Ordered by how badly a silent bug there would damage the README's numbers:

| Module | LOC | Untested behaviour that could invalidate every reported metric |
|---|---|---|
| `src/data/dataset.py` | 328 | **Sliding-window construction** — off-by-one in the window/target alignment; `train_val_split` engine-level split (leakage); silent dropping of short engines |
| `src/data/preprocessing.py` | 488 | **RUL cap at 125**; scaler fit/transform separation (leakage); per-regime KMeans fit on train and predict on test |
| `src/models/ensemble.py` | 394 | weighted averaging, uncertainty (std) computation |
| `src/models/advanced_rul.py` | 485 | EnhancedLSTM + TwoStage — the models behind 3 of 4 headline rows |
| `src/digital_twin/predictor.py` | 273 | the entire serving path the dashboard depends on |
| `src/digital_twin/simulator.py` | 339 | degradation simulation |
| `src/digital_twin/state.py` | 145 | — |
| `src/data/feature_engineering.py` | 356 | rolling stats / EMA / trend |
| `src/evaluation/explainability.py` | 631 | — |
| `src/training/trainer.py` | 262 | — |
| `src/api/predict.py` | 242 | the CLI bridge the dashboard shells out to |
| `src/models/{gru,lstm_improved,base}.py` | 633 | — |

Nothing currently catches a windowing off-by-one or a leaky split.

---

## 3. Headline numbers — traceability

Every RMSE/MAE in the README **traces to a committed result file**. Verified by
re-reading `models/*.json` and diffing against the README tables:

| README table row | Source file | Status |
|---|---|---|
| all FD001–FD004 `lstm` / `cnn` / `transformer` / `ensemble` | `models/training_results.json` | traced, matches |
| all `EnhancedLSTM-Weighted` / `-Asymmetric` / `TwoStage` | `models/advanced_training_results.json` | traced, matches |
| badge `best RMSE 11.71` | `advanced_training_results.json` FD003 TwoStage | traced, matches |

No untraceable number. **But three findings:**

### FINDING 1 — the FD004 headline row names the wrong best model

README headline table:

| Dataset | Best Model | RMSE |
|---|---|---|
| FD004 | LSTM | **14.87** |

README's own per-model FD004 table, three screens lower:

| Model | RMSE |
|---|---|
| EnhancedLSTM-Asymmetric | **14.75** |
| LSTM | 14.87 |

`14.75 < 14.87`. The best FD004 model is EnhancedLSTM-Asymmetric, confirmed in
`models/advanced_training_results.json`. The "Models" table repeats the error
("LSTM … wins FD001/FD004"). The README contradicts itself and understates the
result. This is a genuine error in a claim, fixable without retraining.

### FINDING 2 — the evaluation protocol is not the C-MAPSS standard, and the README doesn't say so

`scripts/train_all_models.py:145` builds the test set as:

```python
test_dataset = CMAPSSSequenceDataset(
    test_processed, feature_cols,
    sequence_length=ds_config.sequence_length,
    stride=1,
)
```

and then computes RMSE over **every window** of every test trajectory. The standard
C-MAPSS benchmark scores **one prediction per test engine** (the final window),
against `RUL_FDxxx.txt`. Measured window counts:

| Dataset | Test engines | Windows actually scored |
|---|---|---|
| FD001 | 100 | 10,196 |
| FD002 | 259 | 26,505 |
| FD003 | 100 | 13,696 |
| FD004 | 248 | 34,081 |

Most of those extra windows sit early in the trajectory where the target is pinned
at the 125-cycle cap and is trivially predictable, so this protocol yields a lower
RMSE than the published one. The numbers are internally consistent and honestly
computed — but the README frames them next to the C-MAPSS benchmark, and a reader
who knows the benchmark will read `13.48` as a per-engine score. It is not.

**Recommendation: do not retrain and do not change the numbers.** State the protocol
explicitly next to the table, so the figures are precise rather than merely true.

### FINDING 3 — 17 test engines are silently excluded from evaluation

`CMAPSSSequenceDataset._build_sequences` iterates
`range(0, n_samples - sequence_length + 1, stride)`, which yields nothing for an
engine with fewer than 30 cycles. Measured:

| Dataset | Test engines with < 30 cycles | Shortest |
|---|---|---|
| FD001 | 0 | 31 |
| FD002 | **6** | 21 |
| FD003 | 0 | 38 |
| FD004 | **11** | 19 |

17 of 707 test engines contribute zero predictions and are dropped without a
warning. Short trajectories are the hardest cases. Worth documenting; a
`Limitations` entry covers it honestly.

---

## 4. Dashboard

Built and exercised the real artifact: `npm ci && npm run build:static`, served
`web/out/` at the deployed base path, drove all four routes in headless Chromium
and read the console.

```
✓ Compiled successfully
✓ Generating static pages (7/7)
Route (app)          Size     First Load JS
/                    2.92 kB  99.2 kB
/comparison          5.31 kB  102 kB
/prediction          3.24 kB  103 kB
/simulation          3 kB     103 kB
```

**The hosted build works.** All four views render, the prediction runs and shows
RUL 93 ±9.4 vs true 111 with a model breakdown and trajectory chart, the comparison
page renders the bar chart / heatmap / table, and the simulation animates cycle by
cycle. No page errors, no broken images, no failed demo fetches. Demo JSON matches
what the app expects — all `engines_*`, `prediction_*`, `trajectory_*` and
`comparison.json` keys line up with `web/lib/api.js`.

### FINDING 4 — local "live inference" mode is broken for 2 of 3 views

`web/lib/api.js` requests predictions and simulations with **GET**; the routes only
export **POST**. Verified against a running dev server:

```
GET /api/predict?dataset=FD001&engine=9&model=ensemble  →  HTTP 405 Method Not Allowed
GET /api/simulate?initial_rul=150&rate=1&mode=hpc       →  HTTP 405 Method Not Allowed
GET /api/engines?dataset=FD001                          →  {"error":"Failed to get engines"}
GET /api/comparison                                     →  200 OK
```

`withDemoFallback` swallows the 405 and silently serves the demo JSON, so the bug is
invisible: the local dashboard *looks* live while serving precomputed data. The
README's "local: live PyTorch inference via Python" is false for the prediction and
simulation views regardless of whether models are trained.

`/api/engines` additionally spawns the literal binary `python` (`route.js`), which
does not exist on systems where it is `python3`, and the failure surfaces only as a
generic 500.

### FINDING 5 — the hosted demo labels a JavaScript simulation as ML-backed

In demo mode `fetchSimulation` calls `simulateLocally` — an AR(1)-noise curve in
`web/lib/api.js`, no model involved. The UI titles that view
**"ML-Backed Simulation"** (`web/app/simulation/page.jsx:205`) and captions it
**"Uses trained ML models for real-time RUL prediction"** (line 267). Both are
false on the deployed site. The nav `DEMO` badge is present and honest; these two
strings contradict it.

### FINDING 6 — the demo predictions are synthesized, not model outputs

`scripts/export_demo_data.py` says so in its own docstring:

> Model predictions are emulated to match each dataset's reported test RMSE (noise
> around the piecewise-linear RUL target), since trained checkpoints are not stored
> in the repo.

Ground-truth RUL is real C-MAPSS data; the *predictions* are noise calibrated to the
reported RMSE. The README calls this "precomputed outputs", which reads as "cached
real model outputs". It is not that, and the distinction matters to exactly the
audience this repo is for.

### FINDING 7 — cosmetic / consistency

- `web/public/demo/comparison.json` and `models/training_results.json` are published
  with absolute local paths baked in: `/Users/<name>/projects/...`. Visible on the
  live site.
- Home page advertises "**9** DL Models"; README says **7 architectures**. Both are
  defensible countings (the dashboard counts `ImprovedLSTM` and `GRU`, which the
  README omits entirely) but they disagree in public.
- Home stat reads "Best RMSE — 13.5 (FD001)"; the actual best is 11.71 (FD003), which
  is what the README badge shows.
- The page depends on Google Fonts over the network; it degrades to fallback fonts
  without them (not a defect, just a dependency).

---

## 5. Repo weight

The 43 MB figure is the **working tree**, not the clone.

```
$ du -sh data/raw          43M
$ git count-objects -vH    size-pack: 12.69 MiB
```

Per-path pack cost (compressed, what a cloner actually downloads):

| Path | Pack bytes |
|---|---|
| `data/raw/*.txt` (13 files) | **11.55 MB** |
| `data/raw/Damage Propagation Modeling.pdf` (deleted, still in history) | 0.39 MB |
| `docs/screenshots/*.png` | 0.46 MB |
| everything else | ~0.3 MB |
| **total clone** | **12.69 MB** |

So: a clone costs ~13 MB, of which C-MAPSS is ~91%. It is highly compressible ASCII.

**Redistributable? Yes.** C-MAPSS is a NASA Prognostics Center of Excellence dataset
released as US Government open data with no redistribution restriction. Verified the
committed files are byte-identical to the official NASA distribution:

```
$ curl -sSL "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
  → 200, 12,429,152 bytes, sha256 c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2
  → inner CMAPSSData.zip, sha256 74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f
$ sha256 compare vs data/raw/*   →  ALL 13 FILES MATCH
```

(The README's link, `data.nasa.gov/dataset/cmapss-jet-engine-simulated-data`, could
not be reached from this sandbox — the proxy blocks it — so its liveness is
unverified. The S3 URL above is verified reachable and is the canonical mirror.)

**Recommendation: keep the data committed.** 13 MB is a normal clone; the data is
public domain and byte-verified; and removing it would mean a history rewrite plus a
network dependency on every clone, in exchange for ~11 MB. A fetch script is still
worth adding as the *fallback* path (and as proof of provenance), but committed data
is what makes "three commands to a prediction" achievable. Recording the checksums
in-repo turns the committed copy into a verifiable artifact rather than a mystery
blob.

A download-on-demand script, if chosen instead, would be:
`scripts/fetch_data.py` → GET the S3 zip → verify sha256 → extract the inner
`CMAPSSData.zip` → write the 13 files to `data/raw/` → verify each file's sha256.

---

## 6. Script sprawl

Four training scripts, 1,550 lines, heavily overlapping. Traced what each actually does:

| Script | Lines | Models trained | Results written | Status |
|---|---|---|---|---|
| `train_all_models.py` | 512 | lstm, cnn, transformer + ensemble, `--datasets`/`--models` flags, FD001–FD004 | `models/training_results.json` | **LIVE** — the only script the README documents; produced 16 of 28 published numbers |
| `train_advanced_models.py` | 270 | EnhancedLSTM-Weighted, EnhancedLSTM-Asymmetric, TwoStage on FD001–FD004 | `models/advanced_training_results.json` | **LIVE but undocumented** — produced the other 12 published numbers, including the 11.71 headline. Not mentioned in the README at all. |
| `train_improved_models.py` | 238 | ImprovedLSTM, GRU on FD001 only | `models/improved_training_results.json` | **ORPHANED** — its 2 results appear on the dashboard's comparison page but nowhere in the README. Only script that touches `src/models/gru.py` and `src/models/lstm_improved.py`. |
| `train_models.py` | 530 | lstm, cnn, transformer + ensemble on a single dataset | nothing persisted | **SUPERSEDED** — a strict subset of `train_all_models.py` (which adds multi-dataset looping, per-model checkpoint dirs, and JSON output). No result file traces to it. |

Real differences beyond model choice: `train_all_models.py` uses `RULTrainer`
(MLflow-logged, config-driven); `train_advanced_models.py` and
`train_improved_models.py` hand-roll a `pl.Trainer` with their own callbacks and skip
MLflow. So the three live scripts do not share a training path, which is why
"consolidate into one entry point" is worth doing rather than cosmetic.

---

## 7. Repo hygiene

| Item | State |
|---|---|
| `LICENSE` | **Missing.** No license file, no license field in `pyproject.toml`. Legally this is "all rights reserved" — a reviewer cannot reuse or even safely fork it. |
| CI | Only `deploy-pages.yml`. **The 27 tests never run in CI.** No lint, no typecheck, no Next.js build check outside the deploy job (so a broken dashboard is caught only by a failed *deploy*, on `main`, after merge). |
| `docs/` | Contains **only** `screenshots/` (4 PNGs). No architecture doc, no methodology, nothing to link to from the README. |
| `requirements.txt` vs `pyproject.toml` | **Disagree.** `requirements.txt` is missing `mlflow`, `shap`, `pyarrow`, `fastapi`, `uvicorn`, `websockets`, `pydantic`, `tqdm`, `rich`; `pyproject.toml` is missing `plotly`. Neither lists `matplotlib`, which `src/evaluation` imports at package level. Every dependency is an open `>=` bound — nothing is pinned, so a clone in six months resolves differently. |
| Declared-but-unused deps | `fastapi`, `uvicorn`, `websockets`, `pydantic`, `pyarrow` — grepped, zero imports anywhere in `src/`, `scripts/`, or `tests/`. The API is a subprocess CLI, not FastAPI. |
| MLflow | `requirements.txt` omits it, so the documented install path crashes at `MLFlowLogger(...)`: `ModuleNotFoundError: Requirement 'mlflow>=1.0.0' not met`. Confirmed by execution. |
| `.gitignore` | Ignores `data/raw/*.txt` while those files are tracked — an inert, misleading rule. |
| `web/README.md` | **Stale.** Documents a prediction response schema (`predicted_rul`, `confidence_interval`, `model_predictions`) that does not match the actual output (`rul`, `uncertainty`, `individual_predictions`); omits `/api/simulate` entirely; lists `/api/predict` as POST without noting the client calls it with GET. |
| Dead code | No TODO/FIXME/XXX markers, no `console.log`, no debug prints — the code itself is clean. `train_models.py` (530 lines) is the main dead weight. |
| README ordering | Badges and hero image are above the fold and good. The full per-model tables (4 tables, ~40 rows) sit in a `<details>` block directly after the results table, pushing the dashboard section, the architecture explanation and Quick start far down. No "reproduce these results" section anywhere. |
| Local paths leaked | `/Users/<name>/projects/...` in `models/training_results.json` and in the publicly served `web/public/demo/comparison.json`. |

---

## Summary of defects, by severity

**Blocking (repo does not work as documented)**
1. `pip install -e .` fails — hatchling has no package target. *(§1 A)*
2. No documented path from clone to a prediction; no checkpoints committed. *(§1 C)*
3. `import src.evaluation.metrics` requires undeclared `matplotlib` + `shap`. *(§2)*
4. Training crashes without `mlflow`, absent from `requirements.txt`. *(§7)*

**Wrong or misleading claims**
5. FD004 headline names LSTM as best; the repo's own data says EnhancedLSTM-Asymmetric. *(§3.1)*
6. Metrics are per-window, not the per-engine C-MAPSS protocol; unstated. *(§3.2)*
7. 17 short test engines silently excluded. *(§3.3)*
8. "local: live PyTorch inference" — broken by a GET/POST mismatch. *(§4.4)*
9. Hosted demo says "ML-Backed Simulation" over a JavaScript curve. *(§4.5)*
10. "precomputed outputs" describes synthesized predictions. *(§4.6)*
11. README instructs downloading data that is already committed. *(§1 B)*

**Hygiene**
12. No LICENSE. 13. Tests never run in CI. 14. No dashboard build check pre-merge.
15. `docs/` has no docs. 16. Dependencies unpinned and contradictory; 5 unused.
17. `train_models.py` dead (530 lines). 18. `web/README.md` stale.
19. Local absolute paths published. 20. Dashboard says 9 models, README says 7.
