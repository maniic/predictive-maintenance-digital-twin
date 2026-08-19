#!/usr/bin/env python
"""Export precomputed demo data for the static (hosted) dashboard build.

The hosted dashboard has no Python backend, so this script bakes the data the
API routes would normally serve into web/public/demo/*.json:

- comparison.json                  real training results (models/*.json)
- engines_<DS>.json                sampled engines per C-MAPSS dataset
- prediction_<DS>_<ID>.json        RUL prediction snapshot per sampled engine
- trajectory_<DS>_<ID>.json        per-cycle RUL trajectory per sampled engine

Ground-truth RUL comes from the real C-MAPSS test files. The predictions are NOT
model outputs: trained checkpoints are not committed, so each prediction is an
illustrative value drawn around the piecewise-linear RUL target with noise scaled
to that dataset's reported test RMSE. Payloads are tagged "demo": true, the nav
shows a DEMO badge, and the prediction and simulation views carry a banner saying
so. For real inference, train locally and run scripts/predict.py.
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "web" / "public" / "demo"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
# Best reported test RMSE per dataset (see the README results table). Used only
# to scale the illustrative noise, so the demo's error looks like the real one.
DATASET_RMSE = {"FD001": 13.48, "FD002": 16.77, "FD003": 11.71, "FD004": 14.75}
RUL_CAP = 125  # piecewise-linear RUL target used during training
ENGINES_PER_DATASET = 8
MODELS = ["lstm", "cnn", "transformer", "ensemble"]


def load_test_data(dataset: str):
    """Return {engine_id: [(cycle, true_rul), ...]} for a C-MAPSS test set."""
    test_file = RAW_DIR / f"test_{dataset}.txt"
    rul_file = RAW_DIR / f"RUL_{dataset}.txt"
    final_ruls = [
        int(float(line.split()[0])) for line in rul_file.read_text().split("\n") if line.strip()
    ]

    engines: dict[int, list[int]] = {}
    for line in test_file.read_text().split("\n"):
        parts = line.split()
        if len(parts) < 3:
            continue
        engine_id, cycle = int(parts[0]), int(parts[1])
        engines.setdefault(engine_id, []).append(cycle)

    result = {}
    for engine_id, cycles in engines.items():
        max_cycle = max(cycles)
        final_rul = final_ruls[engine_id - 1]
        result[engine_id] = [(c, final_rul + max_cycle - c) for c in sorted(cycles)]
    return result


def emulate_prediction(true_rul: float, rmse: float, rng: random.Random) -> float:
    """A model-like estimate: capped RUL target plus RMSE-scaled noise."""
    target = min(true_rul, RUL_CAP)
    return max(0.0, target + rng.gauss(0, rmse * 0.85))


def pick_engines(engines: dict, n: int, rng: random.Random) -> list[int]:
    """Sample engines spanning short/medium/long remaining life."""
    by_final_rul = sorted(engines, key=lambda e: engines[e][-1][1])
    step = max(1, len(by_final_rul) // n)
    picked = by_final_rul[::step][:n]
    return sorted(picked)


def export_dataset(dataset: str) -> None:
    rng = random.Random(f"demo-{dataset}")
    rmse = DATASET_RMSE[dataset]
    engines = load_test_data(dataset)
    sampled = pick_engines(engines, ENGINES_PER_DATASET, rng)

    (OUT_DIR / f"engines_{dataset}.json").write_text(
        json.dumps({"engines": sampled, "count": len(sampled), "dataset": dataset, "demo": True})
    )

    for engine_id in sampled:
        series = engines[engine_id]
        total_cycles = series[-1][0]
        true_rul = float(series[-1][1])

        # Per-cycle trajectory with AR(1)-correlated noise so it looks like
        # a real model track rather than white noise.
        trajectory = []
        noise = 0.0
        for cycle, rul in series:
            if cycle < 30:  # warm-up window before the model has a full sequence
                continue
            noise = 0.75 * noise + rng.gauss(0, rmse * 0.55)
            predicted = max(0.0, min(rul, RUL_CAP) + noise)
            trajectory.append(
                {
                    "cycle": cycle,
                    "predicted_rul": round(predicted, 2),
                    "uncertainty": round(rmse * rng.uniform(0.5, 0.9), 2),
                    "true_rul": float(rul),
                }
            )
        (OUT_DIR / f"trajectory_{dataset}_{engine_id}.json").write_text(
            json.dumps(
                {
                    "trajectory": trajectory,
                    "total_cycles": total_cycles,
                    "engine_id": engine_id,
                    "dataset": dataset,
                    "demo": True,
                }
            )
        )

        # Final prediction snapshot with per-model spread
        individual = {
            m: round(emulate_prediction(true_rul, rmse, rng), 2) for m in MODELS if m != "ensemble"
        }
        ensemble = round(sum(individual.values()) / len(individual), 2)
        individual["ensemble"] = ensemble
        health = max(0.0, min(1.0, ensemble / RUL_CAP))
        (OUT_DIR / f"prediction_{dataset}_{engine_id}.json").write_text(
            json.dumps(
                {
                    "engine_id": engine_id,
                    "dataset": dataset,
                    "model": "ensemble",
                    "rul": ensemble,
                    "uncertainty": round(rmse * 0.7, 2),
                    "health_score": round(health, 4),
                    "true_rul": true_rul,
                    "error": round(ensemble - true_rul, 2),
                    "individual_predictions": individual,
                    "total_cycles": total_cycles,
                    "demo": True,
                }
            )
        )


def export_comparison() -> None:
    results = []
    for name in [
        "training_results.json",
        "advanced_training_results.json",
        "improved_training_results.json",
    ]:
        path = PROJECT_ROOT / "models" / name
        if path.exists():
            results.extend(json.loads(path.read_text()))
    (OUT_DIR / "comparison.json").write_text(json.dumps({"results": results}))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export_comparison()
    for dataset in DATASETS:
        export_dataset(dataset)
        print(f"exported {dataset}")
    print(f"demo data written to {OUT_DIR}")


if __name__ == "__main__":
    main()
