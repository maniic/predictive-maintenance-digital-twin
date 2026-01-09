#!/usr/bin/env python
"""
Python CLI bridge for Next.js API routes.
Called via subprocess from Node.js to get ML predictions.

Usage:
    python src/api/predict.py --dataset FD001 --engine 1 --model ensemble
    python src/api/predict.py --action engines --dataset FD001
    python src/api/predict.py --action comparison
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def get_engines(dataset: str) -> dict:
    """Get list of available engines for a dataset."""
    from src.data.ingestion import CMAPSSDataLoader, compute_test_rul

    loader = CMAPSSDataLoader(raw_data_dir=str(project_root / "data" / "raw"))
    data = loader.load_dataset(dataset)
    test_df = compute_test_rul(data.test, data.rul)

    engines = sorted(test_df["engine_id"].unique().tolist())
    return {"engines": engines, "count": len(engines), "dataset": dataset}


def get_prediction(dataset: str, engine_id: int, model: str) -> dict:
    """Get RUL prediction for an engine."""
    from src.digital_twin import RULPredictor
    from src.data.ingestion import CMAPSSDataLoader, compute_test_rul

    # Load test data
    loader = CMAPSSDataLoader(raw_data_dir=str(project_root / "data" / "raw"))
    data = loader.load_dataset(dataset)
    test_df = compute_test_rul(data.test, data.rul)

    engine_df = test_df[test_df["engine_id"] == engine_id].copy()
    if engine_df.empty:
        return {"error": f"Engine {engine_id} not found in {dataset}"}

    engine_df = engine_df.sort_values("cycle")
    true_rul = float(engine_df["RUL"].iloc[-1])

    # Load predictor and make prediction
    predictor = RULPredictor(dataset=dataset, models_dir=str(project_root / "models"))
    predictor.load_models()

    result = predictor.predict_from_dataframe(engine_df, engine_id=engine_id)

    return {
        "engine_id": engine_id,
        "dataset": dataset,
        "model": model,
        "rul": float(result.rul),
        "uncertainty": float(result.uncertainty),
        "health_score": float(result.health_score),
        "true_rul": true_rul,
        "error": float(result.rul - true_rul),
        "individual_predictions": result.individual_predictions,
        "total_cycles": int(engine_df["cycle"].max()),
    }


def get_comparison() -> dict:
    """Get model comparison data."""
    results_path = project_root / "models" / "training_results.json"
    advanced_path = project_root / "models" / "advanced_training_results.json"

    results = []

    if results_path.exists():
        with open(results_path) as f:
            results.extend(json.load(f))

    if advanced_path.exists():
        with open(advanced_path) as f:
            results.extend(json.load(f))

    return {"results": results}


def main():
    parser = argparse.ArgumentParser(description="ML prediction CLI")
    parser.add_argument("--action", default="predict", choices=["predict", "engines", "comparison"])
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--engine", type=int, default=1)
    parser.add_argument("--model", default="ensemble")
    args = parser.parse_args()

    try:
        if args.action == "engines":
            result = get_engines(args.dataset)
        elif args.action == "comparison":
            result = get_comparison()
        else:
            result = get_prediction(args.dataset, args.engine, args.model)

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
