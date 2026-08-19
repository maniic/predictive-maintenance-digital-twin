#!/usr/bin/env python
"""Predict remaining useful life for a single C-MAPSS test engine.

Loads the trained checkpoints and the fitted preprocessor for a dataset, runs
the ensemble over that engine's most recent `sequence_length` cycles, and prints
the predicted RUL with an uncertainty band alongside the ground truth.

This is the serving path the dashboard's API routes use, exercised from the
command line. It needs trained checkpoints under models/checkpoints/<DATASET>/,
which `scripts/train.py` produces - checkpoints are not committed to the repo.

Usage:
    python scripts/predict.py --dataset FD001 --engine 24
    python scripts/predict.py --dataset FD001            # picks an engine for you
    python scripts/predict.py --dataset FD001 --engine 24 --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingestion import CMAPSSDataLoader, compute_test_rul  # noqa: E402


def predict_engine(
    dataset: str = "FD001",
    engine_id: Optional[int] = None,
    models_dir: str = "models",
    sequence_length: int = 30,
) -> dict:
    """Predict RUL for one test engine.

    Args:
        dataset: C-MAPSS sub-dataset (FD001..FD004)
        engine_id: test engine to score. If None, picks the first engine with
            at least `sequence_length` cycles.
        models_dir: directory holding checkpoints/ and preprocessors/
        sequence_length: cycles of history the models expect

    Returns:
        Dict with the prediction, its uncertainty, the per-model breakdown and
        the ground-truth RUL.

    Raises:
        FileNotFoundError: no trained checkpoints or preprocessor available.
        ValueError: the requested engine does not exist or is too short.
    """
    from src.digital_twin import RULPredictor

    loader = CMAPSSDataLoader(PROJECT_ROOT / "data" / "raw")
    data = loader.load_dataset(dataset)
    test_df = compute_test_rul(data.test, data.rul)

    lengths = test_df.groupby("engine_id")["cycle"].max()
    usable = lengths[lengths >= sequence_length]
    if usable.empty:
        raise ValueError(f"No {dataset} test engine has {sequence_length} or more cycles.")

    if engine_id is None:
        engine_id = int(usable.index[0])
    elif engine_id not in lengths.index:
        raise ValueError(f"Engine {engine_id} is not in the {dataset} test set.")
    elif engine_id not in usable.index:
        raise ValueError(
            f"Engine {engine_id} has only {int(lengths[engine_id])} cycles; "
            f"{sequence_length} are needed for a prediction."
        )

    engine_df = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")

    predictor = RULPredictor(dataset=dataset, models_dir=str(PROJECT_ROOT / models_dir))
    try:
        predictor.load_models()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\n"
            f"  Nothing has been trained for {dataset} yet. "
            f"Train a model first:  python scripts/train.py --quick"
        ) from exc

    if not predictor.models:
        raise FileNotFoundError(
            f"No trained checkpoints under {models_dir}/checkpoints/{dataset}/.\n"
            f"  Train a model first:  python scripts/train.py --quick"
        )

    result = predictor.predict_from_dataframe(engine_df, engine_id=engine_id)
    true_rul = float(engine_df["RUL"].iloc[-1])

    return {
        "dataset": dataset,
        "engine_id": int(engine_id),
        "rul": float(result.rul),
        "uncertainty": float(result.uncertainty),
        "rul_lower": float(result.rul_lower),
        "rul_upper": float(result.rul_upper),
        "health_score": float(result.health_score),
        "true_rul": true_rul,
        "error": float(result.rul - true_rul),
        "individual_predictions": result.individual_predictions,
        "observed_cycles": int(engine_df["cycle"].max()),
        "models_used": sorted(predictor.models),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--engine", type=int, default=None, help="test engine id")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    args = parser.parse_args()

    try:
        result = predict_engine(args.dataset, args.engine, args.models_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"\n  {result['dataset']} engine {result['engine_id']}")
    print(f"  {result['observed_cycles']} cycles observed\n")
    print(f"  Predicted RUL   {result['rul']:.1f} cycles")
    if len(result["models_used"]) > 1:
        print(f"  95% interval    {result['rul_lower']:.1f} - {result['rul_upper']:.1f}")
    else:
        # Uncertainty is the spread between ensemble members, so a single
        # loaded model always reports zero. Say that instead of printing a
        # degenerate interval.
        print("  95% interval    n/a (uncertainty is ensemble spread; only one model loaded)")
    print(f"  True RUL        {result['true_rul']:.0f} cycles")
    print(f"  Error           {result['error']:+.1f} cycles")
    print(f"  Health score    {result['health_score'] * 100:.0f}%\n")
    print("  Per-model breakdown")
    for name, value in sorted(result["individual_predictions"].items()):
        print(f"    {name:<14} {value:7.1f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
