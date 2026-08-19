#!/usr/bin/env python
"""Train and evaluate RUL prediction models on NASA C-MAPSS.

Single entry point for every architecture in this project. Replaces the four
overlapping scripts that preceded it (train_models, train_all_models,
train_improved_models, train_advanced_models); see docs/models.md for the
mapping from each published result row to the settings that produced it.

Usage:
    # One small model on one dataset, a few minutes on CPU - ends with a
    # worked prediction on a real test engine.
    python scripts/train.py --quick

    # A specific model and dataset
    python scripts/train.py --models lstm --datasets FD001

    # Everything (hours on CPU)
    python scripts/train.py --models all --datasets all

Models:
    lstm cnn transformer                        base architectures
    enhanced-lstm-weighted enhanced-lstm-asymmetric twostage
    improved-lstm gru                           attention variants
    ensemble                                    weighted average of the base
                                                three, needs them trained first

Evaluation protocol: metrics are computed over every sliding window of every
test trajectory, not one prediction per engine. This is NOT the standard
C-MAPSS benchmark protocol - see docs/architecture.md. Engines with fewer than
`sequence_length` cycles produce no windows and are reported as excluded.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    CMAPSSSequenceDataset,
    DatasetConfig,
    train_val_split,
)
from src.data.ingestion import (  # noqa: E402
    CMAPSSDataLoader,
    compute_test_rul,
    compute_train_rul,
)
from src.data.preprocessing import CMAPSSPreprocessor, PreprocessingConfig  # noqa: E402
from src.evaluation.metrics import evaluate_predictions  # noqa: E402
from src.models import (  # noqa: E402
    EnsembleModel,
    GRUModel,
    ImprovedLSTMModel,
    LSTMModel,
    TemporalCNNModel,
    TransformerModel,
)
from src.models.advanced_rul import EnhancedLSTM, TwoStageRULModel  # noqa: E402
from src.training.trainer import RULTrainer, TrainingConfig  # noqa: E402

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# Hyperparameters are carried over verbatim from the scripts that produced the
# published results, so a re-run reproduces the same configuration.
MODEL_REGISTRY = {
    "lstm": {
        "class": LSTMModel,
        "params": {"hidden_size": 128, "num_layers": 2, "dropout": 0.3, "bidirectional": True},
        "epochs": 50,
        "patience": 15,
        "lr": 1e-3,
    },
    "cnn": {
        "class": TemporalCNNModel,
        "params": {"channels": [64, 128, 256], "kernel_sizes": [5, 5, 3], "dropout": 0.3},
        "epochs": 50,
        "patience": 15,
        "lr": 1e-3,
    },
    "transformer": {
        "class": TransformerModel,
        "params": {"d_model": 64, "n_heads": 4, "n_layers": 4, "dropout": 0.2},
        "epochs": 50,
        "patience": 15,
        "lr": 1e-3,
    },
    "enhanced-lstm-weighted": {
        "class": EnhancedLSTM,
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "loss_type": "weighted",
        },
        "epochs": 100,
        "patience": 15,
        "lr": 1e-3,
        "grad_clip": 1.0,
        "report_as": "EnhancedLSTM-Weighted",
    },
    "enhanced-lstm-asymmetric": {
        "class": EnhancedLSTM,
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "loss_type": "asymmetric",
        },
        "epochs": 100,
        "patience": 15,
        "lr": 1e-3,
        "grad_clip": 1.0,
        "report_as": "EnhancedLSTM-Asymmetric",
    },
    "twostage": {
        "class": TwoStageRULModel,
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "critical_threshold": 25.0,
            "degrading_threshold": 75.0,
        },
        "epochs": 100,
        "patience": 15,
        "lr": 1e-3,
        "grad_clip": 1.0,
        "tuple_output": True,
        "report_as": "TwoStage",
    },
    "improved-lstm": {
        "class": ImprovedLSTMModel,
        "params": {
            "hidden_size": 256,
            "num_layers": 3,
            "n_attention_heads": 4,
            "dropout": 0.2,
            "weight_decay": 1e-4,
        },
        "epochs": 150,
        "patience": 20,
        "lr": 5e-4,
        "grad_clip": 1.0,
        "report_as": "ImprovedLSTM",
    },
    "gru": {
        "class": GRUModel,
        "params": {"hidden_size": 256, "num_layers": 3, "dropout": 0.2, "weight_decay": 1e-4},
        "epochs": 150,
        "patience": 20,
        "lr": 5e-4,
        "grad_clip": 1.0,
        "report_as": "GRU",
    },
}

ENSEMBLE_MEMBERS = ["lstm", "cnn", "transformer"]


def header(text: str) -> None:
    print(f"\n{'=' * 70}\n {text}\n{'=' * 70}\n")


def resolve_regimes(dataset: str, mode: str, config_value: int) -> int:
    """Decide how many operating regimes to cluster into.

    'auto' uses 6 regimes only for the genuinely multi-condition datasets
    (FD002/FD004) and global normalization elsewhere, which is what the
    advanced models used. 'config' takes the value from config.yaml for every
    dataset, which is what the base models used. The choice is recorded in the
    results file so a run is self-describing.
    """
    if mode == "config":
        return config_value
    return 6 if dataset in ("FD002", "FD004") else 1


def prepare_data(dataset: str, args, config_path: Path):
    """Load, preprocess and window one C-MAPSS sub-dataset."""
    print(f"Preparing {dataset}")

    loader = CMAPSSDataLoader(PROJECT_ROOT / "data" / "raw")
    data = loader.load_dataset(dataset)
    print(f"  Engines: {data.n_train_engines} train, {data.n_test_engines} test")

    train_df = compute_train_rul(data.train)
    test_df = compute_test_rul(data.test, data.rul)

    prep_config = PreprocessingConfig.from_yaml(config_path)
    prep_config.n_regimes = resolve_regimes(dataset, args.regimes, prep_config.n_regimes)
    prep_config.cluster_by_regime = prep_config.n_regimes > 1

    preprocessor = CMAPSSPreprocessor(prep_config)
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)
    feature_cols = preprocessor.get_feature_names()

    print(
        f"  Regimes: {prep_config.n_regimes}, features: {len(feature_cols)}, "
        f"RUL capped at {prep_config.rul_cap}"
    )

    ds_config = DatasetConfig.from_yaml(config_path)
    ds_config.sequence_length = args.sequence_length
    train_split, val_split = train_val_split(
        train_processed, val_ratio=ds_config.val_ratio, seed=ds_config.seed
    )

    def build(df, stride):
        return CMAPSSSequenceDataset(
            df, feature_cols, sequence_length=ds_config.sequence_length, stride=stride
        )

    train_ds = build(train_split, ds_config.stride)
    val_ds = build(val_split, 1)
    test_ds = build(test_processed, 1)

    # Engines too short to yield a single window contribute nothing to the
    # reported metrics. Say so rather than dropping them silently.
    test_lengths = test_processed.groupby("engine_id")["cycle"].max()
    excluded = int((test_lengths < ds_config.sequence_length).sum())
    if excluded:
        print(
            f"  NOTE: {excluded} test engines are shorter than "
            f"{ds_config.sequence_length} cycles and are excluded from evaluation"
        )

    print(
        f"  Sequences: {len(train_ds):,} train, {len(val_ds):,} val, {len(test_ds):,} test"
    )
    return train_ds, val_ds, test_ds, feature_cols, ds_config, preprocessor, excluded


def evaluate(model, loader, tuple_output: bool = False, ensemble: bool = False) -> dict:
    """Run inference over a loader and compute all metrics."""
    device = next(model.parameters()).device if not ensemble else torch.device("cpu")
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            if ensemble:
                out = model.predict_with_uncertainty(x).mean
            else:
                out = model(x.to(device))
                if tuple_output:
                    out = out[1]
            preds.append(out.cpu())
            targets.append(y)

    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()

    # Argument order matters: cmapss_score(y_pred, y_true). Reversing it
    # inverts the early/late asymmetry the metric exists to express.
    results = evaluate_predictions(y_pred, y_true)

    errors = y_pred - y_true
    range_maes = {}
    for low, high in [(0, 20), (20, 50), (50, 100), (100, 150)]:
        mask = (y_true >= low) & (y_true < high)
        if mask.any():
            range_maes[f"{low}-{high}"] = float(np.abs(errors[mask]).mean())

    return {
        "test_rmse": results.rmse,
        "test_mae": results.mae,
        "test_cmapss": results.cmapss_score,
        "range_maes": range_maes,
        "n_test_sequences": results.n_samples,
    }


def train_one(name: str, dataset: str, datasets_tuple, feature_cols, ds_config, args) -> dict:
    """Train and evaluate a single architecture on a single dataset."""
    spec = MODEL_REGISTRY[name]
    train_ds, val_ds, test_ds = datasets_tuple

    print(f"\n--- {spec.get('report_as', name)} on {dataset} ---")

    params = dict(spec["params"])
    model = spec["class"](
        input_dim=len(feature_cols),
        sequence_length=ds_config.sequence_length,
        learning_rate=args.lr if args.lr else spec["lr"],
        **params,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints" / dataset / name
    trainer = RULTrainer(
        TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr if args.lr else spec["lr"],
            max_epochs=args.epochs if args.epochs else spec["epochs"],
            early_stopping_patience=args.patience if args.patience else spec["patience"],
            checkpoint_dir=str(checkpoint_dir),
            experiment_name=f"cmapss-{dataset.lower()}-{name}",
            gradient_clip_val=spec.get("grad_clip", 0.0),
            use_mlflow=not args.no_mlflow,
        )
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(val_ds, batch_size=args.batch_size),
        "test": DataLoader(test_ds, batch_size=args.batch_size),
    }

    run_name = f"{name}_{dataset}_{datetime.now():%Y%m%d_%H%M%S}"
    trained = trainer.train(model, loaders["train"], loaders["val"], run_name=run_name)

    best_score = trainer._trainer.checkpoint_callback.best_model_score
    best_path = trainer._trainer.checkpoint_callback.best_model_path

    metrics = evaluate(trained, loaders["test"], tuple_output=spec.get("tuple_output", False))
    print(
        f"  RMSE {metrics['test_rmse']:.2f} | MAE {metrics['test_mae']:.2f} "
        f"| C-MAPSS {metrics['test_cmapss']:.0f}"
    )

    return {
        "dataset": dataset,
        "model": spec.get("report_as", name),
        # Store a repo-relative path: absolute paths leak the machine they were
        # trained on and mean nothing to anyone else.
        "checkpoint": str(Path(best_path).relative_to(PROJECT_ROOT)) if best_path else "",
        "val_rmse": float(best_score) ** 0.5 if best_score is not None else 0.0,
        "epochs": trainer._trainer.current_epoch + 1,
        **metrics,
    }


def build_ensemble(dataset: str, member_results: dict, test_ds, args) -> dict:
    """Weighted average of the base models, weighted by inverse validation RMSE."""
    print(f"\n--- Ensemble on {dataset} ---")

    models = {}
    for name, result in member_results.items():
        if name not in ENSEMBLE_MEMBERS or not result["checkpoint"]:
            continue
        models[name] = MODEL_REGISTRY[name]["class"].load_from_checkpoint(
            PROJECT_ROOT / result["checkpoint"]
        )

    if len(models) < 2:
        print("  Skipped: needs at least two of lstm/cnn/transformer trained in this run.")
        return {}

    ensemble = EnsembleModel(models)
    inv = {
        n: 1.0 / member_results[n]["val_rmse"] if member_results[n]["val_rmse"] > 0 else 1.0
        for n in models
    }
    total = sum(inv.values())
    ensemble._update_weights({n: v / total for n, v in inv.items()})
    print(f"  Weights: { {k: round(v, 3) for k, v in ensemble.weight_dict.items()} }")

    metrics = evaluate(
        ensemble, DataLoader(test_ds, batch_size=args.batch_size), ensemble=True
    )
    print(f"  RMSE {metrics['test_rmse']:.2f} | MAE {metrics['test_mae']:.2f}")

    return {
        "dataset": dataset,
        "model": "ensemble",
        "checkpoint": "",
        "val_rmse": 0.0,
        "epochs": 0,
        **metrics,
    }


def show_example_prediction(dataset: str, args) -> None:
    """Print one worked prediction so a training run ends in something concrete."""
    from scripts.predict import predict_engine  # local import: optional path

    try:
        result = predict_engine(dataset=dataset, engine_id=None, models_dir="models")
    except Exception as exc:  # pragma: no cover - convenience output only
        print(f"\n(Skipping example prediction: {exc})")
        return

    print(f"\n{'=' * 70}\n Example prediction\n{'=' * 70}")
    print(
        f"  {dataset} engine {result['engine_id']}: "
        f"RUL {result['rul']:.1f} +/- {result['uncertainty']:.1f} cycles "
        f"(true {result['true_rul']:.0f})"
    )
    print(f"\n  Reproduce: python scripts/predict.py --dataset {dataset} --engine {result['engine_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train C-MAPSS RUL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm"],
        help="model names, or 'all' (default: lstm)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["FD001"],
        help="FD001..FD004, or 'all' (default: FD001)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs")
    parser.add_argument("--patience", type=int, default=None, help="override early-stopping patience")
    parser.add_argument("--lr", type=float, default=None, help="override learning rate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument(
        "--regimes",
        choices=["auto", "config"],
        default="auto",
        help="operating-regime clustering: 'auto' clusters only FD002/FD004 (default), "
        "'config' uses config.yaml for every dataset",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="log to CSV instead of MLflow")
    parser.add_argument("--no-ensemble", action="store_true", help="skip the ensemble step")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small LSTM, 5 epochs, FD001 - a few minutes on CPU, ends with a prediction",
    )
    parser.add_argument("--output", type=Path, default=None, help="results JSON path")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "config.yaml")
    args = parser.parse_args()

    if args.quick:
        args.models = ["lstm"]
        args.datasets = ["FD001"]
        args.epochs = args.epochs or 5
        args.patience = args.patience or 5
        args.no_ensemble = True

    models = list(MODEL_REGISTRY) if "all" in args.models else args.models
    datasets = ALL_DATASETS if "all" in args.datasets else args.datasets

    unknown = [m for m in models if m not in MODEL_REGISTRY]
    if unknown:
        parser.error(f"unknown model(s): {unknown}. Choose from: {list(MODEL_REGISTRY)}")
    unknown = [d for d in datasets if d not in ALL_DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s): {unknown}. Choose from: {ALL_DATASETS}")

    header("C-MAPSS RUL Training")
    print(f"Models:   {', '.join(models)}")
    print(f"Datasets: {', '.join(datasets)}")
    device = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available()
        else "MPS" if torch.backends.mps.is_available() else "CPU"
    )
    print(f"Device:   {device}")

    all_results = []
    for dataset in datasets:
        header(f"Dataset {dataset}")
        train_ds, val_ds, test_ds, feature_cols, ds_config, preprocessor, excluded = prepare_data(
            dataset, args, args.config
        )

        preprocessor_path = PROJECT_ROOT / "models" / "preprocessors" / f"{dataset}_preprocessor.pkl"
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessor.save(preprocessor_path)
        print(f"  Preprocessor saved to {preprocessor_path.relative_to(PROJECT_ROOT)}")

        per_model = {}
        for name in models:
            result = train_one(
                name, dataset, (train_ds, val_ds, test_ds), feature_cols, ds_config, args
            )
            result["regimes_mode"] = args.regimes
            result["test_engines_excluded"] = excluded
            per_model[name] = result
            all_results.append(result)

        if not args.no_ensemble and sum(m in per_model for m in ENSEMBLE_MEMBERS) >= 2:
            ensemble_result = build_ensemble(dataset, per_model, test_ds, args)
            if ensemble_result:
                ensemble_result["regimes_mode"] = args.regimes
                ensemble_result["test_engines_excluded"] = excluded
                all_results.append(ensemble_result)

    header("Results")
    print(f"{'Dataset':<9}{'Model':<26}{'RMSE':>8}{'MAE':>8}{'C-MAPSS':>12}")
    print("-" * 63)
    for r in sorted(all_results, key=lambda r: (r["dataset"], r["test_rmse"])):
        print(
            f"{r['dataset']:<9}{r['model']:<26}{r['test_rmse']:>8.2f}"
            f"{r['test_mae']:>8.2f}{r['test_cmapss']:>12.0f}"
        )
    print(
        "\nMetrics are per sliding window over all test trajectories, not one "
        "prediction per engine.\nThey are not directly comparable to published "
        "C-MAPSS leaderboard figures."
    )

    output = args.output or (
        PROJECT_ROOT / "models" / "training_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {output.relative_to(PROJECT_ROOT)}")

    if args.quick:
        show_example_prediction(datasets[0], args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
