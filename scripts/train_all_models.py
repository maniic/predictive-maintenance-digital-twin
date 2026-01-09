#!/usr/bin/env python
"""
Comprehensive Training Script for All C-MAPSS Datasets

Trains LSTM, CNN, and Transformer models on all C-MAPSS datasets (FD001-FD004)
with proper checkpoint organization and evaluation.

Usage:
    python scripts/train_all_models.py                    # Train all
    python scripts/train_all_models.py --datasets FD001   # Single dataset
    python scripts/train_all_models.py --models lstm cnn  # Specific models
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.ingestion import CMAPSSDataLoader, compute_train_rul, compute_test_rul
from src.data.preprocessing import CMAPSSPreprocessor, PreprocessingConfig
from src.data.dataset import (
    CMAPSSSequenceDataset,
    DatasetConfig,
    train_val_split,
)
from src.models import LSTMModel, TemporalCNNModel, TransformerModel, EnsembleModel
from src.training.trainer import RULTrainer, TrainingConfig
from src.evaluation.metrics import evaluate_predictions


# Model configurations optimized for C-MAPSS
MODEL_CONFIGS = {
    "lstm": {
        "class": LSTMModel,
        "params": {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
        },
    },
    "cnn": {
        "class": TemporalCNNModel,
        "params": {
            "channels": [64, 128, 256],  # Increased capacity
            "kernel_sizes": [5, 5, 3],   # Larger kernels for temporal patterns
            "dropout": 0.3,
        },
    },
    "transformer": {
        "class": TransformerModel,
        "params": {
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 4,  # Increased depth
            "dropout": 0.2,
        },
    },
}


@dataclass
class TrainingResult:
    """Container for training results."""
    dataset: str
    model_name: str
    checkpoint_path: str
    train_loss: float
    val_loss: float
    val_rmse: float
    test_rmse: float
    test_mae: float
    test_cmapss: float
    epochs_trained: int


def print_header(text: str) -> None:
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}\n")


def print_subheader(text: str) -> None:
    print(f"\n{'-'*50}")
    print(f" {text}")
    print(f"{'-'*50}\n")


def prepare_data(
    dataset_name: str,
    raw_data_dir: str = "data/raw",
    config_path: str = "config/config.yaml",
) -> tuple:
    """Load and preprocess data for a single dataset."""
    print_subheader(f"Preparing {dataset_name}")

    # Load raw data
    loader = CMAPSSDataLoader(raw_data_dir)
    dataset = loader.load_dataset(dataset_name)

    print(f"  Train engines: {dataset.n_train_engines}")
    print(f"  Test engines: {dataset.n_test_engines}")

    # Compute RUL labels
    train_df = compute_train_rul(dataset.train)
    test_df = compute_test_rul(dataset.test, dataset.rul)

    # Preprocess
    config = PreprocessingConfig.from_yaml(config_path)
    preprocessor = CMAPSSPreprocessor(config)
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    feature_cols = preprocessor.get_feature_names()
    dropped = preprocessor.get_dropped_sensors()
    print(f"  Features: {len(feature_cols)} (dropped {len(dropped)} zero-variance sensors)")

    # Train/val split
    ds_config = DatasetConfig.from_yaml(config_path)
    train_split, val_split = train_val_split(
        train_processed,
        val_ratio=ds_config.val_ratio,
        seed=ds_config.seed,
    )

    # Create datasets
    train_dataset = CMAPSSSequenceDataset(
        train_split, feature_cols,
        sequence_length=ds_config.sequence_length,
        stride=ds_config.stride,
    )
    val_dataset = CMAPSSSequenceDataset(
        val_split, feature_cols,
        sequence_length=ds_config.sequence_length,
        stride=1,
    )
    test_dataset = CMAPSSSequenceDataset(
        test_processed, feature_cols,
        sequence_length=ds_config.sequence_length,
        stride=1,
    )

    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences: {len(val_dataset):,}")
    print(f"  Test sequences: {len(test_dataset):,}")

    return (
        train_dataset, val_dataset, test_dataset,
        feature_cols, ds_config, preprocessor
    )


def train_single_model(
    model_name: str,
    dataset_name: str,
    train_dataset: CMAPSSSequenceDataset,
    val_dataset: CMAPSSSequenceDataset,
    test_dataset: CMAPSSSequenceDataset,
    feature_cols: list[str],
    ds_config: DatasetConfig,
    training_config: dict,
) -> TrainingResult:
    """Train and evaluate a single model."""
    print_subheader(f"Training {model_name.upper()} on {dataset_name}")

    # Get model config
    model_cfg = MODEL_CONFIGS[model_name]
    model_class = model_cfg["class"]
    model_params = model_cfg["params"].copy()

    # Create model
    input_dim = len(feature_cols)
    sequence_length = ds_config.sequence_length

    model = model_class(
        input_dim=input_dim,
        sequence_length=sequence_length,
        learning_rate=training_config["learning_rate"],
        **model_params,
    )

    print(f"  Model: {model_class.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders
    batch_size = training_config["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    # Create trainer with MODEL-SPECIFIC checkpoint directory
    checkpoint_dir = f"models/checkpoints/{dataset_name}/{model_name}"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    trainer_config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=training_config["learning_rate"],
        max_epochs=training_config["max_epochs"],
        early_stopping_patience=training_config["patience"],
        checkpoint_dir=checkpoint_dir,
        experiment_name=f"cmapss-{dataset_name.lower()}-{model_name}",
    )

    trainer = RULTrainer(trainer_config)

    # Train
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{dataset_name}_{timestamp}"

    trained_model = trainer.train(model, train_loader, val_loader, run_name=run_name)

    # Get training metrics
    best_val_loss = trainer._trainer.checkpoint_callback.best_model_score
    if best_val_loss is not None:
        best_val_loss = float(best_val_loss)
    else:
        best_val_loss = float('inf')

    best_checkpoint = trainer._trainer.checkpoint_callback.best_model_path
    epochs_trained = trainer._trainer.current_epoch + 1

    print(f"  Best val_loss: {best_val_loss:.4f}")
    print(f"  Epochs trained: {epochs_trained}")
    print(f"  Checkpoint: {best_checkpoint}")

    # Evaluate on test set
    print(f"  Evaluating on test set...")
    trained_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            device = next(trained_model.parameters()).device
            x = x.to(device)
            preds = trained_model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    results = evaluate_predictions(y_pred, y_true)

    print(f"  Test RMSE: {results.rmse:.2f}")
    print(f"  Test MAE: {results.mae:.2f}")
    print(f"  Test C-MAPSS: {results.cmapss_score:.0f}")

    return TrainingResult(
        dataset=dataset_name,
        model_name=model_name,
        checkpoint_path=best_checkpoint,
        train_loss=0.0,  # Not easily accessible
        val_loss=best_val_loss,
        val_rmse=np.sqrt(best_val_loss) if best_val_loss != float('inf') else 0,
        test_rmse=results.rmse,
        test_mae=results.mae,
        test_cmapss=results.cmapss_score,
        epochs_trained=epochs_trained,
    )


def evaluate_ensemble(
    dataset_name: str,
    model_results: dict[str, TrainingResult],
    test_dataset: CMAPSSSequenceDataset,
    batch_size: int = 64,
) -> TrainingResult:
    """Create and evaluate ensemble from trained models."""
    print_subheader(f"Evaluating Ensemble on {dataset_name}")

    # Load best models
    models = {}
    for model_name, result in model_results.items():
        model_cfg = MODEL_CONFIGS[model_name]
        model_class = model_cfg["class"]
        models[model_name] = model_class.load_from_checkpoint(result.checkpoint_path)
        print(f"  Loaded {model_name} from {Path(result.checkpoint_path).name}")

    # Create ensemble
    ensemble = EnsembleModel(models)

    # Calibrate weights based on validation RMSE (inverse weighting)
    val_rmses = {name: result.val_rmse for name, result in model_results.items()}
    inv_rmses = {name: 1.0 / rmse if rmse > 0 else 1.0 for name, rmse in val_rmses.items()}
    total = sum(inv_rmses.values())
    weights = {name: v / total for name, v in inv_rmses.items()}

    ensemble._update_weights(weights)
    print(f"  Calibrated weights: {ensemble.weight_dict}")

    # Evaluate
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    all_preds = []
    all_stds = []
    all_targets = []

    for batch in test_loader:
        x, y = batch
        result = ensemble.predict_with_uncertainty(x)
        all_preds.append(result.mean)
        all_stds.append(result.std)
        all_targets.append(y)

    y_pred = torch.cat(all_preds).numpy()
    y_std = torch.cat(all_stds).numpy()
    y_true = torch.cat(all_targets).numpy()

    results = evaluate_predictions(y_pred, y_true)

    print(f"  Test RMSE: {results.rmse:.2f}")
    print(f"  Test MAE: {results.mae:.2f}")
    print(f"  Test C-MAPSS: {results.cmapss_score:.0f}")
    print(f"  Mean uncertainty: {y_std.mean():.2f}")

    return TrainingResult(
        dataset=dataset_name,
        model_name="ensemble",
        checkpoint_path="",
        train_loss=0.0,
        val_loss=0.0,
        val_rmse=0.0,
        test_rmse=results.rmse,
        test_mae=results.mae,
        test_cmapss=results.cmapss_score,
        epochs_trained=0,
    )


def print_results_table(all_results: list[TrainingResult]) -> None:
    """Print formatted results table."""
    print_header("FINAL RESULTS")

    # Group by dataset
    datasets = sorted(set(r.dataset for r in all_results))

    for dataset in datasets:
        print(f"\n{dataset}:")
        print(f"{'Model':<12} {'RMSE':>8} {'MAE':>8} {'C-MAPSS':>12} {'Val RMSE':>10}")
        print("-" * 55)

        dataset_results = [r for r in all_results if r.dataset == dataset]
        for r in sorted(dataset_results, key=lambda x: x.test_rmse):
            val_rmse_str = f"{r.val_rmse:.2f}" if r.val_rmse > 0 else "-"
            print(f"{r.model_name:<12} {r.test_rmse:>8.2f} {r.test_mae:>8.2f} {r.test_cmapss:>12.0f} {val_rmse_str:>10}")

    # Summary across datasets
    print_header("SUMMARY BY MODEL")
    models = sorted(set(r.model_name for r in all_results))

    print(f"{'Model':<12}", end="")
    for dataset in datasets:
        print(f" {dataset:>12}", end="")
    print(f" {'Average':>12}")
    print("-" * (12 + 13 * (len(datasets) + 1)))

    for model in models:
        print(f"{model:<12}", end="")
        rmses = []
        for dataset in datasets:
            result = next((r for r in all_results if r.dataset == dataset and r.model_name == model), None)
            if result:
                print(f" {result.test_rmse:>12.2f}", end="")
                rmses.append(result.test_rmse)
            else:
                print(f" {'-':>12}", end="")
        if rmses:
            print(f" {np.mean(rmses):>12.2f}")
        else:
            print()


def main():
    parser = argparse.ArgumentParser(description="Train all C-MAPSS models")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="Datasets to train on",
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["lstm", "cnn", "transformer"],
        help="Models to train",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--no-ensemble", action="store_true", help="Skip ensemble evaluation")
    args = parser.parse_args()

    print_header("C-MAPSS Full Training Pipeline")
    print(f"Datasets: {args.datasets}")
    print(f"Models: {args.models}")
    print(f"Max epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")

    # Device info
    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        print(f"Device: MPS (Apple Silicon)")
    else:
        print(f"Device: CPU")

    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_epochs": args.epochs,
        "patience": args.patience,
    }

    all_results = []

    for dataset_name in args.datasets:
        print_header(f"DATASET: {dataset_name}")

        # Prepare data
        (train_dataset, val_dataset, test_dataset,
         feature_cols, ds_config, preprocessor) = prepare_data(dataset_name)

        # Save preprocessor
        preprocessor_path = Path(f"models/preprocessors/{dataset_name}_preprocessor.pkl")
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessor.save(preprocessor_path)
        print(f"  Preprocessor saved to: {preprocessor_path}")

        # Train each model
        model_results = {}
        for model_name in args.models:
            result = train_single_model(
                model_name=model_name,
                dataset_name=dataset_name,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_dataset=test_dataset,
                feature_cols=feature_cols,
                ds_config=ds_config,
                training_config=training_config,
            )
            model_results[model_name] = result
            all_results.append(result)

        # Evaluate ensemble
        if not args.no_ensemble and len(model_results) > 1:
            ensemble_result = evaluate_ensemble(
                dataset_name=dataset_name,
                model_results=model_results,
                test_dataset=test_dataset,
                batch_size=args.batch_size,
            )
            all_results.append(ensemble_result)

    # Print final results
    print_results_table(all_results)

    # Save results to JSON
    results_path = Path("models/training_results.json")
    results_data = [
        {
            "dataset": r.dataset,
            "model": r.model_name,
            "checkpoint": r.checkpoint_path,
            "test_rmse": r.test_rmse,
            "test_mae": r.test_mae,
            "test_cmapss": r.test_cmapss,
            "val_rmse": r.val_rmse,
            "epochs": r.epochs_trained,
        }
        for r in all_results
    ]
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    print("\nTraining complete!")
    print("Run 'mlflow ui --backend-store-uri mlruns' to view experiments")

    return all_results


if __name__ == "__main__":
    results = main()
