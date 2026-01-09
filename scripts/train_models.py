#!/usr/bin/env python
"""
Comprehensive Training Script for C-MAPSS RUL Prediction Models

This script trains and evaluates all models (LSTM, CNN, Transformer) on the
C-MAPSS FD001 dataset, creates an ensemble, and reports comparative results.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --dataset FD001 --epochs 50
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

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
    CMAPSSInferenceDataset,
    DatasetConfig,
    train_val_split,
    create_dataloaders,
)
from src.models import LSTMModel, TemporalCNNModel, TransformerModel, EnsembleModel
from src.training.trainer import RULTrainer, TrainingConfig
from src.evaluation.metrics import evaluate_predictions, cmapss_score


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f" {text}")
    print(f"{'='*70}\n")


def print_subheader(text: str) -> None:
    """Print a formatted subheader."""
    print(f"\n{'-'*50}")
    print(f" {text}")
    print(f"{'-'*50}\n")


def load_and_preprocess_data(
    dataset_name: str = "FD001",
    raw_data_dir: str = "data/raw",
    config_path: str = "config/config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], CMAPSSPreprocessor]:
    """Load and preprocess C-MAPSS data.

    Returns:
        Tuple of (train_df, val_df, test_df, feature_columns, preprocessor)
    """
    print_subheader(f"Loading {dataset_name} dataset")

    # Load raw data
    loader = CMAPSSDataLoader(raw_data_dir)
    dataset = loader.load_dataset(dataset_name)

    print(f"Dataset loaded: {dataset}")
    print(f"  Train engines: {dataset.n_train_engines}")
    print(f"  Test engines: {dataset.n_test_engines}")

    # Compute RUL labels
    train_df = compute_train_rul(dataset.train)
    test_df = compute_test_rul(dataset.test, dataset.rul)

    print(f"  Train samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")

    # Preprocess
    print_subheader("Preprocessing data")

    config = PreprocessingConfig.from_yaml(config_path)
    preprocessor = CMAPSSPreprocessor(config)

    # Fit on train data only
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    # Print sensor analysis
    preprocessor.print_sensor_summary()

    # Get feature columns
    feature_cols = preprocessor.get_feature_names()
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols[:5]}...")

    # Train/val split
    print_subheader("Creating train/validation split")

    ds_config = DatasetConfig.from_yaml(config_path)
    train_split, val_split = train_val_split(
        train_processed,
        val_ratio=ds_config.val_ratio,
        seed=ds_config.seed,
    )

    print(f"  Train engines: {train_split['engine_id'].nunique()}")
    print(f"  Val engines: {val_split['engine_id'].nunique()}")
    print(f"  Train samples: {len(train_split):,}")
    print(f"  Val samples: {len(val_split):,}")

    return train_split, val_split, test_processed, feature_cols, preprocessor


def create_datasets_and_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    config_path: str = "config/config.yaml",
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, CMAPSSSequenceDataset, DatasetConfig]:
    """Create PyTorch datasets and data loaders."""
    print_subheader("Creating datasets and dataloaders")

    ds_config = DatasetConfig.from_yaml(config_path)

    # Training and validation datasets
    train_dataset = CMAPSSSequenceDataset(
        train_df,
        feature_cols,
        sequence_length=ds_config.sequence_length,
        stride=ds_config.stride,
    )

    val_dataset = CMAPSSSequenceDataset(
        val_df,
        feature_cols,
        sequence_length=ds_config.sequence_length,
        stride=1,
    )

    # Test dataset
    test_dataset = CMAPSSSequenceDataset(
        test_df,
        feature_cols,
        sequence_length=ds_config.sequence_length,
        stride=1,
    )

    print(f"  Train sequences: {len(train_dataset):,}")
    print(f"  Val sequences: {len(val_dataset):,}")
    print(f"  Test sequences: {len(test_dataset):,}")
    print(f"  Sequence length: {ds_config.sequence_length}")
    print(f"  Input features: {train_dataset.n_features}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,  # MPS compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader, test_dataset, ds_config


def train_model(
    model_class,
    model_name: str,
    input_dim: int,
    sequence_length: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
    model_kwargs: dict = None,
) -> tuple:
    """Train a single model and return it with the best checkpoint path."""
    print_subheader(f"Training {model_name}")

    model_kwargs = model_kwargs or {}
    model = model_class(
        input_dim=input_dim,
        sequence_length=sequence_length,
        learning_rate=config.learning_rate,
        **model_kwargs,
    )

    print(f"Model architecture:")
    print(f"  Class: {model.__class__.__name__}")
    print(f"  Input dim: {input_dim}")
    print(f"  Sequence length: {sequence_length}")
    for k, v in model_kwargs.items():
        print(f"  {k}: {v}")

    # Create trainer
    trainer = RULTrainer(config)

    # Train
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"

    trained_model = trainer.train(
        model,
        train_loader,
        val_loader,
        run_name=run_name,
    )

    # Get best checkpoint path
    best_path = trainer._trainer.checkpoint_callback.best_model_path

    return trained_model, best_path, trainer


def evaluate_model(
    model,
    model_name: str,
    test_loader: DataLoader,
    trainer: RULTrainer,
) -> dict:
    """Evaluate a trained model on test data."""
    print_subheader(f"Evaluating {model_name}")

    model.eval()

    # Collect predictions
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(model.device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_targets.append(y)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    # Compute metrics
    results = evaluate_predictions(y_pred, y_true)

    print(f"Results for {model_name}:")
    print(f"  RMSE: {results.rmse:.2f}")
    print(f"  MAE: {results.mae:.2f}")
    print(f"  C-MAPSS Score: {results.cmapss_score:.2f}")
    print(f"  C-MAPSS Score (normalized): {results.cmapss_score_normalized:.4f}")

    return {
        "model_name": model_name,
        "rmse": results.rmse,
        "mae": results.mae,
        "cmapss_score": results.cmapss_score,
        "cmapss_score_normalized": results.cmapss_score_normalized,
        "predictions": y_pred,
        "targets": y_true,
    }


def evaluate_ensemble(
    ensemble: EnsembleModel,
    test_loader: DataLoader,
) -> dict:
    """Evaluate ensemble model on test data."""
    print_subheader("Evaluating Ensemble")

    ensemble.eval()

    # Collect predictions
    all_preds = []
    all_stds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            result = ensemble.predict_with_uncertainty(x)
            all_preds.append(result.mean.cpu())
            all_stds.append(result.std.cpu())
            all_targets.append(y)

    y_pred = torch.cat(all_preds).numpy()
    y_std = torch.cat(all_stds).numpy()
    y_true = torch.cat(all_targets).numpy()

    # Compute metrics
    results = evaluate_predictions(y_pred, y_true)

    print(f"Ensemble Results:")
    print(f"  RMSE: {results.rmse:.2f}")
    print(f"  MAE: {results.mae:.2f}")
    print(f"  C-MAPSS Score: {results.cmapss_score:.2f}")
    print(f"  Mean prediction uncertainty (std): {y_std.mean():.2f}")
    print(f"\nEnsemble weights: {ensemble.weight_dict}")

    return {
        "model_name": "Ensemble",
        "rmse": results.rmse,
        "mae": results.mae,
        "cmapss_score": results.cmapss_score,
        "cmapss_score_normalized": results.cmapss_score_normalized,
        "mean_uncertainty": float(y_std.mean()),
        "predictions": y_pred,
        "uncertainties": y_std,
        "targets": y_true,
    }


def print_comparison_table(results_list: list[dict]) -> None:
    """Print a comparison table of all model results."""
    print_header("MODEL COMPARISON")

    # Create comparison DataFrame
    comparison = pd.DataFrame([
        {
            "Model": r["model_name"],
            "RMSE": f"{r['rmse']:.2f}",
            "MAE": f"{r['mae']:.2f}",
            "C-MAPSS Score": f"{r['cmapss_score']:.0f}",
        }
        for r in results_list
    ])

    print(comparison.to_string(index=False))

    # Find best model
    rmse_values = [r["rmse"] for r in results_list]
    best_idx = np.argmin(rmse_values)
    print(f"\nBest model by RMSE: {results_list[best_idx]['model_name']} (RMSE={rmse_values[best_idx]:.2f})")

    cmapss_values = [r["cmapss_score"] for r in results_list]
    best_cmapss_idx = np.argmin(cmapss_values)
    print(f"Best model by C-MAPSS: {results_list[best_cmapss_idx]['model_name']} (Score={cmapss_values[best_cmapss_idx]:.0f})")


def main():
    parser = argparse.ArgumentParser(description="Train C-MAPSS RUL prediction models")
    parser.add_argument("--dataset", type=str, default="FD001", help="Dataset name (FD001-FD004)")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    print_header(f"C-MAPSS RUL Prediction - Full Training Pipeline")
    print(f"Dataset: {args.dataset}")
    print(f"Max epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")

    # Device info
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = "mps"
        print(f"Device: MPS (Apple Silicon)")
    else:
        device = "cpu"
        print(f"Device: CPU")

    # =========================================================================
    # 1. Load and preprocess data
    # =========================================================================
    train_df, val_df, test_df, feature_cols, preprocessor = load_and_preprocess_data(
        dataset_name=args.dataset,
    )

    # Save preprocessor for later use
    preprocessor_path = Path("models/preprocessor.pkl")
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor.save(preprocessor_path)
    print(f"\nPreprocessor saved to: {preprocessor_path}")

    # =========================================================================
    # 2. Create datasets and loaders
    # =========================================================================
    train_loader, val_loader, test_loader, test_dataset, ds_config = create_datasets_and_loaders(
        train_df, val_df, test_df, feature_cols,
        batch_size=args.batch_size,
    )

    input_dim = len(feature_cols)
    sequence_length = ds_config.sequence_length

    # =========================================================================
    # 3. Training configuration
    # =========================================================================
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        early_stopping_patience=args.patience,
        checkpoint_dir=f"models/checkpoints/{args.dataset}",
        experiment_name=f"cmapss-{args.dataset.lower()}",
    )

    # Storage for models and results
    trained_models = {}
    checkpoint_paths = {}
    all_results = []

    # =========================================================================
    # 4. Train LSTM
    # =========================================================================
    lstm_model, lstm_path, lstm_trainer = train_model(
        model_class=LSTMModel,
        model_name="LSTM",
        input_dim=input_dim,
        sequence_length=sequence_length,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        model_kwargs={
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "bidirectional": True,
        },
    )
    trained_models["lstm"] = lstm_model
    checkpoint_paths["lstm"] = lstm_path

    lstm_results = evaluate_model(lstm_model, "LSTM", test_loader, lstm_trainer)
    all_results.append(lstm_results)

    # =========================================================================
    # 5. Train CNN
    # =========================================================================
    cnn_model, cnn_path, cnn_trainer = train_model(
        model_class=TemporalCNNModel,
        model_name="CNN",
        input_dim=input_dim,
        sequence_length=sequence_length,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        model_kwargs={
            "channels": [32, 64, 128],
            "kernel_sizes": [3, 3, 3],
            "dropout": 0.3,
        },
    )
    trained_models["cnn"] = cnn_model
    checkpoint_paths["cnn"] = cnn_path

    cnn_results = evaluate_model(cnn_model, "CNN", test_loader, cnn_trainer)
    all_results.append(cnn_results)

    # =========================================================================
    # 6. Train Transformer
    # =========================================================================
    transformer_model, transformer_path, transformer_trainer = train_model(
        model_class=TransformerModel,
        model_name="Transformer",
        input_dim=input_dim,
        sequence_length=sequence_length,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        model_kwargs={
            "d_model": 64,
            "n_heads": 4,
            "n_layers": 3,
            "dropout": 0.2,
        },
    )
    trained_models["transformer"] = transformer_model
    checkpoint_paths["transformer"] = transformer_path

    transformer_results = evaluate_model(transformer_model, "Transformer", test_loader, transformer_trainer)
    all_results.append(transformer_results)

    # =========================================================================
    # 7. Create and evaluate ensemble
    # =========================================================================
    print_subheader("Creating Ensemble Model")

    ensemble = EnsembleModel(trained_models)
    print(f"Ensemble created: {ensemble}")

    # Calibrate weights on validation data
    print("Calibrating ensemble weights on validation data...")
    calibrated_weights = ensemble.calibrate_weights(val_loader, method="inverse_rmse")
    print(f"Calibrated weights: {calibrated_weights}")

    ensemble_results = evaluate_ensemble(ensemble, test_loader)
    all_results.append(ensemble_results)

    # =========================================================================
    # 8. Final comparison
    # =========================================================================
    print_comparison_table(all_results)

    # Save checkpoint paths for future use
    print_header("SAVED ARTIFACTS")
    print(f"Preprocessor: {preprocessor_path}")
    for name, path in checkpoint_paths.items():
        print(f"{name.upper()} checkpoint: {path}")

    print("\nTraining complete!")
    print(f"Run 'mlflow ui --backend-store-uri mlruns' to view experiments")

    return all_results, trained_models, ensemble


if __name__ == "__main__":
    results, models, ensemble = main()
