#!/usr/bin/env python
"""
Train improved RUL prediction models.

Trains ImprovedLSTM and GRU models with enhanced training configuration.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.ingestion import CMAPSSDataLoader, compute_train_rul, compute_test_rul
from src.data.preprocessing import CMAPSSPreprocessor, PreprocessingConfig
from src.data.dataset import CMAPSSSequenceDataset, train_val_split
from src.models.lstm_improved import ImprovedLSTMModel
from src.models.gru import GRUModel
from src.evaluation.metrics import cmapss_score


def prepare_data(dataset_name: str = "FD001"):
    """Load and prepare data for training."""
    print(f"Loading {dataset_name} dataset...")

    loader = CMAPSSDataLoader(raw_data_dir=str(project_root / "data" / "raw"))
    data = loader.load_dataset(dataset_name)

    train_df = compute_train_rul(data.train)
    test_df = compute_test_rul(data.test, data.rul)

    # Preprocess with default settings
    config = PreprocessingConfig(
        drop_strategy="auto",
        rul_cap=125,
        normalization="minmax",
        cluster_by_regime=True,
        n_regimes=6,
    )
    preprocessor = CMAPSSPreprocessor(config)
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    # Train/val split
    train_split, val_split = train_val_split(train_processed, val_ratio=0.2, seed=42)

    feature_cols = preprocessor.get_feature_names()
    print(f"Features: {len(feature_cols)}")
    print(f"Train engines: {train_split['engine_id'].nunique()}")
    print(f"Val engines: {val_split['engine_id'].nunique()}")
    print(f"Test engines: {test_processed['engine_id'].nunique()}")

    return train_split, val_split, test_processed, feature_cols, preprocessor


def create_dataloaders(train_df, val_df, test_df, feature_cols, sequence_length=30, batch_size=64):
    """Create data loaders."""
    train_ds = CMAPSSSequenceDataset(train_df, feature_cols, sequence_length, stride=1)
    val_ds = CMAPSSSequenceDataset(val_df, feature_cols, sequence_length, stride=1)
    test_ds = CMAPSSSequenceDataset(test_df, feature_cols, sequence_length, stride=1)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, model_name: str, max_epochs: int = 150):
    """Train a model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    checkpoint_dir = project_root / "models" / "checkpoints_improved"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,  # More patience
            mode="min",
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename=f"{model_name}-{{epoch:02d}}-{{val_loss:.4f}}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Gradient clipping
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint
    best_path = callbacks[1].best_model_path
    if best_path:
        print(f"Loading best checkpoint: {best_path}")
        model = type(model).load_from_checkpoint(best_path)

    return model, trainer


def evaluate_model(model, test_loader, model_name: str):
    """Evaluate model on test set."""
    # Determine device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            preds = model(x)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    # Compute metrics
    errors = preds - targets
    rmse = np.sqrt((errors ** 2).mean())
    mae = np.abs(errors).mean()
    score = cmapss_score(targets, preds)

    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  C-MAPSS Score: {score:.0f}")

    # Error by RUL range
    print("\n  Error by RUL range:")
    for low, high in [(0, 20), (20, 50), (50, 100), (100, 150)]:
        mask = (targets >= low) & (targets < high)
        if mask.sum() > 0:
            range_mae = np.abs(errors[mask]).mean()
            print(f"    RUL {low:>3}-{high:<3}: MAE = {range_mae:.2f} ({mask.sum()} samples)")

    return {
        "model": model_name,
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_cmapss": float(score),
    }


def main():
    """Main training loop."""
    print("="*60)
    print("Training Improved RUL Prediction Models")
    print("="*60)

    # Prepare data
    train_df, val_df, test_df, feature_cols, preprocessor = prepare_data("FD001")

    # Create data loaders
    sequence_length = 30
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, feature_cols,
        sequence_length=sequence_length, batch_size=64
    )

    input_dim = len(feature_cols)
    results = []

    # Train ImprovedLSTM
    improved_lstm = ImprovedLSTMModel(
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_size=256,
        num_layers=3,
        n_attention_heads=4,
        dropout=0.2,
        learning_rate=5e-4,
        weight_decay=1e-4,
    )
    improved_lstm, _ = train_model(improved_lstm, train_loader, val_loader, "improved_lstm", max_epochs=150)
    result = evaluate_model(improved_lstm, test_loader, "ImprovedLSTM")
    result["dataset"] = "FD001"
    results.append(result)

    # Train GRU
    gru = GRUModel(
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_size=256,
        num_layers=3,
        dropout=0.2,
        learning_rate=5e-4,
        weight_decay=1e-4,
    )
    gru, _ = train_model(gru, train_loader, val_loader, "gru", max_epochs=150)
    result = evaluate_model(gru, test_loader, "GRU")
    result["dataset"] = "FD001"
    results.append(result)

    # Save results
    results_path = project_root / "models" / "improved_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['model']:20s} RMSE={r['test_rmse']:.2f}  MAE={r['test_mae']:.2f}")


if __name__ == "__main__":
    main()
