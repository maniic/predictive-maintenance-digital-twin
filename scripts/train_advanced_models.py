#!/usr/bin/env python
"""
Train advanced RUL prediction models with multiple strategies.

Trains:
1. EnhancedLSTM with weighted loss
2. EnhancedLSTM with asymmetric loss
3. TwoStageRULModel
4. All on FD001-FD004 datasets
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
from src.models.advanced_rul import EnhancedLSTM, TwoStageRULModel
from src.evaluation.metrics import cmapss_score


def prepare_data(dataset_name: str = "FD001"):
    """Load and prepare data for training."""
    print(f"Loading {dataset_name} dataset...")

    loader = CMAPSSDataLoader(raw_data_dir=str(project_root / "data" / "raw"))
    data = loader.load_dataset(dataset_name)

    train_df = compute_train_rul(data.train)
    test_df = compute_test_rul(data.test, data.rul)

    # Determine n_regimes based on dataset
    n_regimes = 6 if dataset_name in ["FD002", "FD004"] else 1

    config = PreprocessingConfig(
        drop_strategy="auto",
        rul_cap=125,
        normalization="minmax",
        cluster_by_regime=True,
        n_regimes=n_regimes,
    )
    preprocessor = CMAPSSPreprocessor(config)
    train_processed = preprocessor.fit_transform(train_df)
    test_processed = preprocessor.transform(test_df)

    train_split, val_split = train_val_split(train_processed, val_ratio=0.2, seed=42)

    feature_cols = preprocessor.get_feature_names()
    print(f"  Features: {len(feature_cols)}, Train: {train_split['engine_id'].nunique()} engines")

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


def train_model(model, train_loader, val_loader, model_name: str, max_epochs: int = 100, patience: int = 15):
    """Train a model."""
    print(f"\n  Training {model_name}...")

    checkpoint_dir = project_root / "models" / "checkpoints_advanced"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=False),
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
        gradient_clip_val=1.0,
        enable_model_summary=False,
    )

    trainer.fit(model, train_loader, val_loader)

    best_path = callbacks[1].best_model_path
    if best_path:
        model = type(model).load_from_checkpoint(best_path)

    return model


def evaluate_model(model, test_loader, model_name: str, is_two_stage: bool = False):
    """Evaluate model on test set."""
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

            if is_two_stage:
                _, preds = model(x)
            else:
                preds = model(x)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    errors = preds - targets
    rmse = np.sqrt((errors ** 2).mean())
    mae = np.abs(errors).mean()
    score = cmapss_score(targets, preds)

    # Error by RUL range
    range_maes = {}
    for low, high in [(0, 20), (20, 50), (50, 100), (100, 150)]:
        mask = (targets >= low) & (targets < high)
        if mask.sum() > 0:
            range_maes[f"{low}-{high}"] = float(np.abs(errors[mask]).mean())

    print(f"    {model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, Score={score:.0f}")
    print(f"      Range MAEs: 0-20={range_maes.get('0-20', 'N/A'):.2f}, "
          f"20-50={range_maes.get('20-50', 'N/A'):.2f}, "
          f"50-100={range_maes.get('50-100', 'N/A'):.2f}")

    return {
        "model": model_name,
        "test_rmse": float(rmse),
        "test_mae": float(mae),
        "test_cmapss": float(score),
        "range_maes": range_maes,
    }


def main():
    """Main training loop."""
    print("=" * 70)
    print("Training Advanced RUL Prediction Models")
    print("=" * 70)

    all_results = []
    datasets = ["FD001", "FD002", "FD003", "FD004"]

    for dataset_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print("=" * 70)

        # Prepare data
        train_df, val_df, test_df, feature_cols, preprocessor = prepare_data(dataset_name)
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, feature_cols,
            sequence_length=30, batch_size=64
        )

        input_dim = len(feature_cols)
        dataset_results = []

        # 1. EnhancedLSTM with weighted loss
        model_weighted = EnhancedLSTM(
            input_dim=input_dim,
            sequence_length=30,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            learning_rate=1e-3,
            loss_type="weighted",
        )
        model_weighted = train_model(model_weighted, train_loader, val_loader,
                                      f"enhanced_lstm_weighted_{dataset_name}", max_epochs=100, patience=15)
        result = evaluate_model(model_weighted, test_loader, "EnhancedLSTM-Weighted")
        result["dataset"] = dataset_name
        dataset_results.append(result)

        # 2. EnhancedLSTM with asymmetric loss
        model_asymm = EnhancedLSTM(
            input_dim=input_dim,
            sequence_length=30,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            learning_rate=1e-3,
            loss_type="asymmetric",
        )
        model_asymm = train_model(model_asymm, train_loader, val_loader,
                                   f"enhanced_lstm_asymmetric_{dataset_name}", max_epochs=100, patience=15)
        result = evaluate_model(model_asymm, test_loader, "EnhancedLSTM-Asymmetric")
        result["dataset"] = dataset_name
        dataset_results.append(result)

        # 3. TwoStageRULModel
        model_twostage = TwoStageRULModel(
            input_dim=input_dim,
            sequence_length=30,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            learning_rate=1e-3,
            critical_threshold=25.0,
            degrading_threshold=75.0,
        )
        model_twostage = train_model(model_twostage, train_loader, val_loader,
                                      f"two_stage_{dataset_name}", max_epochs=100, patience=15)
        result = evaluate_model(model_twostage, test_loader, "TwoStage", is_two_stage=True)
        result["dataset"] = dataset_name
        dataset_results.append(result)

        all_results.extend(dataset_results)

        # Save preprocessor for this dataset
        preprocessor_path = project_root / "models" / "preprocessors" / f"{dataset_name}_preprocessor_advanced.pkl"
        preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
        preprocessor.save(str(preprocessor_path))

    # Save all results
    results_path = project_root / "models" / "advanced_training_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY - All Datasets")
    print("=" * 70)
    print(f"{'Model':<25} {'Dataset':<8} {'RMSE':>8} {'MAE':>8} {'MAE 0-20':>10}")
    print("-" * 70)

    for r in all_results:
        mae_0_20 = r['range_maes'].get('0-20', 'N/A')
        if isinstance(mae_0_20, float):
            mae_0_20 = f"{mae_0_20:.2f}"
        print(f"{r['model']:<25} {r['dataset']:<8} {r['test_rmse']:>8.2f} {r['test_mae']:>8.2f} {mae_0_20:>10}")


if __name__ == "__main__":
    main()
