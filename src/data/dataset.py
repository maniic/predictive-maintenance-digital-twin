"""
PyTorch Dataset Module for C-MAPSS Data

This module provides PyTorch Dataset classes for training deep learning
models on C-MAPSS time series data. Features include:
- Sliding window sequences for LSTM/Transformer models
- Configurable sequence length and stride
- Train/validation split by engine (no data leakage)
- Support for both training and inference modes

The datasets are designed for RUL (Remaining Useful Life) prediction
as a regression task.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import yaml


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    
    # Sequence length for sliding window
    sequence_length: int = 30
    
    # Stride between consecutive sequences (1 = max overlap)
    stride: int = 1
    
    # Validation split ratio (by engine, not by sample)
    val_ratio: float = 0.2
    
    # Random seed for reproducibility
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DatasetConfig":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        ds_config = config.get("dataset", {})
        return cls(
            sequence_length=ds_config.get("sequence_length", 30),
            stride=ds_config.get("stride", 1),
            val_ratio=ds_config.get("val_ratio", 0.2),
            seed=ds_config.get("seed", 42),
        )


class CMAPSSSequenceDataset(Dataset):
    """PyTorch Dataset for C-MAPSS time series sequences.
    
    Creates sliding window sequences from engine sensor data for
    training RUL prediction models. Each sample consists of:
    - X: (sequence_length, n_features) tensor of sensor readings
    - y: scalar RUL value at the end of the sequence
    
    Sequences are created per-engine to avoid mixing data from
    different engines within a single sequence.
    
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int = 30,
        stride: int = 1,
        target_column: str = "RUL",
    ):
        """Initialize the dataset.
        
        Args:
            df: DataFrame with engine_id, cycle, features, and RUL columns
            feature_columns: List of column names to use as features
            sequence_length: Number of time steps in each sequence
            stride: Step size between consecutive sequences
            target_column: Name of the target column (default: "RUL")
        """
        self.sequence_length = sequence_length
        self.stride = stride
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Build sequences
        self.sequences, self.targets = self._build_sequences(df)
    
    def _build_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sliding window sequences from DataFrame.
        
        Returns:
            Tuple of (sequences, targets) arrays
            - sequences: shape (n_samples, sequence_length, n_features)
            - targets: shape (n_samples,)
        """
        sequences = []
        targets = []
        
        for engine_id, group in df.groupby("engine_id"):
            # Sort by cycle
            group = group.sort_values("cycle").reset_index(drop=True)
            
            # Extract feature matrix and target vector
            features = group[self.feature_columns].values
            rul = group[self.target_column].values
            
            n_samples = len(group)
            
            # Create sequences with sliding window
            for i in range(0, n_samples - self.sequence_length + 1, self.stride):
                seq = features[i : i + self.sequence_length]
                target = rul[i + self.sequence_length - 1]  # RUL at end of sequence
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.from_numpy(self.sequences[idx])
        y = torch.tensor(self.targets[idx])
        return X, y
    
    @property
    def n_features(self) -> int:
        """Number of input features."""
        return self.sequences.shape[2] if len(self.sequences) > 0 else 0
    
    def get_sample_weights(self, method: str = "linear") -> torch.Tensor:
        """Compute sample weights based on RUL values.
        
        Lower RUL values (closer to failure) can be weighted higher
        to improve predictions in the critical region.
        
        Args:
            method: Weighting method - "linear", "inverse", or "exp"
            
        Returns:
            Tensor of sample weights
        """
        rul = self.targets
        
        if method == "linear":
            # Higher weight for lower RUL
            max_rul = rul.max()
            weights = (max_rul - rul) / max_rul + 0.1
        elif method == "inverse":
            # Inverse of RUL
            weights = 1 / (rul + 1)
        elif method == "exp":
            # Exponential decay
            weights = np.exp(-rul / 50)
        else:
            weights = np.ones_like(rul)
        
        # Normalize
        weights = weights / weights.sum() * len(weights)
        
        return torch.from_numpy(weights.astype(np.float32))


class CMAPSSInferenceDataset(Dataset):
    """Dataset for inference on new/test data.
    
    Unlike CMAPSSSequenceDataset, this creates sequences for every
    possible endpoint (not just the last one per engine), which is
    useful for generating predictions across the full trajectory.
    
    Can also handle single sequences for real-time prediction.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        sequence_length: int = 30,
    ):
        """Initialize inference dataset.
        
        Args:
            df: DataFrame with engine_id, cycle, and feature columns
            feature_columns: List of column names to use as features
            sequence_length: Number of time steps in each sequence
        """
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        
        # Store sequences with metadata for reconstruction
        self.sequences, self.metadata = self._build_sequences(df)
    
    def _build_sequences(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, list[dict]]:
        """Build sequences with metadata for result mapping."""
        sequences = []
        metadata = []
        
        for engine_id, group in df.groupby("engine_id"):
            group = group.sort_values("cycle").reset_index(drop=True)
            features = group[self.feature_columns].values
            cycles = group["cycle"].values
            
            n_samples = len(group)
            
            # For inference, we want prediction at every valid position
            for i in range(self.sequence_length - 1, n_samples):
                start_idx = i - self.sequence_length + 1
                seq = features[start_idx : i + 1]
                
                sequences.append(seq)
                metadata.append({
                    "engine_id": engine_id,
                    "cycle": cycles[i],
                    "is_last": i == n_samples - 1,
                })
        
        return np.array(sequences, dtype=np.float32), metadata
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.sequences[idx])
    
    def get_metadata(self) -> pd.DataFrame:
        """Get metadata DataFrame for mapping predictions back to engines."""
        return pd.DataFrame(self.metadata)


def train_val_split(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/val by engine ID.
    
    Splitting by engine prevents data leakage where sequences from
    the same engine appear in both train and validation sets.
    
    Args:
        df: DataFrame with engine_id column
        val_ratio: Fraction of engines to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df)
    """
    rng = np.random.RandomState(seed)
    
    engine_ids = df["engine_id"].unique()
    n_val = int(len(engine_ids) * val_ratio)
    
    # Shuffle and split
    rng.shuffle(engine_ids)
    val_engines = set(engine_ids[:n_val])
    train_engines = set(engine_ids[n_val:])
    
    train_df = df[df["engine_id"].isin(train_engines)].copy()
    val_df = df[df["engine_id"].isin(val_engines)].copy()
    
    return train_df, val_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    config: Optional[DatasetConfig] = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        feature_columns: Feature column names
        config: Dataset configuration
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    config = config or DatasetConfig()
    
    train_dataset = CMAPSSSequenceDataset(
        train_df,
        feature_columns,
        sequence_length=config.sequence_length,
        stride=config.stride,
    )
    
    val_dataset = CMAPSSSequenceDataset(
        val_df,
        feature_columns,
        sequence_length=config.sequence_length,
        stride=1,  # Use stride=1 for validation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader