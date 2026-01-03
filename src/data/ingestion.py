"""
Data Ingestion Module for NASA C-MAPSS Dataset

This module handles loading and parsing the raw C-MAPSS turbofan engine
degradation data files. The dataset contains run-to-failure sensor 
measurements from simulated aircraft engines.

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


@dataclass
class CMAPSSDataset:
    """Container for a single C-MAPSS sub-dataset (e.g., FD001).
    
    Attributes:
        name: Dataset identifier (FD001, FD002, FD003, or FD004)
        train: Training data with full run-to-failure trajectories
        test: Test data with truncated trajectories
        rul: Ground truth RUL values for test data (one per engine)
        n_conditions: Number of operating conditions (1 or 6)
        n_fault_modes: Number of fault modes (1 or 2)
    """
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.Series
    n_conditions: int
    n_fault_modes: int
    
    @property
    def n_train_engines(self) -> int:
        """Number of engines in training set."""
        return self.train["engine_id"].nunique()
    
    @property
    def n_test_engines(self) -> int:
        """Number of engines in test set."""
        return self.test["engine_id"].nunique()
    
    def __repr__(self) -> str:
        return (
            f"CMAPSSDataset({self.name}: "
            f"train={self.n_train_engines} engines, "
            f"test={self.n_test_engines} engines, "
            f"conditions={self.n_conditions}, "
            f"fault_modes={self.n_fault_modes})"
        )


class CMAPSSDataLoader:
    """Loader for NASA C-MAPSS turbofan engine degradation dataset.
    
    The C-MAPSS dataset consists of 4 sub-datasets with different
    operating conditions and fault modes:
    
    - FD001: 1 condition, 1 fault mode (HPC degradation)
    - FD002: 6 conditions, 1 fault mode (HPC degradation)  
    - FD003: 1 condition, 2 fault modes (HPC + Fan degradation)
    - FD004: 6 conditions, 2 fault modes (HPC + Fan degradation)
    
    Each sub-dataset contains:
    - train_FDXXX.txt: Complete run-to-failure trajectories
    - test_FDXXX.txt: Truncated trajectories (ends before failure)
    - RUL_FDXXX.txt: True RUL at end of each test trajectory
    """
    
    # Column names based on NASA documentation
    COLUMN_NAMES = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    
    # Dataset metadata
    DATASET_INFO = {
        "FD001": {"n_conditions": 1, "n_fault_modes": 1},
        "FD002": {"n_conditions": 6, "n_fault_modes": 1},
        "FD003": {"n_conditions": 1, "n_fault_modes": 2},
        "FD004": {"n_conditions": 6, "n_fault_modes": 2},
    }
    
    def __init__(self, raw_data_dir: str | Path, config_path: Optional[str | Path] = None):
        """Initialize the data loader.
        
        Args:
            raw_data_dir: Path to directory containing raw C-MAPSS text files
            config_path: Optional path to config.yaml for additional settings
        """
        self.raw_dir = Path(raw_data_dir)
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {self.raw_dir}")
        
        self.config = self._load_config(config_path) if config_path else {}
    
    def _load_config(self, config_path: str | Path) -> dict:
        """Load configuration from YAML file."""
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def _load_txt_file(self, file_path: Path) -> pd.DataFrame:
        """Load a space-separated C-MAPSS text file.
        
        The raw files have 26 space-separated columns with no header.
        """
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python"
        )
        
        df = df.dropna(axis=1, how="all")
        
        if df.shape[1] != len(self.COLUMN_NAMES):
            raise ValueError(
                f"Expected {len(self.COLUMN_NAMES)} columns in {file_path.name}, "
                f"got {df.shape[1]}"
            )
        
        df.columns = self.COLUMN_NAMES
        
        # Ensure proper dtypes
        df["engine_id"] = df["engine_id"].astype(np.int32)
        df["cycle"] = df["cycle"].astype(np.int32)
        
        # Sort by engine and cycle for consistency
        df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        
        return df
    
    def _load_rul_file(self, file_path: Path) -> pd.Series:
        """Load RUL ground truth file.
        
        The RUL file contains one integer per line, representing the
        remaining cycles until failure for each engine in the test set.
        Engine IDs are implicit (1-indexed, sequential).
        """
        rul = pd.read_csv(file_path, header=None, names=["RUL"])
        return rul["RUL"].astype(np.int32)
    
    def load_dataset(self, dataset_name: str) -> CMAPSSDataset:
        """Load a single C-MAPSS sub-dataset.
        
        Args:
            dataset_name: One of "FD001", "FD002", "FD003", "FD004"
            
        Returns:
            CMAPSSDataset containing train, test, and RUL data
        """
        if dataset_name not in self.DATASET_INFO:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Must be one of {list(self.DATASET_INFO.keys())}"
            )
        
        train_path = self.raw_dir / f"train_{dataset_name}.txt"
        test_path = self.raw_dir / f"test_{dataset_name}.txt"
        rul_path = self.raw_dir / f"RUL_{dataset_name}.txt"
        
        for path in [train_path, test_path, rul_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")
        
        train = self._load_txt_file(train_path)
        test = self._load_txt_file(test_path)
        rul = self._load_rul_file(rul_path)
        
        # Validate RUL length matches number of test engines
        n_test_engines = test["engine_id"].nunique()
        if len(rul) != n_test_engines:
            raise ValueError(
                f"RUL file has {len(rul)} entries but test data has "
                f"{n_test_engines} engines"
            )
        
        info = self.DATASET_INFO[dataset_name]
        
        return CMAPSSDataset(
            name=dataset_name,
            train=train,
            test=test,
            rul=rul,
            n_conditions=info["n_conditions"],
            n_fault_modes=info["n_fault_modes"],
        )
    
    def load_all_datasets(self) -> dict[str, CMAPSSDataset]:
        """Load all four C-MAPSS sub-datasets.
        
        Returns:
            Dictionary mapping dataset names to CMAPSSDataset objects
        """
        datasets = {}
        for name in self.DATASET_INFO.keys():
            try:
                datasets[name] = self.load_dataset(name)
            except FileNotFoundError as e:
                print(f"Warning: Could not load {name}: {e}")
        return datasets
    
    def get_combined_train(self, dataset_names: Optional[list[str]] = None) -> pd.DataFrame:
        """Combine training data from multiple sub-datasets.
        
        Useful for training on all available data. Engine IDs are made
        unique by prefixing with the dataset name.
        
        Args:
            dataset_names: List of datasets to combine. If None, uses all.
            
        Returns:
            Combined DataFrame with unique engine IDs
        """
        if dataset_names is None:
            dataset_names = list(self.DATASET_INFO.keys())
        
        dfs = []
        for name in dataset_names:
            ds = self.load_dataset(name)
            df = ds.train.copy()
            # Create unique engine ID by combining dataset and original ID
            df["dataset"] = name
            df["engine_id"] = df["dataset"] + "_" + df["engine_id"].astype(str)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)


def compute_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RUL labels for training data (each engine runs until failure).
    
    The RUL at each cycle is computed as: max_cycle - current_cycle.
    
    Args:
        df: Training DataFrame with engine_id and cycle columns
        
    Returns:
        DataFrame with added RUL column
    """
    df = df.copy()
    
    # Find maximum cycle (failure point) for each engine
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    
    # RUL = cycles remaining until failure
    df["RUL"] = max_cycles - df["cycle"]
    
    return df


def compute_test_rul(df: pd.DataFrame, rul_values: pd.Series) -> pd.DataFrame:
    """Compute RUL labels for test data.
    
    In test data, trajectories are truncated before failure. The ground
    truth RUL at the last cycle is provided. RUL at earlier cycles is
    computed by adding the remaining cycles to reach the end of trajectory.
    
    Args:
        df: Test DataFrame with engine_id and cycle columns
        rul_values: Series of RUL values at end of each trajectory
        
    Returns:
        DataFrame with added RUL column
    """
    df = df.copy()
    
    # Get sorted unique engine IDs
    engine_ids = sorted(df["engine_id"].unique())
    
    if len(engine_ids) != len(rul_values):
        raise ValueError(
            f"Number of engines ({len(engine_ids)}) doesn't match "
            f"number of RUL values ({len(rul_values)})"
        )
    
    # Map engine ID to ground truth RUL at end of trajectory
    rul_mapping = dict(zip(engine_ids, rul_values))
    
    # Get max cycle (last observation) for each engine
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    
    # RUL = (remaining cycles in trajectory) + (RUL at trajectory end)
    df["RUL_end"] = df["engine_id"].map(rul_mapping)
    df["RUL"] = (max_cycles - df["cycle"]) + df["RUL_end"]
    df = df.drop(columns=["RUL_end"])
    
    return df