"""
Data Preprocessing Module for C-MAPSS Dataset

This module handles preprocessing of raw sensor data including:
- Smart sensor filtering (only truly zero-variance sensors)
- Operating regime identification and clustering
- Feature normalization (per-regime or global)
- RUL label transformation (piecewise linear capping)

The preprocessing follows a conservative approach. We only drop sensors
that have literally zero variance, as even "near-constant" sensors may
contain subtle degradation signals important for RUL prediction.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import yaml


@dataclass
class SensorAnalysis:
    """Results of sensor variance analysis."""
    name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    cv: float  # Coefficient of variation
    unique_values: int
    is_zero_variance: bool  # std == 0
    
    def __repr__(self) -> str:
        status = "ZERO-VAR" if self.is_zero_variance else "OK"
        return f"{self.name}: mean={self.mean:.2f}, std={self.std:.4f}, cv={self.cv:.6f} [{status}]"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Strategy for dropping sensors: "none", "auto", or "manual"
    drop_strategy: Literal["none", "auto", "manual"] = "auto"
    
    # Manual list of sensors to drop (only used if drop_strategy="manual")
    drop_sensors: list[str] = field(default_factory=list)
    
    # RUL capping value (piecewise linear assumption)
    rul_cap: int = 125
    
    # Normalization method
    normalization: str = "minmax"  # "minmax", "standard", or "robust"
    
    # Whether to cluster by operating regime
    cluster_by_regime: bool = True
    n_regimes: int = 6
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "PreprocessingConfig":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        prep_config = config.get("preprocessing", {})
        return cls(
            drop_strategy=prep_config.get("drop_strategy", "auto"),
            drop_sensors=prep_config.get("drop_sensors", []),
            rul_cap=prep_config.get("rul_cap", 125),
            normalization=prep_config.get("normalization", "minmax"),
            cluster_by_regime=prep_config.get("cluster_by_regime", True),
            n_regimes=prep_config.get("n_regimes", 6),
        )
    
    @classmethod
    def default(cls) -> "PreprocessingConfig":
        """Create default configuration - conservative approach keeping most sensors."""
        return cls(
            drop_strategy="auto",  # Only drop truly zero-variance sensors
            drop_sensors=[],
            rul_cap=125,
            normalization="minmax",
            cluster_by_regime=True,
            n_regimes=6,
        )
    
    @classmethod
    def keep_all_sensors(cls) -> "PreprocessingConfig":
        """Configuration that keeps all sensors - let the model decide importance."""
        return cls(
            drop_strategy="none",
            drop_sensors=[],
            rul_cap=125,
            normalization="minmax",
            cluster_by_regime=True,
            n_regimes=6,
        )


def analyze_sensors(df: pd.DataFrame) -> list[SensorAnalysis]:
    """Analyze all sensors in the DataFrame to identify their variance characteristics.
    
    This function examines each sensor to determine:
    - Basic statistics (mean, std, min, max)
    - Coefficient of variation (CV = std/mean)
    - Whether the sensor has exactly zero variance
    
    Args:
        df: DataFrame with sensor columns (sensor_1, sensor_2, etc.)
        
    Returns:
        List of SensorAnalysis objects with statistics for each sensor
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    
    analyses = []
    for col in sorted(sensor_cols):
        values = df[col]
        mean = values.mean()
        std = values.std()
        cv = std / abs(mean) if mean != 0 else 0
        
        analysis = SensorAnalysis(
            name=col,
            mean=mean,
            std=std,
            min_val=values.min(),
            max_val=values.max(),
            cv=cv,
            unique_values=values.nunique(),
            is_zero_variance=(std == 0) or (values.nunique() == 1),
        )
        analyses.append(analysis)
    
    return analyses


def find_zero_variance_sensors(df: pd.DataFrame) -> list[str]:
    """Find sensors with exactly zero variance.
    
    Only returns sensors that have literally no variation. Near-constant sensors are
    NOT included as they may contain subtle but important signals.
    
    Args:
        df: DataFrame with sensor columns
        
    Returns:
        List of sensor column names with zero variance
    """
    analyses = analyze_sensors(df)
    return [a.name for a in analyses if a.is_zero_variance]


class CMAPSSPreprocessor:
    """Preprocessor for C-MAPSS turbofan engine data.
    
    This class handles the full preprocessing pipeline:
    1. Optionally remove zero-variance sensors
    2. Identify operating regimes via clustering
    3. Normalize features (optionally per-regime)
    4. Apply piecewise linear RUL capping
    
    The preprocessor uses a conservative approach to sensor dropping:
    - "auto" mode only drops sensors with EXACTLY zero variance
    - "none" mode keeps all sensors, letting the model learn importance
    - "manual" mode uses a user-specified list
    
    Example:
        >>> preprocessor = CMAPSSPreprocessor(config)
        >>> train_processed = preprocessor.fit_transform(train_df)
        >>> test_processed = preprocessor.transform(test_df)
    """
    
    SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize the preprocessor.
        
        Args:
            config: Preprocessing configuration. Uses defaults if None.
        """
        self.config = config or PreprocessingConfig.default()
        
        # Fitted components (populated by fit())
        self.scalers_: dict[int, MinMaxScaler | StandardScaler | RobustScaler] = {}
        self.regime_model_: Optional[KMeans] = None
        self.feature_cols_: list[str] = []
        self.dropped_sensors_: list[str] = []  # Sensors actually dropped
        self.sensor_analysis_: list[SensorAnalysis] = []  # Analysis results
        self.is_fitted_: bool = False
    
    def _get_scaler(self):
        """Create a new scaler instance based on config."""
        if self.config.normalization == "minmax":
            return MinMaxScaler()
        elif self.config.normalization == "standard":
            return StandardScaler()
        elif self.config.normalization == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization: {self.config.normalization}")
    
    def _determine_sensors_to_drop(self, df: pd.DataFrame) -> list[str]:
        """Determine which sensors to drop based on strategy."""
        if self.config.drop_strategy == "none":
            return []
        elif self.config.drop_strategy == "manual":
            return self.config.drop_sensors.copy()
        elif self.config.drop_strategy == "auto":
            # Only drop truly zero-variance sensors
            return find_zero_variance_sensors(df)
        else:
            raise ValueError(f"Unknown drop_strategy: {self.config.drop_strategy}")
    
    def _identify_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Identify sensor and setting columns to use as features."""
        all_cols = df.columns.tolist()
        
        # Start with settings
        feature_cols = [c for c in self.SETTING_COLS if c in all_cols]
        
        # Add sensors (excluding dropped ones)
        sensor_cols = [c for c in all_cols if c.startswith("sensor_")]
        sensor_cols = [c for c in sensor_cols if c not in self.dropped_sensors_]
        
        feature_cols.extend(sensor_cols)
        return feature_cols
    
    def _cluster_regimes(self, df: pd.DataFrame) -> np.ndarray:
        """Cluster operating regimes based on operational settings.
        
        Operating conditions in C-MAPSS are defined by altitude, Mach number,
        and throttle resolver angle (settings 1-3).
        """
        settings = df[self.SETTING_COLS].values
        
        if self.regime_model_ is None:
            self.regime_model_ = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=42,
                n_init=10,
            )
            regimes = self.regime_model_.fit_predict(settings)
        else:
            regimes = self.regime_model_.predict(settings)
        
        return regimes
    
    def _normalize_features(
        self, 
        df: pd.DataFrame, 
        fit: bool = False
    ) -> pd.DataFrame:
        """Normalize features, optionally per operating regime.
        
        Args:
            df: DataFrame with features to normalize
            fit: Whether to fit the scalers (True for training data)
            
        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        
        for col in self.feature_cols_:
            if col in df.columns:
                df[col] = df[col].astype(np.float64)
        
        if self.config.cluster_by_regime and self.config.n_regimes > 1:
            # Normalize per operating regime
            regimes = self._cluster_regimes(df)
            df["regime"] = regimes
            
            for regime in range(self.config.n_regimes):
                mask = df["regime"] == regime
                if not mask.any():
                    continue
                
                if fit:
                    self.scalers_[regime] = self._get_scaler()
                    df.loc[mask, self.feature_cols_] = self.scalers_[regime].fit_transform(
                        df.loc[mask, self.feature_cols_]
                    )
                else:
                    if regime in self.scalers_:
                        df.loc[mask, self.feature_cols_] = self.scalers_[regime].transform(
                            df.loc[mask, self.feature_cols_]
                        )
            
            # Keep regime as a feature (useful for models)
        else:
            # Global normalization
            if fit:
                self.scalers_[0] = self._get_scaler()
                df[self.feature_cols_] = self.scalers_[0].fit_transform(df[self.feature_cols_])
            else:
                df[self.feature_cols_] = self.scalers_[0].transform(df[self.feature_cols_])
        
        return df
    
    def _cap_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply piecewise linear RUL capping.
        
        Early in an engine's life, the RUL can be very high (100s of cycles).
        Research shows that engines are essentially "healthy" during this period
        and the exact RUL doesn't matter much for prediction. Capping creates
        a piecewise linear target:
        
        - RUL = min(actual_rul, cap) 
        - This makes the problem more tractable and improves model performance
        
        Standard practice is to cap at 125 cycles based on NASA recommendations.
        """
        df = df.copy()
        if "RUL" in df.columns:
            df["RUL"] = df["RUL"].clip(upper=self.config.rul_cap)
        return df
    
    def fit(self, df: pd.DataFrame) -> "CMAPSSPreprocessor":
        """Fit the preprocessor on training data.
        
        This learns:
        - Which sensors to drop (based on strategy)
        - Which features to use
        - Operating regime clusters (if enabled)
        - Normalization parameters (per-regime or global)
        
        Args:
            df: Training DataFrame
            
        Returns:
            self (for method chaining)
        """
        # Analyze sensors
        self.sensor_analysis_ = analyze_sensors(df)
        
        # Determine which sensors to drop
        self.dropped_sensors_ = self._determine_sensors_to_drop(df)
        
        # Identify feature columns (after dropping)
        self.feature_cols_ = self._identify_feature_columns(df)
        
        # Fit normalization (includes regime clustering if enabled)
        _ = self._normalize_features(df, fit=True)
        
        self.is_fitted_ = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessor.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform()")
        
        df = df.copy()
        
        # Drop determined sensors
        cols_to_drop = [c for c in self.dropped_sensors_ if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Normalize features
        df = self._normalize_features(df, fit=False)
        
        # Cap RUL if present
        df = self._cap_rul(df)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform data in one step.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.fit(df)
        
        df = df.copy()
        
        # Drop determined sensors
        cols_to_drop = [c for c in self.dropped_sensors_ if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Normalize features (already fitted)
        df = self._normalize_features(df, fit=False)
        
        # Cap RUL if present
        df = self._cap_rul(df)
        
        return df
    
    def get_feature_names(self) -> list[str]:
        """Get list of feature column names after preprocessing."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted first")
        
        features = self.feature_cols_.copy()
        if self.config.cluster_by_regime:
            features.append("regime")
        return features
    
    def get_dropped_sensors(self) -> list[str]:
        """Get list of sensors that were dropped."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted first")
        return self.dropped_sensors_.copy()
    
    def get_sensor_analysis(self) -> list[SensorAnalysis]:
        """Get detailed analysis of all sensors."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted first")
        return self.sensor_analysis_.copy()
    
    def print_sensor_summary(self) -> None:
        """Print a summary of sensor analysis and dropping decisions."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted first")
        
        print(f"\n{'='*60}")
        print("SENSOR ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"Strategy: {self.config.drop_strategy}")
        print(f"Total sensors analyzed: {len(self.sensor_analysis_)}")
        print(f"Sensors dropped: {len(self.dropped_sensors_)}")
        print(f"Sensors kept: {len([a for a in self.sensor_analysis_ if a.name not in self.dropped_sensors_])}")
        
        if self.dropped_sensors_:
            print(f"\nDropped sensors: {self.dropped_sensors_}")
        
        print(f"\n{'Sensor':<12} {'Mean':>10} {'Std':>10} {'CV':>12} {'Status':<10}")
        print("-" * 60)
        
        for a in self.sensor_analysis_:
            status = "DROPPED" if a.name in self.dropped_sensors_ else "kept"
            if a.is_zero_variance:
                status += " (zero-var)"
            print(f"{a.name:<12} {a.mean:>10.2f} {a.std:>10.4f} {a.cv:>12.6f} {status:<10}")
    
    def save(self, path: str | Path) -> None:
        """Save fitted preprocessor to disk."""
        import pickle
        
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted preprocessor")
        
        state = {
            "config": self.config,
            "scalers": self.scalers_,
            "regime_model": self.regime_model_,
            "feature_cols": self.feature_cols_,
            "dropped_sensors": self.dropped_sensors_,
            "sensor_analysis": self.sensor_analysis_,
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "CMAPSSPreprocessor":
        """Load a fitted preprocessor from disk."""
        import pickle
        
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        preprocessor = cls(config=state["config"])
        preprocessor.scalers_ = state["scalers"]
        preprocessor.regime_model_ = state["regime_model"]
        preprocessor.feature_cols_ = state["feature_cols"]
        preprocessor.dropped_sensors_ = state.get("dropped_sensors", [])
        preprocessor.sensor_analysis_ = state.get("sensor_analysis", [])
        preprocessor.is_fitted_ = True
        
        return preprocessor