"""
Feature Engineering Module for C-MAPSS Dataset

This module creates derived features from raw sensor data to improve
model performance. Features include:
- Rolling window statistics (mean, std, min, max, trend)
- Exponential moving averages
- Health indicators derived from degradation patterns
- Lag features for temporal context

These features capture the degradation trajectory of engine components
and help models learn patterns leading to failure.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
import yaml


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    
    # Window sizes for rolling statistics
    window_sizes: list[int] = field(default_factory=lambda: [5, 10, 20, 30])
    
    # Statistics to compute over rolling windows
    rolling_stats: list[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "trend"])
    
    # Spans for exponential moving averages
    ema_spans: list[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Number of lag features to create
    n_lags: int = 0
    
    # Whether to include interaction features
    include_interactions: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "FeatureConfig":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        feat_config = config.get("features", {})
        return cls(
            window_sizes=feat_config.get("window_sizes", [5, 10, 20, 30]),
            rolling_stats=feat_config.get("rolling_stats", ["mean", "std", "min", "max", "trend"]),
            ema_spans=feat_config.get("ema_spans", [5, 10, 20]),
            n_lags=feat_config.get("n_lags", 0),
            include_interactions=feat_config.get("include_interactions", False),
        )
    
    @classmethod
    def default(cls) -> "FeatureConfig":
        """Create default configuration."""
        return cls()
    
    @classmethod
    def minimal(cls) -> "FeatureConfig":
        """Minimal feature set for quick experiments."""
        return cls(
            window_sizes=[10, 20],
            rolling_stats=["mean", "std"],
            ema_spans=[10],
            n_lags=0,
            include_interactions=False,
        )


class FeatureEngineer:
    """Feature engineering for C-MAPSS time series data.
    
    Creates derived features from sensor readings to capture degradation
    patterns. All features are computed per-engine to avoid data leakage.
    
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the feature engineer.
        
        Args:
            config: Feature configuration. Uses defaults if None.
        """
        self.config = config or FeatureConfig.default()
    
    def _get_sensor_columns(self, df: pd.DataFrame) -> list[str]:
        """Identify sensor columns in the DataFrame."""
        return [c for c in df.columns if c.startswith("sensor_")]
    
    def _compute_rolling_stats(
        self, 
        group: pd.DataFrame, 
        sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Compute rolling window statistics for a single engine.
        
        Args:
            group: DataFrame for a single engine (sorted by cycle)
            sensor_cols: List of sensor column names
            
        Returns:
            DataFrame with added rolling feature columns
        """
        result = group.copy()
        
        for window in self.config.window_sizes:
            for col in sensor_cols:
                prefix = f"{col}_w{window}"
                
                # Rolling window (min_periods=1 to handle start of sequence)
                rolling = group[col].rolling(window=window, min_periods=1)
                
                if "mean" in self.config.rolling_stats:
                    result[f"{prefix}_mean"] = rolling.mean()
                
                if "std" in self.config.rolling_stats:
                    result[f"{prefix}_std"] = rolling.std().fillna(0)
                
                if "min" in self.config.rolling_stats:
                    result[f"{prefix}_min"] = rolling.min()
                
                if "max" in self.config.rolling_stats:
                    result[f"{prefix}_max"] = rolling.max()
                
                if "trend" in self.config.rolling_stats:
                    # Linear regression slope over the window
                    result[f"{prefix}_trend"] = self._compute_trend(group[col], window)
        
        return result
    
    def _compute_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling linear regression slope (trend).
        
        The slope indicates whether sensor values are increasing or
        decreasing, which is crucial for degradation detection.
        """
        def slope(x):
            if len(x) < 2:
                return 0
            # Use simple linear regression: slope = cov(x,y) / var(x)
            n = len(x)
            indices = np.arange(n)
            x_arr = np.asarray(x)
            
            # Handle constant arrays
            if np.std(x_arr) == 0:
                return 0
            
            slope, _, _, _, _ = stats.linregress(indices, x_arr)
            return slope
        
        return series.rolling(window=window, min_periods=2).apply(slope, raw=False).fillna(0)
    
    def _compute_ema(
        self, 
        group: pd.DataFrame, 
        sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Compute exponential moving averages for a single engine."""
        result = group.copy()
        
        for span in self.config.ema_spans:
            for col in sensor_cols:
                result[f"{col}_ema{span}"] = group[col].ewm(span=span, min_periods=1).mean()
        
        return result
    
    def _compute_lags(
        self, 
        group: pd.DataFrame, 
        sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Compute lag features for a single engine."""
        result = group.copy()
        
        for lag in range(1, self.config.n_lags + 1):
            for col in sensor_cols:
                result[f"{col}_lag{lag}"] = group[col].shift(lag)
        
        return result
    
    def _compute_health_indicators(
        self, 
        group: pd.DataFrame, 
        sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Compute health indicator features for a single engine.
        
        Health indicators are derived metrics that capture overall
        engine health based on multiple sensor deviations from
        baseline (early cycle) values.
        """
        result = group.copy()
        
        # Use first few cycles as baseline (healthy state)
        baseline_cycles = 10
        
        for col in sensor_cols:
            baseline = group[col].iloc[:baseline_cycles].mean()
            baseline_std = group[col].iloc[:baseline_cycles].std()
            
            if baseline_std > 0:
                # Z-score deviation from baseline
                result[f"{col}_deviation"] = (group[col] - baseline) / baseline_std
            else:
                result[f"{col}_deviation"] = 0
        
        # Aggregate health indicator (average absolute deviation across sensors)
        deviation_cols = [f"{col}_deviation" for col in sensor_cols]
        result["health_indicator"] = result[deviation_cols].abs().mean(axis=1)
        
        # Cumulative degradation (integral of deviations)
        result["cumulative_degradation"] = result["health_indicator"].cumsum()
        
        return result
    
    def _compute_interactions(
        self, 
        group: pd.DataFrame, 
        sensor_cols: list[str]
    ) -> pd.DataFrame:
        """Compute interaction features between key sensors.
        
        Some sensor combinations provide additional predictive signal.
        """
        result = group.copy()
        
        # Only compute for a subset of important sensors to limit feature explosion
        important_sensors = [c for c in sensor_cols if c in [
            "sensor_2", "sensor_3", "sensor_4", "sensor_7", 
            "sensor_8", "sensor_9", "sensor_11", "sensor_12",
            "sensor_13", "sensor_14", "sensor_15"
        ]]
        
        # Pairwise ratios for physically meaningful combinations
        if "sensor_8" in group.columns and "sensor_9" in group.columns:
            # Fan speed / Core speed ratio
            result["speed_ratio"] = group["sensor_8"] / group["sensor_9"].replace(0, np.nan)
        
        if "sensor_11" in group.columns and "sensor_7" in group.columns:
            # Pressure ratio
            result["pressure_ratio"] = group["sensor_11"] / group["sensor_7"].replace(0, np.nan)
        
        if "sensor_3" in group.columns and "sensor_4" in group.columns:
            # Temperature difference (HPC outlet - LPT outlet)
            result["temp_diff_34"] = group["sensor_3"] - group["sensor_4"]
        
        return result
    
    def create_features(self, df: pd.DataFrame, include_health: bool = True) -> pd.DataFrame:
        """Create all engineered features.
        
        Features are computed per-engine to prevent data leakage.
        
        Args:
            df: Input DataFrame with engine_id, cycle, and sensor columns
            include_health: Whether to include health indicator features
            
        Returns:
            DataFrame with all original and engineered features
        """
        sensor_cols = self._get_sensor_columns(df)
        
        if not sensor_cols:
            raise ValueError("No sensor columns found in DataFrame")
        
        # Process each engine separately to avoid leakage
        results = []
        
        for engine_id, group in df.groupby("engine_id"):
            # Sort by cycle
            group = group.sort_values("cycle").reset_index(drop=True)
            
            # Add rolling statistics
            group = self._compute_rolling_stats(group, sensor_cols)
            
            # Add EMAs
            group = self._compute_ema(group, sensor_cols)
            
            # Add lag features
            if self.config.n_lags > 0:
                group = self._compute_lags(group, sensor_cols)
            
            # Add health indicators
            if include_health:
                group = self._compute_health_indicators(group, sensor_cols)
            
            # Add interaction features
            if self.config.include_interactions:
                group = self._compute_interactions(group, sensor_cols)
            
            results.append(group)
        
        result_df = pd.concat(results, ignore_index=True)
        
        # Fill any remaining NaN from lag/rolling operations
        result_df = result_df.bfill().fillna(0)
        
        return result_df
    
    def get_feature_names(self, base_sensor_cols: list[str]) -> list[str]:
        """Get list of all feature names that will be created.
        
        Useful for understanding feature dimensionality before creating features.
        """
        features = list(base_sensor_cols)
        
        # Rolling features
        for window in self.config.window_sizes:
            for col in base_sensor_cols:
                prefix = f"{col}_w{window}"
                for stat in self.config.rolling_stats:
                    features.append(f"{prefix}_{stat}")
        
        # EMA features
        for span in self.config.ema_spans:
            for col in base_sensor_cols:
                features.append(f"{col}_ema{span}")
        
        # Lag features
        for lag in range(1, self.config.n_lags + 1):
            for col in base_sensor_cols:
                features.append(f"{col}_lag{lag}")
        
        # Health indicators
        for col in base_sensor_cols:
            features.append(f"{col}_deviation")
        features.extend(["health_indicator", "cumulative_degradation"])
        
        # Interactions (if enabled)
        if self.config.include_interactions:
            features.extend(["speed_ratio", "pressure_ratio", "temp_diff_34"])
        
        return features


def create_full_feature_pipeline(
    df: pd.DataFrame,
    feature_config: Optional[FeatureConfig] = None,
) -> pd.DataFrame:
    """Convenience function to run full feature engineering pipeline.
    
    Args:
        df: Input DataFrame (should already have RUL column if training data)
        feature_config: Feature configuration. Uses defaults if None.
        
    Returns:
        DataFrame with all engineered features
    """
    engineer = FeatureEngineer(feature_config)
    return engineer.create_features(df)