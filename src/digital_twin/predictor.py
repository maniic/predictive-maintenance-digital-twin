"""
RUL Predictor for Digital Twin

Loads trained models and preprocessors, provides predictions with uncertainty.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import CMAPSSPreprocessor
from src.models import (
    LSTMModel,
    TemporalCNNModel,
    TransformerModel,
    EnsembleModel,
)


@dataclass
class PredictionResult:
    """Result of a RUL prediction.

    Attributes:
        rul: Predicted remaining useful life (cycles)
        uncertainty: Standard deviation of prediction
        individual_predictions: Predictions from each model
        weights: Ensemble weights used
        health_score: Derived health score (0-1)
    """
    rul: float
    uncertainty: float
    individual_predictions: dict[str, float]
    weights: dict[str, float]
    health_score: float

    @property
    def rul_lower(self) -> float:
        """Lower bound of 95% CI."""
        return max(0, self.rul - 1.96 * self.uncertainty)

    @property
    def rul_upper(self) -> float:
        """Upper bound of 95% CI."""
        return self.rul + 1.96 * self.uncertainty


class RULPredictor:
    """Predictor for Remaining Useful Life.

    Loads trained models and preprocessor for a specific dataset,
    provides predictions with uncertainty quantification.

    Example:
        predictor = RULPredictor(dataset="FD001")
        predictor.load_models()

        # Predict from sequence
        result = predictor.predict(sequence)
        print(f"RUL: {result.rul:.1f} +/- {result.uncertainty:.1f}")
    """

    # Model class mapping
    MODEL_CLASSES = {
        "lstm": LSTMModel,
        "cnn": TemporalCNNModel,
        "transformer": TransformerModel,
    }

    def __init__(
        self,
        dataset: str = "FD001",
        models_dir: str = "models",
        device: Optional[str] = None,
    ):
        """Initialize predictor.

        Args:
            dataset: C-MAPSS dataset name (FD001, FD002, FD003, FD004)
            models_dir: Base directory for models and preprocessors
            device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.dataset = dataset
        self.models_dir = Path(models_dir)
        self.device = device or self._detect_device()

        self.preprocessor: Optional[CMAPSSPreprocessor] = None
        self.models: dict[str, torch.nn.Module] = {}
        self.ensemble: Optional[EnsembleModel] = None
        self._loaded = False

    def _detect_device(self) -> str:
        """Detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_models(self) -> None:
        """Load preprocessor and all models for the dataset."""
        # Load preprocessor
        preprocessor_path = self.models_dir / "preprocessors" / f"{self.dataset}_preprocessor.pkl"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        self.preprocessor = CMAPSSPreprocessor.load(str(preprocessor_path))

        # Load individual models
        checkpoint_dir = self.models_dir / "checkpoints" / self.dataset
        for model_name, model_class in self.MODEL_CLASSES.items():
            model_dir = checkpoint_dir / model_name
            if model_dir.exists():
                # Find best checkpoint
                checkpoints = list(model_dir.glob("*.ckpt"))
                if checkpoints:
                    # Use most recent checkpoint
                    checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                    try:
                        model = model_class.load_from_checkpoint(
                            str(checkpoint),
                            map_location=self.device,
                        )
                        model.eval()
                        self.models[model_name] = model
                    except Exception as e:
                        print(f"Warning: Could not load {model_name}: {e}")

        # Create ensemble from loaded models
        if self.models:
            self.ensemble = EnsembleModel(
                models=self.models,
                learnable_weights=False,
            )

        self._loaded = True

    @property
    def feature_columns(self) -> list[str]:
        """Get feature column names from preprocessor (includes regime if enabled)."""
        if self.preprocessor is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self.preprocessor.get_feature_names()

    @property
    def sequence_length(self) -> int:
        """Get expected sequence length."""
        return 30  # From config

    @property
    def input_dim(self) -> int:
        """Get input dimension (number of features)."""
        return len(self.feature_columns)

    def predict(self, sequence: np.ndarray) -> PredictionResult:
        """Make RUL prediction from preprocessed sequence.

        Args:
            sequence: Preprocessed sequence of shape (sequence_length, input_dim)
                     or (batch, sequence_length, input_dim)

        Returns:
            PredictionResult with RUL, uncertainty, and individual predictions
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Ensure 3D tensor (batch, seq_len, features)
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :, :]

        # Convert to tensor
        x = torch.tensor(sequence, dtype=torch.float32)

        # Get ensemble prediction with uncertainty
        with torch.no_grad():
            result = self.ensemble.predict_with_uncertainty(x)

        # Extract values
        rul = float(result.mean.item())
        uncertainty = float(result.std.item())
        individual = {k: float(v.item()) for k, v in result.individual.items()}
        weights = result.weights

        # Compute health score (0-1 based on RUL)
        # RUL capped at 125, so health = min(1, rul/125)
        health_score = min(1.0, max(0.0, rul / 125.0))

        return PredictionResult(
            rul=rul,
            uncertainty=uncertainty,
            individual_predictions=individual,
            weights=weights,
            health_score=health_score,
        )

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        engine_id: Optional[int] = None,
    ) -> PredictionResult:
        """Make prediction from raw sensor DataFrame.

        Args:
            df: DataFrame with sensor readings (requires last sequence_length rows)
            engine_id: Optional engine ID to filter by

        Returns:
            PredictionResult for the latest timestep
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Filter by engine if specified
        if engine_id is not None and "unit_id" in df.columns:
            df = df[df["unit_id"] == engine_id]

        # Get last sequence_length rows
        if len(df) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} rows, got {len(df)}"
            )
        df_seq = df.tail(self.sequence_length)

        # Apply preprocessing
        df_processed = self.preprocessor.transform(df_seq)

        # Extract feature columns
        features = df_processed[self.feature_columns].values

        return self.predict(features)

    def predict_from_readings(
        self,
        readings_history: list[dict[str, float]],
    ) -> PredictionResult:
        """Make prediction from list of sensor readings.

        Args:
            readings_history: List of sensor reading dicts (last sequence_length)

        Returns:
            PredictionResult
        """
        if len(readings_history) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} readings, got {len(readings_history)}"
            )

        # Convert to DataFrame
        df = pd.DataFrame(readings_history[-self.sequence_length:])

        # Add required columns if missing
        if "unit_id" not in df.columns:
            df["unit_id"] = 1
        if "cycle" not in df.columns:
            df["cycle"] = range(1, len(df) + 1)

        return self.predict_from_dataframe(df)

    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "dataset": self.dataset,
            "device": self.device,
            "models_loaded": list(self.models.keys()),
            "input_dim": self.input_dim if self._loaded else None,
            "sequence_length": self.sequence_length,
            "ensemble_weights": self.ensemble.weight_dict if self.ensemble else None,
        }
