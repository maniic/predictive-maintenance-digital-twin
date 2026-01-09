"""
Ensemble Model for RUL Prediction

Combines multiple models (LSTM, CNN, Transformer) with weighted averaging
and provides uncertainty quantification via prediction variance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseRULModel


@dataclass
class EnsemblePrediction:
    """Container for ensemble prediction results.

    Attributes:
        mean: Weighted mean prediction across models
        std: Standard deviation of predictions (uncertainty)
        individual: Dictionary of individual model predictions
        weights: Weights used for each model
    """
    mean: Tensor
    std: Tensor
    individual: dict[str, Tensor]
    weights: dict[str, float]


class EnsembleModel(nn.Module):
    """Ensemble of RUL prediction models.

    Combines predictions from multiple models using weighted averaging.
    Provides uncertainty quantification through prediction variance.

    Two modes of operation:
    1. Fixed weights: Manually specified weights for each model
    2. Learnable weights: Weights optimized on validation data

    Example usage:
        ```python
        # Load pre-trained models
        lstm = LSTMModel.load_from_checkpoint("lstm.ckpt")
        cnn = TemporalCNNModel.load_from_checkpoint("cnn.ckpt")
        transformer = TransformerModel.load_from_checkpoint("transformer.ckpt")

        # Create ensemble with equal weights
        ensemble = EnsembleModel({
            "lstm": lstm,
            "cnn": cnn,
            "transformer": transformer,
        })

        # Or with custom weights
        ensemble = EnsembleModel(
            models={"lstm": lstm, "cnn": cnn, "transformer": transformer},
            weights={"lstm": 0.4, "cnn": 0.3, "transformer": 0.3},
        )

        # Get predictions with uncertainty
        result = ensemble.predict_with_uncertainty(x)
        print(f"RUL: {result.mean} +/- {result.std}")
        ```
    """

    def __init__(
        self,
        models: dict[str, BaseRULModel],
        weights: Optional[dict[str, float]] = None,
        learnable_weights: bool = False,
    ):
        """Initialize ensemble model.

        Args:
            models: Dictionary mapping model names to model instances
            weights: Optional dictionary of weights per model. If None, uses equal weights.
                    Weights will be normalized to sum to 1.
            learnable_weights: If True, weights are learnable parameters optimized during training
        """
        super().__init__()

        if not models:
            raise ValueError("At least one model is required")

        self.model_names = list(models.keys())

        # Store models as ModuleDict for proper parameter registration
        self.models = nn.ModuleDict(models)

        # Freeze all model parameters (ensemble uses pre-trained models)
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False

        # Initialize weights
        n_models = len(models)
        if weights is None:
            # Equal weights by default
            weight_values = [1.0 / n_models] * n_models
        else:
            # Normalize provided weights
            total = sum(weights.get(name, 1.0) for name in self.model_names)
            weight_values = [weights.get(name, 1.0) / total for name in self.model_names]

        if learnable_weights:
            # Learnable weights (in log space for unconstrained optimization)
            self._log_weights = nn.Parameter(
                torch.log(torch.tensor(weight_values, dtype=torch.float32))
            )
        else:
            # Fixed weights
            self.register_buffer(
                "_fixed_weights",
                torch.tensor(weight_values, dtype=torch.float32)
            )

        self.learnable_weights = learnable_weights

    @property
    def weights(self) -> Tensor:
        """Get normalized weights for each model.

        Returns:
            Tensor of weights that sum to 1
        """
        if self.learnable_weights:
            # Softmax over log weights ensures positive weights that sum to 1
            return torch.softmax(self._log_weights, dim=0)
        else:
            return self._fixed_weights

    @property
    def weight_dict(self) -> dict[str, float]:
        """Get weights as a dictionary.

        Returns:
            Dictionary mapping model names to their weights
        """
        weights = self.weights.detach().cpu().tolist()
        return dict(zip(self.model_names, weights))

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning weighted mean prediction.

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Weighted mean RUL prediction (batch_size,)
        """
        predictions = self._get_individual_predictions(x)
        weights = self.weights.cpu()  # Ensure weights are on CPU to match predictions

        # Stack predictions: (n_models, batch_size)
        stacked = torch.stack([predictions[name] for name in self.model_names], dim=0)

        # Weighted average: (batch_size,)
        weighted_mean = torch.sum(stacked * weights.unsqueeze(1), dim=0)

        return weighted_mean

    def _get_individual_predictions(self, x: Tensor) -> dict[str, Tensor]:
        """Get predictions from each individual model.

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            Dictionary mapping model names to their predictions
        """
        predictions = {}
        for name, model in self.models.items():
            # Move data to model's device
            device = next(model.parameters()).device
            x_device = x.to(device)
            with torch.no_grad():
                predictions[name] = model(x_device).cpu()
        return predictions

    def predict_with_uncertainty(self, x: Tensor) -> EnsemblePrediction:
        """Predict RUL with uncertainty quantification.

        Uncertainty is computed as the weighted standard deviation of
        predictions across models, which captures model disagreement.

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            EnsemblePrediction containing mean, std, individual predictions, and weights
        """
        self.eval()

        predictions = self._get_individual_predictions(x)
        weights = self.weights.cpu()  # Ensure weights are on CPU to match predictions
        weight_dict = self.weight_dict

        # Stack predictions: (n_models, batch_size)
        stacked = torch.stack([predictions[name] for name in self.model_names], dim=0)

        # Weighted mean: (batch_size,)
        weighted_mean = torch.sum(stacked * weights.unsqueeze(1), dim=0)

        # Weighted variance: E[X^2] - E[X]^2
        weighted_sq_mean = torch.sum(stacked ** 2 * weights.unsqueeze(1), dim=0)
        variance = weighted_sq_mean - weighted_mean ** 2

        # Handle numerical issues (variance should be non-negative)
        variance = torch.clamp(variance, min=0)
        std = torch.sqrt(variance)

        return EnsemblePrediction(
            mean=weighted_mean,
            std=std,
            individual=predictions,
            weights=weight_dict,
        )

    def predict_rul(self, x: Tensor) -> Tensor:
        """Predict RUL (convenience method matching BaseRULModel interface).

        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)

        Returns:
            RUL predictions (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def calibrate_weights(
        self,
        val_loader: "torch.utils.data.DataLoader",
        method: str = "inverse_rmse",
    ) -> dict[str, float]:
        """Calibrate model weights based on validation performance.

        Args:
            val_loader: Validation data loader
            method: Calibration method:
                - "inverse_rmse": Weight inversely proportional to RMSE
                - "inverse_mse": Weight inversely proportional to MSE
                - "softmax_rmse": Softmax over negative RMSE

        Returns:
            Dictionary of calibrated weights
        """
        self.eval()
        device = next(self.parameters()).device

        # Compute per-model errors
        model_errors = {name: [] for name in self.model_names}

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                for name, model in self.models.items():
                    pred = model(x)
                    errors = (pred - y) ** 2
                    model_errors[name].append(errors)

        # Compute MSE for each model
        model_mse = {}
        for name in self.model_names:
            all_errors = torch.cat(model_errors[name])
            model_mse[name] = all_errors.mean().item()

        # Compute weights based on method
        if method == "inverse_rmse":
            rmse = {name: mse ** 0.5 for name, mse in model_mse.items()}
            inv_rmse = {name: 1.0 / r for name, r in rmse.items()}
            total = sum(inv_rmse.values())
            weights = {name: v / total for name, v in inv_rmse.items()}

        elif method == "inverse_mse":
            inv_mse = {name: 1.0 / mse for name, mse in model_mse.items()}
            total = sum(inv_mse.values())
            weights = {name: v / total for name, v in inv_mse.items()}

        elif method == "softmax_rmse":
            import math
            rmse = {name: mse ** 0.5 for name, mse in model_mse.items()}
            exp_neg_rmse = {name: math.exp(-r) for name, r in rmse.items()}
            total = sum(exp_neg_rmse.values())
            weights = {name: v / total for name, v in exp_neg_rmse.items()}

        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Update weights
        self._update_weights(weights)

        return weights

    def _update_weights(self, weights: dict[str, float]) -> None:
        """Update model weights.

        Args:
            weights: Dictionary mapping model names to weights
        """
        weight_values = [weights[name] for name in self.model_names]

        if self.learnable_weights:
            # Update log weights
            with torch.no_grad():
                self._log_weights.copy_(
                    torch.log(torch.tensor(weight_values, dtype=torch.float32))
                )
        else:
            # Update fixed weights
            self._fixed_weights = torch.tensor(
                weight_values, dtype=torch.float32, device=self._fixed_weights.device
            )

    def get_model(self, name: str) -> BaseRULModel:
        """Get a specific model by name.

        Args:
            name: Model name

        Returns:
            The requested model
        """
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found. Available: {self.model_names}")
        return self.models[name]

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: dict[str, Union[str, Path]],
        model_classes: dict[str, type],
        weights: Optional[dict[str, float]] = None,
        learnable_weights: bool = False,
        map_location: Optional[str] = None,
    ) -> "EnsembleModel":
        """Create ensemble from checkpoint files.

        Args:
            checkpoint_paths: Dictionary mapping model names to checkpoint paths
            model_classes: Dictionary mapping model names to model classes
            weights: Optional weights for each model
            learnable_weights: Whether weights should be learnable
            map_location: Device to load models onto

        Returns:
            EnsembleModel instance with loaded models

        Example:
            ```python
            ensemble = EnsembleModel.from_checkpoints(
                checkpoint_paths={
                    "lstm": "checkpoints/lstm_best.ckpt",
                    "cnn": "checkpoints/cnn_best.ckpt",
                    "transformer": "checkpoints/transformer_best.ckpt",
                },
                model_classes={
                    "lstm": LSTMModel,
                    "cnn": TemporalCNNModel,
                    "transformer": TransformerModel,
                },
            )
            ```
        """
        models = {}
        for name, path in checkpoint_paths.items():
            if name not in model_classes:
                raise ValueError(f"No model class provided for '{name}'")

            model_cls = model_classes[name]
            models[name] = model_cls.load_from_checkpoint(
                str(path),
                map_location=map_location,
            )

        return cls(
            models=models,
            weights=weights,
            learnable_weights=learnable_weights,
        )

    def __repr__(self) -> str:
        """String representation of ensemble."""
        weight_str = ", ".join(
            f"{name}={w:.3f}" for name, w in self.weight_dict.items()
        )
        return f"EnsembleModel(models=[{', '.join(self.model_names)}], weights=[{weight_str}])"
