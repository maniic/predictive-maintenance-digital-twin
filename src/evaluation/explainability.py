"""
Explainability Module for RUL Prediction

Provides tools for understanding model predictions:
- SHAP-based feature importance
- Attention weight visualization
- Sensor contribution analysis
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from src.models.base import BaseRULModel


@dataclass
class FeatureImportance:
    """Container for feature importance results.

    Attributes:
        feature_names: Names of features
        importance_mean: Mean absolute SHAP value per feature
        importance_std: Standard deviation of SHAP values
        shap_values: Raw SHAP values matrix (samples x features)
    """
    feature_names: list[str]
    importance_mean: np.ndarray
    importance_std: np.ndarray
    shap_values: Optional[np.ndarray] = None

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features.

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        # Flatten if 2D (e.g., from non-aggregated time)
        importance = self.importance_mean.flatten() if self.importance_mean.ndim > 1 else self.importance_mean
        indices = np.argsort(importance)[::-1][:n]
        return [(self.feature_names[int(i)], float(importance[int(i)])) for i in indices]

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary mapping feature names to importance."""
        importance = self.importance_mean.flatten() if self.importance_mean.ndim > 1 else self.importance_mean
        return dict(zip(self.feature_names, importance.tolist()))


@dataclass
class AttentionAnalysis:
    """Container for attention weight analysis.

    Attributes:
        weights: Attention weights (batch x sequence_length)
        timestep_importance: Mean attention per timestep
        peak_timesteps: Indices of highest attention timesteps
    """
    weights: np.ndarray
    timestep_importance: np.ndarray
    peak_timesteps: np.ndarray


@dataclass
class SensorContribution:
    """Per-sensor contribution to RUL prediction.

    Attributes:
        sensor_names: Names of sensors
        contributions: Mean contribution per sensor (positive = increases RUL)
        contribution_std: Standard deviation of contributions
        temporal_contributions: Contributions over time (sensors x timesteps)
    """
    sensor_names: list[str]
    contributions: np.ndarray
    contribution_std: np.ndarray
    temporal_contributions: Optional[np.ndarray] = None


class RULExplainer:
    """Explainability toolkit for RUL prediction models.

    Provides SHAP-based feature importance, attention visualization,
    and sensor contribution analysis.

    Example:
        ```python
        explainer = RULExplainer(model, feature_names, background_data)

        # Get feature importance
        importance = explainer.feature_importance(test_data)
        print(importance.top_features(10))

        # Visualize attention
        explainer.plot_attention(sample_input)

        # Analyze sensor contributions
        contributions = explainer.sensor_contributions(test_data)
        explainer.plot_sensor_contributions(contributions)
        ```
    """

    def __init__(
        self,
        model: BaseRULModel,
        feature_names: list[str],
        background_data: Optional[Tensor] = None,
        device: Optional[str] = None,
    ):
        """Initialize explainer.

        Args:
            model: Trained RUL prediction model
            feature_names: Names of input features
            background_data: Background samples for SHAP (optional, recommended ~100 samples)
            device: Device to run computations on
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self._shap_explainer = None

    def _get_shap_explainer(self, background_data: Tensor):
        """Get or create SHAP explainer.

        Uses GradientExplainer for neural networks as it's more
        efficient than KernelExplainer for deep models.
        """
        import shap

        if self._shap_explainer is None or self.background_data is None:
            self.background_data = background_data.to(self.device).requires_grad_(True)

            # Wrap model to ensure output shape is (batch, 1) for SHAP
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    out = self.model(x)
                    if out.dim() == 1:
                        out = out.unsqueeze(-1)
                    return out

            wrapped_model = ModelWrapper(self.model)

            # Use DeepExplainer for PyTorch models
            self._shap_explainer = shap.DeepExplainer(
                wrapped_model,
                self.background_data,
            )

        return self._shap_explainer

    def feature_importance(
        self,
        data: Tensor,
        background_data: Optional[Tensor] = None,
        aggregate_time: bool = True,
    ) -> FeatureImportance:
        """Compute SHAP-based feature importance.

        Args:
            data: Input data to explain (batch x seq_len x features)
            background_data: Background samples for SHAP (uses stored if None)
            aggregate_time: If True, aggregate importance across timesteps

        Returns:
            FeatureImportance with per-feature importance scores
        """
        import shap

        # Ensure data requires grad for SHAP gradient computation
        data = data.to(self.device).requires_grad_(True)

        if background_data is None:
            if self.background_data is None:
                # Use subset of data as background
                n_background = min(100, len(data))
                background_data = data[:n_background].detach().requires_grad_(True)
            else:
                background_data = self.background_data
        else:
            background_data = background_data.to(self.device).requires_grad_(True)

        explainer = self._get_shap_explainer(background_data)

        # Compute SHAP values (requires gradients)
        shap_values = explainer.shap_values(data)

        # Convert to numpy
        if isinstance(shap_values, Tensor):
            shap_values = shap_values.cpu().numpy()
        elif isinstance(shap_values, list):
            shap_values = shap_values[0]
            if isinstance(shap_values, Tensor):
                shap_values = shap_values.cpu().numpy()

        # shap_values shape: (batch, seq_len, features)
        if aggregate_time:
            # Aggregate across time: mean of absolute values per feature
            importance = np.abs(shap_values).mean(axis=(0, 1))  # (features,)
            importance_std = np.abs(shap_values).std(axis=(0, 1))
            aggregated_shap = np.abs(shap_values).mean(axis=1)  # (batch, features)
        else:
            # Return per-timestep importance
            importance = np.abs(shap_values).mean(axis=0)  # (seq_len, features)
            importance_std = np.abs(shap_values).std(axis=0)
            aggregated_shap = shap_values

        return FeatureImportance(
            feature_names=self.feature_names,
            importance_mean=importance,
            importance_std=importance_std,
            shap_values=aggregated_shap,
        )

    def attention_analysis(self, data: Tensor) -> Optional[AttentionAnalysis]:
        """Analyze attention weights from the model.

        Only works with models that implement get_attention_weights().

        Args:
            data: Input data (batch x seq_len x features)

        Returns:
            AttentionAnalysis if model supports attention, None otherwise
        """
        data = data.to(self.device)

        with torch.no_grad():
            # Forward pass to populate attention weights
            _ = self.model(data)
            weights = self.model.get_attention_weights()

        if weights is None:
            return None

        weights = weights.cpu().numpy()

        # Compute statistics
        timestep_importance = weights.mean(axis=0)
        peak_timesteps = np.argsort(timestep_importance)[::-1][:5]

        return AttentionAnalysis(
            weights=weights,
            timestep_importance=timestep_importance,
            peak_timesteps=peak_timesteps,
        )

    def sensor_contributions(
        self,
        data: Tensor,
        sensor_names: Optional[list[str]] = None,
        background_data: Optional[Tensor] = None,
    ) -> SensorContribution:
        """Compute per-sensor contribution to predictions.

        Args:
            data: Input data (batch x seq_len x features)
            sensor_names: Names of sensors (uses feature_names if None)
            background_data: Background data for SHAP

        Returns:
            SensorContribution with per-sensor analysis
        """
        importance = self.feature_importance(
            data,
            background_data=background_data,
            aggregate_time=False,
        )

        if sensor_names is None:
            sensor_names = self.feature_names

        # Aggregate across time for overall sensor importance
        # shap_values shape: (batch, seq_len, features)
        if importance.shap_values is not None and len(importance.shap_values.shape) == 3:
            temporal = importance.shap_values.mean(axis=0)  # (seq_len, features)
            contributions = np.abs(temporal).mean(axis=0)  # (features,)
            contributions_std = np.abs(temporal).std(axis=0)
        else:
            temporal = None
            contributions = importance.importance_mean
            contributions_std = importance.importance_std

        return SensorContribution(
            sensor_names=sensor_names,
            contributions=contributions,
            contribution_std=contributions_std,
            temporal_contributions=temporal,
        )

    def plot_feature_importance(
        self,
        importance: FeatureImportance,
        top_n: int = 15,
        figsize: tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot feature importance bar chart.

        Args:
            importance: FeatureImportance from feature_importance()
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save figure (optional)

        Returns:
            Matplotlib figure
        """
        top = importance.top_features(top_n)
        names = [t[0] for t in top]
        values = [t[1] for t in top]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(names))

        ax.barh(y_pos, values, color="steelblue", edgecolor="navy", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Feature Importance for RUL Prediction")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_attention(
        self,
        data: Tensor,
        sample_idx: int = 0,
        figsize: tuple[int, int] = (12, 4),
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[plt.Figure]:
        """Plot attention weights over sequence.

        Args:
            data: Input data (batch x seq_len x features)
            sample_idx: Which sample to visualize
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure if model supports attention, None otherwise
        """
        analysis = self.attention_analysis(data)
        if analysis is None:
            print("Model does not provide attention weights")
            return None

        weights = analysis.weights[sample_idx]
        seq_len = len(weights)

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(range(seq_len), weights, color="coral", edgecolor="darkred", alpha=0.8)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Attention Weight")
        ax.set_title("Attention Weights Over Sequence")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Mark peak timesteps
        for idx in analysis.peak_timesteps[:3]:
            ax.axvline(x=idx, color="red", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_attention_heatmap(
        self,
        data: Tensor,
        n_samples: int = 20,
        figsize: tuple[int, int] = (14, 8),
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[plt.Figure]:
        """Plot attention heatmap across multiple samples.

        Args:
            data: Input data (batch x seq_len x features)
            n_samples: Number of samples to show
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure if model supports attention
        """
        analysis = self.attention_analysis(data[:n_samples])
        if analysis is None:
            return None

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(analysis.weights, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Sample")
        ax.set_title("Attention Weights Heatmap")

        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_sensor_contributions(
        self,
        contributions: SensorContribution,
        top_n: int = 15,
        figsize: tuple[int, int] = (10, 6),
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Plot sensor contribution bar chart.

        Args:
            contributions: SensorContribution from sensor_contributions()
            top_n: Number of top sensors to show
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure
        """
        # Sort by contribution
        indices = np.argsort(contributions.contributions)[::-1][:top_n]
        names = [contributions.sensor_names[i] for i in indices]
        values = contributions.contributions[indices]
        errors = contributions.contribution_std[indices]

        fig, ax = plt.subplots(figsize=figsize)
        y_pos = np.arange(len(names))

        ax.barh(
            y_pos, values,
            xerr=errors,
            color="teal",
            edgecolor="darkcyan",
            alpha=0.8,
            capsize=3,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Mean Contribution to RUL")
        ax.set_title("Sensor Contributions to RUL Prediction")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_temporal_importance(
        self,
        contributions: SensorContribution,
        top_sensors: int = 5,
        figsize: tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None,
    ) -> Optional[plt.Figure]:
        """Plot how sensor importance changes over time.

        Args:
            contributions: SensorContribution with temporal data
            top_sensors: Number of top sensors to show
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib figure if temporal data available
        """
        if contributions.temporal_contributions is None:
            print("No temporal contribution data available")
            return None

        temporal = contributions.temporal_contributions  # (seq_len, features)
        seq_len = temporal.shape[0]

        # Get top sensors by overall contribution
        top_indices = np.argsort(contributions.contributions)[::-1][:top_sensors]

        fig, ax = plt.subplots(figsize=figsize)

        for idx in top_indices:
            ax.plot(
                range(seq_len),
                np.abs(temporal[:, idx]),
                label=contributions.sensor_names[idx],
                linewidth=2,
            )

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Absolute Contribution")
        ax.set_title("Sensor Importance Over Time")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def explain_prediction(
        self,
        sample: Tensor,
        background_data: Optional[Tensor] = None,
    ) -> dict:
        """Get comprehensive explanation for a single prediction.

        Args:
            sample: Single sample (1 x seq_len x features) or (seq_len x features)
            background_data: Background data for SHAP

        Returns:
            Dictionary with prediction, feature importance, and attention
        """
        if sample.dim() == 2:
            sample = sample.unsqueeze(0)

        sample = sample.to(self.device)

        # Get prediction
        with torch.no_grad():
            prediction = self.model(sample).item()
            attention_weights = self.model.get_attention_weights()

        # Get feature importance
        importance = self.feature_importance(sample, background_data)

        result = {
            "prediction": prediction,
            "top_features": importance.top_features(10),
            "feature_importance": importance.to_dict(),
        }

        if attention_weights is not None:
            result["attention_weights"] = attention_weights.cpu().numpy()[0].tolist()
            result["peak_attention_timesteps"] = (
                np.argsort(attention_weights.cpu().numpy()[0])[::-1][:5].tolist()
            )

        return result


def create_shap_summary_plot(
    model: BaseRULModel,
    data: Tensor,
    feature_names: list[str],
    background_data: Optional[Tensor] = None,
    max_display: int = 20,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Create SHAP summary plot (beeswarm).

    Args:
        model: Trained model
        data: Data to explain
        feature_names: Feature names
        background_data: Background samples for SHAP
        max_display: Maximum features to display
        save_path: Path to save figure
    """
    import shap

    device = next(model.parameters()).device
    data = data.to(device)

    if background_data is None:
        background_data = data[:min(100, len(data))]
    else:
        background_data = background_data.to(device)

    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(data)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if isinstance(shap_values, Tensor):
        shap_values = shap_values.cpu().numpy()

    # Aggregate across time dimension
    shap_aggregated = shap_values.mean(axis=1)  # (batch, features)
    data_aggregated = data.cpu().numpy().mean(axis=1)  # (batch, features)

    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_aggregated,
        data_aggregated,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
