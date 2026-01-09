"""
Advanced RUL Prediction Models

Implements multiple strategies for improved RUL prediction:
1. Weighted loss function (penalize low-RUL errors more)
2. Two-stage model (classify then regress)
3. RUL-range specialized models
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl

from src.models.base import BaseRULModel


class WeightedMSELoss(nn.Module):
    """MSE loss that weighs errors based on RUL value.

    Errors at low RUL are penalized more heavily since they're
    more operationally critical.
    """

    def __init__(self, alpha: float = 0.5, rul_threshold: float = 50.0):
        """
        Args:
            alpha: Weight increase factor for low RUL predictions
            rul_threshold: RUL below which errors are weighted more
        """
        super().__init__()
        self.alpha = alpha
        self.rul_threshold = rul_threshold

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute weighted MSE loss."""
        errors = (pred - target) ** 2

        # Weight based on target RUL (lower RUL = higher weight)
        # Weight = 1 + alpha * max(0, 1 - target/threshold)
        weights = 1.0 + self.alpha * torch.clamp(1.0 - target / self.rul_threshold, min=0)

        # Also penalize late predictions (predicting higher than actual) more
        late_mask = (pred > target).float()
        weights = weights * (1.0 + 0.3 * late_mask)  # 30% extra penalty for late predictions

        weighted_errors = errors * weights
        return weighted_errors.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss inspired by C-MAPSS scoring.

    Late predictions (predicting failure later than actual) are
    penalized more than early predictions.
    """

    def __init__(self, a1: float = 10.0, a2: float = 13.0):
        """
        Args:
            a1: Exponential factor for late predictions
            a2: Exponential factor for early predictions
        """
        super().__init__()
        self.a1 = a1
        self.a2 = a2

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute asymmetric loss."""
        diff = pred - target  # positive = late prediction

        # Smooth approximation of asymmetric penalty
        # Late: exp(d/a1) - 1, Early: exp(-d/a2) - 1
        late_loss = torch.where(
            diff >= 0,
            (torch.exp(diff / self.a1) - 1),
            (torch.exp(-diff / self.a2) - 1)
        )

        # Combine with MSE for stability
        mse_loss = diff ** 2

        return 0.5 * mse_loss.mean() + 0.5 * late_loss.mean()


class FocalRULLoss(nn.Module):
    """Focal loss adapted for RUL regression.

    Focuses training on hard examples (typically mid-range RUL).
    """

    def __init__(self, gamma: float = 2.0, rul_cap: float = 125.0):
        super().__init__()
        self.gamma = gamma
        self.rul_cap = rul_cap

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """Compute focal RUL loss."""
        # Normalize error by RUL cap
        normalized_error = torch.abs(pred - target) / self.rul_cap
        normalized_error = torch.clamp(normalized_error, max=1.0)

        # Focal weight: (1 - p)^gamma where p is accuracy
        focal_weight = normalized_error ** self.gamma

        mse = (pred - target) ** 2
        return (focal_weight * mse).mean()


class EnhancedLSTM(BaseRULModel):
    """LSTM with enhanced training and weighted loss."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_type: str = "weighted",  # "mse", "weighted", "asymmetric", "focal"
    ):
        super().__init__(
            input_dim=input_dim,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.loss_type = loss_type

        self.save_hyperparameters()

        # Choose loss function
        if loss_type == "weighted":
            self.loss_fn = WeightedMSELoss(alpha=1.0, rul_threshold=50.0)
        elif loss_type == "asymmetric":
            self.loss_fn = AsymmetricLoss()
        elif loss_type == "focal":
            self.loss_fn = FocalRULLoss()
        else:
            self.loss_fn = nn.MSELoss()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * self.num_directions

        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(lstm_output_size)

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Output with skip connection from last timestep
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 1),
        )

        self._attention_weights = None

    def forward(self, x: Tensor) -> Tensor:
        # Input projection
        x = self.input_projection(x)

        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)

        # Attention
        scores = self.attention(lstm_out).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        self._attention_weights = weights.detach()

        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)

        # Combine with last timestep
        last_hidden = lstm_out[:, -1, :]
        combined = torch.cat([context, last_hidden], dim=-1)

        output = self.output_layers(combined)
        return output.squeeze(-1)

    def get_attention_weights(self) -> Optional[Tensor]:
        return self._attention_weights

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class TwoStageRULModel(pl.LightningModule):
    """Two-stage model: first classify health state, then regress RUL.

    Stage 1: Classify into health categories (healthy, degrading, critical)
    Stage 2: Regress RUL within each category using specialized heads
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        # RUL thresholds for classification
        critical_threshold: float = 25.0,
        degrading_threshold: float = 75.0,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.critical_threshold = critical_threshold
        self.degrading_threshold = degrading_threshold

        self.save_hyperparameters()

        # Shared encoder
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        lstm_output_size = hidden_size * 2
        self.lstm_norm = nn.LayerNorm(lstm_output_size)

        # Classification head (3 classes: healthy, degrading, critical)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

        # Separate regression heads for each health state
        self.rul_head_critical = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.rul_head_degrading = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.rul_head_healthy = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        # Loss functions
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.rul_loss_fn = WeightedMSELoss(alpha=1.0, rul_threshold=50.0)

    def _get_health_labels(self, rul: Tensor) -> Tensor:
        """Convert RUL to health class labels."""
        labels = torch.zeros_like(rul, dtype=torch.long)
        labels[rul <= self.critical_threshold] = 0  # Critical
        labels[(rul > self.critical_threshold) & (rul <= self.degrading_threshold)] = 1  # Degrading
        labels[rul > self.degrading_threshold] = 2  # Healthy
        return labels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (class_logits, rul_prediction)."""
        # Encode
        x = self.input_projection(x)
        lstm_out, _ = self.lstm(x)
        features = self.lstm_norm(lstm_out[:, -1, :])  # Last timestep

        # Classify
        class_logits = self.classifier(features)
        class_probs = F.softmax(class_logits, dim=-1)

        # RUL predictions from each head
        rul_critical = self.rul_head_critical(features).squeeze(-1)
        rul_degrading = self.rul_head_degrading(features).squeeze(-1)
        rul_healthy = self.rul_head_healthy(features).squeeze(-1)

        # Weighted combination based on class probabilities
        rul_pred = (
            class_probs[:, 0] * rul_critical +
            class_probs[:, 1] * rul_degrading +
            class_probs[:, 2] * rul_healthy
        )

        return class_logits, rul_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        class_logits, rul_pred = self(x)

        # Classification loss
        health_labels = self._get_health_labels(y)
        cls_loss = self.cls_loss_fn(class_logits, health_labels)

        # RUL regression loss
        rul_loss = self.rul_loss_fn(rul_pred, y)

        # Combined loss
        loss = cls_loss + rul_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_cls_loss", cls_loss)
        self.log("train_rul_loss", rul_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        class_logits, rul_pred = self(x)

        health_labels = self._get_health_labels(y)
        cls_loss = self.cls_loss_fn(class_logits, health_labels)
        rul_loss = self.rul_loss_fn(rul_pred, y)
        loss = cls_loss + rul_loss

        rmse = torch.sqrt(F.mse_loss(rul_pred, y))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)

        return loss

    def predict_rul(self, x: Tensor) -> Tensor:
        """Get just the RUL prediction."""
        _, rul_pred = self(x)
        return rul_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


class LearnedEnsemble(pl.LightningModule):
    """Ensemble with learned weights based on input features."""

    def __init__(
        self,
        models: list[nn.Module],
        input_dim: int,
        sequence_length: int,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
    ):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.learning_rate = learning_rate

        # Freeze base models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # Weight prediction network
        # Uses the last timestep features to predict weights
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_models),
            nn.Softmax(dim=-1),
        )

        self.loss_fn = WeightedMSELoss(alpha=1.0, rul_threshold=50.0)

    def forward(self, x: Tensor) -> Tensor:
        # Get predictions from all base models
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
            preds.append(pred)

        preds = torch.stack(preds, dim=-1)  # (batch, n_models)

        # Predict weights based on last timestep features
        last_features = x[:, -1, :]  # (batch, input_dim)
        weights = self.weight_net(last_features)  # (batch, n_models)

        # Weighted combination
        ensemble_pred = (preds * weights).sum(dim=-1)

        return ensemble_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        rmse = torch.sqrt(F.mse_loss(pred, y))
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_rmse", rmse, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.weight_net.parameters(), lr=self.learning_rate)
