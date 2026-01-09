"""
GRU Model for RUL Prediction

GRU (Gated Recurrent Unit) is often more efficient than LSTM
and can work better for certain time series tasks.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseRULModel


class TemporalAttention(nn.Module):
    """Temporal attention that learns to focus on important time steps."""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.scale = hidden_size ** -0.5
        self.dropout = nn.Dropout(dropout)

        # Learnable query for final prediction
        self.final_query = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply temporal attention.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)

        Returns:
            Tuple of (context_vector, attention_weights)
        """
        batch_size = x.size(0)

        # Use learnable query expanded to batch
        q = self.final_query.expand(batch_size, -1, -1)  # (batch, 1, hidden)
        q = self.query(q)

        k = self.key(x)  # (batch, seq_len, hidden)
        v = self.value(x)  # (batch, seq_len, hidden)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, 1, seq_len)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Weighted sum
        context = torch.matmul(weights, v)  # (batch, 1, hidden)

        return context.squeeze(1), weights.squeeze(1)


class GRUModel(BaseRULModel):
    """Bidirectional GRU with temporal attention for RUL prediction.

    GRU advantages over LSTM:
    1. Fewer parameters (more efficient)
    2. Often trains faster
    3. Can capture long-term dependencies effectively
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True,
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
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

        self.save_hyperparameters()

        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stacked GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        gru_output_size = hidden_size * self.num_directions

        # Layer normalization after GRU
        self.gru_norm = nn.LayerNorm(gru_output_size)

        # Temporal attention
        self.attention = TemporalAttention(gru_output_size, dropout)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(gru_output_size * 2, hidden_size),  # concat last + attention
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
        """Forward pass."""
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, hidden_size)

        # GRU forward
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden * directions)
        gru_out = self.gru_norm(gru_out)

        # Temporal attention
        context, weights = self.attention(gru_out)
        self._attention_weights = weights.detach()

        # Combine with last hidden state
        last_hidden = gru_out[:, -1, :]
        combined = torch.cat([context, last_hidden], dim=-1)

        # Output
        output = self.output_layers(combined)

        return output.squeeze(-1)

    def get_attention_weights(self) -> Optional[Tensor]:
        """Get attention weights from the last forward pass."""
        return self._attention_weights

    def configure_optimizers(self):
        """Configure optimizer with OneCycleLR scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
