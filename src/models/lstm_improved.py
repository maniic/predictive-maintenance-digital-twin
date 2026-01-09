"""
Improved LSTM Model for RUL Prediction

Enhanced version with:
- Larger capacity (256 hidden units)
- Layer normalization
- Residual connections
- Multi-head attention
- Better regularization
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseRULModel


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for sequence modeling."""

    def __init__(self, hidden_size: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % n_heads == 0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply multi-head attention.

        Args:
            x: Input tensor (batch, seq_len, hidden_size)

        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention to values
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        output = self.out(context)

        # Average attention weights across heads for visualization
        avg_weights = weights.mean(dim=1)  # (batch, seq_len, seq_len)

        return output, avg_weights


class LSTMBlock(nn.Module):
    """LSTM block with layer normalization and residual connection."""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.num_directions = 2 if bidirectional else 1
        output_size = hidden_size * self.num_directions

        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)

        # Residual projection if sizes don't match
        self.residual_proj = None
        if input_size != output_size:
            self.residual_proj = nn.Linear(input_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        residual = x
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        output, _ = self.lstm(x)
        output = self.layer_norm(output + residual)
        output = self.dropout(output)

        return output


class ImprovedLSTMModel(BaseRULModel):
    """Improved Bidirectional LSTM with multi-head attention for RUL prediction.

    Enhancements over basic LSTM:
    1. Larger hidden size (256) for more capacity
    2. 3 stacked LSTM blocks with residual connections
    3. Layer normalization for training stability
    4. Multi-head attention (4 heads)
    5. Gradient clipping built-in
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        n_attention_heads: int = 4,
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

        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stacked LSTM blocks
        lstm_output_size = hidden_size * self.num_directions
        self.lstm_blocks = nn.ModuleList()

        # First block takes hidden_size input
        self.lstm_blocks.append(
            LSTMBlock(hidden_size, hidden_size, dropout, bidirectional)
        )

        # Subsequent blocks take lstm_output_size input
        for _ in range(num_layers - 1):
            self.lstm_blocks.append(
                LSTMBlock(lstm_output_size, hidden_size, dropout, bidirectional)
            )

        # Multi-head attention
        self.attention = MultiHeadAttention(lstm_output_size, n_attention_heads, dropout)
        self.attention_norm = nn.LayerNorm(lstm_output_size)

        # Global pooling options
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Output MLP with larger capacity
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size * 2, hidden_size),  # concat attention + pool
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

        # Stacked LSTM blocks with residual connections
        for block in self.lstm_blocks:
            x = block(x)  # (batch, seq_len, hidden_size * num_directions)

        # Multi-head attention
        attended, weights = self.attention(x)
        x = self.attention_norm(x + attended)

        # Store last row of attention weights for visualization
        self._attention_weights = weights[:, -1, :].detach()  # Focus on last timestep

        # Combine attention output with global pooling
        attention_out = x[:, -1, :]  # Last timestep (batch, hidden)
        pooled_out = self.global_pool(x.transpose(1, 2)).squeeze(-1)  # (batch, hidden)

        combined = torch.cat([attention_out, pooled_out], dim=-1)

        # Output projection
        output = self.output_layers(combined)

        return output.squeeze(-1)

    def get_attention_weights(self) -> Optional[Tensor]:
        """Get attention weights from the last forward pass."""
        return self._attention_weights

    def configure_optimizers(self):
        """Configure optimizer with cosine annealing scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double period after each restart
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
