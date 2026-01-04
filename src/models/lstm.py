"""
LSTM Model for RUL Prediction

Bidirectional LSTM with self-attention mechanism for capturing
long-range temporal dependencies in sensor sequences.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.base import BaseRULModel


class Attention(nn.Module):
    """Self-attention mechanism for sequence modeling.
    
    Computes attention weights over LSTM hidden states to focus
    on the most relevant time steps for RUL prediction.
    """
    
    def __init__(self, hidden_size: int):
        """Initialize attention layer.
        
        Args:
            hidden_size: Size of hidden states to attend over
        """
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, lstm_output: Tensor) -> tuple[Tensor, Tensor]:
        """Compute attention-weighted representation.
        
        Args:
            lstm_output: LSTM hidden states (batch, seq_len, hidden_size)
            
        Returns:
            Tuple of:
            - context: Weighted sum of hidden states (batch, hidden_size)
            - weights: Attention weights (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        # Normalize with softmax
        weights = F.softmax(scores, dim=1)  # (batch, seq_len)
        
        # Compute weighted sum
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return context, weights


class LSTMModel(BaseRULModel):
    """Bidirectional LSTM with attention for RUL prediction.
    
    Architecture:
    1. Input projection layer
    2. Bidirectional LSTM layers
    3. Self-attention over hidden states
    4. Fully connected output layers
    
    The attention mechanism allows the model to focus on the most
    relevant time steps (those closer to failure).
    """
    
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
    ):
        """Initialize LSTM model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        super().__init__(
            input_dim=input_dim,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Save additional hyperparameters
        self.save_hyperparameters()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * self.num_directions
        self.attention = Attention(lstm_output_size)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions (batch_size,)
        """
        batch_size = x.size(0)
        
        # Project input features
        x = self.input_projection(x)  # (batch, seq_len, hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size * num_directions)
        
        # Apply attention
        context, weights = self.attention(lstm_out)  # (batch, hidden_size * num_directions)
        
        # Store attention weights for explainability
        self._attention_weights = weights.detach()
        
        # Output projection
        output = self.output_layers(context)  # (batch, 1)
        
        return output.squeeze(-1)  # (batch,)
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get attention weights from the last forward pass.
        
        Returns:
            Attention weights of shape (batch_size, sequence_length)
        """
        return self._attention_weights

