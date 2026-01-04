"""
Transformer Model for RUL Prediction

Transformer encoder with positional encoding for capturing
complex temporal dependencies in sensor sequences.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseRULModel


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position information."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(BaseRULModel):
    """Transformer encoder for RUL prediction.
    
    Architecture:
    1. Input projection to model dimension
    2. Positional encoding
    3. Transformer encoder layers with multi-head self-attention
    4. Global average pooling or CLS token
    5. Fully connected output layers
    
    The self-attention mechanism allows the model to learn which
    time steps are most relevant for predicting RUL.
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """Initialize Transformer model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            d_model: Model/embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer encoder layers
            d_ff: Feed-forward network dimension
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        super().__init__(
            input_dim=input_dim,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length + 1, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 4, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions (batch_size,)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Output projection
        output = self.output_layers(x)  # (batch, 1)
        
        return output.squeeze(-1)  # (batch,)
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get attention weights from transformer layers.
        
        Note: Extracting attention weights from nn.TransformerEncoder
        requires hooks. For now, returns None. Implement custom
        transformer layers for full attention visualization.
        """
        return None

