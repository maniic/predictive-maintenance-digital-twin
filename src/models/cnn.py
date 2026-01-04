"""
Temporal CNN Model for RUL Prediction

1D Convolutional Neural Network with multi-scale feature extraction
for capturing local temporal patterns in sensor sequences.
"""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import BaseRULModel


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        """Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            dilation: Dilation factor for dilated convolutions
            dropout: Dropout probability
        """
        super().__init__()
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, channels, seq_len)
            
        Returns:
            Output tensor (batch, out_channels, seq_len)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TemporalCNNModel(BaseRULModel):
    """1D Temporal CNN for RUL prediction.
    
    Architecture:
    1. Input projection to channel dimension
    2. Stacked 1D convolutions with increasing dilation
    3. Global average pooling
    4. Fully connected output layers
    
    The dilated convolutions capture patterns at multiple time scales
    without losing resolution, making it effective for degradation patterns.
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        channels: list[int] = None,
        kernel_sizes: list[int] = None,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """Initialize Temporal CNN model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            channels: List of channel sizes for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
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
        
        # Default architecture
        if channels is None:
            channels = [32, 64, 128, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3]
        
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Input projection (features -> channels)
        self.input_projection = nn.Linear(input_dim, channels[0])
        
        # Convolutional layers with increasing dilation
        conv_layers = []
        in_channels = channels[0]
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            dilation = 2 ** i  # Exponentially increasing dilation
            conv_layers.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        
        self.conv_layers = nn.ModuleList(conv_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(channels[-1], channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] // 2, channels[-1] // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(channels[-1] // 4, 1),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions (batch_size,)
        """
        # Project input features to channel dimension
        x = self.input_projection(x)  # (batch, seq_len, channels[0])
        
        # Transpose for 1D convolution: (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, channels[-1])
        
        # Output projection
        output = self.output_layers(x)  # (batch, 1)
        
        return output.squeeze(-1)  # (batch,)

