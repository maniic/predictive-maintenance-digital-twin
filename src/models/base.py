"""
Base Model for RUL Prediction

Abstract base class that defines the interface for all RUL prediction models to ensure
consistent training, evaluation, and inference interfaces.
"""

from abc import abstractmethod
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor


class BaseRULModel(pl.LightningModule):
    """Abstract base class for RUL prediction models.
    
    All models must implement:
    - forward(): Forward pass returning RUL predictions
    - get_attention_weights(): Return attention weights if available
    
    The base class provides:
    - Standard training/validation/test step implementations
    - Loss computation (MSE by default)
    - Metric logging
    - Optimizer configuration
    """
    
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """Initialize the base model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Store attention weights for explainability
        self._attention_weights: Optional[Tensor] = None
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions of shape (batch_size,)
        """
        pass
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get attention weights from the last forward pass.
        
        Returns:
            Attention weights tensor if available, None otherwise.
            Shape depends on model architecture.
        """
        return self._attention_weights
    
    def predict_rul(self, x: Tensor) -> Tensor:
        """Predict RUL for input sequences.
        
        Convenience method that handles eval mode and no_grad context.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            RUL predictions of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def _compute_loss(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Compute loss between predictions and targets.
        
        Args:
            y_pred: Predicted RUL values
            y_true: True RUL values
            
        Returns:
            Loss value
        """
        return self.loss_fn(y_pred, y_true)
    
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_pred = self.forward(x)
        loss = self._compute_loss(y_pred, y)
        
        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_rmse", torch.sqrt(loss), on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_pred = self.forward(x)
        loss = self._compute_loss(y_pred, y)
        
        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rmse", torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute MAE
        mae = torch.mean(torch.abs(y_pred - y))
        self.log("val_mae", mae, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Test step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        x, y = batch
        y_pred = self.forward(x)
        loss = self._compute_loss(y_pred, y)
        
        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_rmse", torch.sqrt(loss), on_step=False, on_epoch=True)
        
        mae = torch.mean(torch.abs(y_pred - y))
        self.log("test_mae", mae, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

