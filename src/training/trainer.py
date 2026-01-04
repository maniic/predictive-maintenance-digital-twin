"""
Training Infrastructure for RUL Prediction

PyTorch Lightning trainer wrapper with MLflow integration for
experiment tracking, model checkpointing, and early stopping.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Type
import platform

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger
import torch
from torch.utils.data import DataLoader

from src.models.base import BaseRULModel


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    
    # Checkpointing
    checkpoint_dir: str = "models/checkpoints"
    save_top_k: int = 3
    
    # MLflow
    mlflow_tracking_uri: str = "mlruns"
    experiment_name: str = "cmapss-rul-prediction"
    
    # Hardware
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    
    # Reproducibility
    seed: int = 42
    
    # DataLoader
    num_workers: int = 0  # Set to 0 for MPS compatibility
    
    def __post_init__(self):
        """Adjust settings based on platform."""
        # Disable pin_memory on MPS (Apple Silicon)
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            self.pin_memory = False
        else:
            self.pin_memory = True


class RULTrainer:
    """Trainer for RUL prediction models.
    
    Wraps PyTorch Lightning trainer with:
    - MLflow experiment tracking
    - Model checkpointing (save best models)
    - Early stopping
    - Learning rate monitoring
    
    Example:
        >>> trainer = RULTrainer(config)
        >>> model = trainer.train(model, train_loader, val_loader)
        >>> results = trainer.test(model, test_loader)
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """Initialize trainer.
        
        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or TrainingConfig()
        self._trainer: Optional[pl.Trainer] = None
        self._logger: Optional[MLFlowLogger] = None
        
        # Set random seed for reproducibility
        pl.seed_everything(self.config.seed, workers=True)
    
    def _create_callbacks(self) -> list:
        """Create training callbacks."""
        callbacks = []
        
        # Early stopping
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode="min",
                verbose=True,
            )
        )
        
        # Model checkpointing
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="{epoch}-{val_loss:.4f}-{val_rmse:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=self.config.save_top_k,
                save_last=True,
            )
        )
        
        # Learning rate monitor
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
        
        return callbacks
    
    def _create_logger(self, run_name: Optional[str] = None) -> MLFlowLogger:
        """Create MLflow logger for experiment tracking."""
        return MLFlowLogger(
            experiment_name=self.config.experiment_name,
            tracking_uri=self.config.mlflow_tracking_uri,
            run_name=run_name,
        )
    
    def _create_trainer(self, run_name: Optional[str] = None) -> pl.Trainer:
        """Create PyTorch Lightning trainer."""
        self._logger = self._create_logger(run_name)
        
        return pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            precision=self.config.precision,
            callbacks=self._create_callbacks(),
            logger=self._logger,
            enable_progress_bar=True,
            log_every_n_steps=10,
            deterministic=True,
        )
    
    def train(
        self,
        model: BaseRULModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_name: Optional[str] = None,
    ) -> BaseRULModel:
        """Train a model.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            run_name: Optional name for this training run
            
        Returns:
            Trained model
        """
        self._trainer = self._create_trainer(run_name)
        
        # Log hyperparameters to MLflow
        if self._logger is not None:
            self._logger.log_hyperparams({
                "model_class": model.__class__.__name__,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_epochs": self.config.max_epochs,
                **model.hparams,
            })
        
        # Train
        self._trainer.fit(model, train_loader, val_loader)
        
        # Load best checkpoint
        best_model_path = self._trainer.checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from: {best_model_path}")
            model = model.__class__.load_from_checkpoint(best_model_path)
        
        return model
    
    def test(
        self,
        model: BaseRULModel,
        test_loader: DataLoader,
    ) -> dict:
        """Test a model.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            
        Returns:
            Dictionary of test metrics
        """
        if self._trainer is None:
            self._trainer = self._create_trainer()
        
        results = self._trainer.test(model, test_loader)
        return results[0] if results else {}
    
    def predict(
        self,
        model: BaseRULModel,
        dataloader: DataLoader,
    ) -> torch.Tensor:
        """Generate predictions.
        
        Args:
            model: Trained model
            dataloader: Data loader for prediction
            
        Returns:
            Tensor of predictions
        """
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                
                # Move to same device as model
                x = x.to(model.device)
                pred = model(x)
                predictions.append(pred.cpu())
        
        return torch.cat(predictions, dim=0)
    
    @staticmethod
    def load_model(
        model_class: Type[BaseRULModel],
        checkpoint_path: str | Path,
    ) -> BaseRULModel:
        """Load a model from checkpoint.
        
        Args:
            model_class: Model class to instantiate
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded model
        """
        return model_class.load_from_checkpoint(checkpoint_path)

