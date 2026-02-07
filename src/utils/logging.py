"""Logging utilities for experiment tracking."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger for experiments."""
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        """Initialize logger.
        
        Args:
            log_dir: Directory to save logs.
            experiment_name: Name of the experiment.
            use_tensorboard: Whether to use TensorBoard logging.
            use_wandb: Whether to use Weights & Biases logging.
            wandb_project: W&B project name.
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = self.log_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
        
        # Setup TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            tb_dir = self.experiment_dir / "tensorboard"
            self.tb_writer = SummaryWriter(str(tb_dir))
        
        # Setup W&B
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project or "transformers-for-rl",
                    name=self.experiment_name,
                    dir=str(self.experiment_dir),
                )
            except ImportError:
                logging.warning("wandb not available, skipping W&B logging")
    
    def _setup_file_logging(self) -> None:
        """Setup file logging."""
        log_file = self.experiment_dir / "experiment.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        
        self.logger = logging.getLogger(self.experiment_name)
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value.
        
        Args:
            name: Name of the metric.
            value: Value to log.
            step: Step number.
        """
        if self.tb_writer:
            self.tb_writer.add_scalar(name, value, step)
        
        if self.wandb_run:
            self.wandb_run.log({name: value}, step=step)
    
    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log multiple scalar values.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Step number.
        """
        for name, value in metrics.items():
            self.log_scalar(name, value, step)
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram.
        
        Args:
            name: Name of the histogram.
            values: Values to log.
            step: Step number.
        """
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)
    
    def log_model(self, model: torch.nn.Module, input_tensor: torch.Tensor) -> None:
        """Log model graph.
        
        Args:
            model: PyTorch model.
            input_tensor: Example input tensor.
        """
        if self.tb_writer:
            self.tb_writer.add_graph(model, input_tensor)
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            epoch: Current epoch.
            metrics: Additional metrics to save.
            filename: Custom filename.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        filename = filename or f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = self.experiment_dir / "checkpoints" / filename
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
    ) -> int:
        """Load model checkpoint.
        
        Args:
            model: PyTorch model.
            optimizer: Optimizer.
            checkpoint_path: Path to checkpoint.
            
        Returns:
            Epoch number.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        epoch = checkpoint["epoch"]
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
        
        return epoch
    
    def close(self) -> None:
        """Close logger and cleanup."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.wandb_run:
            self.wandb_run.finish()
        
        self.logger.info(f"Experiment {self.experiment_name} completed")
