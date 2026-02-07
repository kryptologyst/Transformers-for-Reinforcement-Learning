"""Training utilities for transformer-based RL models."""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models.decision_transformer import DecisionTransformer
from ..models.trajectory_transformer import TrajectoryTransformer
from ..utils.device import get_device, set_seed
from ..utils.logging import Logger


class TrajectoryDataset(Dataset):
    """Dataset for storing and sampling trajectory data."""
    
    def __init__(
        self,
        trajectories: List[Dict[str, np.ndarray]],
        max_length: int = 20,
        normalize_rewards: bool = True,
    ):
        """Initialize trajectory dataset.
        
        Args:
            trajectories: List of trajectory dictionaries.
            max_length: Maximum sequence length.
            normalize_rewards: Whether to normalize rewards.
        """
        self.trajectories = trajectories
        self.max_length = max_length
        self.normalize_rewards = normalize_rewards
        
        # Compute reward statistics for normalization
        if normalize_rewards:
            all_rewards = np.concatenate([traj["rewards"] for traj in trajectories])
            self.reward_mean = np.mean(all_rewards)
            self.reward_std = np.std(all_rewards) + 1e-8
        else:
            self.reward_mean = 0.0
            self.reward_std = 1.0
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get trajectory sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary of trajectory tensors.
        """
        traj = self.trajectories[idx]
        
        # Extract trajectory components
        states = traj["states"]
        actions = traj["actions"]
        rewards = traj["rewards"]
        
        # Normalize rewards
        if self.normalize_rewards:
            rewards = (rewards - self.reward_mean) / self.reward_std
        
        # Compute returns-to-go
        returns_to_go = np.zeros_like(rewards)
        running_return = 0.0
        for i in reversed(range(len(rewards))):
            running_return += rewards[i]
            returns_to_go[i] = running_return
        
        # Pad or truncate to max_length
        if len(states) > self.max_length:
            # Randomly sample a subsequence
            start_idx = random.randint(0, len(states) - self.max_length)
            states = states[start_idx:start_idx + self.max_length]
            actions = actions[start_idx:start_idx + self.max_length]
            rewards = rewards[start_idx:start_idx + self.max_length]
            returns_to_go = returns_to_go[start_idx:start_idx + self.max_length]
        else:
            # Pad with zeros
            pad_len = self.max_length - len(states)
            states = np.pad(states, ((0, pad_len), (0, 0)), mode="constant")
            actions = np.pad(actions, ((0, pad_len), (0, 0)), mode="constant")
            rewards = np.pad(rewards, ((0, pad_len), (0, 0)), mode="constant")
            returns_to_go = np.pad(returns_to_go, ((0, pad_len), (0, 0)), mode="constant")
        
        # Create timesteps
        timesteps = np.arange(self.max_length)
        
        return {
            "states": torch.FloatTensor(states),
            "actions": torch.FloatTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "returns_to_go": torch.FloatTensor(returns_to_go),
            "timesteps": torch.LongTensor(timesteps),
        }


class TransformerTrainer:
    """Trainer for transformer-based RL models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: TrajectoryDataset,
        val_dataset: Optional[TrajectoryDataset] = None,
        learning_rate: float = 1e-4,
        batch_size: int = 64,
        num_epochs: int = 100,
        device: Optional[torch.device] = None,
        logger: Optional[Logger] = None,
        save_dir: str = "checkpoints",
    ):
        """Initialize trainer.
        
        Args:
            model: Transformer model to train.
            train_dataset: Training dataset.
            val_dataset: Validation dataset.
            learning_rate: Learning rate.
            batch_size: Batch size.
            num_epochs: Number of training epochs.
            device: Device to use for training.
            logger: Logger for experiment tracking.
            save_dir: Directory to save checkpoints.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device or get_device()
        self.logger = logger
        self.save_dir = save_dir
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        else:
            self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            if isinstance(self.model, DecisionTransformer):
                action_logits, value_estimates = self.model(
                    batch["states"],
                    batch["actions"],
                    batch["rewards"],
                    batch["returns_to_go"],
                    batch["timesteps"],
                )
                
                # Compute loss
                action_loss = F.cross_entropy(
                    action_logits.view(-1, action_logits.size(-1)),
                    batch["actions"].long().view(-1),
                )
                value_loss = F.mse_loss(value_estimates, batch["returns_to_go"])
                loss = action_loss + value_loss
                
            elif isinstance(self.model, TrajectoryTransformer):
                loss_dict = self.model.compute_loss(
                    batch["states"],
                    batch["actions"],
                    batch["rewards"],
                    batch["timesteps"],
                )
                loss = loss_dict["total_loss"]
            else:
                raise ValueError(f"Unknown model type: {type(self.model)}")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                if isinstance(self.model, DecisionTransformer):
                    action_logits, value_estimates = self.model(
                        batch["states"],
                        batch["actions"],
                        batch["rewards"],
                        batch["returns_to_go"],
                        batch["timesteps"],
                    )
                    
                    # Compute loss
                    action_loss = F.cross_entropy(
                        action_logits.view(-1, action_logits.size(-1)),
                        batch["actions"].long().view(-1),
                    )
                    value_loss = F.mse_loss(value_estimates, batch["returns_to_go"])
                    loss = action_loss + value_loss
                    
                elif isinstance(self.model, TrajectoryTransformer):
                    loss_dict = self.model.compute_loss(
                        batch["states"],
                        batch["actions"],
                        batch["rewards"],
                        batch["timesteps"],
                    )
                    loss = loss_dict["total_loss"]
                else:
                    raise ValueError(f"Unknown model type: {type(self.model)}")
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {"val_loss": avg_loss}
    
    def train(self) -> None:
        """Train the model."""
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            if self.logger:
                for name, value in train_metrics.items():
                    self.logger.log_scalar(name, value, epoch)
                for name, value in val_metrics.items():
                    self.logger.log_scalar(name, value, epoch)
            
            # Save checkpoint
            if val_metrics and val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
            
            # Print progress
            print(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"Epoch {epoch}: Val Loss: {val_metrics['val_loss']:.4f}")
    
    def save_checkpoint(self, filename: str) -> str:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
            
        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = f"{self.save_dir}/{filename}"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_val_loss": self.best_val_loss,
        }, checkpoint_path)
        
        if self.logger:
            self.logger.logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        if self.logger:
            self.logger.logger.info(f"Checkpoint loaded from {checkpoint_path}")
