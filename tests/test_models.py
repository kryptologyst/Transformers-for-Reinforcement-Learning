"""Basic tests for the transformers-for-rl package."""

import pytest
import torch
import numpy as np

from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_transformer import TrajectoryTransformer
from src.utils.device import get_device, set_seed


class TestDecisionTransformer:
    """Test Decision Transformer model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DecisionTransformer(
            state_dim=4,
            action_dim=2,
            max_length=10,
            hidden_size=64,
            n_heads=4,
            n_layers=2,
        )
        
        assert model.state_dim == 4
        assert model.action_dim == 2
        assert model.max_length == 10
        assert model.hidden_size == 64
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = DecisionTransformer(
            state_dim=4,
            action_dim=2,
            max_length=5,
            hidden_size=32,
            n_heads=2,
            n_layers=1,
        )
        
        batch_size = 2
        seq_len = 5
        
        states = torch.randn(batch_size, seq_len, 4)
        actions = torch.randint(0, 2, (batch_size, seq_len))
        rewards = torch.randn(batch_size, seq_len, 1)
        returns_to_go = torch.randn(batch_size, seq_len, 1)
        timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        action_logits, value_estimates = model(
            states, actions, rewards, returns_to_go, timesteps
        )
        
        assert action_logits.shape == (batch_size, seq_len, 2)
        assert value_estimates.shape == (batch_size, seq_len, 1)
    
    def test_get_action(self):
        """Test action generation."""
        model = DecisionTransformer(
            state_dim=4,
            action_dim=2,
            max_length=5,
            hidden_size=32,
            n_heads=2,
            n_layers=1,
        )
        
        batch_size = 1
        seq_len = 3
        
        states = torch.randn(batch_size, seq_len, 4)
        actions = torch.randint(0, 2, (batch_size, seq_len))
        rewards = torch.randn(batch_size, seq_len, 1)
        returns_to_go = torch.randn(batch_size, seq_len, 1)
        timesteps = torch.arange(seq_len).unsqueeze(0)
        
        action = model.get_action(states, actions, rewards, returns_to_go, timesteps)
        
        assert action.shape == (batch_size, 2)  # Action probabilities


class TestTrajectoryTransformer:
    """Test Trajectory Transformer model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = TrajectoryTransformer(
            state_dim=4,
            action_dim=2,
            max_length=10,
            hidden_size=64,
            n_heads=4,
            n_layers=2,
        )
        
        assert model.state_dim == 4
        assert model.action_dim == 2
        assert model.max_length == 10
        assert model.hidden_size == 64
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = TrajectoryTransformer(
            state_dim=4,
            action_dim=2,
            max_length=5,
            hidden_size=32,
            n_heads=2,
            n_layers=1,
        )
        
        batch_size = 2
        seq_len = 5
        
        states = torch.randn(batch_size, seq_len, 4)
        actions = torch.randint(0, 2, (batch_size, seq_len))
        rewards = torch.randn(batch_size, seq_len, 1)
        timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        state_pred, action_pred, reward_pred = model(
            states, actions, rewards, timesteps
        )
        
        assert state_pred.shape == (batch_size, seq_len, 4)
        assert action_pred.shape == (batch_size, seq_len, 2)
        assert reward_pred.shape == (batch_size, seq_len, 1)
    
    def test_compute_loss(self):
        """Test loss computation."""
        model = TrajectoryTransformer(
            state_dim=4,
            action_dim=2,
            max_length=5,
            hidden_size=32,
            n_heads=2,
            n_layers=1,
        )
        
        batch_size = 2
        seq_len = 5
        
        states = torch.randn(batch_size, seq_len, 4)
        actions = torch.randint(0, 2, (batch_size, seq_len))
        rewards = torch.randn(batch_size, seq_len, 1)
        timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
        
        loss_dict = model.compute_loss(states, actions, rewards, timesteps)
        
        assert "total_loss" in loss_dict
        assert "state_loss" in loss_dict
        assert "action_loss" in loss_dict
        assert "reward_loss" in loss_dict
        
        assert loss_dict["total_loss"] > 0


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Check that seeds are set
        torch_rand = torch.rand(1).item()
        np_rand = np.random.rand()
        
        # Reset and set seed again
        set_seed(42)
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        
        # Should be the same
        assert torch_rand == torch_rand2
        assert np_rand == np_rand2


if __name__ == "__main__":
    pytest.main([__file__])
