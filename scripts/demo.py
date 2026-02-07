#!/usr/bin/env python3
"""Example script demonstrating Transformers for Reinforcement Learning."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from models.decision_transformer import DecisionTransformer
from models.trajectory_transformer import TrajectoryTransformer
from utils.device import get_device, set_seed


def create_simple_model(model_type: str = "decision_transformer") -> nn.Module:
    """Create a simple transformer model for demonstration.
    
    Args:
        model_type: Type of model to create.
        
    Returns:
        Transformer model.
    """
    if model_type == "decision_transformer":
        model = DecisionTransformer(
            state_dim=4,
            action_dim=2,
            max_length=10,
            hidden_size=64,
            n_heads=4,
            n_layers=2,
        )
    elif model_type == "trajectory_transformer":
        model = TrajectoryTransformer(
            state_dim=4,
            action_dim=2,
            max_length=10,
            hidden_size=64,
            n_heads=4,
            n_layers=2,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def demonstrate_model(model: nn.Module, model_type: str) -> None:
    """Demonstrate model functionality.
    
    Args:
        model: Transformer model.
        model_type: Type of model.
    """
    print(f"\n=== {model_type.upper()} DEMONSTRATION ===")
    
    # Create sample data
    batch_size = 2
    seq_len = 5
    
    states = torch.randn(batch_size, seq_len, 4)
    actions = torch.randint(0, 2, (batch_size, seq_len))
    rewards = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    
    print(f"Input shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        if model_type == "decision_transformer":
            returns_to_go = torch.randn(batch_size, seq_len, 1)
            action_logits, value_estimates = model(
                states, actions, rewards, returns_to_go, timesteps
            )
            
            print(f"\nOutput shapes:")
            print(f"  Action logits: {action_logits.shape}")
            print(f"  Value estimates: {value_estimates.shape}")
            
            # Get action for last timestep
            last_action = model.get_action(
                states, actions, rewards, returns_to_go, timesteps
            )
            print(f"  Last action: {last_action.shape}")
            
        elif model_type == "trajectory_transformer":
            state_pred, action_pred, reward_pred = model(
                states, actions, rewards, timesteps
            )
            
            print(f"\nOutput shapes:")
            print(f"  State predictions: {state_pred.shape}")
            print(f"  Action predictions: {action_pred.shape}")
            print(f"  Reward predictions: {reward_pred.shape}")
            
            # Compute loss
            loss_dict = model.compute_loss(states, actions, rewards, timesteps)
            print(f"\nLoss components:")
            for name, loss in loss_dict.items():
                print(f"  {name}: {loss.item():.4f}")


def main():
    """Main demonstration function."""
    print("🤖 Transformers for Reinforcement Learning - Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Demonstrate Decision Transformer
    dt_model = create_simple_model("decision_transformer")
    dt_model.to(device)
    demonstrate_model(dt_model, "decision_transformer")
    
    # Demonstrate Trajectory Transformer
    tt_model = create_simple_model("trajectory_transformer")
    tt_model.to(device)
    demonstrate_model(tt_model, "trajectory_transformer")
    
    # Model comparison
    print(f"\n=== MODEL COMPARISON ===")
    dt_params = sum(p.numel() for p in dt_model.parameters())
    tt_params = sum(p.numel() for p in tt_model.parameters())
    
    print(f"Decision Transformer parameters: {dt_params:,}")
    print(f"Trajectory Transformer parameters: {tt_params:,}")
    print(f"Parameter difference: {abs(dt_params - tt_params):,}")
    
    # Environment demonstration
    print(f"\n=== ENVIRONMENT DEMONSTRATION ===")
    env = gym.make("CartPole-v1")
    print(f"Environment: {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few random steps
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}")
        
        if terminated or truncated:
            break
        
        state = next_state
    
    print(f"Total reward: {total_reward:.2f}")
    env.close()
    
    print(f"\n=== DEMO COMPLETED ===")
    print("To train models, run: python scripts/train.py")
    print("To launch demo, run: streamlit run demo/app.py")


if __name__ == "__main__":
    main()
