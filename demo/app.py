"""Streamlit demo for Transformers for Reinforcement Learning."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from envs.data_collection import EnvironmentWrapper
from eval.evaluator import TransformerEvaluator
from models.decision_transformer import DecisionTransformer
from models.trajectory_transformer import TrajectoryTransformer
from utils.device import get_device, set_seed


def load_model(model_path: str, config_path: str) -> nn.Module:
    """Load a trained model.
    
    Args:
        model_path: Path to model checkpoint.
        config_path: Path to configuration file.
        
    Returns:
        Loaded model.
    """
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Create model
    if config.model.type == "decision_transformer":
        model = DecisionTransformer(
            state_dim=config.model.state_dim,
            action_dim=config.model.action_dim,
            max_length=config.model.max_length,
            hidden_size=config.model.hidden_size,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )
    elif config.model.type == "trajectory_transformer":
        model = TrajectoryTransformer(
            state_dim=config.model.state_dim,
            action_dim=config.model.action_dim,
            max_length=config.model.max_length,
            hidden_size=config.model.hidden_size,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            dropout=config.model.dropout,
            activation=config.model.activation,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def create_environment(env_name: str) -> gym.Env:
    """Create environment.
    
    Args:
        env_name: Environment name.
        
    Returns:
        Environment.
    """
    env = gym.make(env_name)
    return EnvironmentWrapper(env, normalize_observations=True)


def run_episode(
    model: nn.Module,
    env: gym.Env,
    target_return: Optional[float] = None,
    max_steps: int = 500,
    render: bool = False,
) -> Dict[str, List]:
    """Run a single episode.
    
    Args:
        model: Trained model.
        env: Environment.
        target_return: Target return for Decision Transformer.
        max_steps: Maximum number of steps.
        render: Whether to render.
        
    Returns:
        Episode data.
    """
    # Create evaluator
    evaluator = TransformerEvaluator(model, env, device=get_device())
    
    # Run episode
    metrics = evaluator.evaluate_episode(
        seed=random.randint(0, 10000),
        render=render,
        target_return=target_return,
    )
    
    return {
        "reward": metrics["episode_reward"],
        "length": metrics["episode_length"],
        "success": metrics["success"],
    }


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Transformers for RL Demo",
        page_icon="🤖",
        layout="wide",
    )
    
    st.title("🤖 Transformers for Reinforcement Learning")
    st.markdown("""
    This demo showcases transformer-based models for reinforcement learning, including:
    - **Decision Transformer**: Models actions conditioned on desired returns
    - **Trajectory Transformer**: Models complete trajectories autoregressively
    
    **⚠️ Disclaimer**: This is a research/educational demo. Not for production control of real systems.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["decision_transformer", "trajectory_transformer"],
        help="Choose the transformer model architecture"
    )
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
        help="Choose the environment to test on"
    )
    
    # Model file selection
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        if checkpoint_files:
            checkpoint_file = st.sidebar.selectbox(
                "Model Checkpoint",
                checkpoint_files,
                format_func=lambda x: x.name,
                help="Choose a trained model checkpoint"
            )
        else:
            st.sidebar.error("No model checkpoints found. Please train a model first.")
            return
    else:
        st.sidebar.error("Checkpoints directory not found. Please train a model first.")
        return
    
    # Configuration file
    config_file = st.sidebar.selectbox(
        "Configuration",
        ["configs/default.yaml"],
        help="Choose the configuration file"
    )
    
    # Evaluation parameters
    st.sidebar.header("Evaluation Parameters")
    
    num_episodes = st.sidebar.slider(
        "Number of Episodes",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of episodes to evaluate"
    )
    
    target_return = None
    if model_type == "decision_transformer":
        target_return = st.sidebar.slider(
            "Target Return",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=10.0,
            help="Desired return for the episode"
        )
    
    render_episodes = st.sidebar.checkbox(
        "Render Episodes",
        value=False,
        help="Show episode visualization (may be slow)"
    )
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model(str(checkpoint_file), config_file)
            model.eval()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {str(e)}")
        return
    
    # Create environment
    try:
        env = create_environment(env_name)
        st.sidebar.success("Environment created successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to create environment: {str(e)}")
        return
    
    # Main content
    st.header("Model Evaluation")
    
    # Run evaluation
    if st.button("🚀 Run Evaluation", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        episode_results = []
        
        for i in range(num_episodes):
            status_text.text(f"Running episode {i+1}/{num_episodes}...")
            
            # Run episode
            result = run_episode(
                model=model,
                env=env,
                target_return=target_return,
                render=render_episodes,
            )
            
            episode_results.append(result)
            progress_bar.progress((i + 1) / num_episodes)
        
        status_text.text("Evaluation completed!")
        
        # Display results
        st.header("Results")
        
        # Summary statistics
        rewards = [r["reward"] for r in episode_results]
        lengths = [r["length"] for r in episode_results]
        successes = [r["success"] for r in episode_results]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Mean Reward",
                f"{np.mean(rewards):.2f}",
                delta=f"±{np.std(rewards):.2f}"
            )
        
        with col2:
            st.metric(
                "Mean Length",
                f"{np.mean(lengths):.2f}",
                delta=f"±{np.std(lengths):.2f}"
            )
        
        with col3:
            st.metric(
                "Success Rate",
                f"{np.mean(successes):.2%}"
            )
        
        with col4:
            st.metric(
                "Best Reward",
                f"{np.max(rewards):.2f}"
            )
        
        # Episode details
        st.subheader("Episode Details")
        
        episode_data = []
        for i, result in enumerate(episode_results):
            episode_data.append({
                "Episode": i + 1,
                "Reward": f"{result['reward']:.2f}",
                "Length": result["length"],
                "Success": "✅" if result["success"] else "❌",
            })
        
        st.table(episode_data)
        
        # Visualization
        st.subheader("Reward Distribution")
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Reward histogram
        ax1.hist(rewards, bins=min(10, len(rewards)), alpha=0.7, color="skyblue", edgecolor="black")
        ax1.set_xlabel("Episode Reward")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Reward Distribution")
        ax1.grid(True, alpha=0.3)
        
        # Episode length vs reward
        ax2.scatter(lengths, rewards, alpha=0.7, color="coral")
        ax2.set_xlabel("Episode Length")
        ax2.set_ylabel("Episode Reward")
        ax2.set_title("Length vs Reward")
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # Model information
    st.header("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Architecture")
        st.write(f"**Model Type**: {model_type}")
        st.write(f"**State Dimension**: {model.state_dim}")
        st.write(f"**Action Dimension**: {model.action_dim}")
        st.write(f"**Max Sequence Length**: {model.max_length}")
        st.write(f"**Hidden Size**: {model.hidden_size}")
        st.write(f"**Number of Heads**: {model.n_heads}")
        st.write(f"**Number of Layers**: {model.n_layers}")
    
    with col2:
        st.subheader("Parameters")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        st.write(f"**Total Parameters**: {total_params:,}")
        st.write(f"**Trainable Parameters**: {trainable_params:,}")
        st.write(f"**Model Size**: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Transformers for Reinforcement Learning** - A modern implementation showcasing 
    transformer architectures for sequential decision-making in RL environments.
    
    Built with PyTorch, Gymnasium, and Streamlit.
    """)


if __name__ == "__main__":
    main()
