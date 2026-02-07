"""Main training script for Transformers for Reinforcement Learning."""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from envs.data_collection import DataCollector, EnvironmentWrapper, create_expert_model
from eval.evaluator import TransformerEvaluator
from models.decision_transformer import DecisionTransformer
from models.trajectory_transformer import TrajectoryTransformer
from train.trainer import TrajectoryDataset, TransformerTrainer
from utils.config import Config
from utils.device import get_device, set_seed
from utils.logging import Logger


def create_model(config: Config) -> nn.Module:
    """Create transformer model based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Transformer model.
    """
    model_config = config.cfg.model
    
    if model_config.type == "decision_transformer":
        model = DecisionTransformer(
            state_dim=model_config.state_dim,
            action_dim=model_config.action_dim,
            max_length=model_config.max_length,
            hidden_size=model_config.hidden_size,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout,
            activation=model_config.activation,
        )
    elif model_config.type == "trajectory_transformer":
        model = TrajectoryTransformer(
            state_dim=model_config.state_dim,
            action_dim=model_config.action_dim,
            max_length=model_config.max_length,
            hidden_size=model_config.hidden_size,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout,
            activation=model_config.activation,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config.type}")
    
    return model


def collect_training_data(config: Config, env: gym.Env) -> List[Dict[str, np.ndarray]]:
    """Collect training data.
    
    Args:
        config: Configuration object.
        env: Environment.
        
    Returns:
        List of trajectory dictionaries.
    """
    data_config = config.cfg.data_collection
    env_config = config.cfg.env
    
    # Wrap environment
    wrapped_env = EnvironmentWrapper(
        env,
        normalize_observations=env_config.normalize_observations,
        normalize_rewards=env_config.normalize_rewards,
        clip_actions=env_config.clip_actions,
    )
    
    # Create data collector
    collector = DataCollector(
        wrapped_env,
        max_episode_length=env_config.max_episode_length,
        device=get_device(),
    )
    
    # Create expert model if needed
    expert_model = None
    if data_config.expert_ratio > 0:
        print("Training expert model...")
        expert_model = create_expert_model(
            wrapped_env,
            algorithm=data_config.expert_algorithm,
            total_timesteps=data_config.expert_timesteps,
            device=get_device(),
        )
        print("Expert model training completed!")
    
    # Collect trajectories
    print("Collecting training data...")
    if data_config.expert_ratio == 0:
        trajectories = collector.collect_random_trajectories(
            num_trajectories=data_config.num_trajectories,
            seeds=list(range(data_config.num_trajectories)),
        )
    elif data_config.expert_ratio == 1:
        trajectories = collector.collect_expert_trajectories(
            expert_model=expert_model,
            num_trajectories=data_config.num_trajectories,
            seeds=list(range(data_config.num_trajectories)),
        )
    else:
        trajectories = collector.collect_mixed_trajectories(
            expert_model=expert_model,
            num_trajectories=data_config.num_trajectories,
            expert_ratio=data_config.expert_ratio,
            seeds=list(range(data_config.num_trajectories)),
        )
    
    print(f"Collected {len(trajectories)} trajectories")
    return trajectories


def train_model(
    config: Config,
    model: nn.Module,
    trajectories: List[Dict[str, np.ndarray]],
    logger: Logger,
) -> nn.Module:
    """Train the transformer model.
    
    Args:
        config: Configuration object.
        model: Transformer model.
        trajectories: Training trajectories.
        logger: Logger for experiment tracking.
        
    Returns:
        Trained model.
    """
    training_config = config.cfg.training
    model_config = config.cfg.model
    
    # Create dataset
    train_dataset = TrajectoryDataset(
        trajectories,
        max_length=model_config.max_length,
        normalize_rewards=True,
    )
    
    # Split into train/validation
    train_size = int(0.8 * len(trajectories))
    val_size = len(trajectories) - train_size
    
    train_trajectories = trajectories[:train_size]
    val_trajectories = trajectories[train_size:]
    
    train_dataset = TrajectoryDataset(
        train_trajectories,
        max_length=model_config.max_length,
        normalize_rewards=True,
    )
    
    val_dataset = TrajectoryDataset(
        val_trajectories,
        max_length=model_config.max_length,
        normalize_rewards=True,
    )
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        num_epochs=training_config.num_epochs,
        device=get_device(),
        logger=logger,
        save_dir=config.cfg.paths.checkpoint_dir,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    return model


def evaluate_model(
    config: Config,
    model: nn.Module,
    env: gym.Env,
) -> Dict[str, float]:
    """Evaluate the trained model.
    
    Args:
        config: Configuration object.
        model: Trained model.
        env: Environment.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    eval_config = config.cfg.evaluation
    env_config = config.cfg.env
    
    # Wrap environment
    wrapped_env = EnvironmentWrapper(
        env,
        normalize_observations=env_config.normalize_observations,
        normalize_rewards=env_config.normalize_rewards,
        clip_actions=env_config.clip_actions,
    )
    
    # Create evaluator
    evaluator = TransformerEvaluator(
        model=model,
        env=wrapped_env,
        device=get_device(),
        max_episode_length=env_config.max_episode_length,
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluator.evaluate(
        num_episodes=eval_config.num_episodes,
        seeds=eval_config.eval_seeds,
        target_return=eval_config.target_return,
    )
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Success Rate: {metrics['success_rate']:.2%}")
    print(f"  Mean Episode Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Transformers for RL")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--env", type=str, help="Environment name (overrides config)")
    parser.add_argument("--model", type=str, help="Model type (overrides config)")
    parser.add_argument("--epochs", type=int, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--seed", type=int, help="Random seed (overrides config)")
    parser.add_argument("--device", type=str, help="Device (overrides config)")
    parser.add_argument("--experiment-name", type=str, help="Experiment name (overrides config)")
    parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--render", action="store_true", help="Render during evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Override config with command line arguments
    if args.env:
        config.set("env.name", args.env)
    if args.model:
        config.set("model.type", args.model)
    if args.epochs:
        config.set("training.num_epochs", args.epochs)
    if args.batch_size:
        config.set("training.batch_size", args.batch_size)
    if args.learning_rate:
        config.set("training.learning_rate", args.learning_rate)
    if args.seed:
        config.set("device.seed", args.seed)
    if args.device:
        config.set("device.device", args.device)
    if args.experiment_name:
        config.set("logging.experiment_name", args.experiment_name)
    if args.use_wandb:
        config.set("logging.use_wandb", True)
    if args.render:
        config.set("evaluation.render", True)
    
    # Set random seed
    seed = config.get("device.seed", 42)
    set_seed(seed)
    
    # Create environment
    env_name = config.get("env.name", "CartPole-v1")
    env = gym.make(env_name)
    
    # Update model configuration with environment info
    config.set("model.state_dim", env.observation_space.shape[0])
    config.set("model.action_dim", env.action_space.n)
    
    # Create logger
    logger = Logger(
        log_dir=config.get("logging.log_dir", "logs"),
        experiment_name=config.get("logging.experiment_name"),
        use_tensorboard=config.get("logging.use_tensorboard", True),
        use_wandb=config.get("logging.use_wandb", False),
        wandb_project=config.get("logging.wandb_project", "transformers-for-rl"),
    )
    
    # Log configuration
    logger.logger.info(f"Configuration: {config.to_dict()}")
    
    # Create model
    model = create_model(config)
    logger.logger.info(f"Created {config.get('model.type')} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Collect training data
    trajectories = collect_training_data(config, env)
    
    # Train model
    trained_model = train_model(config, model, trajectories, logger)
    
    # Evaluate model
    metrics = evaluate_model(config, trained_model, env)
    
    # Log final metrics
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.log_scalar(f"final_{name}", value, 0)
    
    # Save final model
    final_checkpoint = logger.save_checkpoint(
        model=trained_model,
        optimizer=torch.optim.AdamW(trained_model.parameters()),
        epoch=config.get("training.num_epochs", 100),
        metrics=metrics,
        filename="final_model.pt",
    )
    
    logger.logger.info(f"Training completed! Final model saved to {final_checkpoint}")
    logger.close()
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
