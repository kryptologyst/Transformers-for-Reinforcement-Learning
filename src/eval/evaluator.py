"""Evaluation utilities for transformer-based RL models."""

import numpy as np
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F

from ..models.decision_transformer import DecisionTransformer
from ..models.trajectory_transformer import TrajectoryTransformer
from ..utils.device import get_device, set_env_seed


class TransformerEvaluator:
    """Evaluator for transformer-based RL models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        env: gym.Env,
        device: Optional[torch.device] = None,
        max_episode_length: int = 1000,
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained transformer model.
            env: Environment for evaluation.
            device: Device to use for evaluation.
            max_episode_length: Maximum episode length.
        """
        self.model = model
        self.env = env
        self.device = device or get_device()
        self.max_episode_length = max_episode_length
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_episode(
        self,
        seed: Optional[int] = None,
        render: bool = False,
        target_return: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate a single episode.
        
        Args:
            seed: Random seed for the episode.
            render: Whether to render the episode.
            target_return: Target return for Decision Transformer.
            
        Returns:
            Dictionary of episode metrics.
        """
        if seed is not None:
            set_env_seed(self.env, seed)
        
        # Reset environment
        state, _ = self.env.reset(seed=seed)
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        # Initialize trajectory buffers
        states = [state]
        actions = []
        rewards = []
        
        while not done and episode_length < self.max_episode_length:
            # Get action from model
            if isinstance(self.model, DecisionTransformer):
                action = self._get_decision_transformer_action(
                    states, actions, rewards, target_return
                )
            elif isinstance(self.model, TrajectoryTransformer):
                action = self._get_trajectory_transformer_action(states, actions, rewards)
            else:
                raise ValueError(f"Unknown model type: {type(self.model)}")
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Update trajectory
            actions.append(action)
            rewards.append(reward)
            states.append(next_state)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            # Render if requested
            if render:
                self.env.render()
            
            # Update state
            state = next_state
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "success": episode_reward > 0,  # Simple success criterion
        }
    
    def _get_decision_transformer_action(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        target_return: Optional[float] = None,
    ) -> int:
        """Get action from Decision Transformer.
        
        Args:
            states: List of states.
            actions: List of actions.
            rewards: List of rewards.
            target_return: Target return.
            
        Returns:
            Action to take.
        """
        # Prepare inputs
        seq_len = len(states)
        max_len = self.model.max_length
        
        # Pad or truncate sequences
        if seq_len > max_len:
            states = states[-max_len:]
            actions = actions[-max_len:]
            rewards = rewards[-max_len:]
            seq_len = max_len
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(0).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(0).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # Compute returns-to-go
        if target_return is None:
            target_return = 100.0  # Default target return
        
        returns_to_go = torch.full((1, seq_len, 1), target_return, device=self.device)
        timesteps = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        # Get action prediction
        with torch.no_grad():
            action_logits, _ = self.model(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                returns_to_go,
                timesteps,
            )
            
            # Get the last action prediction
            last_action_logits = action_logits[0, -1, :]
            
            # Sample action
            if self.model.action_dim == 1:
                # Continuous action space
                action = torch.tanh(last_action_logits).item()
            else:
                # Discrete action space
                action_probs = F.softmax(last_action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
        
        return action
    
    def _get_trajectory_transformer_action(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
    ) -> int:
        """Get action from Trajectory Transformer.
        
        Args:
            states: List of states.
            actions: List of actions.
            rewards: List of rewards.
            
        Returns:
            Action to take.
        """
        # Prepare inputs
        seq_len = len(states)
        max_len = self.model.max_length
        
        # Pad or truncate sequences
        if seq_len > max_len:
            states = states[-max_len:]
            actions = actions[-max_len:]
            rewards = rewards[-max_len:]
            seq_len = max_len
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).unsqueeze(0).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(0).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(0).unsqueeze(-1).to(self.device)
        timesteps = torch.arange(seq_len, device=self.device).unsqueeze(0)
        
        # Get action prediction
        with torch.no_grad():
            _, action_logits, _ = self.model(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                timesteps,
            )
            
            # Get the last action prediction
            last_action_logits = action_logits[0, -1, :]
            
            # Sample action
            if self.model.action_dim == 1:
                # Continuous action space
                action = torch.tanh(last_action_logits).item()
            else:
                # Discrete action space
                action_probs = F.softmax(last_action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
        
        return action
    
    def evaluate(
        self,
        num_episodes: int = 10,
        seeds: Optional[List[int]] = None,
        target_return: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate the model over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate.
            seeds: List of seeds for each episode.
            target_return: Target return for Decision Transformer.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if seeds is None:
            seeds = [None] * num_episodes
        
        episode_metrics = []
        
        for i in range(num_episodes):
            metrics = self.evaluate_episode(
                seed=seeds[i],
                target_return=target_return,
            )
            episode_metrics.append(metrics)
        
        # Compute aggregate metrics
        episode_rewards = [m["episode_reward"] for m in episode_metrics]
        episode_lengths = [m["episode_length"] for m in episode_metrics]
        success_rate = np.mean([m["success"] for m in episode_metrics])
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": success_rate,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
        }
    
    def compare_with_baseline(
        self,
        baseline_model: torch.nn.Module,
        num_episodes: int = 10,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Compare transformer model with baseline.
        
        Args:
            baseline_model: Baseline model for comparison.
            num_episodes: Number of episodes to evaluate.
            seeds: List of seeds for each episode.
            
        Returns:
            Dictionary of comparison metrics.
        """
        # Evaluate transformer model
        transformer_metrics = self.evaluate(num_episodes, seeds)
        
        # Evaluate baseline model
        baseline_evaluator = TransformerEvaluator(
            baseline_model, self.env, self.device, self.max_episode_length
        )
        baseline_metrics = baseline_evaluator.evaluate(num_episodes, seeds)
        
        # Compute comparison metrics
        reward_improvement = (
            transformer_metrics["mean_reward"] - baseline_metrics["mean_reward"]
        )
        relative_improvement = (
            reward_improvement / baseline_metrics["mean_reward"] * 100
        )
        
        return {
            "transformer_mean_reward": transformer_metrics["mean_reward"],
            "baseline_mean_reward": baseline_metrics["mean_reward"],
            "reward_improvement": reward_improvement,
            "relative_improvement": relative_improvement,
            "transformer_success_rate": transformer_metrics["success_rate"],
            "baseline_success_rate": baseline_metrics["success_rate"],
        }
