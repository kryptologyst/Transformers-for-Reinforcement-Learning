"""Environment utilities and data collection for transformer-based RL."""

import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv

from ..utils.device import get_device, set_seed, set_env_seed


class DataCollector:
    """Collect trajectory data from environments."""
    
    def __init__(
        self,
        env: gym.Env,
        max_episode_length: int = 1000,
        device: Optional[torch.device] = None,
    ):
        """Initialize data collector.
        
        Args:
            env: Environment to collect data from.
            max_episode_length: Maximum episode length.
            device: Device for computation.
        """
        self.env = env
        self.max_episode_length = max_episode_length
        self.device = device or get_device()
    
    def collect_random_trajectories(
        self,
        num_trajectories: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """Collect random trajectories.
        
        Args:
            num_trajectories: Number of trajectories to collect.
            seeds: List of seeds for each trajectory.
            
        Returns:
            List of trajectory dictionaries.
        """
        trajectories = []
        
        for i in range(num_trajectories):
            seed = seeds[i] if seeds else None
            trajectory = self._collect_single_trajectory(seed=seed, random_policy=True)
            trajectories.append(trajectory)
        
        return trajectories
    
    def collect_expert_trajectories(
        self,
        expert_model: torch.nn.Module,
        num_trajectories: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """Collect expert trajectories.
        
        Args:
            expert_model: Expert model for data collection.
            num_trajectories: Number of trajectories to collect.
            seeds: List of seeds for each trajectory.
            
        Returns:
            List of trajectory dictionaries.
        """
        trajectories = []
        
        for i in range(num_trajectories):
            seed = seeds[i] if seeds else None
            trajectory = self._collect_single_trajectory(
                seed=seed, expert_model=expert_model
            )
            trajectories.append(trajectory)
        
        return trajectories
    
    def collect_mixed_trajectories(
        self,
        expert_model: torch.nn.Module,
        num_trajectories: int = 100,
        expert_ratio: float = 0.5,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, np.ndarray]]:
        """Collect mixed trajectories (expert + random).
        
        Args:
            expert_model: Expert model for data collection.
            num_trajectories: Number of trajectories to collect.
            expert_ratio: Ratio of expert trajectories.
            seeds: List of seeds for each trajectory.
            
        Returns:
            List of trajectory dictionaries.
        """
        trajectories = []
        num_expert = int(num_trajectories * expert_ratio)
        num_random = num_trajectories - num_expert
        
        # Collect expert trajectories
        for i in range(num_expert):
            seed = seeds[i] if seeds else None
            trajectory = self._collect_single_trajectory(
                seed=seed, expert_model=expert_model
            )
            trajectories.append(trajectory)
        
        # Collect random trajectories
        for i in range(num_random):
            seed = seeds[num_expert + i] if seeds else None
            trajectory = self._collect_single_trajectory(seed=seed, random_policy=True)
            trajectories.append(trajectory)
        
        # Shuffle trajectories
        random.shuffle(trajectories)
        
        return trajectories
    
    def _collect_single_trajectory(
        self,
        seed: Optional[int] = None,
        expert_model: Optional[torch.nn.Module] = None,
        random_policy: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Collect a single trajectory.
        
        Args:
            seed: Random seed.
            expert_model: Expert model for action selection.
            random_policy: Whether to use random policy.
            
        Returns:
            Dictionary containing trajectory data.
        """
        if seed is not None:
            set_env_seed(self.env, seed)
        
        # Reset environment
        state, _ = self.env.reset(seed=seed)
        
        # Initialize trajectory buffers
        states = [state]
        actions = []
        rewards = []
        dones = []
        
        episode_length = 0
        done = False
        
        while not done and episode_length < self.max_episode_length:
            # Select action
            if random_policy:
                action = self.env.action_space.sample()
            elif expert_model is not None:
                action = self._get_expert_action(expert_model, state)
            else:
                raise ValueError("Either expert_model or random_policy must be specified")
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            states.append(next_state)
            
            # Update state
            state = next_state
            episode_length += 1
        
        # Convert to numpy arrays
        trajectory = {
            "states": np.array(states[:-1]),  # Exclude last state
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "next_states": np.array(states[1:]),  # Exclude first state
        }
        
        return trajectory
    
    def _get_expert_action(self, expert_model: torch.nn.Module, state: np.ndarray) -> int:
        """Get action from expert model.
        
        Args:
            expert_model: Expert model.
            state: Current state.
            
        Returns:
            Action to take.
        """
        expert_model.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if hasattr(expert_model, "predict"):
                # Stable Baselines3 model
                action, _ = expert_model.predict(state_tensor.cpu().numpy(), deterministic=True)
                return action[0]
            else:
                # Custom PyTorch model
                if hasattr(expert_model, "forward"):
                    output = expert_model(state_tensor)
                    if output.dim() > 1:
                        output = output.squeeze(0)
                    
                    if expert_model.action_dim == 1:
                        # Continuous action space
                        action = torch.tanh(output).item()
                    else:
                        # Discrete action space
                        action_probs = torch.softmax(output, dim=-1)
                        action = torch.multinomial(action_probs, 1).item()
                    
                    return action
                else:
                    raise ValueError("Expert model must have predict or forward method")


class EnvironmentWrapper:
    """Wrapper for environment preprocessing."""
    
    def __init__(
        self,
        env: gym.Env,
        normalize_observations: bool = True,
        normalize_rewards: bool = False,
        clip_actions: bool = True,
    ):
        """Initialize environment wrapper.
        
        Args:
            env: Environment to wrap.
            normalize_observations: Whether to normalize observations.
            normalize_rewards: Whether to normalize rewards.
            clip_actions: Whether to clip actions.
        """
        self.env = env
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.clip_actions = clip_actions
        
        # Initialize running statistics
        if normalize_observations:
            self.obs_mean = np.zeros(env.observation_space.shape)
            self.obs_std = np.ones(env.observation_space.shape)
            self.obs_count = 0
        
        if normalize_rewards:
            self.reward_mean = 0.0
            self.reward_std = 1.0
            self.reward_count = 0
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info
    
    def step(self, action):
        """Take environment step."""
        if self.clip_actions:
            action = self._clip_action(action)
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update running statistics
        if self.normalize_observations:
            self._update_obs_stats(obs)
        
        if self.normalize_rewards:
            self._update_reward_stats(reward)
        
        return (
            self._process_observation(obs),
            self._process_reward(reward),
            terminated,
            truncated,
            info,
        )
    
    def _process_observation(self, obs: np.ndarray) -> np.ndarray:
        """Process observation."""
        if self.normalize_observations:
            return (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return obs
    
    def _process_reward(self, reward: float) -> float:
        """Process reward."""
        if self.normalize_rewards:
            return (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return reward
    
    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to valid range."""
        if hasattr(self.env.action_space, "low") and hasattr(self.env.action_space, "high"):
            return np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action
    
    def _update_obs_stats(self, obs: np.ndarray) -> None:
        """Update observation statistics."""
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        delta2 = obs - self.obs_mean
        self.obs_std += (delta * delta2 - self.obs_std) / self.obs_count
    
    def _update_reward_stats(self, reward: float) -> None:
        """Update reward statistics."""
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_std += (delta * delta2 - self.reward_std) / self.reward_count
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped environment."""
        return getattr(self.env, name)


def create_expert_model(
    env: gym.Env,
    algorithm: str = "PPO",
    total_timesteps: int = 100000,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Create an expert model using stable-baselines3.
    
    Args:
        env: Environment to train on.
        algorithm: Algorithm to use ("PPO" or "DQN").
        total_timesteps: Number of training timesteps.
        device: Device for training.
        
    Returns:
        Trained expert model.
    """
    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, device=device)
    elif algorithm == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, device=device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train the model
    model.learn(total_timesteps=total_timesteps)
    
    return model
