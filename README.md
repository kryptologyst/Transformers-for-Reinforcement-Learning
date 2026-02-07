# Transformers for Reinforcement Learning

Research-ready implementation of transformer-based models for reinforcement learning, featuring Decision Transformers and Trajectory Transformers.

## ⚠️ Important Disclaimer

**This project is for research and educational purposes only. It is NOT intended for production control of real-world systems, especially in safety-critical domains such as robotics, healthcare, finance, or energy systems.**

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Transformers-for-Reinforcement-Learning.git
cd Transformers-for-Reinforcement-Learning

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```bash
# Train a Decision Transformer on CartPole
python scripts/train.py --env CartPole-v1 --model decision_transformer

# Train a Trajectory Transformer
python scripts/train.py --env CartPole-v1 --model trajectory_transformer

# Run the interactive demo
streamlit run demo/app.py
```

## Features

### Core Models

- **Decision Transformer**: Models actions conditioned on desired returns-to-go
- **Trajectory Transformer**: Models complete trajectories autoregressively
- **Modern Architecture**: Built with PyTorch 2.x and latest transformer components

### Key Capabilities

- **Offline RL**: Train on pre-collected trajectory data
- **Expert Data Collection**: Automatic expert policy training using stable-baselines3
- **Mixed Data**: Combine expert and random trajectories for robust training
- **Comprehensive Evaluation**: Multiple metrics including success rate, sample efficiency
- **Interactive Demo**: Streamlit-based visualization and testing interface

### Technical Features

- **Device Agnostic**: Automatic CUDA/MPS/CPU detection
- **Reproducible**: Deterministic seeding across all components
- **Configurable**: OmegaConf-based configuration management
- **Logging**: TensorBoard and Weights & Biases integration
- **Type Safe**: Full type hints and modern Python practices

## Project Structure

```
transformers-for-rl/
├── src/
│   ├── algorithms/          # RL algorithms and training logic
│   ├── models/              # Transformer model implementations
│   ├── envs/                # Environment utilities and data collection
│   ├── train/               # Training utilities and datasets
│   ├── eval/                # Evaluation framework
│   └── utils/                # Utility functions (device, config, logging)
├── configs/                 # Configuration files
├── scripts/                 # Training and evaluation scripts
├── demo/                    # Streamlit demo application
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks for analysis
├── assets/                  # Generated plots, videos, and results
└── data/                    # Training data and datasets
```

## 🔧 Configuration

The project uses OmegaConf for flexible configuration management. Key configuration options:

### Environment Settings
```yaml
env:
  name: "CartPole-v1"
  max_episode_length: 500
  normalize_observations: true
```

### Model Settings
```yaml
model:
  type: "decision_transformer"  # or "trajectory_transformer"
  hidden_size: 128
  n_heads: 8
  n_layers: 3
  max_length: 20
```

### Training Settings
```yaml
training:
  learning_rate: 1e-4
  batch_size: 64
  num_epochs: 100
```

## Supported Environments

- **CartPole-v1**: Classic control problem
- **MountainCar-v0**: Sparse reward environment
- **Acrobot-v1**: Underactuated system
- **Custom Environments**: Easy to add new environments

## Model Architectures

### Decision Transformer

The Decision Transformer models the RL problem as a sequence modeling task where actions are conditioned on desired returns-to-go:

```
Input: [state, action, reward, return-to-go] tokens
Output: Action predictions conditioned on target return
```

Key features:
- Return-to-go conditioning for goal-directed behavior
- Attention mechanism captures long-range dependencies
- Suitable for offline RL and goal-conditioned tasks

### Trajectory Transformer

The Trajectory Transformer models complete trajectories autoregressively:

```
Input: [state, action, reward] tokens
Output: Next state, action, and reward predictions
```

Key features:
- Autoregressive trajectory generation
- Can generate complete episodes from initial state
- Useful for trajectory planning and analysis

## Evaluation Metrics

The framework provides comprehensive evaluation including:

- **Learning Metrics**: Training loss, validation loss, convergence speed
- **Control Metrics**: Episode reward, success rate, episode length
- **Efficiency Metrics**: Sample efficiency, time-to-threshold
- **Robustness Metrics**: Reward variance, sensitivity to seeds
- **Comparison Metrics**: Performance vs baseline methods

## Interactive Demo

The Streamlit demo provides:

- **Model Selection**: Choose between Decision and Trajectory Transformers
- **Environment Testing**: Test on different environments
- **Parameter Tuning**: Adjust target returns and evaluation settings
- **Real-time Visualization**: See model performance and episode details
- **Model Inspection**: View architecture details and parameter counts

## Research Applications

This implementation is suitable for:

- **Offline RL Research**: Training on fixed datasets
- **Sequence Modeling**: Understanding transformer capabilities in RL
- **Goal-Conditioned RL**: Using return-to-go conditioning
- **Trajectory Analysis**: Studying decision-making patterns
- **Educational Purposes**: Learning transformer architectures

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ scripts/ demo/
ruff check src/ scripts/ demo/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## References

- **Decision Transformer**: Chen, L., et al. "Decision Transformer: Reinforcement Learning via Sequence Modeling." NeurIPS 2021.
- **Trajectory Transformer**: Janner, M., et al. "Offline Reinforcement Learning as One Big Sequence Modeling Problem." NeurIPS 2021.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the original Decision Transformer paper
- The PyTorch team for the excellent deep learning framework
- The Gymnasium team for the RL environment standard
- Stable Baselines3 for expert policy implementations

## Support

For questions, issues, or contributions, please:

1. Check existing issues and discussions
2. Create a new issue with detailed information
3. Follow the contribution guidelines

---

**Remember**: This is a research/educational project. Always ensure appropriate safety measures when applying RL techniques to real-world systems.
# Transformers-for-Reinforcement-Learning
