"""
Configuration module for RL Highway environment and training hyperparameters.

Uses dataclasses for clean, type-safe configuration management.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class EnvConfig:
    """Environment configuration for highway-env."""

    # Environment identifier
    env_id: str = "highway-v0"

    # Observation configuration
    observation_type: str = "Kinematics"  # 'Kinematics' or 'OccupancyGrid'
    observation_vehicles: int = 5  # Number of vehicles in observation
    observation_features: int = 4  # Features per vehicle: [x, y, vx, vy]

    # Action configuration
    action_type: str = "DiscreteMetaAction"  # 'DiscreteMetaAction' or 'ContinuousAction'
    action_lateral_actions: int = 3  # Number of possible lane changes (0=IDLE, 1=LEFT, 2=RIGHT)

    # Reward configuration
    reward_speed_range: tuple = field(default_factory=lambda: (20, 40))  # m/s
    collision_reward: float = -1.0
    lane_change_reward: float = -0.05
    right_lane_reward: float = 0.0
    high_speed_reward: float = 0.4
    offroad_terminal_reward: float = -1.0

    # Vehicle dynamics
    vehicle_count: int = 20  # Number of other vehicles on road
    initial_lane_bias: int = 0  # Start in which lane (0=left)

    # Episode configuration
    episode_duration: int = 200  # Steps per episode
    render: bool = False  # Render environment
    render_mode: Optional[str] = None  # None, 'rgb_array', or 'human'

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for environment creation."""
        return {
            "observation": {
                "type": self.observation_type,
                "vehicles_count": self.observation_vehicles,
                "features_count": self.observation_features,
            },
            "action": {
                "type": self.action_type,
            },
            "reward_speed_range": self.reward_speed_range,
            "collision_reward": self.collision_reward,
            "lane_change_reward": self.lane_change_reward,
            "right_lane_reward": self.right_lane_reward,
            "high_speed_reward": self.high_speed_reward,
            "offroad_terminal_reward": self.offroad_terminal_reward,
            "vehicles_count": self.vehicle_count,
            "initial_lane_bias": self.initial_lane_bias,
            "duration": self.episode_duration,
        }


@dataclass
class HyperParameters:
    """Training hyperparameters for stable-baselines3 agents."""

    # Algorithm selection
    algorithm: str = "PPO"  # 'PPO' or 'DQN'

    # Training parameters (optimized for laptop CPU)
    total_timesteps: int = 50_000  # Total training steps
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_steps: int = 512  # For PPO: steps per epoch

    # PPO-specific parameters
    n_epochs: int = 10  # PPO epochs per update
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_range: float = 0.2  # PPO clipping range
    ent_coef: float = 0.01  # Entropy coefficient

    # DQN-specific parameters
    exploration_fraction: float = 0.1  # Fraction of total_timesteps for exploration
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    target_update_interval: int = 10_000
    learning_starts: int = 1_000

    # General RL parameters
    max_grad_norm: float = 0.5
    use_sde: bool = False  # Use state-dependent exploration noise

    # Checkpoint and logging
    save_interval: int = 5_000  # Save checkpoint every N steps
    log_interval: int = 1_000  # Log metrics every N steps
    checkpoint_dir: str = "checkpoints"
    tensorboard_log_dir: Optional[str] = "logs"  # Set to None to disable tensorboard

    # Device configuration
    device: str = "cpu"  # 'cpu' or 'cuda'
    n_envs: int = 1  # Number of parallel environments (1 for laptop)

    def to_dict(self) -> Dict[str, Any]:
        """Convert hyperparameters to dictionary for agent initialization."""
        if self.algorithm == "PPO":
            return {
                "learning_rate": self.learning_rate,
                "n_steps": self.n_steps,
                "batch_size": self.batch_size,
                "n_epochs": self.n_epochs,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_range": self.clip_range,
                "ent_coef": self.ent_coef,
                "max_grad_norm": self.max_grad_norm,
                "use_sde": self.use_sde,
                "device": self.device,
            }
        elif self.algorithm == "DQN":
            return {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "gamma": self.gamma,
                "exploration_fraction": self.exploration_fraction,
                "exploration_initial_eps": self.exploration_initial_eps,
                "exploration_final_eps": self.exploration_final_eps,
                "target_update_interval": self.target_update_interval,
                "learning_starts": self.learning_starts,
                "max_grad_norm": self.max_grad_norm,
                "device": self.device,
            }
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")


@dataclass
class ProjectConfig:
    """Top-level project configuration combining environment and training settings."""

    env_config: EnvConfig = field(default_factory=EnvConfig)
    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    seed: int = 42
    verbose: int = 1  # 0=no output, 1=standard, 2=debug
