"""
Utility module for RL training: logging, reward shaping, callbacks, and video recording.

Includes:
- Custom reward wrapper for balancing speed and collision penalties
- Callback functions for model checkpointing at training milestones
- Logging setup for training monitoring
"""

import logging
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class RewardWrapper(gym.Wrapper):
    """
    Ödül Şekillendirme - Agresif Sürüş Modeli
    Ajanın sürekli frene basmasını engellemek için düşük hızlarda şiddetli ceza verilir.
    """
    def __init__(
        self,
        env: gym.Env,
        speed_reward_coef: float = 2,
        collision_penalty: float = -1.0,     # Kaza yapmasın ama...
        lane_change_penalty: float = -0.05,
        right_lane_bonus: float = 0.1,
        slow_penalty: float = 0.0,          # YENİ: Yavaşlama cezası!
        speed_range: tuple = (20, 40),
    ) -> None:
        super().__init__(env)
        self.speed_reward_coef = speed_reward_coef
        self.collision_penalty = collision_penalty
        self.lane_change_penalty = lane_change_penalty
        self.right_lane_bonus = right_lane_bonus
        self.slow_penalty = slow_penalty
        self.speed_range = speed_range
        self.prev_action = None

    def step(self, action: int) -> tuple:
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped_reward = reward

        if obs.size > 0:
            ego_vx = obs[0, 2] if obs.ndim > 1 else 0.0
            ego_speed = float(np.sqrt(ego_vx**2)) 

            # KESİN ÇÖZÜM: Hız 15 m/s'nin (54 km/h) altındaysa ajana her adımda işkence et!
            if ego_speed < 15.0:
                shaped_reward += self.slow_penalty

            # Normal hız ödülü
            min_speed, max_speed = self.speed_range
            normalized_speed = np.clip(
                (ego_speed - min_speed) / (max_speed - min_speed), 0, 1
            )
            speed_reward = self.speed_reward_coef * normalized_speed
            shaped_reward += speed_reward

        if info.get("crashed", False):
            shaped_reward += self.collision_penalty

        if self.prev_action is not None and action != self.prev_action:
            shaped_reward += self.lane_change_penalty

        lane = info.get("lane", 0)
        if lane > 0:
            shaped_reward += self.right_lane_bonus

        self.prev_action = action
        return obs, shaped_reward, terminated, truncated, info
    """
    Custom reward wrapper for highway-env that balances speed and penalizes collisions.

    Reward shaping strategy:
    - High positive reward for maintaining high speed in safe conditions
    - Severe negative reward for collisions
    - Minor penalty for lane changes to encourage smooth driving
    - Bonus for staying in right lanes (cooperative behavior)
    """

    def __init__(
        self,
        env: gym.Env,
        speed_reward_coef: float = 1.2,
        collision_penalty: float = -1.0,
        lane_change_penalty: float = 0.0,
        right_lane_bonus: float = 0.1,
        speed_range: tuple = (20, 40),
    ) -> None:
        """
        Initialize the reward wrapper.

        Args:
            env: Base gymnasium environment
            speed_reward_coef: Coefficient for speed-based reward (0-1)
            collision_penalty: Reward penalty for collision
            lane_change_penalty: Penalty for changing lanes
            right_lane_bonus: Bonus for staying in right lane
            speed_range: (min_speed, max_speed) tuple for normalization
        """
        super().__init__(env)
        self.speed_reward_coef = speed_reward_coef
        self.collision_penalty = collision_penalty
        self.lane_change_penalty = lane_change_penalty
        self.right_lane_bonus = right_lane_bonus
        self.speed_range = speed_range
        self.prev_action: Optional[int] = None

    def step(self, action: int) -> tuple:
        """
        Step environment and apply custom reward shaping.

        Args:
            action: Action index from the agent

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Shaped reward starts with base reward
        shaped_reward = reward

        # Extract vehicle state from observation (Kinematics format)
        # obs shape: (num_vehicles, features) where features are [x, y, vx, vy]
        if obs.size > 0:
            ego_vx = obs[0, 2] if obs.ndim > 1 else 0.0
            ego_speed = float(np.sqrt(ego_vx**2))  # Speed in x direction

            # Speed-based reward: normalize to [0, 1] and scale
            min_speed, max_speed = self.speed_range
            normalized_speed = np.clip(
                (ego_speed - min_speed) / (max_speed - min_speed), 0, 1
            )
            speed_reward = self.speed_reward_coef * normalized_speed
            shaped_reward += speed_reward

        # Collision detection
        if info.get("crashed", False):
            shaped_reward += self.collision_penalty

        # Lane change penalty (detect if action changed from previous)
        if self.prev_action is not None and action != self.prev_action:
            shaped_reward += self.lane_change_penalty

        # Right lane bonus (lanes typically numbered 0=left, higher=right)
        lane = info.get("lane", 0)
        if lane > 0:  # Not in leftmost lane
            shaped_reward += self.right_lane_bonus

        self.prev_action = action

        return obs, shaped_reward, terminated, truncated, info


class MilestoneCheckpointCallback(BaseCallback):
    """
    Callback to save model at training milestones: 0%, 50%, and 100%.

    Useful for monitoring training progress and recovery from failures.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        total_timesteps: int = 50_000,
        verbose: int = 1,
    ) -> None:
        """
        Initialize the callback.

        Args:
            save_dir: Directory to save checkpoints
            total_timesteps: Total training timesteps (used to calculate milestones)
            verbose: Verbosity level for logging
        """
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.total_timesteps = total_timesteps
        self.milestones = {
            0: "start",
            total_timesteps // 2: "halfway",
            total_timesteps: "final",
        }
        self.saved_milestones: set = set()
        self._logger = logging.getLogger(__name__)

    def _on_step(self) -> bool:
        """
        Called at each training step. Saves model at milestones.

        Returns:
            bool: Whether to continue training
        """
        current_step = self.model.num_timesteps

        for milestone_step, label in self.milestones.items():
            # Check if we've reached a milestone and haven't saved it yet
            if milestone_step not in self.saved_milestones and current_step >= milestone_step:
                checkpoint_path = self.save_dir / f"model_{label}"
                self.model.save(str(checkpoint_path))
                self.saved_milestones.add(milestone_step)

                if self.verbose >= 1:
                    progress_pct = (current_step / self.total_timesteps) * 100
                    self._logger.info(
                        f"✓ Milestone '{label}' ({progress_pct:.1f}%): "
                        f"Saved to {checkpoint_path}"
                    )

        return True


class TrainingProgressCallback(BaseCallback):
    """
    Callback for logging training progress and statistics.

    Logs episode rewards, episode lengths, and mean values periodically.
    """

    def __init__(self, log_interval: int = 1_000, verbose: int = 1) -> None:
        """
        Initialize the callback.

        Args:
            log_interval: Log statistics every N timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_interval = log_interval
        self._logger = logging.getLogger(__name__)
        self.episode_count = 0

    def _on_step(self) -> bool:
        """Called at each training step."""
        # Log every log_interval steps
        if self.model.num_timesteps % self.log_interval == 0:
            if self.verbose >= 1:
                fps = int(self.model.num_timesteps / (self.model._total_timesteps / 1000)) if hasattr(self.model, '_total_timesteps') else 0
                self._logger.info(
                    f"Timestep: {self.model.num_timesteps:,} | "
                    f"Episodes: {self.episode_count} | "
                    f"FPS: {fps}"
                )

        return True


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    verbose: bool = True,
) -> logging.Logger:
    """
    Set up structured logging for the training process.

    Args:
        log_dir: Directory to save log files
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        verbose: Whether to log to console as well as file

    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("rl_highway")
    logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path / "training.log")
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level if verbose else logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def create_callback_list(
    checkpoint_dir: str = "checkpoints",
    log_dir: str = "logs",
    total_timesteps: int = 50_000,
    log_interval: int = 1_000,
    verbose: int = 1,
) -> list:
    """
    Create a list of callbacks for training.

    Args:
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for training logs
        total_timesteps: Total training timesteps
        log_interval: Logging interval
        verbose: Verbosity level

    Returns:
        List of callback instances
    """
    callbacks = [
        MilestoneCheckpointCallback(
            save_dir=checkpoint_dir,
            total_timesteps=total_timesteps,
            verbose=verbose,
        ),
        TrainingProgressCallback(log_interval=log_interval, verbose=verbose),
    ]
    return callbacks
