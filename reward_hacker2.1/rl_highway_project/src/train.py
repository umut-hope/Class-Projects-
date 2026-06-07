"""
Main training script for RL Highway project.

Initializes highway-env, applies custom reward wrapping, and trains
a Stable-Baselines3 agent (PPO or DQN) with milestone checkpointing.

Usage:
    python train.py
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

from config import ProjectConfig, EnvConfig, HyperParameters
from utils import RewardWrapper, setup_logging, create_callback_list


logger = logging.getLogger(__name__)


class HighwayTrainer:
    """
    Main trainer class for highway-env with Stable-Baselines3.

    Handles environment initialization, wrapping, agent creation, and training.
    """

    def __init__(self, config: ProjectConfig) -> None:
        """
        Initialize the trainer with configuration.

        Args:
            config: ProjectConfig instance containing env and hyperparameter settings
        """
        self.config = config
        self.env: Optional[gym.Env] = None
        self.model = None
        self.logger = logging.getLogger(__name__)

    def _create_env(self) -> gym.Env:
        """
        Create and configure the highway environment.

        Applies custom reward wrapper and sets render mode.

        Returns:
            Configured gymnasium environment
        """
        self.logger.info(f"Initializing environment: {self.config.env_config.env_id}")

        try:
            # Create base environment
            env = gym.make(
                self.config.env_config.env_id,
                render_mode=self.config.env_config.render_mode,
            )

            # Register highway-env configuration
            env.unwrapped.configure(self.config.env_config.to_dict())

            # Apply custom reward wrapper
            env = RewardWrapper(
                env,
                speed_reward_coef=self.config.env_config.high_speed_reward,
                collision_penalty=self.config.env_config.collision_reward,
                lane_change_penalty=self.config.env_config.lane_change_reward,
                right_lane_bonus=self.config.env_config.right_lane_reward,
                speed_range=self.config.env_config.reward_speed_range,
            )

            self.logger.info(
                f"✓ Environment initialized | "
                f"Observation: {self.config.env_config.observation_type} | "
                f"Action: {self.config.env_config.action_type}"
            )
            return env

        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}")
            raise

    def _create_agent(self, env: gym.Env) -> object:
        """
        Create a Stable-Baselines3 agent based on configuration.

        Args:
            env: Gymnasium environment

        Returns:
            Initialized SB3 agent (PPO or DQN)
        """
        algorithm = self.config.hyperparameters.algorithm
        hparams = self.config.hyperparameters

        self.logger.info(f"Creating {algorithm} agent with custom hyperparameters")

        try:
            if algorithm == "PPO":
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    verbose=self.config.verbose,
                    tensorboard_log=hparams.tensorboard_log_dir,
                    seed=self.config.seed,
                    **hparams.to_dict(),
                )
            elif algorithm == "DQN":
                model = DQN(
                    policy="MlpPolicy",
                    env=env,
                    verbose=self.config.verbose,
                    tensorboard_log=hparams.tensorboard_log_dir,
                    seed=self.config.seed,
                    **hparams.to_dict(),
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            self.logger.info(
                f"✓ {algorithm} agent created | "
                f"LR: {hparams.learning_rate} | "
                f"Batch size: {hparams.batch_size} | "
                f"Device: {hparams.device}"
            )
            return model

        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            raise

    def train(self) -> None:
        """
        Execute the main training loop.

        - Initializes environment and agent
        - Runs training with milestone checkpointing
        - Handles interruptions gracefully
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("Starting RL Highway Training")
            self.logger.info("=" * 60)

            # Create environment and agent
            self.env = self._create_env()
            self.model = self._create_agent(self.env)

            # Create callbacks
            hparams = self.config.hyperparameters
            callbacks = create_callback_list(
                checkpoint_dir=hparams.checkpoint_dir,
                log_dir=hparams.tensorboard_log_dir,
                total_timesteps=hparams.total_timesteps,
                log_interval=hparams.log_interval,
                verbose=self.config.verbose,
            )

            self.logger.info(
                f"Training parameters: "
                f"Total timesteps: {hparams.total_timesteps:,} | "
                f"Algorithm: {hparams.algorithm}"
            )

            # Train the model
            self.logger.info("Starting training loop...")
            self.model.learn(
                total_timesteps=hparams.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )

            self.logger.info("=" * 60)
            self.logger.info("✓ Training completed successfully!")
            self.logger.info("=" * 60)

            # Save final model
            final_path = Path(hparams.checkpoint_dir) / "model_final"
            self.model.save(str(final_path))
            self.logger.info(f"Final model saved to {final_path}")

        except KeyboardInterrupt:
            self.logger.warning("\n⚠ Training interrupted by user")
            if self.model is not None:
                emergency_path = Path(self.config.hyperparameters.checkpoint_dir) / "model_emergency_save"
                self.model.save(str(emergency_path))
                self.logger.info(f"Emergency checkpoint saved to {emergency_path}")
            sys.exit(0)

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
            raise

        finally:
            # Clean up
            if self.env is not None:
                self.env.close()
                self.logger.info("Environment closed")


def main() -> None:
    """
    Main entry point for the training script.

    Sets up logging, loads configuration, and starts training.
    """
    # Set up logging
    logger_instance = setup_logging(
        log_dir="logs",
        level=logging.INFO,
        verbose=True,
    )

    try:
        # Load configuration
        config = ProjectConfig()

        # Log configuration
        logger_instance.info(f"Configuration loaded:")
        logger_instance.info(f"  Environment: {config.env_config.env_id}")
        logger_instance.info(f"  Algorithm: {config.hyperparameters.algorithm}")
        logger_instance.info(f"  Total timesteps: {config.hyperparameters.total_timesteps:,}")
        logger_instance.info(f"  Seed: {config.seed}")

        # Create trainer and run
        trainer = HighwayTrainer(config)
        trainer.train()

    except Exception as e:
        logger_instance.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
