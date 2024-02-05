"""
Executable file for training the PPO agent.
"""

import warnings

warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys

import gymnasium as gym
import memory_gym
import memory_maze
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

# os.environ["DISPLAY"] = ":0"
os.sys.path.append(os.path.abspath(path=os.path.join(os.getcwd(), os.pardir)))  # type: ignore


import gpustat
import memory_maze
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

gpustat.print_gpustat()

env_name = sys.argv[1]


if env_name in ["MortarMayhem", "MysteryPath", "SearingSpotlights"]:
    env = gym.make(
        id=env_name + "-v0",
    )

    eval_env = Monitor(
        gym.make(
            env_name,
        )
    )

elif env_name == "MemoryMaze":
    env = memory_maze.tasks.memory_maze_9x9()
    env = DmControlCompatibilityV0(env=env, render_mode=None)

    eval_env = memory_maze.tasks.memory_maze_9x9()
    eval_env = DmControlCompatibilityV0(env=eval_env, render_mode=None)
else:
    raise ValueError("Invalid environment name")
    exit(1)


# # env = gym.make('MortarMayhem-v0', )
# env_func = lambda: gym.make('MortarMayhem-v0', )
# # envs = [gym.make('MortarMayhem-v0') for _ in range(4)]
# env = VecNormalize(DummyVecEnv([env_func]*4))

# # json.loads(env.spec.to_json())


class EpisodicRewardCallback(BaseCallback):
    def __init__(self, verbose=0, plot_freq=10, log_dir="./tensorboard_logs/"):
        super(EpisodicRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.plot_freq = plot_freq
        self.log_dir = log_dir
        self.ep_rewards = 0.0

    def on_rollout_start(self) -> None:
        # print("starting rollout")
        self.ep_rewards = 0.0

    def _on_step(self):
        self.ep_rewards += self.locals["rewards"][-1]

    def _on_rollout_end(self) -> None:
        # print("ending rollout")
        episode_reward = self.ep_rewards
        self.episode_rewards.append(episode_reward)
        # print(f"Episode reward: {episode_reward}")
        with open("rewards.npy", "wb") as f:
            np.save(file=f, arr=np.array(self.episode_rewards))

    # def _on_training_end(self) -> None:
    #     self._plot_rewards()

    # def _plot_rewards(self):
    #     x = np.arange(1, len(self.episode_rewards) + 1)
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(x, self.episode_rewards)
    #     plt.xlabel('Episode')
    #     plt.ylabel('Total Reward')
    #     plt.title('Episodic Rewards')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(self.log_dir, 'episodic_rewards.png'))
    #     plt.show()


def linear_schedule(initial_lr: float, final_lr: float, total_steps: int):
    """
    Linear learning rate schedule from initial_lr to final_lr over total_steps.

    :param initial_lr: Initial learning rate.
    :param final_lr: Final learning rate.
    :param total_steps: Total number of steps over which to adjust the learning rate.
    :return: A function that takes progress_remaining and returns the current learning rate.
    """

    def schedule(progress_remaining: float) -> float:
        """
        Linearly adjust the learning rate from initial_lr to final_lr.

        :param progress_remaining: Fraction of training remaining, decreases from 1 to 0.
        :return: Current learning rate.
        """
        # Calculate the current step based on the progress remaining
        current_step = total_steps * (1 - progress_remaining)
        fraction = current_step / total_steps
        # Linearly interpolate between initial_lr and final_lr
        return initial_lr + fraction * (final_lr - initial_lr)

    return schedule


def custom_entropy_schedule(initial_entropy_coef, final_entropy_coef, total_steps):
    def entropy_coef_schedule(progress):
        fraction = min(progress / total_steps, 1.0)
        return torch.tensor(
            initial_entropy_coef
            + fraction * (final_entropy_coef - initial_entropy_coef),
            dtype=torch.float32,
        )

    return entropy_coef_schedule


TOTAL_TIMESTEPS = 170_000_000


if env_name != "MemoryMaze":
    from sb3 import model as model_lib
else:
    from sb3 import model_vis_vec as model_lib

policy = model_lib.CustomGRUPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    lr_schedule=linear_schedule(
        initial_lr=3e-4, final_lr=1e-4, total_steps=TOTAL_TIMESTEPS
    ),
)  # custom policy network


def find_latest_checkpoint(checkpoint_path, name_prefix):
    checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith(name_prefix)]
    if not checkpoints:
        return None

    return os.path.join(
        checkpoint_path, max(checkpoints, key=lambda f: int(f.split("_")[-2]))
    )


# Find the latest checkpoint
latest_checkpoint = find_latest_checkpoint("sb3/models/PPO/saved/" + env_name, "rl_model")  # type: ignore


if latest_checkpoint:
    print(f"Loading model from {latest_checkpoint}")
    model = PPO.load(latest_checkpoint, env, verbose=0)
else:
    print("No checkpoint found. Creating a new model.")
    model = PPO(
        # "MultiInputPolicy" if env_name == "MemoryMaze" else "MlpPolicy",
        model_lib.CustomGRUPolicy,
        env,
        verbose=0,
        # n_steps=TOTAL_TIMESTEPS,
        n_epochs=3,
        vf_coef=0.5,
        clip_range=0.2,
        batch_size=2048,  # 16384,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        gamma=0.99,
        # ent_coef=custom_entropy_schedule(1e-4, 1e-5, TOTAL_TIMESTEPS),
        seed=1234,
        tensorboard_log="./tensorboard_logs/",
        learning_rate=linear_schedule(
            initial_lr=3e-4, final_lr=1e-4, total_steps=TOTAL_TIMESTEPS
        ),
    )


e_callback = EpisodicRewardCallback(plot_freq=10, log_dir="tensorboard_logs/test/")
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10_000_000,
    save_path="saved/" + env_name,
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="sb3/logs/" + env_name + "test",
    log_path="sb3/logs/" + env_name + "test",
    eval_freq=30000,
    deterministic=True,
    render=False,
)

callback = CallbackList([e_callback, checkpoint_callback, eval_callback])


model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True,
    callback=callback,
    tb_log_name=env_name + "_ppo" + "test",
    reset_num_timesteps=False,
)


# callback.plot_rewards()
