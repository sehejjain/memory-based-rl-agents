# import memory_gym
import json
import os

import gymnasium as gym

os.environ["DISPLAY"] = ":0"
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    Logger,
    TensorBoardOutputFormat,
    configure,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
    VecEnvWrapper,
)
from tqdm.notebook import tqdm

os.sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
import memory_maze
from shimmy.dm_control_compatibility import DmControlCompatibilityV0

from sb3 import helm as helm_lib
from sb3 import model as model_lib

# os.environ["MUJOCO_GL"] = "glfw"

env_name = "MortarMayhem"

if env_name in ["MortarMayhem", "MysteryPath", "SearingSpotlights"]:
    env = gym.make(
        id="memory_gym:" + env_name + "-v0",
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


# env = memory_maze.tasks.memory_maze_9x9()
# env = DmControlCompatibilityV0(env=env, render_mode=None)
# env = gym.make("memory_gym:MortarMayhem-v0")
print("Hello", env.observation_space)

model = helm_lib.HELMPPO(
    MultiInputActorCriticPolicy,
    env,
    verbose=2,
    # n_steps=total_timesteps,
    n_epochs=3,
    vf_coef=0.5,
    clip_range=0.2,
    batch_size=64,
    max_grad_norm=0.5,
    gae_lambda=0.95,
    gamma=0.99,
    # ent_coef=custom_entropy_schedule(1e-4, 1e-5, total_timesteps),
    seed=1234,
    tensorboard_log="/common/home/sj1030/Projects/memory-gym/sb3/tensorboard_logs/",
    clip_decay="none",
    _init_setup_model=False,
)

model.learn(total_timesteps=2048 * 2, progress_bar=True)
