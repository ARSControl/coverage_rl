import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import imageio
from pathlib import Path

# custom env
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from envs.grid_world import GridWorldEnv
from envs.coverage_grid_env import CoverageGridEnv

path = Path().resolve()
video_folder = str(path/"videos")
video_length = 100

# Create env
num_envs = 4
env = CoverageGridEnv(size=5)
print("Action space shape: ", env.action_space)
print("Observation space shape: ", env.observation_space)
print("Action sample: ", env.action_space.sample())
print("Observation sample: ", env.observation_space.sample())