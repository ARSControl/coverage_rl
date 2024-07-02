import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
from stable_baselines3 import PPO, DQN, A2C, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
# import imageio
from pathlib import Path
import matplotlib.pyplot as plt


# custom env
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from envs.grid_world import GridWorldEnv
from envs.coverage_grid_env import CoverageGridEnv
from envs.cont_env import ContinuousEnv
from envs.dict_env import ContinuousEnvDict
# from envs.cooperative_env import CooperativeEnv
from envs.multi_env import MultiEnv
from envs.multiobs_env import MultiObsEnv
from envs.singleobs_env import SingleObsEnv
from envs.grid_mates_env import GridMatesEnv
from envs.centroid_env import CentroidEnv

path = Path().resolve()
video_folder = str(path/"videos")
model_folder = path/"models"
video_length = 100


# Create env
num_envs = 4
env = CentroidEnv()
# env = make_vec_env(CooperativeEnv, n_envs=num_envs)
# env = FlattenObservation(env)
print("Action space shape: ", env.action_space)
print("Observation space shape: ", env.observation_space)


obs, info = env.reset()
# print("Obs shape: ", obs.shape)
print("Observation: ", obs)

# Save a checkpoint every 10000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=50000,
  save_path=str(model_folder),
  name_prefix="temp",
  save_replay_buffer=False,
  save_vecnormalize=False,
)


# Train agent
policy_kwargs = {"normalize_images": False}
model = PPO("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
total_timesteps = 3_000_000
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
model.save(str(model_folder/"MultiEnv_A2C"))
env.close()



env = SingleObsEnv(render_mode="human", local_vis=False)
# env = FlattenObservation(env)
obs, info = env.reset()


for i in range(video_length+1):
    actions = env.action_space.sample()
    # action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(actions)
    print("Reward: ", reward)
    env.render()
    if terminated or truncated:
        break

print("Number of steps: ", i+1)
env.close()
