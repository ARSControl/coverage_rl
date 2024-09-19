import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
from stable_baselines3 import PPO, DQN, A2C, DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import preprocessing
# import imageio
from pathlib import Path
import matplotlib.pyplot as plt


# custom env
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from envs.grid_world import GridWorldEnv
from envs.simple_env import SimpleEnv


path = Path().resolve()
video_folder = str(path/"videos")
model_folder = path/"models"
video_length = 100

TRAIN = True
EVAL = True


if TRAIN:
  # Create env
  num_envs = 4
  # env = LocalEnv()
  env = make_vec_env(SimpleEnv, n_envs=num_envs)
  # env = FlattenObservation(env)
  print("Environment: UniformEnv")
  print("Action space shape: ", env.action_space)
  # print("Observation space: ", env.observation_space.items())
  # for key, subspace in env.observation_space.spaces.items():
  #   print("Key: ", key)
  #   print("Subspace. ", subspace)
  #   print("Image? ", preprocessing.is_image_space(subspace))

  # obs, info = env.reset()
  # print("Obs shape: ", obs.shape)
  # print("Observation: ", obs)

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
  model = PPO("MlpPolicy", env, verbose=1)
  # model = PPO.load(str(model_folder/"LocalEnv_DQN_15M"))
  # model.set_env(env)
  # print("Model:" , model)
  total_timesteps = 3_000_000
  model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
  model.save(str(model_folder/"SimpleEnv_obs_t_PPO_3M"))
  env.close()



if EVAL:
  env = SimpleEnv(render_mode="human", local_vis=True)
  # env = FlattenObservation(env)
  obs, info = env.reset()

  # model = PPO("MlpPolicy", env, verbose=1)
  # model = PPO.load(str(model_folder/"UniformEnv_PPO_15M"))
  # model.set_env(env)


  for i in range(video_length+1):
      actions = env.action_space.sample()
      # actions, _ = model.predict(obs)
      obs, reward, terminated, truncated, info = env.step(actions)
      print("Reward: ", reward)
      print("obs shape: ", obs.shape)
      env.render()
      if terminated or truncated:
          break

  print("Number of steps: ", i+1)
  env.close()
