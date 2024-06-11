import matplotlib.pyplot as plt
import numpy as np
import os, sys
import imageio
from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C, DDPG
from gymnasium.wrappers import FlattenObservation


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from envs.cont_env import ContinuousEnv
from envs.dict_env import ContinuousEnvDict


path = Path().resolve()
video_folder = str(path/"videos")
model_folder = path/"models"
video_length = 100

env = ContinuousEnvDict(render_mode="human", local_vis=False)
env = FlattenObservation(env)
model = A2C.load(str(model_folder/"CoverageEnvDict_A2C"))
obs, info = env.reset()

alpha = 0.5
prev_action = None
for i in range(video_length+1):
    # action = env.action_space.sample()
    action, _ = model.predict(obs)
    if prev_action is None:
        prev_action = action
    filtered_action = alpha * action + (1 - alpha) * prev_action
    prev_action = filtered_action
    obs, reward, terminated, truncated, info = env.step(filtered_action)
    # print("Reward: ", reward)
    env.render()
    if terminated or truncated:
        break

print("Number of steps: ", i+1)
env.close()