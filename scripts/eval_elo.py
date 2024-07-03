import matplotlib.pyplot as plt
import numpy as np
import os, sys
import imageio
from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C, DDPG
from gymnasium.wrappers import FlattenObservation


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from envs.grid_world_gaussians_Elo import GridWorldEnv



path = Path().resolve()
video_folder = str(path/"videos")
model_folder = path/"models/"
video_length = 100

env = GridWorldEnv(size=10, render_mode="human")
env = FlattenObservation(env)
model = DQN.load(str(model_folder/"MultiEnv_DQN_Elo"))
obs, info = env.reset()

alpha = 1.0
prev_action = None
for i in range(video_length+1):
    # action = env.action_space.sample()
    actions, _ = model.predict(obs)
    print("Actions: ", actions)
    # if prev_action is None:
    #     prev_action = action
    # filtered_action = alpha * action + (1 - alpha) * prev_action
    # prev_action = filtered_action
    obs, reward, terminated, truncated, info = env.step(actions)
    # print("Reward: ", reward)
    env.render()
    if terminated or truncated:
        break

print("Number of steps: ", i+1)
env.close()