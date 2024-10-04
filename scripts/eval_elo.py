import matplotlib.pyplot as plt
import numpy as np
import os, sys
import imageio
from pathlib import Path
from stable_baselines3 import PPO, DQN, A2C, DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.vec_env import VecVideoRecorder


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
from envs.simple_env import SimpleEnv
from envs.grid_world_gaussians_Elo import GridWorldEnv



path = Path().resolve()
video_folder = str(path/"videos")
model_folder = path/"models/"
video_length = 100
RECORD_VIDEO = True
render_mode = "rgb_array" if RECORD_VIDEO else "human"


env = GridWorldEnv(render_mode=render_mode)
obs, info = env.reset()
print("Obs shape : ", obs)
# env = FlattenObservation(env)
model = PPO.load(str(model_folder/"SimpleEnv_PPO_1M_Elo"), env)
# model.set_env(env)
obs, info = env.reset()
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1)
# print(f"Mean reward: {mean_reward}, std: {std_reward}")

alpha = 1
prev_action = None
"""
for i in range(video_length+1):
    for r in range(3):
        # action = env.action_space.sample()
        action, _ = model.predict(obs[r])
        print("Actions: ", action)
        if prev_action is None:
            prev_action = action
        filtered_action = alpha * action + (1 - alpha) * prev_action
        prev_action = filtered_action
        obs[r], reward, terminated, truncated, info = env.step(filtered_action, r)
    # print("Reward: ", reward)
    env.render()
    if terminated or truncated:
        break
"""

dt = 1
rewards = []
imgs = []
for i in range(video_length+1):
    action, _ = model.predict(obs)
    # print("ACtion: ", action)
    if prev_action is None:
        prev_action = action
    filtered_action = alpha * action + (1 - alpha) * prev_action
    obs, reward, terminated, truncated, info = env.step(filtered_action*dt)
    rewards.append(reward)
    print("Reward: ", reward)
    img = env.render()
    imgs.append(img)
    if terminated or truncated:
        break



print("Number of steps: ", i+1)
imageio.mimsave(os.path.join(video_folder, "simple_3obs_elo.mp4"), [np.array(img) for i, img in enumerate(imgs)], fps=5)
env.close()