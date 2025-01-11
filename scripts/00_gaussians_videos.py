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
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs'))
from grid_world_gaussians_15 import GridWorldEnv


path = Path().resolve()
video_folder = str(path.parent/"videos")
model_folder = path.parent/"models"
video_length = 100

# Create env
num_envs = 4
env = GridWorldEnv(size=5)
env = FlattenObservation(env)
model = A2C("MlpPolicy", env, verbose=1)
# observation, info = env.reset()
total_timesteps = 10_000
model.learn(total_timesteps=total_timesteps)
#model.save(str(model_folder/"MultiEnv_A2C_Elo_test3"))
env.close()

images = []
test_env = GridWorldEnv(render_mode="rgb_array", size=10)
test_env = FlattenObservation(test_env)
#test_ennv = DummyVecEnv(test_env)
obs, info = test_env.reset()
frame = test_env.render()  # Render the frame

for i in range(video_length+1):
    # images.append(img)
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = test_env.step(action)
    frame = test_env.render()
    if frame is not None:
        images.append(frame)
    else:
        raise ValueError("Render returned None. Ensure render_mode='rgb_array' is set correctly.")
    if terminated or truncated:
        break

video_path = str(video_folder+"/simulation_video_15.mp4")
imageio.mimsave(video_path, images, fps=3)  # Adjust fps as needed
print("Number of steps : ", i+1)
# imageio.mimsave(os.path.join(video_folder, "test.gif"), [np.array(img) for i, img in enumerate(images) if i%2==0], fps=29)
test_env.close()