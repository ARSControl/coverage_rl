import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from envs.grid_world import GridWorldEnv

class CoverageGridEnv(GridWorldEnv):
    def __init__(self, render_mode=None, size=5):
        super(CoverageGridEnv, self).__init__(render_mode, size)

        # change observations
        self.observation_space = spaces.Box(0, 1, shape=(2,))
        self.action_space = spaces.Box(0, 2, shape=(2,))
