'''sensing range = 2, obstacles and reward equal to the sum of pdfs in the observation'''
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

def gauss_pdf(point, mean, covariance):

    # Calculate the multivariate Gaussian probability
    point = np.expand_dims(point, 0)
    exponent = -0.5 * np.sum((point - mean) @ np.linalg.inv(covariance) * (point - mean), axis=1)
    coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
    prob = coefficient * np.exp(exponent)

    return prob

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size                # size of the square grid
        self.window_size = 512          # size of the pygame window

        self.observation_space = spaces.Dict(
            {
                "sensing": spaces.Box(low=-10, high=1,
                    shape=(5, 5),
                    dtype=np.float64),
                "obstacles": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8)
            }
        )

        self.sensing_range = 2


        # we have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(9)

        # The following dictionary maps abstract actions from self.action_space to the direction we will walk in if that action is taken.
        # i.e. 0 corresponds to "right", 1 to "up", ...
        self._action_to_direction = {
            6: np.array([-1, 1]), #up left
            7: np.array([0, 1]), #up
            8: np.array([1, 1]), #up right
            11: np.array([-1, 0]), #left
            12: np.array([0, 0]), #still
            13: np.array([1, 0]), #right
            16: np.array([-1, -1]), #down left
            17: np.array([0, -1]), #down
            18: np.array([1, -1]) #down right
        }

        self._action_to_direction = np.array([[-1, 1], [0, 1], [1, 1], [-1, 0], [0, 0], [1, 0], [-1, -1], [0, -1], [1, -1]])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # if human-rendering is used, self.window will be a reference to the window.
        # self.clock will be a clock used to ensure rendering at correct framerate.
        # They will remain None until human-mode is used for the 1st time.
        self.window = None
        self.clock = None

        sigmaxx = 5
        sigmayy = 4
        self.covariance = np.array([[sigmaxx, 0], [0, sigmayy]])

    # Get observations (to be used in reset() and step() )
    def _get_obs(self):
        x = self._agent_location[0]
        y = self._agent_location[1]

        xmin = max(0, x - self.sensing_range)
        xmax = min(self.size - 1, x + self.sensing_range)
        ymin = max(0, y - self.sensing_range)
        ymax = min(self.size - 1, y + self.sensing_range)

        # Adjust the size of the observation window based on the sensing range
        obs = np.full((2 * self.sensing_range + 1, 2 * self.sensing_range + 1), 0)
        obs_obst = np.full((2 * self.sensing_range + 1, 2 * self.sensing_range + 1), 0)

        # Calculate the valid ranges in the observation array (obs) and the grid (self.grid)
        obs_xmin = max(0, self.sensing_range - x)
        obs_xmax = obs_xmin + (xmax - xmin + 1)
        obs_ymin = max(0, self.sensing_range - y)
        obs_ymax = obs_ymin + (ymax - ymin + 1)

        # Fill the valid region of the observation with the corresponding grid values
        obs[obs_xmin:obs_xmax, obs_ymin:obs_ymax] = self.grid[xmin:(xmax+1), ymin:(ymax+1)]
        obs_obst[obs_xmin:obs_xmax, obs_ymin:obs_ymax] = self.grid_obst[xmin:(xmax+1), ymin:(ymax+1)]

        dict_obs = {"sensing": obs, "obstacles": obs_obst}

        return dict_obs

    # Similar for info returned by step and reset
    def _get_info(self):
        #return {"agent": self._agent_location, "target": self._target_location}
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }

    def get_obstacles(self):
        return self.obstacles

    def reset(self, seed=None, options=None):
        # Seed RNG
        super().reset(seed=seed)

        # Choose agent's location random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # sample target's location randomly until it does not coincide with the agent
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        #add obstacles
        self.obstacles = self.np_random.integers(0, self.size, size=(3,2), dtype=int)
        while any(np.array_equal(obstacle, self._agent_location) for obstacle in self.obstacles) or any(np.array_equal(obstacle, self._target_location) for obstacle in self.obstacles):
            self.obstacles = self.np_random.integers(0, self.size, size=(3,2), dtype=int)

        self.old_pdf = gauss_pdf(self._agent_location,self._target_location,self.covariance)

        self.grid = np.zeros((self.size, self.size))
        for i in range(0, self.size):
            for j in range(0, self.size):
                self.grid[i, j] = gauss_pdf(np.array([i,j]), self._target_location, self.covariance)
        #normalize values
        self.grid /= self.grid.max()

        self.grid_obst = np.zeros((self.size, self.size))
        for obst in self.obstacles:
            self.grid_obst[obst[0],obst[1]] = 1 
        #normalize values
        self.grid /= self.grid.max()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            info = self._render_frame()

        return observation, info


    def step(self, action):
        #calculate current pdf
        self.old_pdf = self.grid[self._agent_location[0], self._agent_location[1]]

        # map the action to walk direction
        direction = self._action_to_direction[action]

        # clip to be sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        #calculate pdf after the step
        self.pdf = self.grid[self._agent_location[0], self._agent_location[1]]

        #reward depends on the difference
        observation = self._get_obs()
        obst_obs = observation.get("obstacles")
        sensing = observation.get("sensing")

        if np.any(obst_obs):
            reward = -10
        else:
            reward = np.sum(sensing)

        # episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        if terminated:
            reward = 100

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
    
        return observation, reward, terminated, False, info
    

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        else:
            self._render_frame()

    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.size)        # size of a single grid square in pixels

        for i in range(0,self.size):
            for j in range(0,self.size):
                rect = pygame.Rect(
                    pix_square_size * np.array((i,j)),
                    (pix_square_size, pix_square_size),
                )
                #represent gaussian with colorful level curves
                pdf_value = self.grid[i,j]

                if pdf_value >= 0.8:
                    canvas.fill((255, 153, 51),rect)
                elif pdf_value < 0.8 and pdf_value >= 0.6:
                    canvas.fill((255, 255, 51),rect)
                elif pdf_value < 0.6 and pdf_value >= 0.4:
                    canvas.fill((153, 255, 51),rect)
                elif pdf_value < 0.4 and pdf_value >= 0.2:
                    canvas.fill((0, 255, 255),rect)
                else:
                    canvas.fill((255, 255, 255),rect)


        # first draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # now draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        #draw the obstacles
        for i in range(0,self.size):
            for j in range(0,self.size):
                if self.grid_obst[i,j] == 1:
                    pygame.draw.circle(
                        canvas,
                        0,
                        (np.array((i,j)) + 0.5) * pix_square_size,
                        pix_square_size / 3,
                    )

        # finally, add some grid lines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # copy drawing from canvas to our visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # ensure rendering at predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable
            self.clock.tick(self.metadata["render_fps"])
        else:   #rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
