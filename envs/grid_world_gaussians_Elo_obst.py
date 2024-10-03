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

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of  {0, ..., `size`}^2, i.e. MultiDiscrete([size, size])
        #self.observation_space = spaces.Dict(
        #     {
        #        "agent": spaces.Box(0, size-1, shape=(2,), dtype=int),
        #        "target": spaces.Box(0, size-1, shape=(2,), dtype=int),
        #     }
        #)
        self.observation_space = spaces.Box(low=-10, high=1,
                    shape=(3, 3),
                    dtype=np.float64)
        self.sensing_range = 1


        # we have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(9)

        # The following dictionary maps abstract actions from self.action_space to the direction we will walk in if that action is taken.
        # i.e. 0 corresponds to "right", 1 to "up", ...
        self._action_to_direction = {
            0: np.array([-1, 1]), #up left
            1: np.array([0, 1]), #up
            2: np.array([1, 1]), #up right
            3: np.array([-1, 0]), #left
            4: np.array([0, 0]), #still
            5: np.array([1, 0]), #right
            6: np.array([-1, -1]), #down left
            7: np.array([0, -1]), #down
            8: np.array([1, -1]) #down right
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

        xmin = x - self.sensing_range
        xmax = x + self.sensing_range
        ymin = y - self.sensing_range
        ymax = y + self.sensing_range


        obs = np.full((3,3),-10)
        if xmin >= 0 and xmax<self.size and ymax < self.size and ymin >= 0:
            obs = self.grid[xmin:(xmax+1),ymin:(ymax+1)]
            for i in range(3):
                for j in range(3):
                    for obstacle in self.obstacles:
                        if np.array_equal(np.array((i,j)),obstacle):
                            obs[i,j] = -10
            
        else:
            rangemaxx= 3
            rangemaxy = 3
            rangeminx = 0
            rangeminy = 0
            if ymax >= self.size:
                ymax -= 1
                rangemaxy = 2
            if xmax >= self.size:
                xmax -= 1
                rangemaxx = 2
            if ymin < 0:
                ymin += 1
                rangeminy = 1
            if xmin < 0:
                xmin += 1
                rangeminx = 1

            obs[rangeminx:rangemaxx,rangeminy:rangemaxy] = self.grid[xmin:(xmax+1),ymin:(ymax+1)]
            for i in range(rangeminx,rangemaxx):
                for j in range(rangeminy,rangemaxy):
                    for obstacle in self.obstacles:
                        if np.array_equal(np.array((i,j)),obstacle):
                            obs[i,j] = -10
        return obs

    # Similar for info returned by step and reset
    def _get_info(self):
        #return {"agent": self._agent_location, "target": self._target_location}
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }


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
        self.obstacles = []
        while len(self.obstacles) < 3:
            obstacle = self.np_random.integers(0, self.size, size=2, dtype=int)
            if not np.array_equal(obstacle, self._agent_location) and not np.array_equal(obstacle,self._target_location):
                self.obstacles.append(obstacle)
        self.obstacles = np.array(self.obstacles)

        self.old_pdf = gauss_pdf(self._agent_location,self._target_location,self.covariance)

        self.grid = np.zeros((self.size, self.size))
        for i in range(0, self.size):
            for j in range(0, self.size):
                self.grid[i, j] = gauss_pdf(np.array([i,j]), self._target_location, self.covariance)
        #normalize values
        self.grid /= self.grid.max()

        for obstacle in self.obstacles:
            self.grid[obstacle[0],obstacle[1]] = -10

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
        if self.old_pdf < self.pdf:
            reward = 1
        elif self.old_pdf > self.pdf:
            reward = -1
        else:
            reward = 0

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
        for i in range(3):
            pygame.draw.circle(
                canvas,
                (0, 0, 0),
                (self.obstacles[i,:] + 0.5) * pix_square_size,
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
