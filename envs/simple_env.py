import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces



def gauss_pdf(x, y, mean, covariance):

  # points = np.column_stack([x.flatten(), y.flatten()])
  points = np.array([x, y])
  points = np.expand_dims(points, 0)
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob



class SimpleEnv(gym.Env):
    metadata =  {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, robot_range=3, sigma=2, discr=0.2, render_mode=None, local_vis=True, size=100, width=10):
        self.size = size
        self.window_size = 512
        self.sensing_range = robot_range
        self.width = width
        self.local_vis = local_vis
        self.discretize_precision = width / size
        self.covariance = np.array([[sigma, 0], [0, sigma]])
        obs_shape = int(2*self.sensing_range/self.discretize_precision)
        self.obs_shape = obs_shape
        self.CONVERGENCE_TOLERANCE = 0.2
        
        print("Discretize precision: ", self.discretize_precision)
        print("Shape: ", obs_shape)
        # Observations space: robot's position
        # self.observation_space = spaces.Box(low=0.0, high=self.size-1, shape=(2,), dtype=np.float32)

        # Observation space: R cells around the robot (--> shape = R+1)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(obs_shape, obs_shape),
                                            dtype=np.float32)

        # ACtion space: x and y direction in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None




    def _get_obs(self):
        x, y = self._robot_position
        half_obs_size = self.obs_shape // 2

        xmin = x - self.sensing_range
        xmax = x + self.sensing_range
        ymin = y - self.sensing_range
        ymax = y + self.sensing_range

        
        # convert min and max x,y to indices (+ i_start to stay inside the env)
        imin = int(xmin / self.discretize_precision) + self.i_start
        imax = int(xmax / self.discretize_precision) + self.i_start
        jmin = int(ymin / self.discretize_precision) + self.i_start
        jmax = int(ymax / self.discretize_precision) + self.i_start

        # extract observation from grid        
        obs = self.grid[imin:imax, jmin:jmax]
        pad_i = self.obs_shape - obs.shape[0]
        pad_j = self.obs_shape - obs.shape[1]
        if pad_i > 0:
            obs = np.concatenate((np.zeros((pad_i, obs.shape[1])), obs), 0)
        if pad_j > 0:
            obs = np.concatenate((np.zeros((obs.shape[0], pad_j)), obs), 1)
        
        # print("Obs shape: ", obs.shape)
        
        # range_obs[:obs.shape[0], :obs.shape[1]] = obs

        # print("Obs shape: ", obs.shape)

        return obs
        # return np.expand_dims(obs, 0)

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._robot_position - self._mean_pt)
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.t = 0

        # random robot's position
        self._robot_position = self.width * np.random.rand(2)
        # self._robot_position = np.array([1.0, 1.0])
        self._mean_pt = self.width * np.random.rand(2)
        # self._mean_pt = np.array([8.0, 2.0])
        self.obstacle = self.width * np.random.rand(2)


        # define prob values of the grid
        # env size: (size, size), with additional size/2 cells on each side
        self.grid = np.zeros((2*self.size, 2*self.size))
        self.i_start = self.size // 2               # index of the 1st cell inside the env (both for rows and cols)
        
        for i in range(0, self.size):
            for j in range(0, self.size):
                self.grid[self.i_start+i, self.i_start+j] = gauss_pdf(i*self.discretize_precision, j*self.discretize_precision, self._mean_pt, self.covariance)

        # Normalize values
        self.grid -= self.grid.min()
        self.grid /= self.grid.max()

        '''
        # obstacles
        for i in range(0, self.size):
            for j in range(0, self.size):
                x_i = np.array([i*self.discretize_precision, j*self.discretize_precision])
                dist = np.linalg.norm(x_i - self.obstacle)
                if dist < 1.0:
                    self.grid[self.i_start+i, self.i_start+j] = -(10 - 10*dist)
        '''
        
        return self._get_obs(), self._get_info()

    
    def step(self, action):
        # update robot position
        self._robot_position = np.clip(self._robot_position + action, 0, self.width)
        # self._robot_position += action
        self.t += 1

        # episode is done iff the agent has reached the target
        terminated = np.linalg.norm(self._robot_position - self._mean_pt) < self.CONVERGENCE_TOLERANCE
        truncated = self.t > 1000
        x, y = self._robot_position             # [meters]
        xc, yc = int(x/self.discretize_precision), int(y/self.discretize_precision)       # cell
        observation = self._get_obs()
        # reward = np.sum(observation) - 10*self.t            # reward = sum of values in sensing range
        reward = self.grid[self.i_start + xc, self.i_start + yc] #- 0.01*self.t # - collision_penalty

        info = self._get_info()

        """
        if x < 0.0 or x > self.width or y < 0.0 or y > self.width:
            reward = -1000
            truncated = True
        """

        if terminated:
            reward = 100

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

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
        pix_square_size = self.window_size / self.size

        # colormap_name = 'viridis'
        # scalar_mappable = plt.cm.ScalarMappable(cmap=colormap_name)


        # Draw sensing range values
        # obs = self._get_obs()
        # for i in range(obs.shape[0]):
        #     for j in range(obs.shape[1]):

        
        # draw probability density
        if not self.local_vis:
            for i in range(0, self.size):
                for j in range(0, self.size):
                    pygame.draw.rect(
                        canvas,
                        (0, 0, max(0, 255*self.grid[self.i_start+i,self.i_start+j])),
                        pygame.Rect(
                            pix_square_size * np.array([i, j]),
                            (pix_square_size, pix_square_size),
                        ),
                    )
        
        else:
            obs = self._get_obs()
            for i in range(obs.shape[0]):
                for j in range(obs.shape[1]):
                    pygame.draw.rect(
                        canvas,
                        (0, 0, 255*obs[i,j]),
                        pygame.Rect(
                            pix_square_size * np.array([self._robot_position[0]/self.discretize_precision+i-self.obs_shape/2, self._robot_position[1]/self.discretize_precision+j-self.obs_shape/2]),
                            (pix_square_size, pix_square_size),
                        )
                    )

        # Draw the robot
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            self._robot_position * pix_square_size / self.discretize_precision,
            pix_square_size / 3 * 5,
        )

        # Draw mean pt
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._mean_pt * pix_square_size / self.discretize_precision,
            pix_square_size / 3 * 5,
        )

        # draw obstacle
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            self.obstacle * pix_square_size / self.discretize_precision,
            pix_square_size / 3 * 5,
        )


        if self.render_mode == "human":
            # copy drawing from canvas to our visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # ensure rendering at predefined fps
            self.clock.tick(self.metadata["render_fps"])
        else:       # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


