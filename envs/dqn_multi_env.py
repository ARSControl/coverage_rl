import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import Polygon, Point, intersection, get_coordinates



def mirror(points, width):
    mirrored_points = []
    

    # Define the corners of the square
    square_corners = [(0.0, 0.0), (width, 0.0), (width, width), (0.0, width)]

    # Mirror points across each edge of the square
    for edge_start, edge_end in zip(square_corners, square_corners[1:] + [square_corners[0]]):
        edge_vector = (edge_end[0] - edge_start[0], edge_end[1] - edge_start[1])

        for point in points:
            # Calculate the vector from the edge start to the point
            point_vector = (point[0] - edge_start[0], point[1] - edge_start[1])

            # Calculate the mirrored point by reflecting across the edge
            mirrored_vector = (point_vector[0] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[0],
                               point_vector[1] - 2 * (point_vector[0] * edge_vector[0] + point_vector[1] * edge_vector[1]) / (edge_vector[0]**2 + edge_vector[1]**2) * edge_vector[1])

            # Translate the mirrored vector back to the absolute coordinates
            mirrored_point = (edge_start[0] + mirrored_vector[0], edge_start[1] + mirrored_vector[1])

            # Add the mirrored point to the result list
            mirrored_points.append(mirrored_point)

    return mirrored_points

def gauss_pdf(x, y, mean, covariance):

  # points = np.column_stack([x.flatten(), y.flatten()])
  points = np.array([x, y])
  points = np.expand_dims(points, 0)
  # Calculate the multivariate Gaussian probability
  exponent = -0.5 * np.sum((points - mean) @ np.linalg.inv(covariance) * (points - mean), axis=1)
  coefficient = 1 / np.sqrt((2 * np.pi) ** 2 * np.linalg.det(covariance))
  prob = coefficient * np.exp(exponent)

  return prob



class MultiDQNEnv(gym.Env):
    metadata =  {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, robot_range=3, robots_num=3, safety_dist=1, sigma=2, discr=0.2, render_mode=None, local_vis=True, size=100, width=10):
        self.size = size
        self.window_size = 512
        self.sensing_range = robot_range
        self.safety_dist = safety_dist
        self.robots_num = robots_num
        self.mates_num = robots_num - 1
        self.width = width
        self.local_vis = local_vis
        self.discretize_precision = width / size
        self.covariance = np.array([[sigma, 0], [0, sigma]])
        obs_shape = int(2*self.sensing_range/self.discretize_precision)
        self.obs_shape = obs_shape
        self.CONVERGENCE_TOLERANCE = 0.2
        self.dt = 0.2
        
        print("Discretize precision: ", self.discretize_precision)
        print("Shape: ", obs_shape)
        # Observations space: robot's position
        # self.observation_space = spaces.Box(low=0.0, high=self.size-1, shape=(2,), dtype=np.float32)


        grid_obs = spaces.Box(low=-100, high=100, shape=(self.obs_shape, self.obs_shape), dtype=np.float32)
        mates_obs = spaces.Box(low=-self.width, high=self.width, shape=(self.robots_num-1, 2), dtype=np.float32)
        # self.observation_space = spaces.Dict({"grid": grid_obs, "mates": mates_obs})
        self.observation_space = grid_obs

        # ACtion space: x and y direction in range [-1, 1]
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # self.action_space = spaces.Discrete(9)                  # 8 directions + idle
        self.action_space = spaces.MultiDiscrete([9, 9, 9])
        print("Action space sample: ", self.action_space.sample())


        self._action_to_direction = {
            1: np.array([-1, 1]),
            2: np.array([0, 1]),
            3: np.array([1, 1]),
            4: np.array([-1, 0]),
            5: np.array([1, 0]),
            6: np.array([-1, -1]),
            7: np.array([0, -1]),
            8: np.array([1, -1]),
            0: np.array([0, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None



    def _get_obs(self, i):

        x, y = self._robots_positions[i]
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

        global_map = self.grid
        for x_j in np.delete(self._robots_positions, i, axis=0):
            d = np.linalg.norm(self._robots_positions[i] - x_j)
            if d < self.sensing_range:
                xmin = x_j[0] - self.safety_dist
                xmax = x_j[0] + self.safety_dist
                ymin = x_j[1] - self.safety_dist
                ymax = x_j[1] + self.safety_dist
                iimin = int(xmin / self.discretize_precision) + self.i_start
                iimax = int(xmax / self.discretize_precision) + self.i_start
                jjmin = int(ymin / self.discretize_precision) + self.i_start
                jjmax = int(ymax / self.discretize_precision) + self.i_start
                global_map[iimin:iimax, jjmin:jjmax] = -1

        # extract observation from grid        
        obs = global_map[imin:imax, jmin:jmax]
        pad_i = self.obs_shape - obs.shape[0]
        pad_j = self.obs_shape - obs.shape[1]
        if pad_i > 0:
            obs = np.concatenate((-1*np.ones((pad_i, obs.shape[1])), obs), 0)
        if pad_j > 0:
            obs = np.concatenate((-1*np.ones((obs.shape[0], pad_j)), obs), 1)

        norm_obs = 100*obs
        return norm_obs




    def _get_info(self):
        return {
            "positions": [self._robots_positions[i] for i in range(self.robots_num)]
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.stop_count = np.zeros((self.robots_num))

        # random robot's position
        # self._robot_position = np.array([1.0, 1.0])
        self._robots_positions = [1 + (self.width-2) * np.random.rand(2) for _ in range(self.robots_num)]
        self._mean_pt = 1 + (self.width-2) * np.random.rand(2)
        self._mates_positions = []
        self.centroid = np.zeros((2,))
        # for i in range(self.mates_num):
        #     ready = False
        #     while not ready:
        #         m = 1 + (self.width-2) * np.random.rand(2)
        #         ready = np.linalg.norm(m - self._robot_position) > 1.0
        #     self._mates_positions.append(m)
        
        # self._mean_pt = np.array([8.0, 2.0])
        # self.mates = self.width * np.random.rand(2, self.mates_num)


        # define prob values of the grid
        # env size: (size, size), with additional size/2 cells on each side
        self.grid = np.zeros((2*self.size, 2*self.size))
        self.i_start = self.size // 2               # index of the 1st cell inside the env (both for rows and cols)
        
        for i in range(0, self.size):
            for j in range(0, self.size):
                self.grid[self.i_start+i, self.i_start+j] = gauss_pdf(i*self.discretize_precision, j*self.discretize_precision, self._mean_pt, self.covariance)

        # Normalize values
        # self.grid -= self.grid.min()
        self.grid /= self.grid.max()

        # Set in range [0, 255]
        # self.grid = (255*self.grid).astype(int)

        # set values outside env to -1
        # self.grid[:self.i_start, :] = -1.0
        # self.grid[self.i_start+self.size:, :] = -1.0
        # self.grid[:, :self.i_start] = -1.0
        # self.grid[:, self.i_start+self.size:] = -1.0

        




        '''
        # obstacles
        for i in range(0, self.size):
            for j in range(0, self.size):
                x_i = np.array([i*self.discretize_precision, j*self.discretize_precision])
                dist = np.linalg.norm(x_i - self.obstacle)
                if dist < 1.0:
                    self.grid[self.i_start+i, self.i_start+j] = -(10 - 10*dist)
        '''

        obs = np.zeros((self.robots_num, self.obs_shape, self.obs_shape))
        for r in range(self.robots_num):
            obs[r, :, :] = self._get_obs(r)
        return obs, self._get_info()


    
    def step(self, action, i):
        '''
        # update robot position
        self._robot_position = np.clip(self._robot_position + action*self.dt, 0, self.size)
        # self._robot_position += action * self.dt
        x, y = self._robot_position             # [meters]
        '''
        if i == 0:
            self.t += 1
        if action == 0:
            self.stop_count[i] += 1
        else:
            self.stop_count[i] = 0

        move = self._action_to_direction[action] * self.discretize_precision
        self._robots_positions[i] = np.clip(self._robots_positions[i] + move, 0.5, self.width-0.5)

        x, y = self._robots_positions[i]

        xmin = max(0, x - self.sensing_range)
        xmax = min(self.width, x + self.sensing_range)
        ymin = max(0, y - self.sensing_range)
        ymax = min(self.width, y + self.sensing_range)

        
        # convert min and max x,y to indices (+ i_start to stay inside the env)
        imin = int(xmin / self.discretize_precision) + self.i_start
        imax = int(xmax / self.discretize_precision) + self.i_start
        jmin = int(ymin / self.discretize_precision) + self.i_start
        jmax = int(ymax / self.discretize_precision) + self.i_start

        observation = self._get_obs(i)
        info = self._get_info()

        reward = self.grid[int(x/self.discretize_precision) + self.i_start, int(y/self.discretize_precision) + self.i_start]*100

        # reward -= self.t
        for x_j in np.delete(self._robots_positions, i, axis=0):
            d = np.linalg.norm(self._robots_positions[i] - x_j)
            if d < self.sensing_range:
                reward -= 10 - 10/self.sensing_range * d


        # dist = np.linalg.norm(centr - x_ij)
        # print("Distance to centroid: ", dist)

        # episode is done iff the agent has reached the target
        # terminated = self.t > 1000
        terminated = self.stop_count.all() > 3 
        truncated = self.t > 1e6
        if terminated:
            reward = 1000
        elif truncated:
            reward = -100
        # xc, yc = int(x/self.discretize_precision), int(y/self.discretize_precision)       # cell
        # observations = [self._get_obs(i) for i in range(self.robots_num)]
        # reward = np.sum(observation) - 10*self.t            # reward = sum of values in sensing range
        observation = self._get_obs(i)
        info = self._get_info()

        # if x < 0.0 or x > self.width or y < 0.0 or y > self.width:
        #     reward = -1000
        #     truncated = True
        
        # if terminated:
        #     reward = 100

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
            obs = self._get_obs()["grid"]
            for i in range(obs.shape[1]):
                for j in range(obs.shape[2]):
                    pygame.draw.rect(
                        canvas,
                        (0, 0, obs[0,i,j]),
                        pygame.Rect(
                            pix_square_size * np.array([self._robot_position[0]/self.discretize_precision+i-self.obs_shape/2, self._robot_position[1]/self.discretize_precision+j-self.obs_shape/2]),
                            (pix_square_size, pix_square_size),
                        )
                    )

        """
        # Draw the teammates
        for i in range(self.mates_num):
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                self._mates_positions[i] * pix_square_size / self.discretize_precision,
                pix_square_size / 3 * 5,
            )
        """

        # draw the robots
        for i in range(self.robots_num):
            pygame.draw.circle(
                canvas,
                (0, 255, 0),
                self._robots_positions[i] * pix_square_size / self.discretize_precision,
                pix_square_size / 3 * 5,
            )

        # Draw mean pt
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self._mean_pt * pix_square_size / self.discretize_precision,
            pix_square_size / 3 * 5,
        )

        # Draw centroid
        # pygame.draw.circle(
        #     canvas,
        #     (255, 0, 0),
        #     self.centroid * pix_square_size / self.discretize_precision,
        #     pix_square_size / 3 * 5,
        # )

        # draw mates
        # for i in range(self.mates_num):
        #     pygame.draw.circle(
        #         canvas,
        #         (255, 0, 0),
        #         self.mates[:, i] * pix_square_size / self.discretize_precision,
        #         pix_square_size / 3 * 5,
        #     )


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


