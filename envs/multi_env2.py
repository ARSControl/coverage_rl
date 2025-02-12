import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely import Polygon, Point, intersection



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



class MultiEnv(gym.Env):
    metadata =  {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, robot_range=3, robots_num=3, sigma=2, discr=0.2, render_mode=None, local_vis=True, size=100, width=10):
        self.size = size
        self.window_size = 512
        self.sensing_range = robot_range
        self.robots_num = robots_num
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

        # Observation space: obstacle + sensing area
        ''' 
        self.observation_space = spaces.Dict(
            {
                "sensing": spaces.Box(low=0, high=1,
                                                shape=(obs_shape, obs_shape),
                                                dtype=np.float32),
                "mates": spaces.Box(low=0, high=width-1, shape=(2, self.robots_num-1), dtype=np.float32),
                "agent": spaces.Box(low=0, high=width-1, shape=(2,), dtype=np.float32)
            }
        )
        '''
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_shape, self.obs_shape), dtype=np.float32)

        # ACtion space: x and y direction in range [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None



    def _get_obs(self, idx):
        x, y = self._robots_positions[idx]
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

        mates = self._robots_positions[:idx] + self._robots_positions[idx+1:]
        mates = np.array(mates)
        for mate in mates:
            xn, yn = mate
            ix = int(xn / self.discretize_precision) + self.i_start
            iy = int(yn / self.discretize_precision) + self.i_start
            obs[ix, iy] = -1.0

        return obs

    def _get_info(self, idx):
        return {
            "position": self._robots_positions[idx]
        }

    def _reset_agent(self, idx):
        self._robots_positions[idx] = self.width * np.random.rand(2)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0

        # random robot's position
        self._robots_positions = [self._reset_agent(i) for i in range(self.robots_num)]
        # self._robot_position = np.array([1.0, 1.0])
        self._mean_pt = self.width * np.random.rand(2)
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

        observations = [self._get_obs(i) for i in range(self.robots_num)]
        infos = [self._get_info(i) for i in range(self.robots_num)] 
        return observations, infos

    
    def step(self, actions):
        rewards = [0.0 for _ in range(self.robots_num)]
        terms = [False for _ in range(self.robots_num)]
        infos = [self._get_info(i) for i in range(self.robots_num)]

        self.t += 1
        for i in range(self.robots_num):
            self._robots_positions[i] = np.clip(self._robots_positions[i] + actions[i, :], 0, self.size)

        # Voronoi partitioning
        try:
            pts = np.array(self._robots_positions)
            dummy_points = np.zeros((5*self.robots_num, 2))
            dummy_points[:self.robots_num, :] = pts
            mirrored_points = mirror(pts, self.width)
            mir_pts = np.array(mirrored_points)
            dummy_points[self.robots_num:, :] = mir_pts            
            lim_regions = []
            
            # calculate limited voronoi partitioning
            vor = Voronoi(dummy_points)
            for idx in range(self.robots_num):
                x, y = self._robots_positions[idx]
                region = vor.point_region[idx]
                poly_vert = []
                for vert in vor.regions[region]:
                    v = vor.vertices[vert]
                    poly_vert.append(v)

                poly = Polygon(poly_vert)           # global Voronoi cell
                
                # Intersect with robot range
                range_pts = []
                for th in np.arange(0.0, 2*np.pi, 0.5):
                    xi = x + self.sensing_range * np.cos(th)
                    yi = y + self.sensing_range * np.sin(th)
                    pt = Point(xi, yi)
                    range_pts.append(pt)
                
                range_poly = Polygon(range_pts)
                lim_region = intersection(poly, range_poly)
                lim_regions.append(lim_region)

                for i in range(self.size):
                    for j in range(self.size):
                        x_pt = Point(i*self.discretize_precision, j*self.discretize_precision)
                        x_ij = np.array([i*self.discretize_precision, j*self.discretize_precision])
                        if lim_region.contains(x_pt):
                            d = np.linalg.norm(x_ij - self._robot_position)
                            rewards[idx] += d**2 * self.grid[self.i_start+i, self.i_start+j]
                            if d < self.CONVERGENCE_TOLERANCE:
                                terms[idx] = True
            
        except:
            terminated = False
            truncated = True
            rewards = -1000
            observations = [self._get_obs(i) for i in range(self.robots_num)]
            infos = [self._get_info(i) for i in range(self.robots_num)]
            return observations, reward, terminated, truncated, infos


        # episode is done iff the agent has reached the target
        # terminated = np.linalg.norm(self._robot_position - self._mean_pt) < self.CONVERGENCE_TOLERANCE
        terminated = all(terms)
        truncated = self.t > 1000
        observations = [self._get_obs(i) for i in range(self.robots_num)]
        infos = [self._get_info(i) for i in range(self.robots_num)]

        if self.render_mode == "human":
            self._render_frame()

        return observations, reward, terminated, truncated, infos

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

        # Draw the robots
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


