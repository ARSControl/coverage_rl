import numpy as np
import pygame
import scipy
import gymnasium as gym
from gymnasium import spaces
from sklearn.datasets import make_spd_matrix


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=10):
        self.size = size                # size of the square grid
        self.window_size = 512          # size of the pygame window

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of  {0, ..., `size`}^2, i.e. MultiDiscrete([size, size])
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,3), dtype= np.float64)

        # we have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(8)

        # The following dictionary maps abstract actions from self.action_space to the direction we will walk in if that action is taken.
        # i.e. 0 corresponds to "right", 1 to "up", ...
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            #diagonals
            4: np.array([1, 1]), #up right 
            5: np.array([1, -1]), #down right
            6: np.array([-1, -1]), #down left
            7: np.array([-1, 1]) #up left
        }

        self._action_to_direction = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, -1], [-1, 1] ])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.cov = make_spd_matrix(n_dim=2, random_state=1000)
        self.cov *= 50
        #self.cov = np.array([20.0, 0], [0, 20.0])

        # if human-rendering is used, self.window will be a reference to the window.
        # self.clock will be a clock used to ensure rendering at correct framerate.
        # They will remain None until human-mode is used for the 1st time.
        self.window = None
        self.clock = None


    # Get observations (to be used in reset() and step() )
    def _get_obs(self, current_location, obstacles_locations):
        #if one of the values goes outside boundaries or corresponds to an obstacle add a negative reward
        obs_matrix = np.zeros((3,3))
        pdfs = []

        #calculate max_value
        max_pdf = 0.00
        for i in range(0, self.size):
            for j in range(0, self.size):
                point = np.array((i,j))
                pdf = float(self.eval_pdf(self._target_location,point,self.cov))
                if pdf > max_pdf:
                    max_pdf = pdf

        for i in range(current_location[0]-1,current_location[0]+2):
            for j in range(current_location[1]-1,current_location[1]+2):
                if self.is_within_bounds(np.array((i,j))) and not self.is_on_obstacle(obstacles_locations, np.array((i,j))) :
                    pdf_value = float(self.eval_pdf(self._target_location,np.array((i,j)),self.cov))
                    #normalizing 
                    #pdf_value = float(pdf_value/max_pdf)
                    pdfs.append(pdf_value)
                else:
                    pdfs.append(-10.0)

        for i in range(0,3):
            for j in range(0,3):
                I = int(i*3+j)
                obs_matrix[i,j] = pdfs[I]
        
        obs_matrix[1,1] = 0.0

        return obs_matrix

    # Similar for info returned by step and reset
    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }

    def eval_pdf(self, target, current_pos, cov):
        pdf = scipy.stats.multivariate_normal.pdf(current_pos, target, cov)
        return pdf
    
    def is_within_bounds(self, point):
        b = 0
        if (point >= 0).all() and (point < self.size).all():
            b = 1
        return b
    
    def is_on_obstacle(self,obstacles,point):
        b = False
        for i in range(3):
            if np.array_equal(obstacles[i,:], point):
                b = True
        return b

    def reset(self, seed=None, options=None):
        # Seed RNG
        super().reset(seed=seed)

        # Choose agent's location random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Initialize the obstacles' locations and ensure they don't overlap with the agent's location
        self._obstacles_locations = self.create_obstacles(self._agent_location)
        
        self._target_location = self._agent_location
        # Sample target's location randomly until it does not coincide with the agent's location or any obstacle's location
        while np.array_equal(self._target_location, self._agent_location) or self.is_on_obstacle(self._obstacles_locations, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        observation = self._get_obs(self._agent_location, self._obstacles_locations)
        info = self._get_info()

        if self.render_mode == "human":
            info = self._render_frame()

        self.cov = make_spd_matrix(n_dim=2, random_state=1000)
        self.cov *= 50
        #self.cov = np.array([20.0, 0], [0, 20.0])

        return observation, info

    def create_obstacles(self, agent_loc):
        while True:
            obstacles_locations = self.np_random.integers(0, self.size, size=(3, 2), dtype=int)
            if not self.is_on_obstacle(obstacles_locations, agent_loc):
                break
        return obstacles_locations

    def get_obstacles(self):
        return self._obstacles_locations
    
    def step(self, action):
        # map the action to walk direction
        direction = self._action_to_direction[action]

        # clip to be sure we don't leave the grid
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size - 1)

        #evaluate the pdf in the current agent position
        pdf = self.eval_pdf(self._target_location, self._agent_location, self.cov)

        if self.is_on_obstacle(self.get_obstacles(),self._agent_location):
            r_value = -100.0
        else:
            r_value = pdf

        # episode is done if the agent has reached the target
        observation = self._get_obs(self._agent_location,self._obstacles_locations)

        #the reward is equal to the sum of all the observations around the robot
        #sum = np.sum(observation)
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 100 if terminated else r_value
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
        pix_square_size = (self.window_size / self.size)

        #calculate max_value
        max_pdf = 0.00
        for i in range(0, self.size):
            for j in range(0, self.size):
                point = np.array((i,j))
                pdf = float(self.eval_pdf(self._target_location,point,self.cov))
                if pdf > max_pdf:
                    max_pdf = pdf

        for i in range(0,self.size):
            for j in range(0,self.size):
                rect = pygame.Rect(
                    pix_square_size * np.array((i,j)),
                    (pix_square_size, pix_square_size),
                )
                eval_point = np.array((i,j))

                #represent gaussian with colorful level curves, target with red, obstacles will be 
                if not self.is_on_obstacle(self._obstacles_locations, np.array((i,j))):

                    pdf_value = float(self.eval_pdf(self._target_location,eval_point,self.cov))

                    #normalizing 
                    pdf_value = float(pdf_value/max_pdf)

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
                else:
                    canvas.fill((0, 0, 0),rect)

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
