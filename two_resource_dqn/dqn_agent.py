import random

import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, n_action, action_size, shape_vector_obs, shape_obs_image):
        self.action_size = action_size
        self.shape_obs_image = shape_obs_image  # Like (64, 64, 3)
        self.shape_vector_obs = shape_vector_obs  # Like  (2,)
        self.n_action = n_action

    def step(self, observation, done):
        # plt.imshow(observation[0])
        # plt.pause(0.0001)
        # print(f"Vector observations : {observation[1]}")
        return random.choice(range(self.n_action))

