import random
from collections import deque
from enum import Enum, auto

import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, n_images, vector_dim, n_action):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3 * n_images, out_channels=32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc_vec = nn.Linear(in_features=vector_dim, out_features=256)
        self.fc1 = nn.Linear(in_features=5664, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=n_action)

        self.act_conv = nn.Softplus()
        self.act_fc = nn.Softplus()

    def forward(self, image_tensor, vector_tensor):
        x_im = self.act_conv(self.bn1(self.conv1(image_tensor)))
        x_im = self.act_conv(self.bn2(self.conv2(x_im)))
        x_im = x_im.flatten()
        x_int = self.act_fc(self.fc_vec(vector_tensor))
        x = torch.cat([x_im, x_int], dim=0)
        x = self.act_fc(self.fc1(x))
        output = self.fc2(x)
        return output


class BufferType(Enum):
    observation = 0
    action = 1


class ExperienceData:
    def __init__(self, observation, action, next_observation):
        self._obs = observation
        self._act = action
        self._next_obs = next_observation

    @property
    def observation(self):
        return self._obs

    @property
    def action(self):
        return self._act

    @property
    def next_observation(self):
        return self._next_obs


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        buffer_obs = deque(maxlen=buffer_size)
        buffer_act = deque(maxlen=buffer_size)
        self._buffer = {
            BufferType.observation: buffer_obs,
            BufferType.action: buffer_act
        }
        self.experience_size = len(self._buffer[BufferType.observation])

    def append(self, observation, action):
        self._buffer[BufferType.observation].append(observation)
        self._buffer[BufferType.action].append(action)
        self.experience_size = len(self._buffer[BufferType.observation])

    def get_single_experience(self, time_step):
        """
        Return a single experience instance
        :param time_step:
        :return:
        """
        assert self.experience_size - 1 > time_step, "Sample time step must be less than number of experience minus one."
        experience = ExperienceData(
            observation=self._buffer[BufferType.observation][time_step],
            action=self._buffer[BufferType.action][time_step],
            next_observation=self._buffer[BufferType.observation][time_step + 1],
        )
        return experience

    def get_batch_experience(self, batch_size):
        """
        Return batch list of experience instance
        :param batch_size:
        :return:
        """
        batch = []
        for i in range(batch_size):
            index = random.choice(range(self.experience_size - 1))
            batch.append(self.get_single_experience(index))
        return batch

    def clear(self):
        self._buffer[BufferType.observation].clear()
        self._buffer[BufferType.action].clear()
        self.experience_size = len(self._buffer[BufferType.observation])


class DQNAgent:
    """ Classical Deep Q Network Agent
    """
    def __init__(self, config, n_action, action_size, shape_vector_obs, shape_obs_image):
        self.learning_rate = float(config["dnn"]["learning_rate"])
        self.adam_eps = float(config["dnn"]["adam_eps"])
        self.batch_size = int(config["dnn"]["batch_size"])
        self.replay_buffer = int(config["qn"]["replay_buffer"])
        self.iteration = int(config["qn"]["iteration"])
        self.exploration = float(config["qn"]["exploration"])
        self.input_time_horizon = int(config["qn"]["input_time_horizon"])
        self.action_size = action_size
        self.shape_obs_image = shape_obs_image  # Like (64, 64, 3)
        self.shape_vector_obs = shape_vector_obs  # Like  (2,)
        self.n_action = n_action

        self.qnet = QNet(n_images=self.input_time_horizon,
                         vector_dim=2,
                         n_action=n_action)
        self.qnet_support = QNet(n_images=1,
                                 vector_dim=2,
                                 n_action=n_action)
        self.qnet_support.load_state_dict(self.qnet.state_dict())
        self.time_tick = 0

    def step(self, observation, done) -> torch.Tensor:
        # plt.imshow(observation[0])
        # plt.pause(0.0001)
        # print(f"Vector observations : {observation[1]}")

        im_tensor, vec_tensor = self.obs_to_tensor(observation)
        q_val = self.qnet(im_tensor, vec_tensor)
        print(q_val)

        # Copy Q net to the support network
        if self.time_tick == self.iteration:
            self.time_tick = 0
            self.qnet_support.load_state_dict(self.qnet.state_dict())
        self.time_tick += 1
        return random.choice(range(self.n_action))

    @staticmethod
    def obs_to_tensor(raw_observation):
        im_tensor = torch.tensor(raw_observation[0]).view((1, 3, 64, 64))
        vec_tensor = torch.tensor(raw_observation[1])
        return im_tensor, vec_tensor
