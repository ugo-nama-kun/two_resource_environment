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

        self.fc_vec = nn.Linear(in_features=vector_dim, out_features=128)

        self.fc1 = nn.Linear(in_features=928, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=n_action)

        self.act_conv = nn.Softplus()
        self.act_fc = nn.Softplus()

    def forward(self, image_tensor: torch.Tensor, vector_tensor: torch.Tensor) -> torch.Tensor:
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


class DataType(Enum):
    image = 0
    vector = 1


class Observation:
    def __init__(self, image, vector):
        self._obs = {
            DataType.image: image,
            DataType.vector: vector
        }

    @property
    def image(self):
        return self._obs[DataType.image]

    @property
    def vector(self):
        return self._obs[DataType.vector]


class ExperienceData:
    def __init__(self, observation: Observation, action, next_observation: Observation):
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
        self.n_experience = len(self._buffer[BufferType.observation])

    def append(self, observation: Observation, action):
        self._buffer[BufferType.observation].append(observation)
        self._buffer[BufferType.action].append(action)
        self.n_experience = len(self._buffer[BufferType.observation])

    def get_single_experience(self, time_step):
        """
        Return a single experience instance
        :param time_step:
        :return:
        """
        assert self.n_experience - 1 > time_step, "Sample time step must be less than number of experience minus one."
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
            index = random.choice(range(self.n_experience - 1))
            batch.append(self.get_single_experience(index))
        return batch

    def clear(self):
        self._buffer[BufferType.observation].clear()
        self._buffer[BufferType.action].clear()
        self.n_experience = len(self._buffer[BufferType.observation])


class DQNAgent:
    """ Classical Deep Q Network Agent
    """

    def __init__(self, config, n_action, action_size, shape_vector_obs, shape_obs_image, eps_start, device):
        # Network
        self.input_time_horizon = int(config["qn"]["input_time_horizon"])
        self.action_size = action_size
        self.shape_obs_image = shape_obs_image  # Like (64, 64, 3)
        self.shape_vector_obs = shape_vector_obs  # Like  (2,)
        self.n_action = n_action
        self._device = device
        self.qnet = QNet(n_images=self.input_time_horizon,
                         vector_dim=2,
                         n_action=n_action).to(device)
        self.qnet_support = QNet(n_images=1,
                                 vector_dim=2,
                                 n_action=n_action).to(device)
        self.qnet_support.load_state_dict(self.qnet.state_dict())

        # Optimization
        self.training_start_from = int(config["qn"]["training_start_from"])
        self.learning_rate = float(config["dnn"]["learning_rate"])
        self.adam_eps = float(config["dnn"]["adam_eps"])
        self.batch_size = int(config["dnn"]["batch_size"])
        self.iteration = int(config["qn"]["iteration"])
        self._optimizer = torch.optim.Adam(
            params=self.qnet.parameters(),
            lr=self.learning_rate,
            eps=self.adam_eps
        )

        # Replay Buffer
        self.replay_buffer_size = int(config["qn"]["replay_buffer_size"])
        self.replay_buffer = ReplayBuffer(buffer_size=self.replay_buffer_size)

        # Other initialization
        self._reward_discount = float(config["qn"]["reward_discount"])
        self.__eps_e_greedy = eps_start
        self.time_tick = 0

    def step(self, observation, done) -> torch.Tensor:
        im_tensor, vec_tensor = self.obs_to_tensor(observation)
        greedy_action = self.get_greedy_action(im_tensor.to(self._device), vec_tensor.to(self._device))
        if random.random() < self.eps_e_greedy:
            next_action = random.choice(range(self.n_action))
        else:
            next_action = greedy_action

        # Stock into the replay buffer
        self.replay_buffer.append(
            observation=Observation(image=im_tensor, vector=vec_tensor),
            action=next_action
        )

        # Train Agent
        if self.replay_buffer.n_experience > self.training_start_from:
            self.train()
        else:
            print(f"Buffer : {self.replay_buffer.n_experience}/{self.replay_buffer_size}")

        # Copy Q net to the support network
        if self.time_tick == self.iteration:
            self.time_tick = 0
            self.qnet_support.load_state_dict(self.qnet.state_dict())
        self.time_tick += 1

        print(f"next action :{next_action}")
        return next_action

    def train(self):
        self._optimizer.zero_grad()
        data_batch = self.replay_buffer.get_batch_experience(batch_size=self.batch_size)
        loss = torch.zeros(1).to(self._device)
        for experience in data_batch:
            action = experience.action
            im_tensor = experience.observation.image.to(self._device)
            vec_tensor = experience.observation.vector.to(self._device)
            q_vec = self.qnet(im_tensor, vec_tensor)

            reward_tensor = self.reward(vec_tensor).to(self._device)
            next_im_tensor = experience.next_observation.image.to(self._device)
            next_vec_tensor = experience.next_observation.vector.to(self._device)
            q_vec_next = self.qnet_support(next_im_tensor, next_vec_tensor).max()
            target = reward_tensor + self._reward_discount * q_vec_next.detach()
            loss += (target - q_vec[action]).pow(2)
        loss /= self.batch_size
        loss.backward()
        self._optimizer.step()

    @staticmethod
    def reward(vector_obs: torch.Tensor):
        reward = - 0.1 * vector_obs.pow(2.0).sum()
        return reward.detach()

    @property
    def eps_e_greedy(self):
        return self.__eps_e_greedy

    @eps_e_greedy.setter
    def eps_e_greedy(self, v):
        self.__eps_e_greedy = v

    def get_greedy_action(self, im_tensor, vec_tensor):
        q_val = self.qnet(im_tensor, vec_tensor).detach()
        _, index = q_val.topk(1)
        return index[0]

    def obs_to_tensor(self, raw_observation):
        im_tensor = torch.tensor(raw_observation[0]).view((
            1,
            self.shape_obs_image[2],
            self.shape_obs_image[0],
            self.shape_obs_image[1]
        ))
        vec_tensor = torch.tensor(raw_observation[1])
        return im_tensor, vec_tensor
