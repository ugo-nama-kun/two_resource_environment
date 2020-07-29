from collections import deque
import random
from typing import NamedTuple, Tuple

import torch


class Observation(NamedTuple):
    image_seq: Tuple[torch.Tensor]
    vector: torch.Tensor


class Experience(NamedTuple):
    observation: Observation
    action: int
    next_observation: Observation


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer_experience = deque(maxlen=buffer_size)
        self.n_experience = len(self.buffer_experience)

    def append(self, observation: Observation, action, next_observation: Observation):
        self.buffer_experience.append(Experience(
            observation=observation,
            action=action,
            next_observation=next_observation
        ))
        self.n_experience = len(self.buffer_experience)

    def get_single_experience(self, time_step):
        """
        Return a single experience instance
        :param time_step:
        :return:
        """
        assert self.n_experience - 1 > time_step, "Sample time step must be less than number of experience minus one."
        return self.buffer_experience[time_step]

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
        self.buffer_experience.clear()
        self.n_experience = len(self.buffer_experience)