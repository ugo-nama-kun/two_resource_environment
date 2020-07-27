import pytest

import torch

from two_resource_dqn.dqn_agent import DQNAgent


def test_reward():
    reward = DQNAgent.reward(torch.tensor([[1.0, 2.0], [2.0, 3.0]]))
    expected = [-0.1 * 5, -0.1 * 13]
    assert pytest.approx(expected[0], reward.numpy()[0], 0.001)
    assert pytest.approx(expected[1], reward.numpy()[1], 0.001)

