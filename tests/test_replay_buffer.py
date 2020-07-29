import pytest
import torch

from two_resource_dqn.replay_buffer import Observation, ReplayBuffer


def test_observation():
    obs = Observation(image_seq=tuple([torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])]),
                      vector=torch.tensor([1, 2, 3]))
    assert pytest.approx(0.0, (obs.image_seq[0] - torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])).sum())
    assert pytest.approx(0.0, (obs.vector - torch.tensor([1, 2, 3])).sum())


def test_append_one():
    buf = ReplayBuffer(buffer_size=10)
    assert len(buf.buffer_experience) == 0

    obs = Observation(image_seq=tuple([torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])]),
                      vector=torch.tensor([1, 2, 3]))
    buf.append(observation=obs, action=1, next_observation=obs)
    assert len(buf.buffer_experience) == 1


def test_append_many():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(15):
        obs = Observation(image_seq=tuple([torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])]),
                          vector=torch.tensor([1, 2, 3]))
        buf.append(observation=obs, action=1, next_observation=obs)

        v = i + 1
        if i >= buf.buffer_size:
            v = buf.buffer_size

        assert len(buf.buffer_experience) == v


def test_get_single_sample():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(10):
        obs = Observation(image_seq=tuple([torch.tensor([i])]), vector=torch.tensor([i]))
        next_obs = Observation(image_seq=tuple([torch.tensor([i+1])]), vector=torch.tensor([i+1]))
        buf.append(observation=obs, action=i, next_observation=next_obs)

    for i in range(5):
        assert buf.get_single_experience(time_step=i).observation.image_seq[0] == torch.tensor([i])
        assert buf.get_single_experience(time_step=i).observation.vector == torch.tensor([i])
        assert buf.get_single_experience(time_step=i).action == i
        assert buf.get_single_experience(time_step=i).next_observation.image_seq[0] == torch.tensor([i + 1])
        assert buf.get_single_experience(time_step=i).next_observation.vector == torch.tensor([i+1])


def test_get_batch():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(10):
        obs = Observation(image_seq=tuple([torch.tensor([i])]), vector=torch.tensor([i]))
        next_obs = Observation(image_seq=tuple([torch.tensor([i+1])]), vector=torch.tensor([i+1]))
        buf.append(observation=obs, action=i, next_observation=next_obs)

    batch = buf.get_batch_experience(batch_size=5)
    assert len(batch) == 5


def test_clear():
    buf = ReplayBuffer(buffer_size=10)
    for i in range(10):
        obs = Observation(image_seq=tuple([torch.tensor([i])]), vector=torch.tensor([i]))
        next_obs = Observation(image_seq=tuple([torch.tensor([i+1])]), vector=torch.tensor([i+1]))
        buf.append(observation=obs, action=i, next_observation=next_obs)

    buf.clear()

    assert buf.n_experience == 0
    assert len(buf.buffer_experience) == 0
    assert buf.buffer_experience.maxlen == 10
