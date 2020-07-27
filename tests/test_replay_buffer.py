import pytest

from two_resource_dqn.dqn_agent import ReplayBuffer, BufferType, Observation


def test_observation():
    obs = Observation(image=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], vector=[1, 2, 3])
    assert obs.image == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert obs.vector == [1, 2, 3]


def test_append_one():
    buf = ReplayBuffer(buffer_size=10)
    assert len(buf._buffer[BufferType.observation]) == 0
    assert len(buf._buffer[BufferType.action]) == 0

    obs = Observation(image=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], vector=[1, 2, 3])
    buf.append(observation=obs, action=1)
    assert len(buf._buffer[BufferType.observation]) == 1
    assert len(buf._buffer[BufferType.action]) == 1


def test_append_many():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(15):
        obs = Observation(image=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], vector=[1, 2, 3])
        buf.append(observation=obs, action=1)

        v = i + 1
        if i >= buf.buffer_size:
            v = buf.buffer_size

        assert len(buf._buffer[BufferType.observation]) == v
        assert len(buf._buffer[BufferType.action]) == v


def test_get_single_sample():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(10):
        obs = Observation(image=i, vector=i)
        buf.append(observation=obs, action=i)

    for i in range(5):
        assert buf.get_single_experience(time_step=i).observation.image == i
        assert buf.get_single_experience(time_step=i).observation.vector == i
        assert buf.get_single_experience(time_step=i).action == i
        assert buf.get_single_experience(time_step=i).next_observation.image == i + 1
        assert buf.get_single_experience(time_step=i).next_observation.vector == i + 1


def test_get_batch():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(10):
        obs = Observation(image=(i, i, i), vector=(i, i, i))
        buf.append(observation=obs, action=(i, i))

    batch = buf.get_batch_experience(batch_size=5)
    assert len(batch) == 5

    for i in range(5):
        assert len(batch[i].observation.image) == 3
        assert len(batch[i].observation.vector) == 3
        assert len(batch[i].action) == 2
        assert len(batch[i].next_observation.image) == 3
        assert len(batch[i].next_observation.vector) == 3


def test_clear():
    buf = ReplayBuffer(buffer_size=10)
    for i in range(10):
        obs = Observation(image=(i, i, i), vector=(i, i, i))
        buf.append(observation=obs, action=(i, i))

    buf.clear()

    assert buf.n_experience == 0
    assert len(buf._buffer[BufferType.observation]) == 0
    assert len(buf._buffer[BufferType.action]) == 0
    assert buf._buffer[BufferType.observation].maxlen == 10
    assert buf._buffer[BufferType.action].maxlen == 10
