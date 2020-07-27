import pytest

from two_resource_dqn.dqn_agent import ReplayBuffer, BufferType


def test_append_one():
    buf = ReplayBuffer(buffer_size=10)
    assert len(buf._buffer[BufferType.observation]) == 0
    assert len(buf._buffer[BufferType.action]) == 0

    buf.append(observation=0, action=1)
    assert len(buf._buffer[BufferType.observation]) == 1
    assert len(buf._buffer[BufferType.action]) == 1


def test_append_many():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(15):
        buf.append(observation=0, action=1)

        v = i + 1
        if i >= buf.buffer_size:
            v = buf.buffer_size

        assert len(buf._buffer[BufferType.observation]) == v
        assert len(buf._buffer[BufferType.action]) == v


def test_get_single_sample():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(10):
        buf.append(observation=i, action=i)

    for i in range(5):
        assert buf.get_single_experience(time_step=i).observation == i
        assert buf.get_single_experience(time_step=i).action == i
        assert buf.get_single_experience(time_step=i).next_observation == i+1


def test_get_batch():
    buf = ReplayBuffer(buffer_size=10)

    for i in range(10):
        buf.append(observation=(i, i, i), action=(i, i))

    batch = buf.get_batch_experience(batch_size=5)
    assert len(batch) == 5

    for i in range(5):
        assert len(batch[i].observation) == 3
        assert len(batch[i].action) == 2
        assert len(batch[i].next_observation) == 3


def test_clear():
    buf = ReplayBuffer(buffer_size=10)
    for i in range(10):
        buf.append(observation=(i, i, i), action=(i, i))

    buf.clear()

    assert buf.experience_size == 0
    assert len(buf._buffer[BufferType.observation]) == 0
    assert len(buf._buffer[BufferType.action]) == 0
    assert buf._buffer[BufferType.observation].maxlen == 10
    assert buf._buffer[BufferType.action].maxlen == 10

