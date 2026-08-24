# Copyright 2026 Gangelia. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for FaultEnvWrapper and maybe_wrap_env_tree."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

from lerobot.faults import FaultEnvWrapper, FaultInjectionConfig, maybe_wrap_env, maybe_wrap_env_tree


class _DummyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.last_action = None
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.last_action = None
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.t += 1
        terminated = self.t >= 10
        return (
            np.zeros(2, dtype=np.float32),
            0.0,
            terminated,
            False,
            {},
        )


def test_maybe_wrap_disabled_returns_same_env():
    env = _DummyEnv()
    out = maybe_wrap_env(env, FaultInjectionConfig(enabled=False))
    assert out is env


def test_maybe_wrap_env_tree_disabled_returns_same_tree():
    tree = {"libero_object": {0: _DummyEnv()}}
    out = maybe_wrap_env_tree(tree, FaultInjectionConfig(enabled=False))
    assert out is tree


def test_wrapper_holds_action(tmp_path):
    cfg = FaultInjectionConfig(
        enabled=True,
        trigger_step=2,
        duration=2,
        probability=1.0,
        seed=0,
        log_path=tmp_path / "events.jsonl",
    )
    env = FaultEnvWrapper(_DummyEnv(), cfg)
    env.reset()
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    env.step(np.array([2.0, 2.0], dtype=np.float32))
    env.step(np.array([3.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, [2.0, 2.0])
    env.step(np.array([4.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, [2.0, 2.0])
    env.step(np.array([5.0, 5.0], dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, [5.0, 5.0])
    env.close()
    assert (tmp_path / "events.jsonl").exists()


def test_wrap_tree_handles_vector_env(tmp_path):
    assert not issubclass(SyncVectorEnv, gym.Env)

    def _make():
        return _DummyEnv()

    vec = SyncVectorEnv([_make])
    cfg = FaultInjectionConfig(
        enabled=True,
        trigger_step=1,
        duration=1,
        log_path=tmp_path / "vec.jsonl",
    )
    tree = {"libero_object": {0: vec}}
    wrapped_tree = maybe_wrap_env_tree(tree, cfg)
    wrapped = wrapped_tree["libero_object"][0]
    assert isinstance(wrapped, FaultEnvWrapper)
    wrapped.reset()
    wrapped.step(np.array([[1.0, 1.0]], dtype=np.float32))
    wrapped.step(np.array([[9.0, 9.0]], dtype=np.float32))
    np.testing.assert_array_equal(wrapped.unwrapped.envs[0].last_action, [1.0, 1.0])
    wrapped.close()


class _ImageEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
                "state": spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.last_action = None
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.last_action = None
        return {
            "pixels": np.full((8, 8, 3), 120, dtype=np.uint8),
            "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }, {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.t += 1
        terminated = self.t >= 10
        return (
            {
                "pixels": np.full((8, 8, 3), 120 + self.t, dtype=np.uint8),
                "state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            },
            0.0,
            terminated,
            False,
            {},
        )


def test_wrapper_jitters_action(tmp_path):
    cfg = FaultInjectionConfig(
        enabled=True,
        type="action_jitter",
        noise_std=0.1,
        seed=0,
        log_path=tmp_path / "jitter_events.jsonl",
    )
    env = FaultEnvWrapper(_DummyEnv(), cfg)
    env.reset()
    proposed = np.array([1.0, 1.0], dtype=np.float32)
    env.step(proposed)
    assert not np.allclose(env.unwrapped.last_action, proposed)
    env.close()
    assert (tmp_path / "jitter_events.jsonl").exists()


def test_wrapper_delays_action(tmp_path):
    cfg = FaultInjectionConfig(
        enabled=True,
        type="action_delay",
        delay_steps=2,
        log_path=tmp_path / "delay_events.jsonl",
    )
    env = FaultEnvWrapper(_DummyEnv(), cfg)
    env.reset()
    env.step(np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, [1.0, 1.0])
    env.step(np.array([2.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, [2.0, 2.0])
    env.step(np.array([3.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, [1.0, 1.0])
    env.step(np.array([4.0, 4.0], dtype=np.float32))
    np.testing.assert_array_equal(env.unwrapped.last_action, [2.0, 2.0])
    env.close()
    assert (tmp_path / "delay_events.jsonl").exists()


def test_wrapper_sensor_dropout_blackens_obs(tmp_path):
    cfg = FaultInjectionConfig(
        enabled=True,
        type="sensor_dropout",
        trigger_step=0,
        duration=2,
        log_path=tmp_path / "sensor_events.jsonl",
        diag_dir=tmp_path / "diag",
    )
    env = FaultEnvWrapper(_ImageEnv(), cfg)
    obs, _ = env.reset()
    assert obs["pixels"].max() == 120
    obs, *_ = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert obs["pixels"].max() == 0
    np.testing.assert_array_equal(obs["state"], [1.0, 2.0, 3.0])
    obs, *_ = env.step(np.array([2.0, 2.0], dtype=np.float32))
    assert obs["pixels"].max() == 0
    obs, *_ = env.step(np.array([3.0, 3.0], dtype=np.float32))
    assert obs["pixels"].max() > 0
    env.close()
    assert (tmp_path / "sensor_events.jsonl").exists()
    if __import__("importlib").util.find_spec("PIL") is not None:
        assert list((tmp_path / "diag").glob("*.png"))
