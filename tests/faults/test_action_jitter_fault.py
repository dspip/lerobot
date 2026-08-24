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

"""Unit tests for ActionJitterFault (no LeRobot import required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.faults import ActionJitterFault, FaultEventLogger, FaultInjectionConfig, make_fault_injector


def _cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="action_jitter",
        noise_std=0.05,
        seed=42,
        env_ids=None,
        log_path=None,
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _action(batch: int, dim: int, fill: float) -> np.ndarray:
    return np.full((batch, dim), fill, dtype=np.float32)


def test_jitter_adds_noise():
    inj = ActionJitterFault(_cfg(noise_std=0.1, seed=0), num_envs=1)
    proposed = _action(1, 3, 1.0)
    out = inj.apply(proposed)
    assert not np.allclose(out, proposed)
    assert out.shape == proposed.shape


def test_same_seed_reproducible():
    def _first_noise(seed: int) -> np.ndarray:
        inj = ActionJitterFault(_cfg(noise_std=0.1, seed=seed), num_envs=1)
        proposed = _action(1, 4, 0.0)
        return inj.apply(proposed).copy()

    a = _first_noise(123)
    b = _first_noise(123)
    c = _first_noise(999)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_noise_mean_near_zero():
    inj = ActionJitterFault(_cfg(noise_std=0.05, seed=7), num_envs=1)
    proposed = np.zeros((1, 1), dtype=np.float32)
    samples = []
    for _ in range(5000):
        out = inj.apply(proposed)
        samples.append(float(out[0, 0]))
    mean = np.mean(samples)
    assert abs(mean) < 0.005


def test_disabled_is_exact_noop():
    cfg = _cfg(enabled=False)
    assert make_fault_injector(cfg, num_envs=2) is None
    inj = ActionJitterFault(cfg, num_envs=1)
    proposed = _action(1, 4, 5.0)
    out = inj.apply(proposed)
    assert out is proposed


def test_noise_std_zero_is_identity_no_log(tmp_path: Path):
    log_path = tmp_path / "jitter.jsonl"
    logger = FaultEventLogger(log_path)
    inj = ActionJitterFault(_cfg(noise_std=0.0), num_envs=1, event_logger=logger)
    proposed = _action(1, 2, 3.0)
    out = inj.apply(proposed)
    np.testing.assert_array_equal(out, proposed)
    logger.close()
    assert log_path.read_text().strip() == ""


def test_vector_envs_maintain_separate_state():
    inj = ActionJitterFault(_cfg(noise_std=0.1, seed=5), num_envs=2)
    proposed = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    out = inj.apply(proposed)
    assert not np.allclose(out[0], out[1])


def test_reset_clears_episode_state():
    inj = ActionJitterFault(_cfg(noise_std=0.1, seed=1), num_envs=1)
    first = inj.apply(_action(1, 2, 1.0)).copy()
    inj.reset()
    second = inj.apply(_action(1, 2, 1.0)).copy()
    np.testing.assert_array_equal(first, second)


def test_logging_on_every_step(tmp_path: Path):
    log_path = tmp_path / "jitter.jsonl"
    logger = FaultEventLogger(log_path)
    inj = ActionJitterFault(_cfg(noise_std=0.05, seed=0), num_envs=1, event_logger=logger)
    for fill in [1.0, 2.0, 3.0]:
        inj.apply(_action(1, 1, fill), episode_ids=[5])
    logger.close()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(line) for line in lines]
    assert events[0]["status"] == "activated"
    assert events[1]["status"] == "active"
    assert events[0]["event"] == "action_jitter"
    assert events[0]["evaluation_episode_id"] == 5
    assert events[0]["noise_std"] == 0.05
    assert len(events[0]["noise"]) == 1
    assert events[0]["proposed_action"] == [1.0]
    assert events[0]["executed_jittered_action"][0] == pytest.approx(
        1.0 + events[0]["noise"][0], rel=1e-5
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"noise_std": -0.1}, "noise_std"),
        ({"env_ids": []}, "env_ids"),
        ({"env_ids": [-1]}, "env_ids"),
        ({"env_ids": [0, 0]}, "duplicates"),
        ({"type": "drop_object"}, "Unsupported fault type"),
    ],
)
def test_invalid_config_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FaultInjectionConfig(enabled=True, **{**dict(type="action_jitter", noise_std=0.05), **kwargs})


def test_action_jitter_ignores_hold_delay_fields():
    cfg = FaultInjectionConfig(
        enabled=True,
        type="action_jitter",
        noise_std=0.05,
        trigger_step=0,
        duration=0,
        delay_steps=0,
        probability=2.0,
    )
    cfg.validate()


def test_notify_dones_marks_finished_no_jitter():
    inj = ActionJitterFault(_cfg(noise_std=0.1, seed=0), num_envs=2)
    out = inj.apply(np.array([[1.0], [10.0]], dtype=np.float32))
    assert not np.allclose(out[0], [1.0])
    inj.notify_dones(np.array([True, False]))
    out = inj.apply(np.array([[2.0], [20.0]], dtype=np.float32))
    np.testing.assert_array_equal(out[0], [2.0])
    assert not np.allclose(out[1], [20.0])


def test_notify_dones_then_reset_restarts_jitter():
    inj = ActionJitterFault(_cfg(noise_std=0.1, seed=0), num_envs=1)
    first = inj.apply(_action(1, 1, 1.0)).copy()
    inj.notify_dones(np.array([True]))
    passthrough = inj.apply(_action(1, 1, 9.0))
    np.testing.assert_array_equal(passthrough, [[9.0]])
    inj.reset(episode_ids=[1])
    second = inj.apply(_action(1, 1, 1.0)).copy()
    np.testing.assert_array_equal(first, second)


def test_make_fault_injector_returns_jitter():
    inj = make_fault_injector(_cfg(noise_std=0.07), num_envs=1)
    assert isinstance(inj, ActionJitterFault)
    assert inj.config.noise_std == 0.07


def test_unselected_envs_pass_through():
    inj = ActionJitterFault(_cfg(noise_std=0.1, seed=0, env_ids=[1]), num_envs=2)
    proposed = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    out = inj.apply(proposed)
    np.testing.assert_array_equal(out[0], [1.0, 1.0])
    assert not np.allclose(out[1], [2.0, 2.0])
