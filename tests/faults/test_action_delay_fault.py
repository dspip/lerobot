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

"""Unit tests for ActionDelayFault (no LeRobot import required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.faults import ActionDelayFault, FaultEventLogger, FaultInjectionConfig, make_fault_injector


def _cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="action_delay",
        delay_steps=3,
        env_ids=None,
        log_path=None,
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _action(batch: int, dim: int, fill: float) -> np.ndarray:
    return np.full((batch, dim), fill, dtype=np.float32)


def test_warmup_passes_through_and_enqueues():
    inj = ActionDelayFault(_cfg(delay_steps=3), num_envs=1)
    for fill in [1.0, 2.0, 3.0]:
        proposed = _action(1, 2, fill)
        out = inj.apply(proposed)
        np.testing.assert_array_equal(out, proposed)


def test_delay_two_executes_action_from_two_steps_ago():
    inj = ActionDelayFault(_cfg(delay_steps=2), num_envs=1)
    outs = []
    for fill in [10.0, 20.0, 30.0, 40.0, 50.0]:
        outs.append(inj.apply(_action(1, 1, fill)).copy())

    np.testing.assert_array_equal(outs[0], [[10.0]])
    np.testing.assert_array_equal(outs[1], [[20.0]])
    np.testing.assert_array_equal(outs[2], [[10.0]])
    np.testing.assert_array_equal(outs[3], [[20.0]])
    np.testing.assert_array_equal(outs[4], [[30.0]])


def test_reset_clears_buffers():
    inj = ActionDelayFault(_cfg(delay_steps=2), num_envs=1)
    inj.apply(_action(1, 1, 1.0))
    inj.apply(_action(1, 1, 2.0))
    delayed = inj.apply(_action(1, 1, 99.0))
    np.testing.assert_array_equal(delayed, [[1.0]])
    inj.reset()
    out = inj.apply(_action(1, 1, 7.0))
    np.testing.assert_array_equal(out, [[7.0]])
    out = inj.apply(_action(1, 1, 8.0))
    np.testing.assert_array_equal(out, [[8.0]])
    out = inj.apply(_action(1, 1, 9.0))
    np.testing.assert_array_equal(out, [[7.0]])


def test_disabled_is_exact_noop():
    cfg = _cfg(enabled=False)
    assert make_fault_injector(cfg, num_envs=2) is None
    inj = ActionDelayFault(cfg, num_envs=1)
    proposed = _action(1, 4, 5.0)
    out = inj.apply(proposed)
    assert out is proposed


def test_vector_envs_maintain_separate_state():
    inj = ActionDelayFault(_cfg(delay_steps=2), num_envs=2)
    a0 = np.array([[1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
    a1 = np.array([[2.0, 2.0], [20.0, 20.0]], dtype=np.float32)
    a2 = np.array([[3.0, 3.0], [30.0, 30.0]], dtype=np.float32)
    assert np.allclose(inj.apply(a0), a0)
    assert np.allclose(inj.apply(a1), a1)
    delayed = inj.apply(a2)
    np.testing.assert_array_equal(delayed[0], [1.0, 1.0])
    np.testing.assert_array_equal(delayed[1], [10.0, 10.0])


def test_logging_on_delayed_steps(tmp_path: Path):
    log_path = tmp_path / "delay.jsonl"
    logger = FaultEventLogger(log_path)
    inj = ActionDelayFault(_cfg(delay_steps=2), num_envs=1, event_logger=logger)
    for fill in [1.0, 2.0, 3.0, 4.0, 5.0]:
        inj.apply(_action(1, 1, fill), episode_ids=[3])
    logger.close()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 3
    events = [json.loads(line) for line in lines]
    assert events[0]["status"] == "activated"
    assert events[0]["event"] == "action_delay"
    assert events[0]["evaluation_episode_id"] == 3
    assert events[0]["executed_delayed_action"] == [1.0]
    assert events[0]["proposed_action"] == [3.0]
    assert events[1]["status"] == "active"
    assert events[2]["status"] == "active"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"delay_steps": 0}, "delay_steps"),
        ({"env_ids": []}, "env_ids"),
        ({"env_ids": [-1]}, "env_ids"),
        ({"env_ids": [0, 0]}, "duplicates"),
        ({"type": "drop_object"}, "Unsupported fault type"),
    ],
)
def test_invalid_config_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FaultInjectionConfig(enabled=True, **{**dict(type="action_delay", delay_steps=2), **kwargs})


def test_action_delay_ignores_hold_specific_fields():
    cfg = FaultInjectionConfig(
        enabled=True,
        type="action_delay",
        delay_steps=2,
        trigger_step=0,
        duration=0,
        probability=2.0,
    )
    cfg.validate()


def test_notify_dones_clears_finished_env():
    inj = ActionDelayFault(_cfg(delay_steps=2), num_envs=2)
    inj.apply(np.array([[1.0], [10.0]], dtype=np.float32))
    inj.apply(np.array([[2.0], [20.0]], dtype=np.float32))
    inj.notify_dones(np.array([True, False]))
    out = inj.apply(np.array([[3.0], [30.0]], dtype=np.float32))
    np.testing.assert_array_equal(out[0], [3.0])
    np.testing.assert_array_equal(out[1], [10.0])


def test_notify_dones_then_reset_restarts_delay():
    inj = ActionDelayFault(_cfg(delay_steps=2), num_envs=1)
    inj.apply(_action(1, 1, 1.0))
    inj.apply(_action(1, 1, 2.0))
    inj.notify_dones(np.array([True]))
    # Finished: pass-through, no buffering.
    np.testing.assert_array_equal(inj.apply(_action(1, 1, 9.0)), [[9.0]])
    inj.reset(episode_ids=[1])
    # New episode warm-up, then delay of a0=1 at third step.
    np.testing.assert_array_equal(inj.apply(_action(1, 1, 1.0)), [[1.0]])
    np.testing.assert_array_equal(inj.apply(_action(1, 1, 2.0)), [[2.0]])
    np.testing.assert_array_equal(inj.apply(_action(1, 1, 3.0)), [[1.0]])


def test_make_fault_injector_returns_delay():
    inj = make_fault_injector(_cfg(delay_steps=4), num_envs=1)
    assert isinstance(inj, ActionDelayFault)
    assert inj.config.delay_steps == 4
