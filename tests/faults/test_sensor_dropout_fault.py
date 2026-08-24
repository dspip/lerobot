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

"""Unit tests for SensorDropoutFault (no LeRobot import required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.faults import (
    FaultEventLogger,
    FaultInjectionConfig,
    SensorDropoutFault,
    make_obs_fault_injector,
)


def _cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="sensor_dropout",
        trigger_step=2,
        duration=2,
        log_path=None,
        diag_dir=None,
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _obs(image_fill: int = 128, proprio_fill: float = 1.5) -> dict:
    return {
        "pixels": {
            "camera1": np.full((64, 64, 3), image_fill, dtype=np.uint8),
            "camera2": np.full((64, 64, 3), image_fill, dtype=np.uint8),
        },
        "state": np.full((7,), proprio_fill, dtype=np.float32),
    }


def test_rotation_matrix_not_treated_as_image():
    from lerobot.faults.observation.utils import is_image_field

    assert not is_image_field("eef_mat", np.zeros((3, 3), dtype=np.float32))
    assert not is_image_field("eef_mat", np.zeros((1, 3, 3), dtype=np.float32))
    assert is_image_field("camera1", np.zeros((64, 64, 3), dtype=np.uint8))


def test_no_blackout_before_trigger():
    fault = SensorDropoutFault(_cfg(trigger_step=3, duration=2), num_envs=1)
    obs = _obs()
    out = fault.apply_obs(obs)
    np.testing.assert_array_equal(out["pixels"]["camera1"], obs["pixels"]["camera1"])
    np.testing.assert_array_equal(out["state"], obs["state"])


def test_blackout_zeros_images_not_proprio():
    fault = SensorDropoutFault(_cfg(trigger_step=0, duration=1), num_envs=1)
    obs = _obs(image_fill=200, proprio_fill=2.5)
    out = fault.apply_obs(obs)
    assert out["pixels"]["camera1"].max() == 0
    assert out["pixels"]["camera2"].max() == 0
    np.testing.assert_array_equal(out["state"], obs["state"])


def test_blackout_respects_duration():
    fault = SensorDropoutFault(_cfg(trigger_step=0, duration=2), num_envs=1)
    first = fault.apply_obs(_obs(image_fill=100))
    assert first["pixels"]["camera1"].max() == 0
    second = fault.apply_obs(_obs(image_fill=100))
    assert second["pixels"]["camera1"].max() == 0
    third = fault.apply_obs(_obs(image_fill=100))
    assert third["pixels"]["camera1"].max() == 100


def test_reset_obs_unchanged():
    fault = SensorDropoutFault(_cfg(trigger_step=0, duration=5), num_envs=1)
    out = fault.apply_obs(_obs(), from_reset=True)
    assert out["pixels"]["camera1"].max() == 128


def test_vector_env_blackout_per_env():
    fault = SensorDropoutFault(_cfg(trigger_step=0, duration=1, env_ids=[1]), num_envs=2)
    obs = {
        "pixels": {
            "camera1": np.full((2, 32, 32, 3), 99, dtype=np.uint8),
        },
        "state": np.ones((2, 4), dtype=np.float32),
    }
    out = fault.apply_obs(obs)
    assert out["pixels"]["camera1"][0].max() == 99
    assert out["pixels"]["camera1"][1].max() == 0
    np.testing.assert_array_equal(out["state"], obs["state"])


def test_logging_on_blackout(tmp_path: Path):
    log_path = tmp_path / "dropout.jsonl"
    logger = FaultEventLogger(log_path)
    fault = SensorDropoutFault(
        _cfg(trigger_step=0, duration=2),
        num_envs=1,
        event_logger=logger,
    )
    fault.apply_obs(_obs())
    fault.apply_obs(_obs())
    logger.close()

    events = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert len(events) == 2
    assert events[0]["event"] == "sensor_dropout"
    assert events[0]["status"] == "activated"
    assert events[1]["status"] == "completed"


def test_diag_writes_pngs(tmp_path: Path):
    pytest.importorskip("PIL")
    diag_dir = tmp_path / "diag"
    fault = SensorDropoutFault(
        _cfg(trigger_step=0, duration=20, diag_dir=diag_dir),
        num_envs=1,
    )
    fault.apply_obs(_obs())
    for _ in range(9):
        fault.apply_obs(_obs())
    fault.apply_obs(_obs())  # step 10 blackout -> second dump

    pngs = sorted(diag_dir.glob("*.png"))
    assert len(pngs) >= 2
    assert pngs[0].stat().st_size > 0


def test_notify_dones_then_reset():
    fault = SensorDropoutFault(_cfg(trigger_step=0, duration=3), num_envs=1)
    black = fault.apply_obs(_obs())
    assert black["pixels"]["camera1"].max() == 0
    fault.notify_dones(np.array([True]))
    passthrough = fault.apply_obs(_obs(image_fill=77))
    assert passthrough["pixels"]["camera1"].max() == 77
    fault.reset()
    black_again = fault.apply_obs(_obs())
    assert black_again["pixels"]["camera1"].max() == 0


def test_make_obs_fault_injector_returns_sensor_dropout():
    inj = make_obs_fault_injector(_cfg(), num_envs=1)
    assert isinstance(inj, SensorDropoutFault)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"trigger_step": -1}, "trigger_step"),
        ({"duration": 0}, "duration"),
        ({"env_ids": []}, "env_ids"),
    ],
)
def test_invalid_config_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FaultInjectionConfig(enabled=True, **{**dict(type="sensor_dropout"), **kwargs})


def test_make_obs_fault_injector_none_for_action_fault():
    assert make_obs_fault_injector(
        FaultInjectionConfig(enabled=True, type="action_hold"),
        num_envs=1,
    ) is None
