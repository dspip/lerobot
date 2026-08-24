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

"""Unit tests for object_slip and eef_bump (mocked sim helpers)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.factory import make_sim_inject_fault
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.sim.eef_bump import EefBumpFault
from lerobot.faults.sim.object_slip import ObjectSlipFault
from lerobot.faults.wrappers import SimFaultEnvWrapper, maybe_wrap_env


def _slip_cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="object_slip",
        t_min=0,
        t_max=5,
        probability=1.0,
        seed=0,
        require_grasp=True,
        min_object_z=0.0,
        slip_pos_std=0.01,
        slip_yaw_std=0.05,
        settle_steps=1,
        object_name="alphabet_soup_1",
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _bump_cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="eef_bump",
        t_min=0,
        t_max=5,
        probability=1.0,
        seed=0,
        bump_force_std=10.0,
        settle_steps=2,
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


@patch("lerobot.faults.sim.object_slip.apply_object_pose_delta")
@patch("lerobot.faults.sim.object_slip.get_eef_pose")
@patch("lerobot.faults.sim.object_slip.get_object_pose")
@patch("lerobot.faults.sim.object_slip.is_object_grasped", return_value=True)
@patch("lerobot.faults.sim.object_slip.get_robosuite_env")
def test_object_slip_triggers_once(mock_rs, mock_grasp, mock_pose, mock_eef, mock_delta, tmp_path: Path):
    mock_pose.return_value = {"pos": np.array([0.0, 0.0, 0.2]), "quat_wxyz": np.zeros(4)}
    mock_eef.return_value = (np.array([0.0, 0.0, 0.2]), np.zeros(4))
    mock_delta.return_value = {
        "object_name": "alphabet_soup_1",
        "dpos": np.array([0.01, 0.0, 0.0]),
        "dyaw": 0.1,
        "pre_object_pos": np.zeros(3),
        "post_object_pos": np.array([0.01, 0.0, 0.0]),
    }
    log_path = tmp_path / "slip.jsonl"
    logger = FaultEventLogger(log_path)
    fault = ObjectSlipFault(_slip_cfg(), num_envs=1, event_logger=logger)
    actions = np.zeros((1, 7), dtype=np.float32)
    env = MagicMock()
    out1 = fault.on_step(env, actions)
    out2 = fault.on_step(env, actions)
    np.testing.assert_array_equal(out1, actions)
    np.testing.assert_array_equal(out2, actions)
    assert mock_delta.call_count == 1
    logger.close()
    events = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert events[0]["event"] == "object_slip"


@patch("lerobot.faults.sim.eef_bump.apply_eef_bump")
@patch("lerobot.faults.sim.eef_bump.get_robosuite_env")
def test_eef_bump_triggers_once(mock_rs, mock_bump, tmp_path: Path):
    mock_bump.return_value = {
        "force": np.array([1.0, 0.0, 0.0]),
        "body_id": 3,
        "eef_pre": np.zeros(3),
        "eef_post": np.array([0.01, 0.0, 0.0]),
        "settle_steps": 2,
    }
    log_path = tmp_path / "bump.jsonl"
    logger = FaultEventLogger(log_path)
    fault = EefBumpFault(_bump_cfg(), num_envs=1, event_logger=logger)
    actions = np.ones((1, 7), dtype=np.float32)
    env = MagicMock()
    fault.on_step(env, actions)
    fault.on_step(env, actions)
    assert mock_bump.call_count == 1
    logger.close()
    events = [json.loads(line) for line in log_path.read_text().strip().splitlines()]
    assert events[0]["event"] == "eef_bump"


def test_factory_and_wrapper_routing():
    slip = make_sim_inject_fault(_slip_cfg(), num_envs=1)
    bump = make_sim_inject_fault(_bump_cfg(), num_envs=1)
    assert isinstance(slip, ObjectSlipFault)
    assert isinstance(bump, EefBumpFault)

    class _Env:
        num_envs = 1
        action_space = None

        def reset(self, **kwargs):
            return np.zeros(2), {}

        def step(self, action):
            return np.zeros(2), 0.0, False, False, {}

        def close(self):
            return None

    wrapped = maybe_wrap_env(_Env(), _slip_cfg())
    assert isinstance(wrapped, SimFaultEnvWrapper)
