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

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.faults.annotation import injector_injection_active
from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from tests.faults.test_midair_drop_fault import _action, _cfg, _mock_rs_env, _setup_drop_mocks


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_trigger_scheduled_drop_uses_dwell_not_immediate_recovery(
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
) -> None:
    mock_get_rs.return_value = _mock_rs_env()
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])
    mock_drop.return_value = {"object_pose_after": {"pos": [0, 0, 0]}}

    inj = MidAirDropFault(
        _cfg(post_drop_dwell_steps=2, post_drop_mode="continue_then_ik"),
        num_envs=1,
    )
    env = MagicMock()
    proposed = np.full(7, 3.0, dtype=np.float32)
    out = inj.trigger_scheduled_drop(env, 0, proposed, reason="path_uniform")
    assert inj._states[0].triggered
    assert not inj._states[0].recovery_active
    np.testing.assert_allclose(out, proposed)
    mock_dest.assert_not_called()


def test_xy_target_crossing_triggers_with_step_jump() -> None:
    from lerobot.faults.recovery.basket_drop_target import basket_distance_target_reached

    assert basket_distance_target_reached(
        prev_m=0.50,
        curr_m=0.30,
        target_m=0.36,
        band_min_m=0.34,
        band_max_m=0.38,
        held_midair=True,
    )


def test_reset_then_ik_zero_dwell_raises_at_config() -> None:
    with pytest.raises(ValueError, match="reset_then_ik"):
        FaultInjectionConfig(
            enabled=True,
            type="midair_drop",
            post_drop_mode="reset_then_ik",
            post_drop_dwell_steps=0,
        )


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_zero_dwell_scheduled_drop_executes_first_recovery_once(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
) -> None:
    _setup_drop_mocks(mock_grasped, mock_get_rs, mock_arm_q, mock_eef, mock_obj_pose, mock_drop, mock_dest)
    inj = MidAirDropFault(
        _cfg(post_drop_dwell_steps=0, post_drop_mode="immediate_ik", require_grasp=False),
        num_envs=1,
    )
    env = MagicMock()
    first = inj.trigger_scheduled_drop(env, 0, _action(1, 7, 5.0)[0], reason="path_uniform")
    assert inj._states[0].recovery_active
    second = inj.on_step(env, _action(1, 7, 99.0))
    np.testing.assert_allclose(first, second[0])
    assert inj._states[0].pending_first_recovery_action is None


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
@pytest.mark.parametrize(
    ("dwell_steps", "post_drop_mode"),
    [(0, "immediate_ik"), (2, "continue_then_ik")],
)
def test_scheduled_external_drop_marks_physical_injection_step(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
    dwell_steps: int,
    post_drop_mode: str,
) -> None:
    _setup_drop_mocks(mock_grasped, mock_get_rs, mock_arm_q, mock_eef, mock_obj_pose, mock_drop, mock_dest)
    inj = MidAirDropFault(
        _cfg(
            post_drop_dwell_steps=dwell_steps,
            post_drop_mode=post_drop_mode,
            require_grasp=False,
        ),
        num_envs=1,
    )
    env = MagicMock()
    proposed = _action(1, 7, 7.0)[0]
    inj.trigger_scheduled_drop(env, 0, proposed, reason="path_uniform")
    assert inj._states[0].mark_drop_injection_on_next_step

    inj.on_step(env, _action(1, 7, 8.0))
    assert inj._states[0].drop_injection_step
    assert inj.loss_mask_for_env(0) == 0.0
    assert injector_injection_active(inj, 0)

    inj.on_step(env, _action(1, 7, 9.0))
    assert not inj._states[0].drop_injection_step
    assert not injector_injection_active(inj, 0)
    if dwell_steps == 0:
        assert inj.loss_mask_for_env(0) == 1.0
    else:
        assert inj.loss_mask_for_env(0) == 0.0
