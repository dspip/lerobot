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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.datagen.controllers.simple_ik import run_simple_ik_episode_loop
from lerobot.faults.datagen.drop_timing import DropDecision
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment
from tests.faults.test_midair_drop_fault import _action, _setup_drop_mocks


def _fault_continue(dwell: int = 2) -> MidAirDropFault:
    return MidAirDropFault(
        FaultInjectionConfig(
            enabled=True,
            type="midair_drop",
            probability=0.0,
            post_drop_dwell_steps=dwell,
            post_drop_mode="continue_then_ik",
            object_name="alphabet_soup_1",
        ),
        num_envs=1,
    )


@patch("lerobot.faults.datagen.controllers.simple_ik._nominal_action", return_value=np.ones(7))
def test_continue_uses_scheduled_drop_without_immediate_recovery(mock_nominal: MagicMock) -> None:
    fault = _fault_continue()
    scheduled = MagicMock(wraps=fault.trigger_scheduled_drop)
    fault.trigger_scheduled_drop = scheduled
    env = MagicMock()
    rs_env = MagicMock()

    paired_plan = SimpleNamespace(
        drop_decision=DropDecision(True, None, "injected"),
        drop_u=0.05,
    )
    carry_path = CarryPath(
        segments=(
            PathSegment("lift", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ),
        requested_transport_offset_m=0.0,
        resolved_transport_offset_m=0.0,
        fallback=False,
    )

    with patch(
        "lerobot.faults.recovery.midair_drop.get_robosuite_env",
        return_value=rs_env,
    ), patch(
        "lerobot.faults.recovery.midair_drop.get_arm_qpos",
        return_value=np.zeros(7),
    ), patch(
        "lerobot.faults.recovery.midair_drop.get_eef_pose",
        return_value=(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
    ), patch(
        "lerobot.faults.recovery.midair_drop.get_object_pose",
        return_value={"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])},
    ), patch(
        "lerobot.faults.recovery.midair_drop.midair_drop",
        return_value={"object_pose_after": {"pos": [0, 0, 0]}},
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_object_pose",
        return_value={"pos": np.array([0.55, 0.0, 0.2])},
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_place_destination",
        return_value=np.array([0.0, 0.0, 0.0]),
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_held_midair",
        return_value=True,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_grasped",
        return_value=False,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_in_basket",
        return_value=False,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik._pause_and_pump",
        return_value=False,
    ):
        run_simple_ik_episode_loop(
            env,
            rs_env,
            fault=fault,
            planner=MagicMock(phase_name="lift", carry_path=carry_path, done=False),
            recipe_drop=MagicMock(
                min_drop_distance_from_basket_m=0.3,
                hard_keepout_floor_m=0.22,
            ),
            object_name="alphabet_soup_1",
            basket_name="basket_1",
            q=1.0,
            drop_rng=np.random.default_rng(0),
            paired_plan=paired_plan,
            max_steps=1,
            gripper_settle_steps=0,
        )

    scheduled.assert_called_once()
    assert not fault._states[0].recovery_active


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_continue_dwell_timeline_matches_automatic_drop(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
) -> None:
    _setup_drop_mocks(
        mock_grasped, mock_get_rs, mock_arm_q, mock_eef, mock_obj_pose, mock_drop, mock_dest
    )
    fault = _fault_continue(dwell=2)
    env = MagicMock()
    proposed = _action(1, 7, 11.0)
    out_drop = fault.trigger_scheduled_drop(env, 0, proposed[0], reason="path_uniform")
    assert fault._states[0].triggered
    assert not fault._states[0].recovery_active
    np.testing.assert_allclose(out_drop, proposed[0])
    mock_dest.assert_not_called()

    fault.on_step(env, _action(1, 7, 12.0))

    for fill in (22.0, 33.0):
        out = fault.on_step(env, _action(1, 7, fill))
        assert not fault._states[0].recovery_active
        np.testing.assert_array_equal(out, _action(1, 7, fill))
        assert fault._states[0].dwell_steps_completed <= 2
    mock_dest.assert_not_called()

    out_rec = fault.on_step(env, _action(1, 7, 99.0))
    assert fault._states[0].recovery_active
    mock_dest.assert_called_once()
    assert not np.allclose(out_rec, _action(1, 7, 99.0))


@patch("lerobot.faults.recovery.midair_drop.is_object_in_basket", return_value=False)
@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_continue_first_post_drop_grasp_clears_suppress_then_skip(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
    mock_in_basket,
) -> None:
    _setup_drop_mocks(
        mock_grasped, mock_get_rs, mock_arm_q, mock_eef, mock_obj_pose, mock_drop, mock_dest
    )
    mock_grasped.side_effect = [False, True, True]

    fault = _fault_continue(dwell=2)
    env = MagicMock()
    fault.trigger_scheduled_drop(env, 0, _action(1, 7, 1.0)[0], reason="path_uniform")
    fault.on_step(env, _action(1, 7, 2.0))
    fault.on_step(env, _action(1, 7, 3.0))
    fault.on_step(env, _action(1, 7, 4.0))
    skipped, reason = fault.post_drop_recovery_skipped(0)
    assert skipped
    assert reason == "regrasp_during_dwell"
    assert not fault._states[0].recovery_active
    mock_dest.assert_not_called()


@patch("lerobot.faults.datagen.controllers.simple_ik.sample_path_drop")
@patch("lerobot.faults.datagen.controllers.simple_ik._nominal_action", return_value=np.ones(7))
def test_paired_no_drop_never_resamples_or_schedules(
    _mock_nominal: MagicMock,
    mock_sample_path_drop: MagicMock,
) -> None:
    fault = _fault_continue()
    scheduled = MagicMock(wraps=fault.trigger_scheduled_drop)
    fault.trigger_scheduled_drop = scheduled
    env = MagicMock()
    rs_env = MagicMock()
    paired_plan = SimpleNamespace(
        drop_decision=DropDecision(False, None, "paired_no_drop"),
        drop_u=0.05,
    )
    carry_path = CarryPath(
        segments=(PathSegment("lift", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        requested_transport_offset_m=0.0,
        resolved_transport_offset_m=0.0,
        fallback=False,
    )
    with patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_object_pose",
        return_value={"pos": np.array([0.55, 0.0, 0.2])},
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_place_destination",
        return_value=np.array([0.0, 0.0, 0.0]),
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_held_midair",
        return_value=True,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_grasped",
        return_value=False,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_in_basket",
        return_value=False,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik._pause_and_pump",
        return_value=False,
    ):
        facts = run_simple_ik_episode_loop(
            env,
            rs_env,
            fault=fault,
            planner=MagicMock(phase_name="lift", carry_path=carry_path, done=False),
            recipe_drop=MagicMock(
                min_drop_distance_from_basket_m=0.3,
                hard_keepout_floor_m=0.22,
            ),
            object_name="alphabet_soup_1",
            basket_name="basket_1",
            q=1.0,
            drop_rng=np.random.default_rng(0),
            paired_plan=paired_plan,
            max_steps=3,
            gripper_settle_steps=0,
        )
    mock_sample_path_drop.assert_not_called()
    scheduled.assert_not_called()
    assert facts.outcome == "nominal_no_drop"
    assert facts.success is False
    assert facts.drop_trigger == {
        "kind": "paired_skipped",
        "reason": "paired_no_drop",
    }


@patch("lerobot.faults.datagen.controllers.simple_ik.sample_path_drop")
@patch("lerobot.faults.datagen.controllers.simple_ik._nominal_action", return_value=None)
def test_paired_no_drop_success_when_object_in_basket(
    _mock_nominal: MagicMock,
    mock_sample_path_drop: MagicMock,
) -> None:
    fault = _fault_continue()
    env = MagicMock()
    rs_env = MagicMock()
    paired_plan = SimpleNamespace(
        drop_decision=DropDecision(False, None, "paired_q_skip"),
        drop_u=0.05,
    )
    carry_path = CarryPath(
        segments=(PathSegment("lift", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        requested_transport_offset_m=0.0,
        resolved_transport_offset_m=0.0,
        fallback=False,
    )
    with patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_object_pose",
        return_value={"pos": np.array([0.55, 0.0, 0.2])},
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_place_destination",
        return_value=np.array([0.0, 0.0, 0.0]),
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_held_midair",
        return_value=True,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_in_basket",
        return_value=True,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik._pause_and_pump",
        return_value=False,
    ):
        facts = run_simple_ik_episode_loop(
            env,
            rs_env,
            fault=fault,
            planner=MagicMock(phase_name="lift", carry_path=carry_path, done=True),
            recipe_drop=MagicMock(
                min_drop_distance_from_basket_m=0.3,
                hard_keepout_floor_m=0.22,
            ),
            object_name="alphabet_soup_1",
            basket_name="basket_1",
            q=1.0,
            drop_rng=np.random.default_rng(0),
            paired_plan=paired_plan,
            max_steps=3,
            gripper_settle_steps=0,
        )
    mock_sample_path_drop.assert_not_called()
    assert facts.outcome == "nominal_no_drop"
    assert facts.success is True
    assert facts.drop_trigger == {
        "kind": "paired_skipped",
        "reason": "paired_q_skip",
    }


@patch("lerobot.faults.recovery.midair_drop.is_object_in_basket", return_value=False)
@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
@patch("lerobot.faults.datagen.controllers.simple_ik._nominal_action", return_value=np.ones(7))
def test_loop_returns_skipped_recovery_outcome(
    mock_nominal: MagicMock,
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
    mock_in_basket,
) -> None:
    _setup_drop_mocks(
        mock_grasped, mock_get_rs, mock_arm_q, mock_eef, mock_obj_pose, mock_drop, mock_dest
    )
    mock_grasped.side_effect = [False, True, True, True, True, True]

    fault = _fault_continue(dwell=2)
    env = MagicMock()
    rs_env = MagicMock()

    def _env_step(action: np.ndarray) -> tuple:
        fault.on_step(env, action)
        return (None, 0.0, False, False, {})

    env.step.side_effect = _env_step

    paired_plan = SimpleNamespace(
        drop_decision=DropDecision(True, None, "injected"),
        drop_u=0.0,
    )
    carry_path = CarryPath(
        segments=(PathSegment("lift", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),),
        requested_transport_offset_m=0.0,
        resolved_transport_offset_m=0.0,
        fallback=False,
    )

    with patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_object_pose",
        return_value={"pos": np.array([1.0, 0.0, 0.2])},
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_place_destination",
        return_value=np.array([0.0, 0.0, 0.0]),
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_held_midair",
        return_value=True,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_grasped",
        return_value=False,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik._pause_and_pump",
        return_value=False,
    ):
        facts = run_simple_ik_episode_loop(
            env,
            rs_env,
            fault=fault,
            planner=MagicMock(phase_name="lift", carry_path=carry_path, done=False),
            recipe_drop=MagicMock(
                min_drop_distance_from_basket_m=0.3,
                hard_keepout_floor_m=0.22,
            ),
            object_name="alphabet_soup_1",
            basket_name="basket_1",
            q=1.0,
            drop_rng=np.random.default_rng(0),
            paired_plan=paired_plan,
            max_steps=20,
            gripper_settle_steps=0,
        )

    assert facts.outcome == "regrasp_during_dwell"
    assert facts.actual_dwell_steps == 1
