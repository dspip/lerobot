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

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.datagen.controllers.simple_ik import run_simple_ik_episode_loop
from lerobot.faults.datagen.drop_timing import DropDecision
from lerobot.faults.datagen.path_drop import PathTrigger
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment


def _fault_continue() -> MidAirDropFault:
    return MidAirDropFault(
        FaultInjectionConfig(
            enabled=True,
            type="midair_drop",
            probability=0.0,
            post_drop_dwell_steps=2,
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
    paired_plan = MagicMock()
    paired_plan.drop_decision = DropDecision(True, None, "injected")
    carry_path = CarryPath(
        segments=(
            PathSegment("lift", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ),
        requested_transport_offset_m=0.0,
        resolved_transport_offset_m=0.0,
        fallback=False,
    )
    paired_plan.path_trigger = PathTrigger(segment_name="lift", segment_order=0, target_t=0.3)

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
        return_value={"pos": np.array([0.4, 0.0, 0.2])},
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.get_place_destination",
        return_value=np.array([0.0, 0.0, 0.0]),
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_held_midair",
        return_value=True,
    ), patch(
        "lerobot.faults.datagen.controllers.simple_ik.is_object_grasped",
        return_value=True,
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
