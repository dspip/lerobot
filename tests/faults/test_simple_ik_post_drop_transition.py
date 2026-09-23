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

from lerobot.faults.datagen.controllers.simple_ik import (
    SimpleIKPostDropAction,
    decide_simple_ik_post_drop,
    run_simple_ik_episode_loop,
)
from lerobot.faults.datagen.recipe import PostDropMode


def test_immediate_ik_requests_recovery_on_drop_step() -> None:
    action = decide_simple_ik_post_drop(
        mode=PostDropMode.IMMEDIATE_IK,
        dwell_steps=80,
        just_dropped=True,
        dwell_elapsed=0,
        recovery_started=False,
        regrasped=False,
        in_basket=False,
    )
    assert action is SimpleIKPostDropAction.REQUEST_RECOVERY_NOW


def test_continue_then_ik_defers_recovery_at_drop() -> None:
    action = decide_simple_ik_post_drop(
        mode=PostDropMode.CONTINUE_THEN_IK,
        dwell_steps=80,
        just_dropped=True,
        dwell_elapsed=0,
        recovery_started=False,
        regrasped=False,
        in_basket=False,
    )
    assert action is SimpleIKPostDropAction.CONTINUE_NOMINAL


def test_continue_then_ik_keeps_nominal_until_dwell_elapsed() -> None:
    action = decide_simple_ik_post_drop(
        mode=PostDropMode.CONTINUE_THEN_IK,
        dwell_steps=80,
        just_dropped=False,
        dwell_elapsed=79,
        recovery_started=False,
        regrasped=False,
        in_basket=False,
    )
    assert action is SimpleIKPostDropAction.CONTINUE_NOMINAL


def test_continue_then_ik_requests_recovery_after_dwell() -> None:
    action = decide_simple_ik_post_drop(
        mode=PostDropMode.CONTINUE_THEN_IK,
        dwell_steps=80,
        just_dropped=False,
        dwell_elapsed=80,
        recovery_started=False,
        regrasped=False,
        in_basket=False,
    )
    assert action is SimpleIKPostDropAction.REQUEST_RECOVERY_NOW


def test_continue_skips_recovery_if_regrasped_during_dwell() -> None:
    action = decide_simple_ik_post_drop(
        mode=PostDropMode.CONTINUE_THEN_IK,
        dwell_steps=80,
        just_dropped=False,
        dwell_elapsed=80,
        recovery_started=False,
        regrasped=True,
        in_basket=False,
    )
    assert action is SimpleIKPostDropAction.SKIP_RECOVERY


def test_continue_skips_recovery_if_in_basket_during_dwell() -> None:
    action = decide_simple_ik_post_drop(
        mode=PostDropMode.CONTINUE_THEN_IK,
        dwell_steps=80,
        just_dropped=False,
        dwell_elapsed=10,
        recovery_started=False,
        regrasped=False,
        in_basket=True,
    )
    assert action is SimpleIKPostDropAction.SKIP_RECOVERY


@patch("lerobot.faults.datagen.controllers.simple_ik._nominal_action", return_value=np.zeros(7))
def test_episode_loop_continue_does_not_request_recovery_on_drop(mock_nominal: MagicMock) -> None:
    fault = MagicMock()
    state = MagicMock()
    state.planner = None
    fault._states = [state]
    fault.trigger_manual_drop.return_value = True

    env = MagicMock()
    rs_env = MagicMock()

    calls: list[str] = []

    def _track_recovery(*_args, **_kwargs):
        calls.append("recovery")
        return None

    fault.request_recovery.side_effect = _track_recovery

    with patch(
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
            planner=MagicMock(phase_name="lift", carry_path=MagicMock(), done=False),
            recipe_drop=MagicMock(
                min_drop_distance_from_basket_m=0.3,
                hard_keepout_floor_m=0.22,
            ),
            object_name="alphabet_soup_1",
            basket_name="basket_1",
            q=1.0,
            drop_rng=np.random.default_rng(0),
            post_drop_mode=PostDropMode.CONTINUE_THEN_IK,
            dwell_steps=2,
            path_trigger=MagicMock(
                fires=lambda **_: True,
                segment_name="lift",
                target_t=0.5,
            ),
            max_steps=1,
            gripper_settle_steps=0,
            on_step_end=lambda **_k: None,
        )

    fault.trigger_manual_drop.assert_called_once()
    fault.request_recovery.assert_not_called()
    assert calls == []
