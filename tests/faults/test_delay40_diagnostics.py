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

"""Unit tests for delay-40 trigger reconstruction (no GPU)."""

from __future__ import annotations

import pytest

from lerobot.faults.recovery.delay40_diagnostics import reconstruct_frames


def _fake_log(*, dist_at_step: dict[int, float]) -> dict:
    max_step = max(dist_at_step)
    basket = [0.0, 0.0, 0.1]
    traj = []
    flags = []
    for i in range(max_step + 1):
        d = dist_at_step.get(i, 0.45)
        traj.append([d, 0.0, 0.15])
        flags.append(True)
    return {
        "t_min": 0,
        "t_max": 500,
        "first_grasp_step": 10,
        "basket_destination": basket,
        "fault_config": {"post_grasp_delay_steps": 40},
        "object_traj": traj,
        "grasp_flags": flags,
    }


def test_block_reason_when_dist_drops_inside_keepout() -> None:
    dist = {i: 0.45 for i in range(60)}
    for i in range(40, 60):
        dist[i] = 0.25
    rows = reconstruct_frames(_fake_log(dist_at_step=dist))
    by_step = {r.episode_frame: r for r in rows}

    # Injector checks keep-out before delay; step 49 is already inside 0.30 m.
    assert by_step[49].block_reason == "optionB_keepout_lt_0.30"
    assert by_step[50].block_reason == "optionB_keepout_lt_0.30"
    assert by_step[50].xy_dist_m == pytest.approx(0.25)


def test_would_trigger_at_grasp_plus_delay_when_far_enough() -> None:
    rows = reconstruct_frames(_fake_log(dist_at_step={i: 0.45 for i in range(60)}))
    by_step = {r.episode_frame: r for r in rows}
    assert by_step[49].block_reason == "delay_not_elapsed"
    assert by_step[50].block_reason == "would_trigger"
    assert by_step[50].xy_dist_m == pytest.approx(0.45)
