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

import numpy as np
import pytest

from lerobot.faults.datagen.path_drop import eligible_path, sample_path_drop
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment


def _carry(*segments: tuple[str, tuple[float, ...], tuple[float, ...]]) -> CarryPath:
    return CarryPath(
        segments=tuple(PathSegment(*segment) for segment in segments),
        requested_transport_offset_m=0.0,
        resolved_transport_offset_m=0.0,
        fallback=False,
    )


def _straight() -> CarryPath:
    return _carry(
        ("lift", (0.5, 0.0, 0.03), (0.5, 0.0, 0.25)),
        ("to_basket_hover", (0.5, 0.0, 0.25), (0.0, 0.0, 0.25)),
    )


def test_vertical_lift_outside_keepout_is_fully_eligible():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)

    lift = path.pieces[0]
    assert lift.segment_name == "lift"
    assert lift.t0 == 0.0
    assert lift.t1 == 1.0
    assert lift.length_m == pytest.approx(0.22)


def test_segment_entering_keepout_is_clipped_at_circle_intersection():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)
    transport = next(p for p in path.pieces if p.segment_name == "to_basket_hover")

    assert transport.t0 == 0.0
    assert transport.t1 == pytest.approx(0.6)
    assert transport.length_m == pytest.approx(0.30)


def test_segment_outside_to_outside_crossing_disk_produces_two_pieces():
    carry = _carry(("cross", (-0.5, 0.0, 0.2), (0.5, 0.0, 0.2)))
    path = eligible_path(carry, basket_xy=np.zeros(2), keepout_m=0.20)

    assert len(path.pieces) == 2
    assert path.pieces[0].t1 == pytest.approx(0.3)
    assert path.pieces[1].t0 == pytest.approx(0.7)


def test_via_path_total_is_sum_of_clipped_piece_lengths():
    carry = _carry(
        ("lift", (0.5, 0.0, 0.03), (0.5, 0.0, 0.25)),
        ("to_basket_via", (0.5, 0.0, 0.25), (0.25, 0.1, 0.25)),
        ("to_basket_hover", (0.25, 0.1, 0.25), (0.0, 0.0, 0.25)),
    )
    path = eligible_path(carry, basket_xy=np.zeros(2), keepout_m=0.20)

    assert path.total == pytest.approx(sum(p.length_m for p in path.pieces))


def test_sampling_frequency_is_proportional_to_piece_length():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)
    rng = np.random.default_rng(0)
    phases = [
        sample_path_drop(1.0, path, rng)[1].segment_name for _ in range(10_000)
    ]

    expected = path.pieces[0].length_m / path.total
    observed = phases.count("lift") / len(phases)
    assert observed == pytest.approx(expected, abs=0.03)


def test_q_zero_never_drops():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)

    decision, trigger = sample_path_drop(0.0, path, np.random.default_rng(0))

    assert not decision.drop
    assert trigger is None


def test_q_one_drops_when_any_piece_exists():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)

    decision, trigger = sample_path_drop(1.0, path, np.random.default_rng(0))

    assert decision.drop
    assert trigger is not None


def test_empty_eligible_path_skips():
    carry = _carry(("lift", (0.1, 0.0, 0.03), (0.1, 0.0, 0.25)))
    path = eligible_path(carry, basket_xy=np.zeros(2), keepout_m=0.20)

    decision, trigger = sample_path_drop(1.0, path, np.random.default_rng(0))

    assert decision.reason == "no_eligible_path"
    assert trigger is None


def test_trigger_uses_projection_progress_on_active_segment():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)
    _, trigger = sample_path_drop(1.0, path, np.random.default_rng(4))
    segment = path.carry_path.segments[trigger.segment_order]
    start = np.asarray(segment.start_xyz)
    end = np.asarray(segment.end_xyz)

    before = start + (trigger.target_t - 0.01) * (end - start)
    after = start + (trigger.target_t + 0.01) * (end - start)
    assert not trigger.fires(
        phase=trigger.segment_name,
        object_xyz=before,
        carry_path=path.carry_path,
    )
    assert trigger.fires(
        phase=trigger.segment_name,
        object_xyz=after,
        carry_path=path.carry_path,
    )


def test_trigger_fires_when_execution_advances_to_a_later_segment():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)
    lift_piece = next(piece for piece in path.pieces if piece.segment_name == "lift")
    from lerobot.faults.datagen.path_drop import PathTrigger

    trigger = PathTrigger("lift", lift_piece.segment_order, 0.99)

    assert trigger.fires(
        phase="to_basket_hover",
        object_xyz=np.array([0.5, 0.0, 0.25]),
        carry_path=path.carry_path,
    )


def test_trigger_waits_at_reached_point_until_object_is_held_midair():
    path = eligible_path(_straight(), basket_xy=np.zeros(2), keepout_m=0.20)
    lift_piece = next(piece for piece in path.pieces if piece.segment_name == "lift")
    from lerobot.faults.datagen.path_drop import PathTrigger

    trigger = PathTrigger("lift", lift_piece.segment_order, 0.1)
    reached = np.array([0.5, 0.0, 0.20])

    assert not trigger.fires(
        phase="lift",
        object_xyz=reached,
        carry_path=path.carry_path,
        held_midair=False,
    )
    assert trigger.fires(
        phase="lift",
        object_xyz=reached,
        carry_path=path.carry_path,
        held_midair=True,
    )


def test_trigger_does_not_fire_on_an_earlier_segment():
    carry = _carry(
        ("lift", (0.5, 0.0, 0.03), (0.5, 0.0, 0.25)),
        ("to_basket_via", (0.5, 0.0, 0.25), (0.3, 0.1, 0.25)),
        ("to_basket_hover", (0.3, 0.1, 0.25), (0.0, 0.0, 0.25)),
    )
    from lerobot.faults.datagen.path_drop import PathTrigger

    trigger = PathTrigger("to_basket_hover", 2, 0.1)

    assert not trigger.fires(
        phase="lift",
        object_xyz=np.array([0.5, 0.0, 0.25]),
        carry_path=carry,
    )
