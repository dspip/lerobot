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

from lerobot.faults.datagen.drop_timing import (
    DropDecision,
    TraceFrame,
    eligible_indices,
    phase_at_step,
    phase_breakdown,
    sample_drop,
)


def _frame(step, phase, obj_xy, basket_xy=(0.0, 0.0), *, grasped=True):
    return TraceFrame(
        step=step,
        phase=phase,
        object_xy=np.array(obj_xy, dtype=np.float64),
        basket_xy=np.array(basket_xy, dtype=np.float64),
        grasped=grasped,
    )


def test_eligible_excludes_keepout_and_non_carry_phases():
    frames = [
        _frame(0, "descend_grasp", (0.5, 0.0)),
        _frame(1, "lift", (0.5, 0.0)),
        _frame(2, "to_basket_hover", (0.10, 0.0)),
        _frame(3, "to_container", (0.40, 0.0)),
        _frame(4, "open_place", (0.40, 0.0)),
    ]

    assert eligible_indices(frames, ("lift", "to_container"), 0.30) == [1, 3]


def test_phase_breakdown_counts_every_frame_by_canonical_phase():
    frames = [
        _frame(0, "lift", (0.5, 0.0)),
        _frame(1, "lift", (0.5, 0.0)),
        _frame(2, "to_container", (0.4, 0.0)),
    ]

    assert phase_breakdown(frames) == {"lift": 2, "to_basket_hover": 1}


def test_phase_breakdown_restricted_to_eligible_steps():
    frames = [
        _frame(0, "lift", (0.5, 0.0)),
        _frame(1, "to_basket_hover", (0.10, 0.0)),
        _frame(2, "to_basket_hover", (0.40, 0.0)),
    ]
    eligible = eligible_indices(frames, ("lift", "to_container"), 0.30)

    assert phase_breakdown(frames, eligible) == {"lift": 1, "to_basket_hover": 1}


def test_phase_at_step_reports_the_drop_phase():
    frames = [_frame(0, "lift", (0.5, 0.0)), _frame(7, "to_container", (0.4, 0.0))]

    assert phase_at_step(frames, 7) == "to_basket_hover"


def test_phase_at_step_of_none_is_none():
    assert phase_at_step([_frame(0, "lift", (0.5, 0.0))], None) is None


def test_hard_keepout_is_always_enforced():
    frames = [_frame(5, "lift", (0.21, 0.0))]

    assert eligible_indices(frames, ("lift",), 0.0) == []


def test_eligible_requires_can_to_be_grasped():
    frames = [_frame(5, "lift", (0.5, 0.0), grasped=False)]

    assert eligible_indices(frames, ("lift",), 0.30) == []


def test_sample_drop_q0_never_drops():
    rng = np.random.default_rng(0)
    for _ in range(50):
        decision = sample_drop(0.0, [10, 20, 30], rng)
        assert decision == DropDecision(drop=False, step=None, reason="skipped_q")


def test_sample_drop_q1_always_in_e():
    rng = np.random.default_rng(1)
    eligible = [10, 20, 30]
    for _ in range(50):
        decision = sample_drop(1.0, eligible, rng)
        assert decision.drop is True
        assert decision.step in eligible
        assert decision.reason == "injected"


def test_sample_drop_empty_e():
    decision = sample_drop(1.0, [], np.random.default_rng(0))

    assert decision == DropDecision(drop=False, step=None, reason="no_eligible_frames")


def test_sample_drop_empirical_rate_and_uniform():
    rng = np.random.default_rng(42)
    eligible = [1, 2, 3, 4]
    n = 4000
    drops = 0
    counts = {step: 0 for step in eligible}

    for _ in range(n):
        decision = sample_drop(0.5, eligible, rng)
        if decision.drop:
            drops += 1
            counts[decision.step] += 1

    assert 0.47 < drops / n < 0.53
    assert max(counts.values()) / min(counts.values()) < 1.4
