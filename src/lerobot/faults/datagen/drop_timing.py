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

"""Exact run-level drop timing over a recorded SimpleIK trace."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.faults.recovery.midair_drop import HARD_BASKET_KEEPOUT_M


@dataclass(frozen=True)
class TraceFrame:
    step: int
    phase: str
    object_xy: np.ndarray
    basket_xy: np.ndarray
    grasped: bool = True


@dataclass(frozen=True)
class DropDecision:
    drop: bool
    step: int | None
    reason: str


def canonicalize_phase(name: str) -> str:
    """Map recipe-facing phase aliases to planner phase names."""
    return "to_basket_hover" if name == "to_container" else name


def keepout_m(
    min_drop_distance_from_basket_m: float,
    hard_floor_m: float = HARD_BASKET_KEEPOUT_M,
) -> float:
    """Return the effective keep-out, including the hard rim pocket.

    ``hard_floor_m`` is per-recipe so this pipeline can go below the module
    default without moving ``HARD_BASKET_KEEPOUT_M``, which ``mix_audit`` uses
    to judge already-recorded datasets.
    """
    return max(float(min_drop_distance_from_basket_m), float(hard_floor_m))


def eligible_indices(
    frames: list[TraceFrame],
    eligible_phases: tuple[str, ...],
    min_drop_distance_from_basket_m: float,
) -> list[int]:
    """Return trace steps eligible for the episode's single optional drop."""
    allowed = {canonicalize_phase(phase) for phase in eligible_phases}
    keepout = keepout_m(min_drop_distance_from_basket_m)
    eligible: list[int] = []
    for frame in frames:
        if not frame.grasped:
            continue
        if canonicalize_phase(frame.phase) not in allowed:
            continue
        distance = float(
            np.linalg.norm(
                np.asarray(frame.object_xy, dtype=np.float64)
                - np.asarray(frame.basket_xy, dtype=np.float64)
            )
        )
        if distance >= keepout:
            eligible.append(int(frame.step))
    return eligible


def phase_breakdown(
    frames: list[TraceFrame],
    steps: list[int] | None = None,
) -> dict[str, int]:
    """Count trace frames per canonical phase, optionally over a step subset.

    Used to show whether a phase contributes any eligible drop frames at all,
    which a bare eligible-count cannot reveal.
    """
    wanted = None if steps is None else set(int(step) for step in steps)
    counts: dict[str, int] = {}
    for frame in frames:
        if wanted is not None and int(frame.step) not in wanted:
            continue
        phase = canonicalize_phase(frame.phase)
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def phase_at_step(frames: list[TraceFrame], step: int | None) -> str | None:
    """Return the canonical phase recorded at ``step``."""
    if step is None:
        return None
    for frame in frames:
        if int(frame.step) == int(step):
            return canonicalize_phase(frame.phase)
    return None


def sample_drop(
    q: float,
    eligible_steps: list[int],
    rng: np.random.Generator,
) -> DropDecision:
    """Sample no drop or one uniformly selected eligible step."""
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if not eligible_steps:
        return DropDecision(drop=False, step=None, reason="no_eligible_frames")
    if float(rng.random()) >= q:
        return DropDecision(drop=False, step=None, reason="skipped_q")
    index = int(rng.integers(0, len(eligible_steps)))
    return DropDecision(drop=True, step=int(eligible_steps[index]), reason="injected")
