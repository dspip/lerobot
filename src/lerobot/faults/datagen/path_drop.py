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

"""Uniform drop sampling over a planner's clipped carry polyline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.faults.datagen.drop_timing import DropDecision
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment

LIFT_PHASE = "lift"
TRANSPORT_PHASE = "to_basket_hover"


@dataclass(frozen=True)
class EligibleArcPiece:
    segment_name: str
    segment_order: int
    t0: float
    t1: float
    length_m: float


@dataclass(frozen=True)
class EligiblePath:
    carry_path: CarryPath
    pieces: tuple[EligibleArcPiece, ...]

    @property
    def total(self) -> float:
        return float(sum(piece.length_m for piece in self.pieces))


@dataclass(frozen=True)
class PathTrigger:
    segment_name: str
    segment_order: int
    target_t: float

    def fires(
        self,
        *,
        phase: str,
        object_xyz: np.ndarray,
        carry_path: CarryPath,
        held_midair: bool = True,
    ) -> bool:
        """Return true after execution reaches or passes the sampled arc point."""
        if not held_midair:
            return False
        order = {segment.name: index for index, segment in enumerate(carry_path.segments)}
        current_order = order.get(phase)
        if current_order is None or current_order < self.segment_order:
            return False
        if current_order > self.segment_order:
            return True
        segment = carry_path.segments[self.segment_order]
        start = np.asarray(segment.start_xyz, dtype=np.float64)
        end = np.asarray(segment.end_xyz, dtype=np.float64)
        direction = end - start
        denom = float(np.dot(direction, direction))
        if denom <= 1e-15:
            return True
        point = np.asarray(object_xyz, dtype=np.float64).reshape(3)
        progress = float(np.clip(np.dot(point - start, direction) / denom, 0.0, 1.0))
        return progress >= self.target_t


def _outside_intervals(
    segment: PathSegment,
    basket_xy: np.ndarray,
    keepout_m: float,
) -> list[tuple[float, float]]:
    start = np.asarray(segment.start_xyz, dtype=np.float64)
    end = np.asarray(segment.end_xyz, dtype=np.float64)
    p0 = start[:2] - basket_xy
    direction = end[:2] - start[:2]
    a = float(np.dot(direction, direction))
    c = float(np.dot(p0, p0) - keepout_m**2)
    if a <= 1e-15:
        return [(0.0, 1.0)] if c >= 0.0 else []

    b = 2.0 * float(np.dot(p0, direction))
    discriminant = b * b - 4.0 * a * c
    boundaries = [0.0, 1.0]
    if discriminant >= 0.0:
        root = float(np.sqrt(discriminant))
        for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if 0.0 < value < 1.0:
                boundaries.append(float(value))
    boundaries = sorted(set(boundaries))

    intervals: list[tuple[float, float]] = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        midpoint = 0.5 * (lo + hi)
        xy = p0 + midpoint * direction
        if float(np.dot(xy, xy)) >= keepout_m**2 - 1e-12:
            intervals.append((lo, hi))
    return intervals


def eligible_path(
    carry_path: CarryPath,
    *,
    basket_xy: np.ndarray,
    keepout_m: float,
) -> EligiblePath:
    """Clip every carry segment against the circular basket keep-out."""
    basket = np.asarray(basket_xy, dtype=np.float64).reshape(2)
    keepout = float(keepout_m)
    pieces: list[EligibleArcPiece] = []
    for order, segment in enumerate(carry_path.segments):
        for t0, t1 in _outside_intervals(segment, basket, keepout):
            length = segment.length * (t1 - t0)
            if length > 1e-12:
                pieces.append(
                    EligibleArcPiece(
                        segment_name=segment.name,
                        segment_order=order,
                        t0=float(t0),
                        t1=float(t1),
                        length_m=float(length),
                    )
                )
    return EligiblePath(carry_path=carry_path, pieces=tuple(pieces))


def path_trigger_at_drop_u(drop_u: float, path: EligiblePath) -> PathTrigger:
    """Map a unit-interval draw onto uniform arc length over ``path``."""
    u = float(drop_u)
    if not 0.0 <= u <= 1.0:
        raise ValueError(f"drop_u must be in [0, 1], got {u}")
    if path.total <= 0.0:
        raise ValueError("path.total must be positive")
    target = u * path.total
    travelled = 0.0
    selected = path.pieces[-1]
    for piece in path.pieces:
        if target <= travelled + piece.length_m:
            selected = piece
            break
        travelled += piece.length_m
    fraction = (target - travelled) / selected.length_m
    target_t = selected.t0 + fraction * (selected.t1 - selected.t0)
    return PathTrigger(
        segment_name=selected.segment_name,
        segment_order=selected.segment_order,
        target_t=float(np.clip(target_t, selected.t0, selected.t1)),
    )


def sample_path_drop(
    q: float,
    path: EligiblePath,
    rng: np.random.Generator,
) -> tuple[DropDecision, PathTrigger | None]:
    """Draw no drop, or one point uniform over eligible polyline arc length."""
    q = float(q)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if path.total <= 0.0:
        return DropDecision(drop=False, step=None, reason="no_eligible_path"), None
    if float(rng.random()) >= q:
        return DropDecision(drop=False, step=None, reason="skipped_q"), None

    drop_u = float(rng.uniform(0.0, 1.0))
    return (
        DropDecision(drop=True, step=None, reason="injected"),
        path_trigger_at_drop_u(drop_u, path),
    )
