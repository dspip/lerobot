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

"""Controller-specific drop trigger sampling and firing rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from lerobot.faults.datagen.path_drop import PathTrigger
from lerobot.faults.datagen.recipe import DropXYBand

__all__ = [
    "BandDistanceTarget",
    "BandDistanceTrigger",
    "PathDropTriggerAdapter",
    "sample_smolvla_band_target",
]


@dataclass(frozen=True)
class BandDistanceTarget:
    band: DropXYBand
    target_m: float


def sample_smolvla_band_target(
    rng: np.random.Generator,
    bands: tuple[DropXYBand, ...],
) -> BandDistanceTarget:
    if not bands:
        raise ValueError("bands must be non-empty")
    index = int(rng.integers(0, len(bands)))
    band = bands[index]
    target_m = float(rng.uniform(band.min_m, band.max_m))
    return BandDistanceTarget(band=band, target_m=target_m)


class DropTrigger(Protocol):
    def evaluate(self, *, held_midair: bool, basket_distance_m: float, prev_basket_distance_m: float) -> bool:
        """Return whether the drop should fire on this control step."""


class BandDistanceTrigger:
    """Fire once the object crosses the sampled basket-distance target while held midair."""

    def __init__(self, target: BandDistanceTarget) -> None:
        self.target = target
        self._consumed = False

    def evaluate(
        self,
        *,
        held_midair: bool,
        basket_distance_m: float,
        prev_basket_distance_m: float,
    ) -> bool:
        if self._consumed or not held_midair:
            return False
        distance = float(basket_distance_m)
        if distance < self.target.band.min_m or distance > self.target.band.max_m:
            return False
        target = float(self.target.target_m)
        prev = float(prev_basket_distance_m)
        crossed = prev > target >= distance
        if crossed:
            self._consumed = True
            return True
        return False


@dataclass(frozen=True)
class PathDropTriggerAdapter:
    """Wrap planned-path triggers for SimpleIK carry geometry."""

    trigger: PathTrigger

    def evaluate_path(
        self,
        *,
        phase: str,
        object_xyz: np.ndarray,
        carry_path,
        held_midair: bool,
    ) -> bool:
        return self.trigger.fires(
            phase=phase,
            object_xyz=object_xyz,
            carry_path=carry_path,
            held_midair=held_midair,
        )
