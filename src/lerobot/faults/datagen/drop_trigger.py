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

"""Drop trigger sampling and evaluation for unified datagen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.faults.datagen.path_drop import PathTrigger
from lerobot.faults.datagen.recipe import DropXYBand
from lerobot.faults.recovery.basket_drop_target import basket_distance_target_reached

__all__ = [
    "BandDistanceTarget",
    "PathDropEvaluator",
    "basket_distance_target_reached",
    "sample_smolvla_band_target",
    "smolvla_fault_drop_fields",
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


def smolvla_fault_drop_fields(target: BandDistanceTarget) -> dict[str, float]:
    return {
        "drop_xy_band_min": float(target.band.min_m),
        "drop_xy_band_max": float(target.band.max_m),
        "drop_xy_target_m": float(target.target_m),
    }


@dataclass(frozen=True)
class PathDropEvaluator:
    """Evaluate planned-path drop triggers for SimpleIK carry geometry."""

    trigger: PathTrigger

    def should_fire(
        self,
        *,
        phase: str,
        object_xyz: np.ndarray,
        carry_path: Any,
        held_midair: bool,
    ) -> bool:
        return self.trigger.fires(
            phase=phase,
            object_xyz=object_xyz,
            carry_path=carry_path,
            held_midair=held_midair,
        )
