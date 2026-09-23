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

"""JSON recipe loading for can-only SimpleIK episode generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RecipeError(ValueError):
    """Raised when a datagen recipe cannot be loaded or validated."""


@dataclass(frozen=True)
class PlacementRecipe:
    xy_range_m: float
    min_basket_clearance_m: float
    distractor_basket_clearance_m: float
    min_pairwise_clearance_m: float
    yaw_range_deg: tuple[float, float]
    max_attempts: int


@dataclass(frozen=True)
class DropRecipe:
    eligible_phases: tuple[str, ...]
    min_drop_distance_from_basket_m: float
    hard_keepout_floor_m: float


@dataclass(frozen=True)
class SimpleIKRecipe:
    trajectory_randomization_enabled: bool
    pickup_via_offset_m: float
    transport_via_offset_m: float
    arm_posture_noise_deg: float
    speed_multiplier_range: tuple[float, float]
    waypoint_blend_radius_m: float


@dataclass(frozen=True)
class DatagenRecipe:
    q: float
    object_name: str
    basket_name: str
    placement: PlacementRecipe
    drop: DropRecipe
    simple_ik: SimpleIKRecipe


def _required(mapping: dict[str, Any], key: str, *, section: str = "recipe") -> Any:
    if key not in mapping:
        raise RecipeError(f"{section}.{key} is required")
    return mapping[key]


def load_recipe(path: Path) -> DatagenRecipe:
    """Load and strictly validate a can-only datagen recipe."""
    path = Path(path)
    if not path.is_file():
        raise RecipeError(f"Recipe file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeError(f"Could not load recipe {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RecipeError("recipe must be a JSON object")

    q = float(_required(raw, "q"))
    if not 0.0 <= q <= 1.0:
        raise RecipeError(f"q must be in [0, 1], got {q}")
    object_name = str(_required(raw, "object_name")).strip()
    basket_name = str(_required(raw, "basket_name")).strip()
    if not object_name:
        raise RecipeError("object_name must be non-empty")
    if not basket_name:
        raise RecipeError("basket_name must be non-empty")

    placement_raw = _required(raw, "placement")
    drop_raw = _required(raw, "drop")
    simple_ik_raw = _required(raw, "simple_ik")
    if not all(isinstance(value, dict) for value in (placement_raw, drop_raw, simple_ik_raw)):
        raise RecipeError("placement, drop, and simple_ik must be JSON objects")

    yaw_values = tuple(float(v) for v in _required(placement_raw, "yaw_range_deg", section="placement"))
    if len(yaw_values) != 2 or yaw_values[1] < yaw_values[0]:
        raise RecipeError("placement.yaw_range_deg must be [min, max] with max >= min")
    placement = PlacementRecipe(
        xy_range_m=float(_required(placement_raw, "xy_range_m", section="placement")),
        min_basket_clearance_m=float(
            _required(placement_raw, "min_basket_clearance_m", section="placement")
        ),
        distractor_basket_clearance_m=float(
            _required(placement_raw, "distractor_basket_clearance_m", section="placement")
        ),
        min_pairwise_clearance_m=float(
            _required(placement_raw, "min_pairwise_clearance_m", section="placement")
        ),
        yaw_range_deg=(yaw_values[0], yaw_values[1]),
        max_attempts=int(_required(placement_raw, "max_attempts", section="placement")),
    )
    if placement.xy_range_m < 0:
        raise RecipeError("placement.xy_range_m must be >= 0")
    if (
        placement.min_basket_clearance_m < 0
        or placement.distractor_basket_clearance_m < 0
        or placement.min_pairwise_clearance_m < 0
    ):
        raise RecipeError("placement clearances must be >= 0")
    if placement.distractor_basket_clearance_m > placement.min_basket_clearance_m:
        raise RecipeError(
            "placement.distractor_basket_clearance_m must be <= min_basket_clearance_m"
        )
    if placement.max_attempts < 1:
        raise RecipeError("placement.max_attempts must be >= 1")

    phases = tuple(str(v).strip() for v in _required(drop_raw, "eligible_phases", section="drop"))
    if not phases or any(not phase for phase in phases):
        raise RecipeError("drop.eligible_phases must be non-empty")
    drop = DropRecipe(
        eligible_phases=phases,
        min_drop_distance_from_basket_m=float(
            _required(drop_raw, "min_drop_distance_from_basket_m", section="drop")
        ),
        hard_keepout_floor_m=float(
            _required(drop_raw, "hard_keepout_floor_m", section="drop")
        ),
    )
    if drop.min_drop_distance_from_basket_m < 0:
        raise RecipeError("drop.min_drop_distance_from_basket_m must be >= 0")
    if drop.hard_keepout_floor_m < 0:
        raise RecipeError("drop.hard_keepout_floor_m must be >= 0")

    speed_values = tuple(
        float(v)
        for v in _required(simple_ik_raw, "speed_multiplier_range", section="simple_ik")
    )
    if len(speed_values) != 2:
        raise RecipeError("simple_ik.speed_multiplier_range must contain [min, max]")
    simple_ik = SimpleIKRecipe(
        trajectory_randomization_enabled=bool(
            _required(simple_ik_raw, "trajectory_randomization_enabled", section="simple_ik")
        ),
        pickup_via_offset_m=float(
            _required(simple_ik_raw, "pickup_via_offset_m", section="simple_ik")
        ),
        transport_via_offset_m=float(
            _required(simple_ik_raw, "transport_via_offset_m", section="simple_ik")
        ),
        arm_posture_noise_deg=float(
            _required(simple_ik_raw, "arm_posture_noise_deg", section="simple_ik")
        ),
        speed_multiplier_range=(speed_values[0], speed_values[1]),
        waypoint_blend_radius_m=float(
            _required(simple_ik_raw, "waypoint_blend_radius_m", section="simple_ik")
        ),
    )
    for field_name in (
        "pickup_via_offset_m",
        "transport_via_offset_m",
        "arm_posture_noise_deg",
        "waypoint_blend_radius_m",
    ):
        if getattr(simple_ik, field_name) < 0.0:
            raise RecipeError(f"simple_ik.{field_name} must be >= 0")
    speed_lo, speed_hi = simple_ik.speed_multiplier_range
    if speed_lo <= 0.0 or speed_hi <= 0.0:
        raise RecipeError("simple_ik.speed_multiplier_range values must be > 0")
    if speed_hi < speed_lo:
        raise RecipeError("simple_ik.speed_multiplier_range max must be >= min")

    return DatagenRecipe(
        q=q,
        object_name=object_name,
        basket_name=basket_name,
        placement=placement,
        drop=drop,
        simple_ik=simple_ik,
    )
