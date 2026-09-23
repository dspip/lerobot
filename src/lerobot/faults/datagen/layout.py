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

"""Collision-aware 2D object layout sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.faults.datagen.recipe import PlacementRecipe


@dataclass(frozen=True)
class ObjectPose2d:
    name: str
    xy: np.ndarray
    yaw_rad: float


def sample_layout(
    *,
    objects: list[ObjectPose2d],
    basket_xy: np.ndarray,
    table_xy_lim: float,
    rng: np.random.Generator,
    placement: PlacementRecipe,
    target_name: str,
) -> list[ObjectPose2d] | None:
    """Sample one legal layout, or return ``None`` after bounded retries.

    The pick target gets the full basket keep-out so its approach stays clear.
    Distractors only have to miss the basket footprint; the stock LIBERO scene
    already places some of them inside the target keep-out, so applying it to
    every object makes the layout near-infeasible.
    """
    basket_xy = np.asarray(basket_xy, dtype=np.float64).reshape(2)
    table_xy_lim = float(table_xy_lim)
    yaw_lo, yaw_hi = np.deg2rad(placement.yaw_range_deg)

    for _ in range(placement.max_attempts):
        sampled: list[ObjectPose2d] = []
        legal = True
        for obj in objects:
            reset_xy = np.asarray(obj.xy, dtype=np.float64).reshape(2)
            xy = reset_xy + rng.uniform(
                -placement.xy_range_m,
                placement.xy_range_m,
                size=2,
            )
            if np.any(np.abs(xy) > table_xy_lim):
                legal = False
                break
            clearance = (
                placement.min_basket_clearance_m
                if obj.name == target_name
                else placement.distractor_basket_clearance_m
            )
            if float(np.linalg.norm(xy - basket_xy)) < clearance:
                legal = False
                break
            if any(
                float(np.linalg.norm(xy - placed.xy)) < placement.min_pairwise_clearance_m
                for placed in sampled
            ):
                legal = False
                break
            sampled.append(
                ObjectPose2d(
                    name=obj.name,
                    xy=xy,
                    yaw_rad=float(rng.uniform(yaw_lo, yaw_hi)),
                )
            )
        if legal:
            return sampled
    return None
