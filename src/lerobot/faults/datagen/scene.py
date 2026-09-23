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

"""LIBERO scene layout sampling and application for drop datagen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.faults.datagen.layout import ObjectPose2d, sample_layout
from lerobot.faults.datagen.recipe import DatagenRecipe, DropDatagenRecipe
from lerobot.faults.datagen.runtime import movable_object_names, rotate_quat_about_world_z
from lerobot.faults.sim.libero import get_object_pose, get_place_destination, set_object_pose

LAYOUT_SETTLE_STEPS = 40
TABLE_XY_LIMIT_M = 0.7


@dataclass(frozen=True)
class ObjectLayoutPose:
    pos: np.ndarray
    quat_wxyz: np.ndarray


def _sample_layout_poses(
    rs_env: Any,
    *,
    basket_name: str,
    placement: Any,
    target_object_name: str,
    rng: np.random.Generator,
) -> dict[str, ObjectLayoutPose] | None:
    names = movable_object_names(
        rs_env,
        basket_name=basket_name,
        required_object_name=target_object_name,
    )
    reset = {name: get_object_pose(rs_env, name) for name in names}
    basket_xy = get_place_destination(
        rs_env,
        target_object_name,
        basket_name=basket_name,
    )[:2]
    sampled = sample_layout(
        objects=[
            ObjectPose2d(name=name, xy=pose["pos"][:2].copy(), yaw_rad=0.0)
            for name, pose in reset.items()
        ],
        basket_xy=basket_xy,
        table_xy_lim=TABLE_XY_LIMIT_M,
        rng=rng,
        placement=placement,
        target_name=target_object_name,
    )
    if sampled is None:
        return None
    layout: dict[str, ObjectLayoutPose] = {}
    for pose2d in sampled:
        source = reset[pose2d.name]
        pos = source["pos"].copy()
        pos[:2] = pose2d.xy
        layout[pose2d.name] = ObjectLayoutPose(
            pos=pos,
            quat_wxyz=rotate_quat_about_world_z(
                source["quat_wxyz"],
                pose2d.yaw_rad,
            ),
        )
    return layout


def sample_object_layout(
    rs_env: Any,
    recipe: DropDatagenRecipe,
    object_name: str,
    rng: np.random.Generator,
) -> dict[str, ObjectLayoutPose] | None:
    return _sample_layout_poses(
        rs_env,
        basket_name=recipe.basket_name,
        placement=recipe.placement,
        target_object_name=object_name,
        rng=rng,
    )


def sample_object_layout_legacy(
    rs_env: Any,
    recipe: DatagenRecipe,
    rng: np.random.Generator,
) -> dict[str, ObjectLayoutPose] | None:
    return _sample_layout_poses(
        rs_env,
        basket_name=recipe.basket_name,
        placement=recipe.placement,
        target_object_name=recipe.object_name,
        rng=rng,
    )


def apply_object_layout(rs_env: Any, layout: dict[str, ObjectLayoutPose]) -> None:
    for name, pose in layout.items():
        set_object_pose(
            rs_env,
            name,
            pos=pose.pos,
            quat_wxyz=pose.quat_wxyz,
            settle_steps=0,
        )
    for _ in range(LAYOUT_SETTLE_STEPS):
        rs_env.sim.step()


def layout_to_serializable(layout: dict[str, ObjectLayoutPose]) -> dict[str, dict[str, list[float]]]:
    return {
        name: {
            "pos": pose.pos.astype(float).tolist(),
            "quat_wxyz": pose.quat_wxyz.astype(float).tolist(),
        }
        for name, pose in layout.items()
    }


def apply_serializable_layout(rs_env: Any, layout: dict[str, dict[str, list[float]]]) -> None:
    typed = {
        name: ObjectLayoutPose(
            pos=np.asarray(values["pos"], dtype=np.float64),
            quat_wxyz=np.asarray(values["quat_wxyz"], dtype=np.float64),
        )
        for name, values in layout.items()
    }
    apply_object_layout(rs_env, typed)
