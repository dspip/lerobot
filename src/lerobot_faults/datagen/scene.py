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

from lerobot_faults.datagen.layout import ObjectPose2d, sample_layout
from lerobot_faults.datagen.recipe import DropDatagenRecipe
from lerobot_faults.datagen.runtime import movable_object_names, rotate_quat_about_world_z
from lerobot_faults.sim.libero import get_object_pose, get_place_destination, set_object_pose

LAYOUT_SETTLE_STEPS = 40
TABLE_XY_LIMIT_M = 0.7


@dataclass(frozen=True)
class ObjectLayoutPose:
    """World-frame position and orientation for one object in a sampled layout."""

    pos: np.ndarray
    quat_wxyz: np.ndarray


def _sample_layout_poses(
    rs_env: Any,
    *,
    basket_name: str,
    placement: Any,
    target_object_name: str,
    rng: np.random.Generator,
    exclude_names: tuple[str, ...] = (),
) -> dict[str, ObjectLayoutPose] | None:
    excluded = set(exclude_names)
    names = [
        name
        for name in movable_object_names(
            rs_env,
            basket_name=basket_name,
            required_object_name=target_object_name,
        )
        if name not in excluded
    ]
    reset = {name: get_object_pose(rs_env, name) for name in names}
    basket_xy = get_place_destination(
        rs_env,
        target_object_name,
        basket_name=basket_name,
    )[:2]
    sampled = sample_layout(
        objects=[
            ObjectPose2d(name=name, xy=pose["pos"][:2].copy(), yaw_rad=0.0) for name, pose in reset.items()
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
    """Sample collision-aware poses for movable objects on the LIBERO table."""
    return _sample_layout_poses(
        rs_env,
        basket_name=recipe.basket_name,
        placement=recipe.placement,
        target_object_name=object_name,
        rng=rng,
        exclude_names=tuple(getattr(recipe, "held_out_object_names", ())),
    )


_HIDE_POSITION = np.array([3.0, 3.0, -1.0], dtype=np.float64)


def _subtree_body_ids(model: Any, root_id: int) -> list[int]:
    """Return ``root_id`` and every body that hangs under it."""
    ids = [int(root_id)]
    parent = np.asarray(model.body_parentid)
    for body_id in range(int(model.nbody)):
        cursor = int(body_id)
        seen: set[int] = set()
        while cursor > 0 and cursor not in seen:
            if cursor == int(root_id):
                ids.append(int(body_id))
                break
            seen.add(cursor)
            cursor = int(parent[cursor])
    return ids


def _resolve_body_id(rs_env: Any, object_name: str) -> int | None:
    sim = rs_env.sim
    root = None
    try:
        obj = rs_env.get_object(object_name)
        root = getattr(obj, "root_body", None)
    except Exception:
        return None
    candidates = [object_name]
    if isinstance(root, str) and root:
        candidates.insert(0, root)
    if not object_name.endswith("_main"):
        candidates.append(f"{object_name}_main")
    for name in candidates:
        try:
            return int(sim.model.body_name2id(name))
        except Exception:
            continue
    return None


def hide_scene_objects(rs_env: Any, object_names: tuple[str, ...] | list[str]) -> None:
    """Make objects invisible and non-colliding, then park them off the table.

    Official BDDL files are not edited. Objects that are not in this scene are
    skipped. Recording calls this after layout so held-out objects never appear
    in saved frames.
    """
    if not object_names:
        return
    sim = rs_env.sim
    model = sim.model
    for object_name in object_names:
        body_id = _resolve_body_id(rs_env, object_name)
        if body_id is None:
            continue
        subtree = set(_subtree_body_ids(model, body_id))
        updated = 0
        for geom_id, geom_body in enumerate(np.asarray(model.geom_bodyid)):
            if int(geom_body) not in subtree:
                continue
            model.geom_rgba[geom_id, 3] = 0.0
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            updated += 1
        moved = False
        try:
            set_object_pose(
                rs_env,
                object_name,
                pos=_HIDE_POSITION,
                settle_steps=0,
            )
            moved = True
        except Exception:
            moved = False
        if updated == 0 and not moved:
            raise RuntimeError(f"could not hide held-out object {object_name!r}")


def apply_object_layout(rs_env: Any, layout: dict[str, ObjectLayoutPose]) -> None:
    """Write sampled poses into the sim and settle with extra physics steps."""
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
    """Convert in-memory layout poses to JSON-friendly float lists."""
    return {
        name: {
            "pos": pose.pos.astype(float).tolist(),
            "quat_wxyz": pose.quat_wxyz.astype(float).tolist(),
        }
        for name, pose in layout.items()
    }


def apply_serializable_layout(rs_env: Any, layout: dict[str, dict[str, list[float]]]) -> None:
    """Restore a shared layout dict produced by :func:`layout_to_serializable`."""
    typed = {
        name: ObjectLayoutPose(
            pos=np.asarray(values["pos"], dtype=np.float64),
            quat_wxyz=np.asarray(values["quat_wxyz"], dtype=np.float64),
        )
        for name, values in layout.items()
    }
    apply_object_layout(rs_env, typed)
