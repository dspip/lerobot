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

"""Small runtime helpers for the LIBERO datagen runner."""

from __future__ import annotations

from typing import Any

import numpy as np


def stabilize_carry_action(
    action: np.ndarray,
    *,
    translation_scale: float = 0.65,
) -> np.ndarray:
    """Slow carry translation and command a fully closed gripper."""
    stabilized = np.asarray(action).copy()
    stabilized[:3] = np.clip(
        stabilized[:3] * float(translation_scale),
        -1.0,
        1.0,
    )
    stabilized[6] = 1.0
    return stabilized


def rotate_quat_about_world_z(quat_wxyz: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Left-compose a world-Z yaw delta with a wxyz quaternion."""
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    half = 0.5 * float(yaw_rad)
    yaw = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
    aw, ax, ay, az = yaw
    bw, bx, by, bz = quat
    result = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(result))
    if norm == 0.0:
        raise ValueError("quat_wxyz must be non-zero")
    return result / norm


def movable_object_names(
    rs_env: Any,
    *,
    basket_name: str,
    required_object_name: str,
) -> list[str]:
    """List free-joint manipulation objects, preserving scene order."""
    objects_dict = getattr(rs_env, "objects_dict", None)
    scene_objects = getattr(rs_env, "objects", None)
    if isinstance(objects_dict, dict):
        candidates = [str(name) for name in objects_dict]
    elif isinstance(scene_objects, dict):
        candidates = [str(name) for name in scene_objects]
    elif scene_objects is not None:
        candidates = [
            str(name)
            for obj in scene_objects
            if (name := getattr(obj, "name", None)) is not None
        ]
    else:
        candidates = []
    if required_object_name not in candidates:
        candidates.insert(0, required_object_name)

    names: list[str] = []
    for name in candidates:
        if name == basket_name or name in names:
            continue
        try:
            obj = rs_env.get_object(name)
        except Exception:
            continue
        if getattr(obj, "joints", None):
            names.append(name)
    if required_object_name not in names:
        raise KeyError(f"Required object {required_object_name!r} has no movable joint")
    return names
