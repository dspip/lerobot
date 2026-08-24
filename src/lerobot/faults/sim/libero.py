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

"""LIBERO / robosuite simulation helpers for inject-only sim faults."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_f64(arr: Any, size: int | None = None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    if size is not None and out.shape != (size,):
        raise ValueError(f"Expected shape ({size},), got {out.shape}")
    return out


def unwrap_libero_env(env: Any) -> Any:
    """Peel Gymnasium / fault wrappers until a LeRobot-style ``LiberoEnv`` is found."""
    seen: set[int] = set()
    current = env
    while True:
        oid = id(current)
        if oid in seen:
            break
        seen.add(oid)

        if hasattr(current, "_env") and getattr(current, "_task_bddl_file", None) is not None:
            return current

        nxt = getattr(current, "env", None)
        if nxt is None:
            nxt = getattr(current, "unwrapped", None)
        if nxt is None or nxt is current:
            break
        current = nxt

    raise TypeError(
        "Could not unwrap to a LiberoEnv-like object (expected `_env` and `_task_bddl_file`). "
        f"Got {type(env)!r}."
    )


def get_robosuite_env(libero_or_vec_env: Any, env_idx: int = 0) -> Any:
    """Return the underlying ``BDDLBaseDomain`` robosuite env."""
    env = libero_or_vec_env
    type_name = type(env).__name__

    if type_name == "AsyncVectorEnv":
        raise TypeError(
            "AsyncVectorEnv does not expose `.envs`; use SyncVectorEnv or a single LiberoEnv."
        )

    if hasattr(env, "envs"):
        try:
            sub_envs = env.envs
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(f"Vector env {type_name!r} has no accessible `.envs`.") from exc
        if env_idx < 0 or env_idx >= len(sub_envs):
            raise IndexError(f"env_idx={env_idx} out of range for {len(sub_envs)} sub-envs.")
        env = sub_envs[env_idx]

    libero_env = unwrap_libero_env(env)
    offscreen = getattr(libero_env, "_env", None)
    if offscreen is None:
        raise RuntimeError(
            "LiberoEnv._env is None — call reset() once so OffScreenRenderEnv is created."
        )

    rs_env = getattr(offscreen, "env", None)
    if rs_env is None:
        raise RuntimeError("OffScreenRenderEnv.env (BDDLBaseDomain) is missing.")
    if not hasattr(rs_env, "sim") or rs_env.sim is None:
        raise RuntimeError("robosuite env sim is not ready — reset the environment first.")
    return rs_env


def _require_object(rs_env: Any, object_name: str) -> Any:
    if not hasattr(rs_env, "get_object"):
        raise TypeError(f"Expected BDDLBaseDomain-like env, got {type(rs_env)!r}.")
    try:
        obj = rs_env.get_object(object_name)
    except Exception as exc:  # pragma: no cover - libero-specific
        raise KeyError(f"Object {object_name!r} not found in scene.") from exc
    if obj is None:
        raise KeyError(f"Object {object_name!r} not found in scene.")
    return obj


def is_object_grasped(rs_env: Any, object_name: str) -> bool:
    """Return whether the robot gripper is grasping ``object_name``."""
    obj = _require_object(rs_env, object_name)
    if not hasattr(rs_env, "_check_grasp"):
        raise TypeError("robosuite env missing `_check_grasp`.")
    if not rs_env.robots:
        raise RuntimeError("No robots in robosuite env.")
    gripper = rs_env.robots[0].gripper
    geoms = getattr(obj, "contact_geoms", obj)
    return bool(rs_env._check_grasp(gripper=gripper, object_geoms=geoms))


def get_object_pose(rs_env: Any, object_name: str) -> dict[str, np.ndarray]:
    """World-frame pose for a named LIBERO object."""
    obj = _require_object(rs_env, object_name)
    sim = rs_env.sim
    body = obj.root_body
    pos = _as_f64(sim.data.get_body_xpos(body), 3)
    quat_wxyz = _as_f64(sim.data.get_body_xquat(body), 4).copy()
    return {"pos": pos, "quat_wxyz": quat_wxyz}


def get_eef_pose(rs_env: Any) -> tuple[np.ndarray, np.ndarray]:
    """End-effector position (3,) and quaternion xyzw (4,)."""
    if hasattr(rs_env, "_get_observations"):
        obs = rs_env._get_observations()
        pos = obs.get("robot0_eef_pos")
        quat = obs.get("robot0_eef_quat")
        if pos is not None and quat is not None:
            return _as_f64(pos, 3), _as_f64(quat, 4)
    if not rs_env.robots:
        raise RuntimeError("No robots in robosuite env.")
    robot = rs_env.robots[0]
    return _as_f64(robot._hand_pos, 3), _as_f64(robot._hand_quat, 4)


def _free_joint_name(rs_env: Any, object_name: str) -> str:
    obj = _require_object(rs_env, object_name)
    joints = getattr(obj, "joints", None)
    if not joints:
        raise RuntimeError(f"Object {object_name!r} has no joints.")
    return joints[-1] if len(joints) > 1 else joints[0]


def apply_object_pose_delta(
    rs_env: Any,
    object_name: str,
    dpos: np.ndarray | list[float],
    dyaw: float = 0.0,
    settle_steps: int = 5,
) -> dict[str, Any]:
    """Nudge object free-joint pose by ``dpos`` and yaw ``dyaw`` (radians)."""
    sim = rs_env.sim
    joint_name = _free_joint_name(rs_env, object_name)
    dpos_arr = _as_f64(dpos, 3)
    pre = get_object_pose(rs_env, object_name)

    if hasattr(sim.data, "get_joint_qpos") and hasattr(sim.data, "set_joint_qpos"):
        q = np.array(sim.data.get_joint_qpos(joint_name), dtype=np.float64, copy=True)
    else:
        qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
        qpos = np.array(sim.data.qpos, copy=True)
        if isinstance(qpos_addr, slice):
            q = qpos[qpos_addr].copy()
        elif isinstance(qpos_addr, tuple) and len(qpos_addr) == 2:
            q = qpos[int(qpos_addr[0]) : int(qpos_addr[1])].copy()
        else:
            raise RuntimeError(f"Unexpected qpos address for joint {joint_name!r}.")

    if q.size < 7:
        raise RuntimeError(f"Joint {joint_name!r} qpos width {q.size} < 7 (need free joint).")

    q[:3] = q[:3] + dpos_arr
    yaw = float(dyaw)
    if abs(yaw) > 0.0:
        half = 0.5 * yaw
        dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
        qw, qx, qy, qz = q[3], q[4], q[5], q[6]
        q[3] = dq[0] * qw - dq[1] * qx - dq[2] * qy - dq[3] * qz
        q[4] = dq[0] * qx + dq[1] * qw + dq[2] * qz - dq[3] * qy
        q[5] = dq[0] * qy - dq[1] * qz + dq[2] * qw + dq[3] * qx
        q[6] = dq[0] * qz + dq[1] * qy - dq[2] * qx + dq[3] * qw
        n = float(np.linalg.norm(q[3:7]))
        if n > 0:
            q[3:7] /= n

    if hasattr(sim.data, "set_joint_qpos"):
        sim.data.set_joint_qpos(joint_name, q)
        if hasattr(sim.data, "set_joint_qvel"):
            sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    else:
        qpos_addr = sim.model.get_joint_qpos_addr(joint_name)
        qpos = np.array(sim.data.qpos, copy=True)
        if isinstance(qpos_addr, slice):
            qpos[qpos_addr] = q
        else:
            start, stop = int(qpos_addr[0]), int(qpos_addr[1])
            qpos[start:stop] = q
        sim.data.qpos[:] = qpos
    sim.forward()
    for _ in range(max(0, int(settle_steps))):
        sim.step()

    post = get_object_pose(rs_env, object_name)
    return {
        "object_name": object_name,
        "dpos": dpos_arr,
        "dyaw": float(dyaw),
        "pre_object_pos": pre["pos"],
        "post_object_pos": post["pos"],
        "pre_pose": pre,
        "post_pose": post,
    }


def apply_eef_bump(
    rs_env: Any,
    force: np.ndarray | list[float],
    settle_steps: int = 10,
) -> dict[str, Any]:
    """Apply a short external force on the robot end-effector body, then clear it."""
    sim = rs_env.sim
    if not rs_env.robots:
        raise RuntimeError("No robots in robosuite env.")
    robot = rs_env.robots[0]
    body_id = None
    if hasattr(robot, "eef_site_id"):
        site_id = int(robot.eef_site_id)
        body_id = int(sim.model.site_bodyid[site_id])
    if body_id is None:
        for name in ("gripper0_eef", "robot0_eef", "eef"):
            try:
                body_id = int(sim.model.body_name2id(name))
                break
            except Exception:
                continue
    if body_id is None:
        raise RuntimeError("Could not resolve end-effector body id for eef_bump.")

    f = _as_f64(force, 3)
    eef_pre, _ = get_eef_pose(rs_env)
    xfrc = np.zeros(6, dtype=np.float64)
    xfrc[:3] = f
    sim.data.xfrc_applied[body_id] = xfrc
    for _ in range(max(0, int(settle_steps))):
        sim.step()
    sim.data.xfrc_applied[body_id] = 0.0
    sim.forward()
    eef_post, _ = get_eef_pose(rs_env)
    return {
        "force": f,
        "body_id": int(body_id),
        "eef_pre": eef_pre,
        "eef_post": eef_post,
        "settle_steps": int(settle_steps),
    }
