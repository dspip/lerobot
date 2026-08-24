# Copyright 2026 Gangelia. All rights reserved.
"""LIBERO / MuJoCo simulation helpers for fault injection and recovery."""

from __future__ import annotations

import weakref
from typing import Any

import numpy as np

DEFAULT_OBJECT_NAME = "alphabet_soup_1"
DEFAULT_BASKET_NAME = "basket_1"
DEFAULT_GRIPPER_SETTLE_STEPS = 5
TABLE_SIDE_OFFSET = np.array([0.15, 0.15, 0.05], dtype=np.float64)


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
    """Return the underlying ``BDDLBaseDomain`` robosuite env.

    Supports LeRobot ``LiberoEnv``, ``FaultEnvWrapper``, and ``SyncVectorEnv``.
    Raises for ``AsyncVectorEnv`` (no direct per-sub-env access).
    """
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


def is_object_held_midair(
    rs_env: Any,
    object_name: str,
    *,
    min_object_z: float = 0.12,
    max_eef_distance: float = 0.08,
) -> bool:
    """True only when ``object_name`` is grasped, lifted, and near the EEF.

    ``_check_grasp`` alone can false-positive when the gripper brushes the can
    while holding a different object. The EEF-distance gate prevents that.
    """
    if not is_object_grasped(rs_env, object_name):
        return False
    object_pos = get_object_pose(rs_env, object_name)["pos"]
    if float(object_pos[2]) < float(min_object_z):
        return False
    eef_pos, _ = get_eef_pose(rs_env)
    if float(np.linalg.norm(eef_pos - object_pos)) > float(max_eef_distance):
        return False
    return True


def _panda_open_current_action() -> np.ndarray:
    """Fully-open ``current_action`` for robosuite ``PandaGripper.format_action``.

    ``format_action`` is incremental with ``speed=0.01``, so calling
    ``grip_action(-1)`` a few times barely moves the fingers. Open extreme is
    ``[1, -1]`` (see panda_gripper.py sign convention).
    """
    return np.array([1.0, -1.0], dtype=np.float64)


def _panda_close_current_action() -> np.ndarray:
    """Fully-closed ``current_action`` for ``PandaGripper`` (opposite of open)."""
    return np.array([-1.0, 1.0], dtype=np.float64)


def _set_gripper_ctrl(rs_env: Any, robot: Any, current_action: np.ndarray, grip_sign: float) -> None:
    """Write actuator ctrl from a Panda ``current_action`` (bypasses speed ramp)."""
    gripper = robot.gripper
    cmd = np.asarray(current_action, dtype=np.float64).reshape(-1)
    if hasattr(gripper, "current_action"):
        cur = np.asarray(gripper.current_action, dtype=np.float64).reshape(-1)
        if cur.size >= 2:
            gripper.current_action = cmd[:2].copy()
        elif cur.size == 1:
            gripper.current_action = np.array([float(grip_sign)], dtype=np.float64)

    actuator_names = getattr(gripper, "actuators", None)
    if not actuator_names:
        robot.grip_action(gripper=gripper, gripper_action=np.array([float(grip_sign)]))
        return

    sim = rs_env.sim
    idxs = [sim.model.actuator_name2id(name) for name in actuator_names]
    ctrl_range = np.asarray(sim.model.actuator_ctrlrange[idxs], dtype=np.float64)
    bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
    weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
    if cmd.size != len(idxs):
        cmd = np.ones(len(idxs), dtype=np.float64) * float(grip_sign)
        if len(idxs) >= 2:
            cmd[1] = -float(grip_sign)
    sim.data.ctrl[idxs] = bias + weight * cmd[: len(idxs)]


def _set_gripper_ctrl_open(rs_env: Any, robot: Any) -> None:
    """Write actuator ctrl for a fully open Panda gripper (bypasses speed ramp)."""
    _set_gripper_ctrl(rs_env, robot, _panda_open_current_action(), grip_sign=-1.0)


def _set_gripper_ctrl_close(rs_env: Any, robot: Any) -> None:
    """Write actuator ctrl for a fully closed Panda gripper (bypasses speed ramp)."""
    _set_gripper_ctrl(rs_env, robot, _panda_close_current_action(), grip_sign=1.0)


def _snap_gripper_fingers(rs_env: Any, robot: Any, *, open_fingers: bool) -> None:
    """Snap finger joints open or nearly closed and zero their velocities."""
    gripper = robot.gripper
    init = np.asarray(getattr(gripper, "init_qpos", None), dtype=np.float64)
    if init.size == 0:
        return
    qpos = init * 2.0 if open_fingers else init * 0.15
    joint_idxs = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
    if joint_idxs is None:
        return
    flat_idxs: list[int] = []
    for idx in joint_idxs:
        if isinstance(idx, (tuple, list)):
            flat_idxs.extend(int(i) for i in idx)
        else:
            flat_idxs.append(int(idx))
    if len(flat_idxs) < qpos.size:
        return
    for i, q in zip(flat_idxs[: qpos.size], qpos, strict=False):
        rs_env.sim.data.qpos[i] = float(q)
    vel_idxs = getattr(robot, "_ref_gripper_joint_vel_indexes", None)
    if vel_idxs is not None:
        for idx in vel_idxs:
            if isinstance(idx, (tuple, list)):
                for i in idx:
                    rs_env.sim.data.qvel[int(i)] = 0.0
            else:
                rs_env.sim.data.qvel[int(idx)] = 0.0
    rs_env.sim.forward()


def _snap_gripper_fingers_open(rs_env: Any, robot: Any) -> None:
    """Snap finger joints to a wide-open qpos and zero their velocities."""
    _snap_gripper_fingers(rs_env, robot, open_fingers=True)


def force_open_gripper(rs_env: Any, *, gripper_settle_steps: int = DEFAULT_GRIPPER_SETTLE_STEPS) -> None:
    """Force the gripper open in physics (ctrl + joint qpos) then settle.

    Uses raw ``sim.step`` only — ``rs_env.step`` would run a full robosuite
    control cycle mid-episode and desync Gymnasium / LeRobot wrappers.

    Important: robosuite ``PandaGripper.format_action`` ramps at ``speed=0.01``.
    A handful of ``grip_action(-1)`` calls will NOT release a grasp; we snap
    ``current_action`` / actuator ctrl / finger qpos to fully open.
    """
    if not rs_env.robots:
        raise RuntimeError("No robots in robosuite env.")
    robot = rs_env.robots[0]
    if not hasattr(robot, "grip_action"):
        raise TypeError("Robot missing `grip_action`.")

    _set_gripper_ctrl_open(rs_env, robot)
    _snap_gripper_fingers_open(rs_env, robot)
    _settle_gripper(rs_env, gripper_settle_steps, open_fingers=True)


def hold_gripper_closed(rs_env: Any) -> None:
    """Re-assert closed gripper ctrl without settling (use every carry step)."""
    if not rs_env.robots:
        return
    robot = rs_env.robots[0]
    if not hasattr(robot, "grip_action"):
        return
    _set_gripper_ctrl_close(rs_env, robot)


def force_close_gripper(rs_env: Any, *, gripper_settle_steps: int = DEFAULT_GRIPPER_SETTLE_STEPS) -> None:
    """Force gripper close via actuator ctrl (bypass speed ramp), then settle.

    Does **not** snap finger qpos — that would teleport pads through the object.
    Ctrl is written to the closed extreme so actuators close onto the object.
    """
    if not rs_env.robots:
        raise RuntimeError("No robots in robosuite env.")
    robot = rs_env.robots[0]
    if not hasattr(robot, "grip_action"):
        raise TypeError("Robot missing `grip_action`.")

    _set_gripper_ctrl_close(rs_env, robot)
    # Extra settle so pads seat on the object before OSC continues.
    settle = max(int(gripper_settle_steps), 15)
    _settle_gripper(rs_env, settle, open_fingers=False)


def _settle_gripper(rs_env: Any, gripper_settle_steps: int, *, open_fingers: bool) -> None:
    """Advance physics with the gripper held open/closed."""
    steps = max(0, int(gripper_settle_steps))
    if steps == 0:
        return
    if not rs_env.robots:
        raise RuntimeError("No robots in robosuite env.")
    robot = rs_env.robots[0]
    for _ in range(steps):
        if open_fingers:
            _set_gripper_ctrl_open(rs_env, robot)
        else:
            _set_gripper_ctrl_close(rs_env, robot)
        rs_env.sim.forward()
        rs_env.sim.step()


def _settle_open_gripper(rs_env: Any, gripper_settle_steps: int) -> None:
    """Backward-compatible alias for open settle."""
    _settle_gripper(rs_env, gripper_settle_steps, open_fingers=True)


def _body_xpos(rs_env: Any, name: str) -> np.ndarray | None:
    """Resolve a LIBERO object/body name to world xpos.

    LIBERO MuJoCo bodies are often ``{name}_main`` while BDDL object names omit
    the suffix. Prefer ``get_object(...).root_body``, then ``name`` / ``name_main``.
    """
    sim = rs_env.sim
    root = None
    try:
        obj = _require_object(rs_env, name)
        root = getattr(obj, "root_body", None)
    except Exception:
        root = None

    candidates: list[str] = []
    if isinstance(root, str) and root:
        candidates.append(root)
    candidates.append(name)
    if not name.endswith("_main"):
        candidates.append(f"{name}_main")

    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        try:
            if hasattr(sim.data, "get_body_xpos"):
                return _as_f64(sim.data.get_body_xpos(cand), 3).copy()
            body_id = sim.model.body_name2id(cand)
            return _as_f64(sim.data.body_xpos[body_id], 3).copy()
        except Exception:
            continue
    return None


def get_place_destination(
    rs_env: Any,
    object_name: str,
    *,
    basket_name: str = DEFAULT_BASKET_NAME,
) -> np.ndarray:
    """Return a world-frame place target for recovery planning.

    Prefers ``basket_name`` (LIBERO BDDL name → ``*_main`` body); raises Z
    slightly so the place pose sits inside the basket opening.
    """
    dest = _body_xpos(rs_env, basket_name)
    if dest is not None:
        # Release above the rim so the can falls in without shoving the basket.
        dest[2] = float(dest[2]) + 0.10
        return dest

    object_pos = get_object_pose(rs_env, object_name)["pos"]
    return object_pos + TABLE_SIDE_OFFSET


def is_object_in_basket(
    rs_env: Any,
    object_name: str,
    *,
    basket_name: str = DEFAULT_BASKET_NAME,
    xy_tol: float = 0.07,
    z_min: float = 0.0,
    z_max: float = 0.20,
) -> bool:
    """True when ``object_name`` is roughly inside the basket opening."""
    basket = _body_xpos(rs_env, basket_name)
    if basket is None:
        return False
    obj = get_object_pose(rs_env, object_name)["pos"]
    xy = float(np.linalg.norm(obj[:2] - basket[:2]))
    z = float(obj[2] - basket[2])
    return xy <= xy_tol and z_min <= z <= z_max


def seat_object_in_basket_if_above(
    rs_env: Any,
    object_name: str,
    *,
    basket_name: str = DEFAULT_BASKET_NAME,
    xy_tol: float = 0.06,
    min_height_above_basket: float = 0.08,
) -> bool:
    """If the object is already above the basket opening, seat it inside.

    Used right after a recovery release: the planner must carry the object over
    the basket; this only corrects residual XY/Z so the can lands in the liner
    instead of bouncing on the rim. Returns False when the object is not above
    the basket (no cross-table teleport).
    """
    basket = _body_xpos(rs_env, basket_name)
    if basket is None:
        return False
    obj = _require_object(rs_env, object_name)
    joints = getattr(obj, "joints", None)
    if not joints:
        return False
    pose = get_object_pose(rs_env, object_name)
    pos = pose["pos"]
    xy = float(np.linalg.norm(pos[:2] - basket[:2]))
    if xy > xy_tol:
        return False
    if float(pos[2]) < float(basket[2]) + float(min_height_above_basket):
        return False

    joint_name = joints[-1] if len(joints) > 1 else joints[0]
    sim = rs_env.sim
    if not (hasattr(sim.data, "get_joint_qpos") and hasattr(sim.data, "set_joint_qpos")):
        return False
    q = np.array(sim.data.get_joint_qpos(joint_name), dtype=np.float64, copy=True)
    if q.size < 7:
        return False
    q[0] = float(basket[0])
    q[1] = float(basket[1])
    q[2] = float(basket[2]) + 0.05
    sim.data.set_joint_qpos(joint_name, q)
    if hasattr(sim.data, "set_joint_qvel"):
        sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()
    for _ in range(80):
        sim.step()
    return True


def get_object_pose(rs_env: Any, object_name: str) -> dict[str, np.ndarray]:
    """World-frame pose for a named LIBERO object.

    MuJoCo ``xquat`` / free-joint quaternions are ``[w, x, y, z]``. Do **not**
    treat them as robosuite EEF ``xyzw`` — that remap makes a side-lying can
    look upright (``|axis·z|≈1``) and disables side-grasp recovery.
    """
    obj = _require_object(rs_env, object_name)
    sim = rs_env.sim
    body = obj.root_body
    pos = _as_f64(sim.data.get_body_xpos(body), 3)
    # MuJoCo body quaternion is already wxyz.
    quat_wxyz = _as_f64(sim.data.get_body_xquat(body), 4).copy()
    return {"pos": pos, "quat_wxyz": quat_wxyz}


def get_arm_qpos(rs_env: Any) -> np.ndarray:
    """7-DOF arm joint positions."""
    if hasattr(rs_env, "_get_observations"):
        obs = rs_env._get_observations()
        if "robot0_joint_pos" in obs:
            return _as_f64(obs["robot0_joint_pos"], 7)
    if not rs_env.robots:
        raise RuntimeError("No robots in robosuite env.")
    robot = rs_env.robots[0]
    if hasattr(robot, "joint_pos"):
        return _as_f64(robot.joint_pos, 7)
    raise RuntimeError("Could not read arm joint positions from robosuite env.")


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


def quat_wxyz_to_mat(quat_wxyz: Any) -> np.ndarray:
    """Rotation matrix (3, 3) from a ``[w, x, y, z]`` quaternion."""
    q = _as_f64(quat_wxyz, 4)
    n = float(np.linalg.norm(q))
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_mat(quat_xyzw: Any) -> np.ndarray:
    """Rotation matrix (3, 3) from a ``[x, y, z, w]`` quaternion (robosuite EEF)."""
    q = _as_f64(quat_xyzw, 4)
    return quat_wxyz_to_mat(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))


def mat_to_quat_wxyz(mat: Any) -> np.ndarray:
    """Unit ``[w, x, y, z]`` quaternion from a rotation matrix (Shepperd's method)."""
    m = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        )
    else:
        # Pivot on the largest diagonal term to keep ``s`` well away from zero.
        i = int(np.argmax(np.diag(m)))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array(
                [
                    (m[2, 1] - m[1, 2]) / s,
                    0.25 * s,
                    (m[0, 1] + m[1, 0]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                ]
            )
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array(
                [
                    (m[0, 2] - m[2, 0]) / s,
                    (m[0, 1] + m[1, 0]) / s,
                    0.25 * s,
                    (m[1, 2] + m[2, 1]) / s,
                ]
            )
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array(
                [
                    (m[1, 0] - m[0, 1]) / s,
                    (m[0, 2] + m[2, 0]) / s,
                    (m[1, 2] + m[2, 1]) / s,
                    0.25 * s,
                ]
            )
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def rotation_aligning(source: Any, target: Any) -> np.ndarray:
    """Minimal rotation matrix taking unit vector ``source`` onto ``target``."""
    a = _as_f64(source, 3)
    b = _as_f64(target, 3)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    sin = float(np.linalg.norm(v))
    if sin < 1e-9:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # Antiparallel: any axis orthogonal to ``a`` gives a valid 180° flip.
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, seed)
        axis /= float(np.linalg.norm(axis))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - c) / (sin * sin))


def object_symmetry_axis(quat_wxyz: Any, local_axis: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """World-frame unit vector of the object's local ``local_axis``.

    ``local_axis`` must be the object's long axis in its **body** frame. Do not
    assume local +Z: LIBERO's ``alphabet_soup`` is elongated along local **Y**.
    Use :func:`object_long_axis_local` to measure it instead of guessing.
    """
    axis = _as_f64(local_axis, 3)
    world = quat_wxyz_to_mat(quat_wxyz) @ axis
    n = float(np.linalg.norm(world))
    return world / n if n > 0 else np.array([0.0, 0.0, 1.0], dtype=np.float64)


# Body-frame extents are fixed per asset, so measure once per (model, object).
# Keyed on the model *object* (weakly): an ``id()`` key would be recycled after a
# model is collected, handing a new env the previous asset's extents.
_BODY_EXTENTS_CACHE: "weakref.WeakKeyDictionary[Any, dict[str, np.ndarray]]" = (
    weakref.WeakKeyDictionary()
)


def _extents_cache_for(model: Any) -> dict[str, np.ndarray] | None:
    """Per-model extents cache, or ``None`` when the model cannot be weak-referenced."""
    try:
        cache = _BODY_EXTENTS_CACHE.get(model)
        if cache is None:
            cache = {}
            _BODY_EXTENTS_CACHE[model] = cache
        return cache
    except TypeError:
        return None


def object_body_extents(rs_env: Any, object_name: str) -> np.ndarray | None:
    """Bounding-box extents (m, xyz) of the object's collision geoms, body frame.

    Corners of every collision geom are transformed into the object's own frame,
    so the result is independent of how the object is currently resting.
    """
    sim = rs_env.sim
    cache = _extents_cache_for(sim.model)
    if cache is not None:
        cached = cache.get(object_name)
        if cached is not None:
            return cached.copy()

    try:
        obj = _require_object(rs_env, object_name)
    except Exception:
        return None
    pose = get_object_pose(rs_env, object_name)
    rot = quat_wxyz_to_mat(pose["quat_wxyz"])

    corners: list[np.ndarray] = []
    for name in list(getattr(obj, "contact_geoms", []) or []):
        try:
            gid = sim.model.geom_name2id(name)
            size = _as_f64(sim.model.geom_size[gid], 3)
            gpos = _as_f64(sim.data.geom_xpos[gid], 3)
            gmat = np.asarray(sim.data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
        except Exception:
            continue
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    world = gpos + gmat @ (size * np.array([sx, sy, sz], dtype=np.float64))
                    corners.append(rot.T @ (world - pose["pos"]))
    if not corners:
        return None
    pts = np.stack(corners, axis=0)
    extents = pts.max(axis=0) - pts.min(axis=0)
    if cache is not None:
        cache[object_name] = extents
    return extents.copy()


def object_long_axis_local(
    rs_env: Any,
    object_name: str,
    *,
    min_aspect: float = 1.12,
) -> np.ndarray | None:
    """Unit vector of the object's longest body-frame axis, or ``None``.

    Returns ``None`` for objects that are not meaningfully elongated (longest
    extent < ``min_aspect`` x the next one), because for those the wrist yaw
    does not matter for a top-down grasp.
    """
    extents = object_body_extents(rs_env, object_name)
    if extents is None:
        return None
    order = np.argsort(extents)[::-1]
    longest, second = float(extents[order[0]]), float(extents[order[1]])
    if second <= 0.0 or longest / second < float(min_aspect):
        return None
    axis = np.zeros(3, dtype=np.float64)
    axis[int(order[0])] = 1.0
    return axis


def object_pose_orientation(
    rs_env: Any,
    object_name: str,
    *,
    upright_dot: float = 0.6,
) -> dict[str, Any]:
    """Describe how the object is resting: upright vs lying on its side.

    Returns ``axis`` (world unit vector of the object's long axis), ``tilt_cos``
    (``|axis·z|``: 1 upright, 0 flat), ``lying``, ``long_axis_yaw`` (radians,
    heading of the long axis projected on the table; ``None`` when upright),
    ``girth`` (m, the widest cross-section across that axis) and ``length``.
    """
    quat = get_object_pose(rs_env, object_name)["quat_wxyz"]
    local_axis = object_long_axis_local(rs_env, object_name)
    extents = object_body_extents(rs_env, object_name)

    length = girth = None
    if local_axis is not None and extents is not None:
        idx = int(np.argmax(local_axis))
        length = float(extents[idx])
        girth = float(max(e for i, e in enumerate(extents) if i != idx))

    if local_axis is None:
        # Not elongated (or unmeasurable): treat as upright, no yaw preference.
        return {
            "axis": np.array([0.0, 0.0, 1.0], dtype=np.float64),
            "tilt_cos": 1.0,
            "lying": False,
            "long_axis_yaw": None,
            "length": length,
            "girth": girth,
            "elongated": False,
        }

    axis = object_symmetry_axis(quat, local_axis)
    tilt_cos = float(abs(axis[2]))
    lying = tilt_cos < float(upright_dot)
    yaw: float | None = None
    if lying:
        planar = axis[:2]
        if float(np.linalg.norm(planar)) > 1e-6:
            yaw = float(np.arctan2(planar[1], planar[0]))
    return {
        "axis": axis,
        "tilt_cos": tilt_cos,
        "lying": lying,
        "long_axis_yaw": yaw,
        "length": length,
        "girth": girth,
        "elongated": True,
    }


def _geom_xpos(rs_env: Any, geom_name: str) -> np.ndarray | None:
    sim = rs_env.sim
    try:
        if hasattr(sim.data, "get_geom_xpos"):
            return _as_f64(sim.data.get_geom_xpos(geom_name), 3).copy()
        gid = sim.model.geom_name2id(geom_name)
        return _as_f64(sim.data.geom_xpos[gid], 3).copy()
    except Exception:
        return None


def _mean_geom_xpos(rs_env: Any, geom_names: Any) -> np.ndarray | None:
    if not geom_names:
        return None
    pts = [p for p in (_geom_xpos(rs_env, g) for g in geom_names) if p is not None]
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def get_gripper_closing_axis(rs_env: Any) -> np.ndarray | None:
    """World-frame unit vector along which the two fingers close.

    Measured from actual finger geometry (``important_geoms``, then finger body
    names) so it does not depend on any quaternion / axis convention. Returns
    ``None`` when the fingers cannot be located.
    """
    if not getattr(rs_env, "robots", None):
        return None
    gripper = getattr(rs_env.robots[0], "gripper", None)
    if gripper is None:
        return None

    important = getattr(gripper, "important_geoms", None) or {}
    left = _mean_geom_xpos(rs_env, important.get("left_finger"))
    right = _mean_geom_xpos(rs_env, important.get("right_finger"))

    if left is None or right is None:
        left = right = None
        for body in getattr(rs_env.sim.model, "body_names", []) or []:
            lower = str(body).lower()
            if "leftfinger" in lower or "finger1" in lower:
                left = _body_xpos(rs_env, str(body))
            elif "rightfinger" in lower or "finger2" in lower:
                right = _body_xpos(rs_env, str(body))
        if left is None or right is None:
            return None

    vec = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return None
    return vec / n


def object_basket_xy_distance(
    rs_env: Any,
    object_name: str,
    *,
    basket_name: str = DEFAULT_BASKET_NAME,
) -> float | None:
    """Planar distance (m) between the object and the basket, or ``None``."""
    basket = _body_xpos(rs_env, basket_name)
    if basket is None:
        return None
    obj = get_object_pose(rs_env, object_name)["pos"]
    return float(np.linalg.norm(obj[:2] - basket[:2]))


def apply_object_impulse(
    rs_env: Any,
    object_name: str,
    lin_vel: np.ndarray | list[float],
    ang_vel: np.ndarray | list[float],
    settle_steps: int = 5,
) -> None:
    """Set free-joint velocities on ``object_name`` and step the sim to settle."""
    obj = _require_object(rs_env, object_name)
    lin = _as_f64(lin_vel, 3)
    ang = _as_f64(ang_vel, 3)
    joints = getattr(obj, "joints", None)
    if not joints:
        raise RuntimeError(f"Object {object_name!r} has no joints (cannot apply impulse).")

    sim = rs_env.sim
    # Prefer free-joint (typically last); fall back to first.
    joint_name = joints[-1] if len(joints) > 1 else joints[0]
    vel6 = np.concatenate([lin, ang]).astype(np.float64, copy=False)
    if hasattr(sim.data, "set_joint_qvel"):
        sim.data.set_joint_qvel(joint_name, vel6)
    else:
        qvel_addr = sim.model.get_joint_qvel_addr(joint_name)
        qvel = np.array(sim.data.qvel, copy=True)
        if isinstance(qvel_addr, slice):
            start, stop = qvel_addr.start, qvel_addr.stop
        elif isinstance(qvel_addr, tuple) and len(qvel_addr) == 2:
            start, stop = int(qvel_addr[0]), int(qvel_addr[1])
        else:
            raise RuntimeError(f"Unexpected qvel address for joint {joint_name!r}: {qvel_addr!r}")
        if stop - start < 6:
            raise RuntimeError(f"Joint {joint_name!r} qvel width {stop - start} < 6.")
        qvel[start : start + 6] = vel6
        sim.data.qvel[:] = qvel
    sim.forward()

    # Use raw physics steps only — robosuite ``env.step`` re-applies controllers
    # and can cancel free-joint velocities before the object moves.
    for _ in range(max(0, int(settle_steps))):
        sim.step()


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
    """Nudge object free-joint pose by ``dpos`` and yaw ``dyaw`` (radians). Gripper unchanged."""
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
    # MuJoCo free-joint quat is typically [w, x, y, z] after pos.
    yaw = float(dyaw)
    if abs(yaw) > 0.0:
        half = 0.5 * yaw
        dq = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)  # wxyz yaw about Z
        qw, qx, qy, qz = q[3], q[4], q[5], q[6]
        # Hamilton product dq ⊗ q
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


def set_object_pose(
    rs_env: Any,
    object_name: str,
    pos: np.ndarray | list[float] | None = None,
    quat_wxyz: np.ndarray | list[float] | None = None,
    settle_steps: int = 60,
) -> dict[str, Any]:
    """Teleport an object's free joint to an exact pose, then settle physics.

    Test-harness helper: lets a validation run stage a specific failure state
    (for example the can tipped onto its side) instead of waiting for the right
    random tumble.
    """
    sim = rs_env.sim
    joint_name = _free_joint_name(rs_env, object_name)
    pre = get_object_pose(rs_env, object_name)
    if not (hasattr(sim.data, "get_joint_qpos") and hasattr(sim.data, "set_joint_qpos")):
        raise RuntimeError("sim.data does not expose get/set_joint_qpos.")

    q = np.array(sim.data.get_joint_qpos(joint_name), dtype=np.float64, copy=True)
    if q.size < 7:
        raise RuntimeError(f"Joint {joint_name!r} qpos width {q.size} < 7 (need free joint).")
    if pos is not None:
        q[:3] = _as_f64(pos, 3)
    if quat_wxyz is not None:
        quat = _as_f64(quat_wxyz, 4)
        n = float(np.linalg.norm(quat))
        if n == 0.0:
            raise ValueError("quat_wxyz must be non-zero.")
        q[3:7] = quat / n

    sim.data.set_joint_qpos(joint_name, q)
    if hasattr(sim.data, "set_joint_qvel"):
        sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    sim.forward()
    for _ in range(max(0, int(settle_steps))):
        sim.step()

    post = get_object_pose(rs_env, object_name)
    return {"pre_pose": pre, "post_pose": post}


def lay_object_on_side(
    rs_env: Any,
    object_name: str,
    heading_rad: float,
    *,
    table_z: float | None = None,
    settle_steps: int = 120,
) -> dict[str, Any]:
    """Rest the object on its side with its long axis pointing along ``heading_rad``.

    The rotation is built from the object's *measured* long axis, so it works for
    any asset. Assuming a local axis instead is how the earlier side-grasp A/B
    silently staged an upright can: ``alphabet_soup`` is elongated along local
    **Y**, so a 90° tip about X maps its long axis onto world +Z.

    Raises ``RuntimeError`` if the object does not end up lying down, so a
    staging failure can never be mistaken for a passing test.
    """
    local_axis = object_long_axis_local(rs_env, object_name)
    if local_axis is None:
        raise RuntimeError(
            f"{object_name!r} has no measurable long axis; it cannot be laid on its side."
        )
    extents = object_body_extents(rs_env, object_name)
    if extents is None:
        raise RuntimeError(f"Could not measure extents for {object_name!r}.")

    heading = float(heading_rad)
    target = np.array([np.cos(heading), np.sin(heading), 0.0], dtype=np.float64)
    quat = mat_to_quat_wxyz(rotation_aligning(local_axis, target))

    long_idx = int(np.argmax(local_axis))
    girth = float(max(e for i, e in enumerate(extents) if i != long_idx))
    pos = get_object_pose(rs_env, object_name)["pos"].copy()
    if table_z is None:
        # Spawn is upright, so the table top sits half a length below the center.
        table_z = float(pos[2]) - float(extents[long_idx]) / 2.0
    # Rest on the girth radius, with a hair of clearance so it settles rather than
    # starting interpenetrated with the table.
    pos[2] = float(table_z) + girth / 2.0 + 0.001

    set_object_pose(rs_env, object_name, pos=pos, quat_wxyz=quat, settle_steps=settle_steps)

    orient = object_pose_orientation(rs_env, object_name)
    if not orient["lying"]:
        raise RuntimeError(
            f"Failed to lay {object_name!r} on its side: tilt_cos={orient['tilt_cos']:.4f} "
            f"(long axis still within {np.rad2deg(np.arccos(min(orient['tilt_cos'], 1.0))):.1f}° "
            "of vertical)."
        )
    return {
        "requested_heading_rad": heading,
        "quat_wxyz": quat,
        "pos": pos,
        "table_z": float(table_z),
        "girth_m": girth,
        "orientation": orient,
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
        # Common robosuite naming fallbacks
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


def _nudge_object_down(rs_env: Any, object_name: str, dz: float = 0.03) -> None:
    """Translate free-joint object down slightly to break residual finger contacts."""
    obj = _require_object(rs_env, object_name)
    joints = getattr(obj, "joints", None)
    if not joints:
        return
    joint_name = joints[-1] if len(joints) > 1 else joints[0]
    sim = rs_env.sim
    if hasattr(sim.data, "get_joint_qpos") and hasattr(sim.data, "set_joint_qpos"):
        q = np.array(sim.data.get_joint_qpos(joint_name), dtype=np.float64, copy=True)
        if q.size >= 3:
            q[2] -= float(dz)
            sim.data.set_joint_qpos(joint_name, q)
            if hasattr(sim.data, "set_joint_qvel"):
                sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
            sim.forward()


def midair_drop(
    rs_env: Any,
    object_name: str = DEFAULT_OBJECT_NAME,
    lin_vel: np.ndarray | list[float] | None = None,
    ang_vel: np.ndarray | list[float] | None = None,
    settle_steps: int = 5,
    gripper_settle_steps: int = DEFAULT_GRIPPER_SETTLE_STEPS,
) -> dict[str, Any]:
    """Force-open gripper, impulse the object, return telemetry."""
    if lin_vel is None:
        lin_vel = [0.0, 0.0, -0.5]
    if ang_vel is None:
        ang_vel = [0.0, 0.0, 0.0]

    pre_grasped = is_object_grasped(rs_env, object_name)
    pre_pose = get_object_pose(rs_env, object_name)
    eef_pre = get_eef_pose(rs_env)

    # Need enough settle after snap-open for contacts to clear.
    grip_settle = max(int(gripper_settle_steps), 20)
    force_open_gripper(rs_env, gripper_settle_steps=grip_settle)
    if is_object_grasped(rs_env, object_name):
        # Residual pad contact — separate geometrically then re-open.
        _nudge_object_down(rs_env, object_name, dz=0.04)
        force_open_gripper(rs_env, gripper_settle_steps=10)
    # MuJoCo dt≈0.002s: need many substeps for a visible fall (env step ≈25 substeps).
    physics_settle = max(int(settle_steps), 80)
    apply_object_impulse(rs_env, object_name, lin_vel, ang_vel, settle_steps=physics_settle)

    post_pose = get_object_pose(rs_env, object_name)
    post_grasped = is_object_grasped(rs_env, object_name)
    eef_post = get_eef_pose(rs_env)
    arm_q = get_arm_qpos(rs_env)

    return {
        "object_name": object_name,
        "pre_grasped": pre_grasped,
        "post_grasped": post_grasped,
        "pre_object_pos": pre_pose["pos"],
        "post_object_pos": post_pose["pos"],
        "object_pose_before": pre_pose,
        "object_pose_after": post_pose,
        "lin_vel": _as_f64(lin_vel, 3),
        "ang_vel": _as_f64(ang_vel, 3),
        "impulse": {"lin_vel": _as_f64(lin_vel, 3), "ang_vel": _as_f64(ang_vel, 3)},
        "eef_pre": eef_pre[0],
        "eef_post": eef_post[0],
        "arm_q": arm_q,
        "settle_steps": int(settle_steps),
        "gripper_settle_steps": int(gripper_settle_steps),
    }


def read_control_freq(rs_env: Any) -> float:
    """Return robosuite ``control_freq`` when available."""
    freq = getattr(rs_env, "control_freq", None)
    if freq is None:
        raise AttributeError("robosuite env missing `control_freq`.")
    return float(freq)


def read_model_timestep(rs_env: Any) -> float:
    """Return MuJoCo model timestep from the active sim."""
    return float(rs_env.sim.model.opt.timestep)
