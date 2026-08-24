"""Side-grasp alignment, lying-pose staging, and recovery-demo honesty gates.

No MuJoCo / LeRobot / GPU required: the sim is faked down to the handful of
``sim.model`` / ``sim.data`` fields the geometry helpers actually read.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from lerobot.faults.recovery.evaluation import evaluate_recovery_episode
from lerobot.faults.sim.libero import (
    get_gripper_closing_axis,
    get_object_pose,
    lay_object_on_side,
    mat_to_quat_wxyz,
    object_body_extents,
    object_long_axis_local,
    object_pose_orientation,
    object_symmetry_axis,
    quat_wxyz_to_mat,
    rotation_aligning,
)
from lerobot.faults.recovery.planner import SimpleIKRecoveryPlanner, _wrap_to_half_pi

UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
OBJ = "alphabet_soup_1"

# Measured from LIBERO's alphabet_soup asset: 6.24 x 7.62 x 6.24 cm, so the long
# axis is local **Y**, not local Z (see outputs/side_grasp_truth.json).
CAN_HALF_EXTENTS = np.array([0.03121, 0.03810, 0.03120])
CAN_LOCAL_LONG_AXIS = np.array([0.0, 1.0, 0.0])


def _standing_quat() -> np.ndarray:
    """Quaternion that stands the can on its end (long axis → world +Z).

    Note this is *not* the identity: with an identity rotation the can's local-Y
    long axis lies along world Y, i.e. the can is already on its side.
    """
    return mat_to_quat_wxyz(rotation_aligning(CAN_LOCAL_LONG_AXIS, [0.0, 0.0, 1.0]))


def _quat_about_x(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])


def _quat_about_z(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _fake_can_env(
    quat_wxyz: np.ndarray,
    pos: np.ndarray | None = None,
    half_extents: np.ndarray | None = None,
) -> MagicMock:
    """Robosuite-shaped stub holding one box geom aligned with the body frame.

    Only the fields the geometry helpers read are populated, so these tests stay
    free of MuJoCo while still exercising the real corner-transform math.
    """
    pos = np.array([0.1, -0.2, 0.04]) if pos is None else np.asarray(pos, dtype=float)
    half = CAN_HALF_EXTENTS if half_extents is None else np.asarray(half_extents, dtype=float)
    rot = quat_wxyz_to_mat(quat_wxyz)

    obj = MagicMock()
    obj.root_body = f"{OBJ}_main"
    obj.contact_geoms = ["g1"]

    rs_env = MagicMock()
    rs_env.get_object.return_value = obj
    rs_env.sim.model = MagicMock()  # distinct per call → distinct extents cache
    rs_env.sim.model.geom_name2id.side_effect = lambda name: {"g1": 0}[name]
    rs_env.sim.model.geom_size = np.array([half])
    rs_env.sim.data.geom_xpos = np.array([pos])
    rs_env.sim.data.geom_xmat = np.array([rot.reshape(9)])
    rs_env.sim.data.get_body_xpos.return_value = pos
    rs_env.sim.data.get_body_xquat.return_value = np.asarray(quat_wxyz, dtype=float)
    return rs_env


# --------------------------------------------------------------------------
# Rotation / quaternion math
# --------------------------------------------------------------------------


def test_quat_wxyz_to_mat_is_orthonormal():
    mat = quat_wxyz_to_mat(_quat_about_z(0.7))
    np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-9)


def test_mat_to_quat_round_trips_through_all_shepperd_branches():
    rng = np.random.default_rng(0)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        mat = quat_wxyz_to_mat(q)
        np.testing.assert_allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(mat)), mat, atol=1e-8)


@pytest.mark.parametrize("angle", [0.0, np.pi])
def test_mat_to_quat_handles_degenerate_traces(angle):
    """trace <= 0 (180° turns) takes the pivot branch, not the fast path."""
    mat = quat_wxyz_to_mat(_quat_about_x(angle))
    np.testing.assert_allclose(quat_wxyz_to_mat(mat_to_quat_wxyz(mat)), mat, atol=1e-9)


def test_rotation_aligning_maps_source_onto_target():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        rot = rotation_aligning(a, b)
        np.testing.assert_allclose(rot @ a, b, atol=1e-9)
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)


def test_rotation_aligning_antiparallel_stays_a_proper_rotation():
    """A 180° flip must not silently become a reflection (det = -1)."""
    for axis in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])):
        rot = rotation_aligning(axis, -axis)
        np.testing.assert_allclose(rot @ axis, -axis, atol=1e-9)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-9)


def test_rotation_aligning_identity_when_already_aligned():
    np.testing.assert_allclose(
        rotation_aligning([0, 0, 1], [0, 0, 1]), np.eye(3), atol=1e-12
    )


# --------------------------------------------------------------------------
# Object orientation measured from geometry
# --------------------------------------------------------------------------


def test_upright_can_symmetry_axis_points_up():
    np.testing.assert_allclose(object_symmetry_axis(UPRIGHT), [0, 0, 1], atol=1e-9)


def test_symmetry_axis_follows_the_supplied_local_axis():
    """Passing local Y must not be silently overridden by a local-Z assumption."""
    axis = object_symmetry_axis(_quat_about_x(np.pi / 2), local_axis=(0.0, 1.0, 0.0))
    np.testing.assert_allclose(axis, [0, 0, 1], atol=1e-9)


def test_get_object_pose_keeps_mujoco_wxyz():
    """Regression: remapping wxyz→xyzw made a tipped can read as upright."""
    tipped = _quat_about_x(np.pi / 2)
    pose = get_object_pose(_fake_can_env(tipped), OBJ)
    np.testing.assert_allclose(pose["quat_wxyz"], tipped, atol=1e-9)


def test_body_extents_are_pose_invariant():
    upright = object_body_extents(_fake_can_env(UPRIGHT), OBJ)
    rolled = object_body_extents(_fake_can_env(_quat_about_z(0.9)), OBJ)
    np.testing.assert_allclose(upright, 2 * CAN_HALF_EXTENTS, atol=1e-9)
    np.testing.assert_allclose(rolled, 2 * CAN_HALF_EXTENTS, atol=1e-9)


def test_long_axis_is_measured_as_local_y_for_the_can():
    axis = object_long_axis_local(_fake_can_env(UPRIGHT), OBJ)
    np.testing.assert_allclose(axis, CAN_LOCAL_LONG_AXIS, atol=1e-12)


def test_near_cubic_object_has_no_long_axis():
    """Below the aspect threshold the wrist yaw does not matter — say so."""
    env = _fake_can_env(UPRIGHT, half_extents=np.array([0.03, 0.031, 0.03]))
    assert object_long_axis_local(env, OBJ) is None
    orient = object_pose_orientation(env, OBJ)
    assert orient["elongated"] is False
    assert orient["lying"] is False


def test_can_standing_on_its_end_reads_upright():
    """Long axis is local Y, so the quat that stands the can up rotates Y→Z."""
    stand = mat_to_quat_wxyz(rotation_aligning(CAN_LOCAL_LONG_AXIS, [0, 0, 1]))
    orient = object_pose_orientation(_fake_can_env(stand), OBJ)
    assert not orient["lying"]
    assert orient["tilt_cos"] == pytest.approx(1.0, abs=1e-9)
    assert orient["long_axis_yaw"] is None


@pytest.mark.parametrize("heading_deg", [0, 30, 45, 90, 135, -60])
def test_can_laid_flat_reports_that_heading(heading_deg):
    heading = np.deg2rad(heading_deg)
    target = np.array([np.cos(heading), np.sin(heading), 0.0])
    quat = mat_to_quat_wxyz(rotation_aligning(CAN_LOCAL_LONG_AXIS, target))
    orient = object_pose_orientation(_fake_can_env(quat), OBJ)
    assert orient["lying"]
    assert orient["tilt_cos"] == pytest.approx(0.0, abs=1e-9)
    assert _wrap_to_half_pi(orient["long_axis_yaw"]) == pytest.approx(
        _wrap_to_half_pi(heading), abs=1e-6
    )
    assert orient["length"] == pytest.approx(2 * CAN_HALF_EXTENTS[1], abs=1e-9)
    assert orient["girth"] == pytest.approx(2 * CAN_HALF_EXTENTS[0], abs=1e-4)


def test_tipping_about_x_leaves_this_can_standing():
    """The bug the old A/B shipped: qx(90°) maps local Y onto world +Z."""
    orient = object_pose_orientation(_fake_can_env(_quat_about_x(np.pi / 2)), OBJ)
    assert not orient["lying"]


# --------------------------------------------------------------------------
# Staging helper
# --------------------------------------------------------------------------


def _lay_env(half_extents: np.ndarray | None = None) -> MagicMock:
    """Fake env, starting upright, whose ``set_object_pose`` rewrites the body quat."""
    env = _fake_can_env(_standing_quat(), half_extents=half_extents)

    def _apply(rs_env, object_name, pos=None, quat_wxyz=None, settle_steps=0):
        if quat_wxyz is not None:
            env.sim.data.get_body_xquat.return_value = np.asarray(quat_wxyz, dtype=float)
        if pos is not None:
            env.sim.data.get_body_xpos.return_value = np.asarray(pos, dtype=float)
        return {}

    return env, _apply


@pytest.mark.parametrize("heading_deg", [0, 45, 90, 135])
def test_lay_object_on_side_requests_a_pose_that_is_actually_lying(heading_deg):
    env, apply = _lay_env()
    with patch("lerobot.faults.sim.libero.set_object_pose", side_effect=apply):
        info = lay_object_on_side(env, OBJ, float(np.deg2rad(heading_deg)))
    assert info["orientation"]["lying"]
    assert _wrap_to_half_pi(info["orientation"]["long_axis_yaw"]) == pytest.approx(
        _wrap_to_half_pi(np.deg2rad(heading_deg)), abs=1e-6
    )
    # Resting height is the girth radius above the table, not the half length.
    assert info["girth_m"] == pytest.approx(2 * CAN_HALF_EXTENTS[0], abs=1e-4)


def test_lay_object_on_side_raises_when_the_object_is_not_elongated():
    env, apply = _lay_env(half_extents=np.array([0.03, 0.031, 0.03]))
    with patch("lerobot.faults.sim.libero.set_object_pose", side_effect=apply):
        with pytest.raises(RuntimeError, match="no measurable long axis"):
            lay_object_on_side(env, OBJ, 0.0)


def test_lay_object_on_side_raises_when_the_pose_does_not_stick():
    """A staging no-op must fail loudly, never be reported as a lying can."""
    env = _fake_can_env(_standing_quat())
    with patch("lerobot.faults.sim.libero.set_object_pose", return_value={}):
        with pytest.raises(RuntimeError, match="Failed to lay"):
            lay_object_on_side(env, OBJ, 0.0)


# --------------------------------------------------------------------------
# Planner: side grasp (takes a measured world-frame axis, never a quaternion)
# --------------------------------------------------------------------------


def _world_axis(heading_deg: float) -> np.ndarray:
    h = np.deg2rad(heading_deg)
    return np.array([np.cos(h), np.sin(h), 0.0])


def _angle_between_headings_deg(a_rad: float, b_rad: float) -> float:
    """Angle between two undirected headings, in [0°, 90°]."""
    return abs(np.rad2deg(_wrap_to_half_pi(a_rad - b_rad)))


def test_upright_can_needs_no_wrist_rotation():
    planner = SimpleIKRecoveryPlanner()
    planner._set_grasp_orientation(np.array([0.0, 0.0, 1.0]))
    assert not planner.object_lying
    assert planner.target_closing_yaw is None
    assert planner.grasp_z_offset == planner.grasp_offset


def test_tipped_can_reaches_lower_than_an_upright_one():
    planner = SimpleIKRecoveryPlanner()
    planner._set_grasp_orientation(_world_axis(0.0))
    assert planner.object_lying
    # A tipped can's center sits lower, so the fingers reach further down.
    assert planner.grasp_z_offset == planner.grasp_offset_lying
    assert planner.grasp_z_offset < planner.grasp_offset


@pytest.mark.parametrize("heading_deg", [0, 30, 45, 90, 135, -60])
def test_target_yaw_is_always_perpendicular_to_the_can(heading_deg):
    """Close across the 6.2 cm girth, never along the 7.6 cm length."""
    planner = SimpleIKRecoveryPlanner()
    planner._set_grasp_orientation(_world_axis(heading_deg))
    assert planner.object_lying
    separation = _angle_between_headings_deg(
        planner.target_closing_yaw, np.deg2rad(heading_deg)
    )
    assert separation == pytest.approx(90.0, abs=1e-6)


def test_side_grasp_can_be_disabled():
    planner = SimpleIKRecoveryPlanner(side_grasp_enabled=False)
    planner._set_grasp_orientation(_world_axis(0.0))
    assert not planner.object_lying
    assert planner.target_closing_yaw is None


def test_yaw_command_drives_fingers_toward_the_short_axis():
    planner = SimpleIKRecoveryPlanner(max_rot_step=0.5)
    planner._set_grasp_orientation(_world_axis(90.0))  # can along Y → close along X

    aligned = planner._yaw_command(np.array([1.0, 0.0, 0.0]))
    assert aligned == pytest.approx(0.0, abs=1e-9)
    assert planner.yaw_error == pytest.approx(0.0, abs=1e-9)

    # Fingers 30° counter-clockwise of the target → command must be negative.
    misaligned = planner._yaw_command(
        np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30)), 0.0])
    )
    assert misaligned < 0
    assert planner.yaw_error == pytest.approx(-np.deg2rad(30), abs=1e-6)

    # A 180°-flipped gripper grasps identically — demand no rotation.
    flipped = planner._yaw_command(np.array([-1.0, 0.0, 0.0]))
    assert flipped == pytest.approx(0.0, abs=1e-9)


def test_planner_emits_wrist_rotation_and_blocks_descent_until_aligned():
    lying_axis = _world_axis(90.0)  # wants the fingers closing along X
    planner = SimpleIKRecoveryPlanner(waypoint_noise_m=0.0)
    object_pos = np.array([0.1, -0.2, 0.03])
    planner.plan(
        eef_pos=np.array([0.1, -0.2, 0.30]),
        eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        object_pos=object_pos,
        object_axis=lying_axis,
        destination_pos=np.array([0.0, 0.26, 0.10]),
        gripper_open=True,
    )
    assert planner.object_lying

    crossed = np.array([0.0, 1.0, 0.0])  # fingers 90° off, and they stay there
    saw_rotation = False
    eef = np.array([0.1, -0.2, 0.30])
    lowest_z = eef[2]
    # Stay inside the per-waypoint stall budget: past it the planner deliberately
    # stops waiting and attempts the grasp rather than freezing.
    for _ in range(planner.max_steps_per_waypoint - 10):
        action = planner.next_action(
            eef_pos=eef,
            object_pos=object_pos,
            closing_axis=crossed,
            object_axis=lying_axis,
        )
        assert action is not None
        saw_rotation = saw_rotation or abs(float(action[5])) > 0.1
        assert planner.phase_name != "close_grasp"
        eef = eef + np.asarray(action[:3], dtype=float) * planner.max_pos_step
        lowest_z = min(lowest_z, float(eef[2]))
    assert saw_rotation
    assert lowest_z > object_pos[2] + planner.grasp_z_offset + 0.02


def test_planner_reaches_grasp_once_wrist_is_aligned():
    lying_axis = _world_axis(90.0)
    planner = SimpleIKRecoveryPlanner(waypoint_noise_m=0.0)
    object_pos = np.array([0.1, -0.2, 0.03])
    planner.plan(
        eef_pos=np.array([0.1, -0.2, 0.30]),
        eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        object_pos=object_pos,
        object_axis=lying_axis,
        destination_pos=np.array([0.0, 0.26, 0.10]),
        gripper_open=True,
    )
    aligned = np.array([1.0, 0.0, 0.0])
    phases = []
    eef = np.array([0.1, -0.2, 0.30])
    for _ in range(400):
        action = planner.next_action(
            eef_pos=eef,
            object_pos=object_pos,
            closing_axis=aligned,
            object_axis=lying_axis,
        )
        if action is None:
            break
        phases.append(planner.phase_name)
        eef = eef + np.asarray(action[:3], dtype=float) * planner.max_pos_step
    assert "close_grasp" in phases


def test_upright_can_leaves_wrist_action_untouched():
    planner = SimpleIKRecoveryPlanner(waypoint_noise_m=0.0)
    object_pos = np.array([0.1, -0.2, 0.05])
    planner.plan(
        eef_pos=np.array([0.1, -0.2, 0.30]),
        eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        object_pos=object_pos,
        object_axis=np.array([0.0, 0.0, 1.0]),
        destination_pos=np.array([0.0, 0.26, 0.10]),
        gripper_open=True,
    )
    action = planner.next_action(
        eef_pos=np.array([0.1, -0.2, 0.30]),
        object_pos=object_pos,
        closing_axis=np.array([0.0, 1.0, 0.0]),
        object_axis=np.array([0.0, 0.0, 1.0]),
    )
    assert action is not None
    assert float(action[5]) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Measured finger-closing axis (convention free)
# --------------------------------------------------------------------------


def _fake_gripper_env(left: np.ndarray, right: np.ndarray) -> MagicMock:
    rs_env = MagicMock()
    gripper = MagicMock()
    gripper.important_geoms = {"left_finger": ["lf"], "right_finger": ["rf"]}
    rs_env.robots = [MagicMock(gripper=gripper)]
    rs_env.sim.data.get_geom_xpos.side_effect = lambda name: {"lf": left, "rf": right}[name]
    return rs_env


def test_closing_axis_measured_from_finger_geometry():
    axis = get_gripper_closing_axis(
        _fake_gripper_env(np.array([0.0, -0.04, 0.2]), np.array([0.0, 0.04, 0.2]))
    )
    np.testing.assert_allclose(axis, [0, 1, 0], atol=1e-9)


def test_closing_axis_none_without_fingers():
    rs_env = MagicMock()
    rs_env.robots = [MagicMock(gripper=None)]
    assert get_gripper_closing_axis(rs_env) is None


# --------------------------------------------------------------------------
# Drop must fire before the can reaches the basket
# --------------------------------------------------------------------------


def _drop_cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="midair_drop",
        t_min=0,
        t_max=500,
        object_name=OBJ,
        require_grasp=True,
        min_object_z=0.0,
        post_grasp_delay_steps=50,
        settle_steps=1,
        seed=7,
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _patched_trigger(basket_dist, cfg: FaultInjectionConfig, steps: int) -> int | None:
    """Run ``_should_trigger`` for ``steps`` and return the firing step."""
    inj = MidAirDropFault(cfg, num_envs=1)
    state = inj._states[0]
    with (
        patch("lerobot.faults.recovery.midair_drop.get_robosuite_env", return_value=MagicMock()),
        patch("lerobot.faults.recovery.midair_drop.is_object_grasped", return_value=True),
        patch(
            "lerobot.faults.recovery.midair_drop.get_object_pose",
            return_value={"pos": np.array([0.1, -0.2, 0.2]), "quat_wxyz": UPRIGHT},
        ),
        patch(
            "lerobot.faults.recovery.midair_drop.get_eef_pose",
            return_value=(np.array([0.1, -0.2, 0.2]), np.array([0.0, 0.0, 0.0, 1.0])),
        ),
        patch(
            "lerobot.faults.recovery.midair_drop.object_basket_xy_distance",
            return_value=basket_dist,
        ),
    ):
        for step in range(steps):
            state.episode_step = step
            if inj._should_trigger(MagicMock(), 0, state):
                return step
    return None


def test_drop_waits_for_the_random_delay_while_far_from_the_basket():
    cfg = _drop_cfg(post_grasp_delay_steps=50, min_drop_distance_from_basket_m=0.18)
    assert _patched_trigger(0.50, cfg, steps=80) == 50


def test_drop_fires_early_when_the_carry_reaches_the_basket():
    cfg = _drop_cfg(post_grasp_delay_steps=50, min_drop_distance_from_basket_m=0.18)
    # Already inside the deadline radius → drop now, not over the basket.
    assert _patched_trigger(0.10, cfg, steps=80) == 0


def test_basket_deadline_can_be_disabled():
    cfg = _drop_cfg(post_grasp_delay_steps=50, min_drop_distance_from_basket_m=0.0)
    assert _patched_trigger(0.01, cfg, steps=80) == 50


def test_unknown_basket_distance_falls_back_to_the_delay():
    cfg = _drop_cfg(post_grasp_delay_steps=50, min_drop_distance_from_basket_m=0.18)
    assert _patched_trigger(None, cfg, steps=80) == 50


def test_negative_basket_deadline_is_rejected():
    with pytest.raises(ValueError, match="min_drop_distance_from_basket_m"):
        _drop_cfg(min_drop_distance_from_basket_m=-0.1)


def test_trigger_records_distance_and_reason():
    cfg = _drop_cfg(post_grasp_delay_steps=50, min_drop_distance_from_basket_m=0.18)
    inj = MidAirDropFault(cfg, num_envs=1)
    state = inj._states[0]
    with (
        patch("lerobot.faults.recovery.midair_drop.get_robosuite_env", return_value=MagicMock()),
        patch("lerobot.faults.recovery.midair_drop.is_object_grasped", return_value=True),
        patch(
            "lerobot.faults.recovery.midair_drop.get_object_pose",
            return_value={"pos": np.array([0.1, -0.2, 0.2]), "quat_wxyz": UPRIGHT},
        ),
        patch(
            "lerobot.faults.recovery.midair_drop.get_eef_pose",
            return_value=(np.array([0.1, -0.2, 0.2]), np.array([0.0, 0.0, 0.0, 1.0])),
        ),
        patch(
            "lerobot.faults.recovery.midair_drop.object_basket_xy_distance",
            return_value=0.12,
        ),
    ):
        assert inj._should_trigger(MagicMock(), 0, state)
    assert state.drop_basket_xy_dist == pytest.approx(0.12)
    assert state.drop_trigger_reason == "basket_deadline"


# --------------------------------------------------------------------------
# Recovery-demo success gates
# --------------------------------------------------------------------------


def _good_episode(**overrides):
    kwargs = dict(
        grasped_before_drop=True,
        was_midair_at_drop=True,
        triggered_at=70,
        drop_moved=True,
        object_visible_after_drop=True,
        recovery_steps=120,
        drop_basket_xy_dist=0.42,
        drop_trigger_reason="delay_elapsed",
        landed_in_basket=False,
        regrasp_step=95,
        seat_assisted=False,
        forbid_seat_assist=False,
        basket_place_ok=True,
    )
    kwargs.update(overrides)
    return evaluate_recovery_episode(**kwargs)


def test_genuine_recovery_is_accepted():
    success, checks = _good_episode()
    assert success
    assert checks["regrasped_after_drop"]
    assert checks["dropped_away_from_basket"]


def test_can_that_fell_into_the_basket_is_rejected():
    success, checks = _good_episode(landed_in_basket=True)
    assert not success
    assert checks["landed_in_basket_no_recovery_needed"]


def test_drop_next_to_the_basket_is_rejected():
    success, checks = _good_episode(drop_basket_xy_dist=0.05)
    assert not success
    assert not checks["dropped_away_from_basket"]


def test_missing_drop_distance_is_rejected():
    success, _ = _good_episode(drop_basket_xy_dist=None)
    assert not success


def test_place_without_a_regrasp_is_rejected():
    success, checks = _good_episode(regrasp_step=None)
    assert not success
    assert not checks["regrasped_after_drop"]


def test_seat_assist_rejected_only_when_forbidden():
    assert _good_episode(seat_assisted=True)[0]
    assert not _good_episode(seat_assisted=True, forbid_seat_assist=True)[0]


def test_can_never_reaching_the_basket_is_rejected():
    assert not _good_episode(basket_place_ok=False)[0]
