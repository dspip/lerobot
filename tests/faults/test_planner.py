"""Unit tests for SimpleIKRecoveryPlanner."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.faults.recovery.planner import BLENDABLE_PHASES, SimpleIKRecoveryPlanner


def _default_poses():
    eef_pos = np.array([0.0, 0.0, 1.0])
    eef_quat = np.array([0.0, 0.0, 0.0, 1.0])
    object_pos = np.array([0.1, 0.0, 0.9])
    destination_pos = np.array([0.3, 0.2, 0.9])
    return eef_pos, eef_quat, object_pos, destination_pos


def _kw(poses):
    """``_default_poses()`` tuple as keyword args for the keyword-only ``plan``."""
    eef_pos, eef_quat, object_pos, destination_pos = poses
    return {
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "object_pos": object_pos,
        "destination_pos": destination_pos,
    }


def test_plan_returns_actions():
    planner = SimpleIKRecoveryPlanner(fps=10, seed=0)
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    actions = planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
    )
    assert actions.ndim == 2
    assert actions.shape[1] == 7
    assert len(actions) > 0


def test_actions_in_valid_range():
    planner = SimpleIKRecoveryPlanner(fps=10, seed=1)
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    actions = planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=False,
    )
    assert np.all(actions[:, :6] >= -1.0)
    assert np.all(actions[:, :6] <= 1.0)
    assert np.all(np.isin(actions[:, 6], [-1.0, 1.0]))


def test_gripper_phases_progress():
    planner = SimpleIKRecoveryPlanner(fps=10, speed_multiplier=2.0, seed=2)
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    actions = planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
    )
    gripper = actions[:, 6]
    assert gripper[0] == -1.0  # starts open
    assert np.any(gripper == 1.0)  # closes during grasp
    assert gripper[-1] == -1.0  # ends open after place


def test_next_action_iterator():
    planner = SimpleIKRecoveryPlanner(fps=10, seed=3)
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    plan = planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
    )
    replay = []
    while True:
        a = planner.next_action()
        if a is None:
            break
        replay.append(a)
    assert len(replay) == len(plan)
    np.testing.assert_allclose(np.stack(replay), plan)


def test_reset_clears_iterator():
    planner = SimpleIKRecoveryPlanner(fps=10)
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
    )
    planner.next_action()
    planner.reset()
    assert planner.next_action() is None


def test_z_motion_includes_lift_and_descend():
    planner = SimpleIKRecoveryPlanner(fps=10, speed_multiplier=3.0, seed=4)
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    actions = planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
    )
    dz = actions[:, 2] * planner.max_pos_step
    assert np.any(dz > 0.01)  # lift segments
    assert np.any(dz < -0.01)  # descend segments


def test_waypoint_noise_changes_plan():
    base = SimpleIKRecoveryPlanner(fps=10, waypoint_noise_m=0.0, seed=5)
    noisy = SimpleIKRecoveryPlanner(fps=10, waypoint_noise_m=0.02, seed=5)
    poses = _default_poses()
    a0 = base.plan(**_kw(poses), gripper_open=True)
    a1 = noisy.plan(**_kw(poses), gripper_open=True)
    n = min(len(a0), len(a1))
    assert n > 0
    assert not np.allclose(a0[:n, :3], a1[:n, :3])


def test_closed_loop_tracks_eef_and_closes():
    planner = SimpleIKRecoveryPlanner(
        fps=10,
        arrive_tol=0.03,
        basket_xy_tol=0.03,
        grasp_hold_steps=2,
        place_hold_steps=2,
        max_steps_per_waypoint=80,
        seed=6,
    )
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
    )

    # Simulate perfect tracking; after grasp, object rides with the EEF so
    # basket-place phases (object-centric XY) can converge.
    pos = eef_pos.astype(np.float64).copy()
    obj = obj_pos.astype(np.float64).copy()
    grasp_offset = pos - obj
    carrying = False
    saw_close = False
    saw_enter_close = False
    for _ in range(800):
        action = planner.next_action(eef_pos=pos, object_pos=obj)
        if action is None:
            break
        if planner.just_entered_close:
            saw_enter_close = True
            carrying = True
            grasp_offset = pos - obj
        if action[6] > 0:
            saw_close = True
        if planner.just_entered_open:
            carrying = False
        pos = pos + action[:3].astype(np.float64) * planner.max_pos_step
        if carrying:
            obj = pos - grasp_offset

    assert saw_close
    assert saw_enter_close
    assert planner.done or action is None


def _rot_commands(planner, poses, *, n_steps=400, object_axis=None):
    """Rotation commands emitted over a closed-loop rollout with a static pose."""
    eef_pos, eef_quat, obj_pos, dest = poses
    planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
        object_axis=object_axis,
    )
    cmds = []
    for _ in range(n_steps):
        action = planner.next_action(
            eef_pos=eef_pos,
            object_pos=obj_pos,
            closing_axis=np.array([1.0, 0.0, 0.0]),
            object_axis=object_axis,
        )
        if action is None:
            break
        cmds.append(np.asarray(action[3:6], dtype=np.float64).copy())
    return np.asarray(cmds)


def test_posture_bias_is_spent_once_not_every_step():
    """OSC reads action[3:6] as a per-step delta, so a constant bias would wind
    the wrist without bound. Only the first step may carry the offset."""
    bias = np.array([-0.008, -0.031, -0.0026])
    planner = SimpleIKRecoveryPlanner(fps=10, seed=0, arm_posture_noise_rad=bias)
    cmds = _rot_commands(planner, _default_poses())

    assert len(cmds) > 50, "need a long enough rollout to expose drift"
    nonzero = np.flatnonzero(np.any(np.abs(cmds) > 1e-9, axis=1))
    assert nonzero.tolist() == [0], f"bias must be emitted only on step 0, got {nonzero[:10]}"


def test_posture_bias_total_rotation_equals_sampled_offset():
    """The one-shot offset must still deliver the sampled posture variation."""
    bias = np.array([0.02, -0.05, 0.01])
    planner = SimpleIKRecoveryPlanner(fps=10, seed=0, arm_posture_noise_rad=bias)
    cmds = _rot_commands(planner, _default_poses())

    commanded_rad = cmds.sum(axis=0) * planner.max_rot_step
    np.testing.assert_allclose(commanded_rad, bias, atol=1e-9)


def test_posture_bias_does_not_accumulate_over_long_recovery():
    """Regression: a 3 deg bias used to integrate to hundreds of degrees."""
    bias = np.deg2rad(np.array([3.0, -3.0, 3.0]))
    planner = SimpleIKRecoveryPlanner(fps=10, seed=0, arm_posture_noise_rad=bias)
    cmds = _rot_commands(planner, _default_poses(), n_steps=600)

    total_deg = np.rad2deg(np.abs(cmds.sum(axis=0) * planner.max_rot_step))
    assert np.all(total_deg <= 3.0 + 1e-6), f"wrist drifted {total_deg} deg"


def test_posture_bias_rearmed_by_replanning():
    """Each fresh recovery gets its posture offset once."""
    bias = np.array([0.02, -0.05, 0.01])
    planner = SimpleIKRecoveryPlanner(fps=10, seed=0, arm_posture_noise_rad=bias)
    first = _rot_commands(planner, _default_poses(), n_steps=5)
    second = _rot_commands(planner, _default_poses(), n_steps=5)
    np.testing.assert_allclose(first[0], second[0], atol=1e-9)
    assert np.allclose(first[1:], 0.0) and np.allclose(second[1:], 0.0)


def test_no_posture_bias_means_no_rotation_command():
    planner = SimpleIKRecoveryPlanner(fps=10, seed=0, arm_posture_noise_rad=None)
    cmds = _rot_commands(planner, _default_poses())
    assert np.allclose(cmds, 0.0)


def test_shaped_plan_inserts_pickup_and_transport_via_points():
    planner = SimpleIKRecoveryPlanner(
        pickup_via_offset_xy_m=(0.02, -0.01),
        transport_via_offset_m=0.06,
        basket_keepout_m=0.05,
        seed=7,
    )
    poses = _default_poses()
    planner.plan(**_kw(poses), gripper_open=True)

    names = [wp.name for wp in planner._waypoints]
    assert names.index("pickup_via") < names.index("approach_hover")
    assert names.index("lift") < names.index("to_basket_via")
    assert names.index("to_basket_via") < names.index("to_basket_hover")


def test_zero_offsets_preserve_existing_phase_sequence():
    planner = SimpleIKRecoveryPlanner(
        pickup_via_offset_xy_m=(0.0, 0.0),
        transport_via_offset_m=0.0,
        seed=7,
    )
    poses = _default_poses()
    planner.plan(**_kw(poses), gripper_open=True)

    names = [wp.name for wp in planner._waypoints]
    assert "pickup_via" not in names
    assert "to_basket_via" not in names


def test_carry_path_is_refreshed_from_live_object_at_lift_entry():
    planner = SimpleIKRecoveryPlanner(
        grasp_hold_steps=1,
        transport_via_offset_m=0.06,
        basket_keepout_m=0.05,
        seed=8,
    )
    eef_pos, eef_quat, obj_pos, dest = _default_poses()
    planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=obj_pos,
        destination_pos=dest,
        gripper_open=True,
    )
    moved = obj_pos + np.array([0.03, -0.02, 0.0])
    while planner.phase_name != "lift":
        target = planner._current_target(moved, eef_pos)
        eef_pos = target.pos.copy()
        planner.next_action(eef_pos=eef_pos, object_pos=moved)

    assert planner.carry_path is not None
    np.testing.assert_allclose(planner.carry_path.segments[0].start_xyz, moved)


def _shaped_planner(**overrides):
    """Planner with both via-points present so every transit leg is exercised."""
    kwargs = {
        "fps": 10,
        "pickup_via_offset_xy_m": (0.02, -0.01),
        "transport_via_offset_m": 0.06,
        "basket_keepout_m": 0.05,
        "grasp_hold_steps": 4,
        "place_hold_steps": 4,
        "seed": 11,
    }
    kwargs.update(overrides)
    return SimpleIKRecoveryPlanner(**kwargs)


def _phase_command_trace(planner, poses, *, n_steps=1200):
    """Per-step ``(phase, translation magnitude)`` under perfect EEF tracking."""
    eef_pos, _, obj_pos, _ = poses
    planner.plan(**_kw(poses), gripper_open=True)

    pos = eef_pos.astype(np.float64).copy()
    obj = obj_pos.astype(np.float64).copy()
    grasp_offset = pos - obj
    carrying = False
    trace: list[tuple[str, float]] = []
    for _ in range(n_steps):
        action = planner.next_action(eef_pos=pos, object_pos=obj)
        if action is None:
            break
        if planner.just_entered_close:
            carrying = True
            grasp_offset = pos - obj
        if planner.just_entered_open:
            carrying = False
        trace.append((planner.phase_name, float(np.linalg.norm(action[:3]))))
        pos = pos + action[:3].astype(np.float64) * planner.max_pos_step
        if carrying:
            obj = pos - grasp_offset
    return trace


def test_transit_waypoints_dwell_without_blending():
    trace = _phase_command_trace(
        _shaped_planner(waypoint_blend_radius_m=0.0),
        _default_poses(),
    )

    dwell = [phase for phase, speed in trace if phase in BLENDABLE_PHASES and speed < 1e-6]

    assert dwell, "unblended planner is expected to stop on free-space via points"


def test_blending_removes_transit_dwell_steps():
    trace = _phase_command_trace(
        _shaped_planner(waypoint_blend_radius_m=0.04),
        _default_poses(),
    )

    dwell = [phase for phase, speed in trace if phase in BLENDABLE_PHASES and speed < 1e-6]

    assert dwell == []


def test_blending_keeps_transit_command_off_the_creep_floor():
    trace = _phase_command_trace(
        _shaped_planner(waypoint_blend_radius_m=0.04),
        _default_poses(),
    )

    speeds = [speed for phase, speed in trace if phase in BLENDABLE_PHASES]

    assert speeds
    assert min(speeds) > 0.3


def test_blending_passes_through_every_transit_phase():
    trace = _phase_command_trace(
        _shaped_planner(waypoint_blend_radius_m=0.04),
        _default_poses(),
    )

    assert BLENDABLE_PHASES <= {phase for phase, _ in trace}


def test_blending_preserves_exact_settling_at_grasp_and_release():
    trace = _phase_command_trace(
        _shaped_planner(waypoint_blend_radius_m=0.04),
        _default_poses(),
    )

    for phase in ("approach_hover", "descend_grasp", "close_grasp", "to_basket_hover"):
        settled = [speed for name, speed in trace if name == phase and speed < 1e-6]
        assert settled, f"{phase} must still settle exactly before advancing"


def test_blending_preserves_grasp_and_place_waypoint_geometry():
    poses = _default_poses()
    exact = _shaped_planner(waypoint_blend_radius_m=0.0)
    blended = _shaped_planner(waypoint_blend_radius_m=0.04)
    exact.plan(**_kw(poses), gripper_open=True)
    blended.plan(**_kw(poses), gripper_open=True)

    exact_targets = {wp.name: wp.pos.copy() for wp in exact._waypoints}
    blended_targets = {wp.name: wp.pos.copy() for wp in blended._waypoints}

    assert exact_targets.keys() == blended_targets.keys()
    for name, target in exact_targets.items():
        np.testing.assert_allclose(blended_targets[name], target)


def test_blend_radius_is_capped_by_the_length_of_the_leg():
    """A radius wider than a short leg would delete that leg's geometry."""
    poses = _default_poses()
    planner = _shaped_planner(waypoint_blend_radius_m=1.0)
    planner.plan(**_kw(poses), gripper_open=True)
    retract = next(wp for wp in planner._waypoints if wp.name == "retract")
    leg_len = float(np.linalg.norm(retract.pos - poses[0]))

    trace = _phase_command_trace(_shaped_planner(waypoint_blend_radius_m=1.0), poses)

    assert [phase for phase, _ in trace if phase == "retract"]
    assert planner._blend_radius_for(retract) <= 0.5 * leg_len


def test_negative_blend_radius_is_rejected():
    with pytest.raises(ValueError, match="waypoint_blend_radius_m"):
        SimpleIKRecoveryPlanner(waypoint_blend_radius_m=-0.01)
