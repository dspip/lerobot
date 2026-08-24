"""Tests for libero_sim helpers (mocked, no MuJoCo)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.faults.recovery.loss_mask import loss_mask_for_step, loss_mask_from_fault
from lerobot.faults.sim.libero import (
    DEFAULT_GRIPPER_SETTLE_STEPS,
    force_open_gripper,
    get_place_destination,
    midair_drop,
)


def test_loss_mask_for_step_only_injection_frame_masked():
    assert loss_mask_for_step(9, t_fault=10) == 1.0
    assert loss_mask_for_step(10, t_fault=10) == 0.0
    assert loss_mask_for_step(11, t_fault=10) == 1.0
    assert loss_mask_for_step(12, t_fault=10, drop_duration=3) == 1.0


def test_loss_mask_from_fault_state():
    assert loss_mask_from_fault(triggered=False, drop_injection_step=False, recovery_active=False) == 1.0
    assert loss_mask_from_fault(triggered=True, drop_injection_step=True, recovery_active=True) == 0.0
    assert loss_mask_from_fault(triggered=True, drop_injection_step=False, recovery_active=True) == 1.0


def test_get_place_destination_prefers_basket():
    rs_env = MagicMock()
    basket = MagicMock()
    basket.root_body = "basket_1_main"
    rs_env.get_object.side_effect = lambda name: basket if name == "basket_1" else (_ for _ in ()).throw(KeyError(name))
    rs_env.sim.data.get_body_xpos.return_value = np.array([0.4, 0.2, 0.85], dtype=float)

    dest = get_place_destination(rs_env, "alphabet_soup_1")
    # Place target is raised above the rim so release doesn't shove the basket.
    np.testing.assert_allclose(dest, [0.4, 0.2, 0.95])


@patch("lerobot.faults.sim.libero.get_object_pose")
def test_get_place_destination_fallback_offset(mock_pose):
    rs_env = MagicMock()
    rs_env.get_object.side_effect = KeyError("missing")
    rs_env.sim.data.get_body_xpos.side_effect = ValueError("missing")
    rs_env.sim.model.body_name2id.side_effect = ValueError("missing")
    mock_pose.return_value = {"pos": np.array([0.1, 0.2, 0.9])}

    dest = get_place_destination(rs_env, "alphabet_soup_1")
    np.testing.assert_allclose(dest, [0.25, 0.35, 0.95])


def test_force_open_gripper_settles_before_return():
    rs_env = MagicMock()
    robot = rs_env.robots[0]
    robot.gripper.init_qpos = np.array([0.02, -0.02], dtype=float)
    robot.gripper.actuators = ["act_f1", "act_f2"]
    robot.gripper.current_action = np.array([0.0, 0.0], dtype=float)
    robot._ref_gripper_joint_pos_indexes = [10, 11]
    robot._ref_gripper_joint_vel_indexes = [10, 11]
    rs_env.sim.model.actuator_name2id.side_effect = lambda n: {"act_f1": 0, "act_f2": 1}[n]
    rs_env.sim.model.actuator_ctrlrange = np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float)
    rs_env.sim.data.ctrl = np.zeros(2, dtype=float)
    rs_env.sim.data.qpos = np.zeros(20, dtype=float)
    rs_env.sim.data.qvel = np.zeros(20, dtype=float)
    settle = 3

    force_open_gripper(rs_env, gripper_settle_steps=settle)

    # Raw physics only — must not call rs_env.step (desyncs Gym wrappers).
    assert rs_env.step.call_count == 0
    assert rs_env.sim.step.call_count == settle
    assert rs_env.sim.forward.call_count >= 1
    # Fully-open command written to ctrl (not a tiny incremental nudge).
    np.testing.assert_allclose(rs_env.sim.data.ctrl, [1.0, -1.0], atol=1e-6)


@patch("lerobot.faults.sim.libero._nudge_object_down")
@patch("lerobot.faults.sim.libero.apply_object_impulse")
@patch("lerobot.faults.sim.libero.force_open_gripper")
@patch("lerobot.faults.sim.libero.get_arm_qpos", return_value=np.zeros(7))
@patch("lerobot.faults.sim.libero.get_eef_pose", return_value=(np.zeros(3), np.ones(4)))
@patch("lerobot.faults.sim.libero.get_object_pose")
@patch("lerobot.faults.sim.libero.is_object_grasped")
def test_midair_drop_opens_then_impulses(
    mock_grasp, mock_pose, mock_eef, mock_arm, mock_open, mock_impulse, mock_nudge
):
    mock_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.ones(4)}
    # pre_grasped, still grasped after open → nudge path, post_grasped
    mock_grasp.side_effect = [True, True, False]
    rs_env = MagicMock()

    midair_drop(rs_env, gripper_settle_steps=7, settle_steps=5)

    assert mock_open.call_count == 2
    mock_nudge.assert_called_once()
    mock_impulse.assert_called_once()
    assert mock_impulse.call_args.kwargs["settle_steps"] == 80


def test_gripper_settle_default_at_least_five():
    assert DEFAULT_GRIPPER_SETTLE_STEPS >= 5
