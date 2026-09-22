"""Unit tests for MidAirDropFault.request_recovery (mocks only)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from tests.faults.test_midair_drop_fault import _cfg, _mock_rs_env


def _action(batch: int, dim: int, fill: float) -> np.ndarray:
    return np.full((batch, dim), fill, dtype=np.float32)


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_request_recovery_starts_planner_without_impulse(
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])

    inj = MidAirDropFault(_cfg(), num_envs=1)
    env = MagicMock()
    first = inj.request_recovery(env, 0, reason="head")

    assert first is not None
    assert inj._states[0].recovery_active
    assert inj._states[0].planner is not None
    assert not inj._states[0].triggered
    assert not inj._states[0].drop_injection_step
    mock_drop.assert_not_called()
    mock_dest.assert_called_once()


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_on_step_after_request_returns_planner_not_policy(
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_dest,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])

    inj = MidAirDropFault(_cfg(t_min=100, t_max=200), num_envs=1)
    env = MagicMock()
    inj.request_recovery(env, 0, reason="head")
    out = inj.on_step(env, _action(1, 7, 99.0))

    assert inj._states[0].recovery_active
    assert out[0, 6] in (-1.0, 1.0)
    assert not np.allclose(out, [[99.0] * 7])
    assert inj.loss_mask_for_env(0) == 1.0


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_request_while_already_recovering_does_not_reset_planner(
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_dest,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])

    inj = MidAirDropFault(_cfg(), num_envs=1)
    env = MagicMock()
    inj.request_recovery(env, 0, reason="head")
    planner = inj._states[0].planner
    assert planner is not None

    inj.request_recovery(env, 0, reason="head")
    assert inj._states[0].planner is planner
    mock_dest.assert_called_once()


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_request_recovery_logs_reason(
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_dest,
    tmp_path: Path,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])

    log_path = tmp_path / "faults.jsonl"
    logger = FaultEventLogger(log_path)
    inj = MidAirDropFault(_cfg(), num_envs=1, event_logger=logger)
    inj.request_recovery(MagicMock(), 0, reason="head")
    logger.close()

    event = json.loads(log_path.read_text().strip())
    assert event["event"] == "midair_drop"
    assert event["status"] == "recovery_requested"
    assert event["drop_trigger_reason"] == "head"
    assert "impulse" not in event


def test_request_recovery_disabled_returns_none():
    inj = MidAirDropFault(_cfg(enabled=False), num_envs=1)
    assert inj.request_recovery(MagicMock(), 0) is None
    assert not inj._states[0].recovery_active


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_request_recovery_can_leave_first_action_for_wrapper(
    mock_get_rs, mock_arm_q, mock_eef, mock_obj_pose, mock_dest
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])
    inj = MidAirDropFault(_cfg(), num_envs=1)
    env = MagicMock()

    assert inj.request_recovery(env, 0, consume_first_action=False) is None
    action = inj.on_step(env, _action(1, 7, 99.0))

    assert inj._states[0].recovery_active
    assert not np.allclose(action, [[99.0] * 7])


@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_manual_drop_uses_fault_lifecycle_without_starting_recovery(
    mock_get_rs, mock_arm_q, mock_drop, tmp_path: Path
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_arm_q.return_value = np.zeros(7)
    mock_drop.return_value = {"object_pose_after": {"pos": [0.0, 0.0, 0.0]}}
    logger = FaultEventLogger(tmp_path / "faults.jsonl")
    inj = MidAirDropFault(_cfg(), num_envs=1, event_logger=logger)

    assert inj.trigger_manual_drop(MagicMock(), 0)
    logger.close()

    state = inj._states[0]
    assert state.triggered
    assert not state.recovery_active
    assert state.last_impulse_lin is not None
    mock_drop.assert_called_once()
    event = json.loads((tmp_path / "faults.jsonl").read_text())
    assert event["status"] == "manual_triggered"
    assert event["drop_trigger_reason"] == "manual"
