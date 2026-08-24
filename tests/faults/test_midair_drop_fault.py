"""Unit tests for MidAirDropFault (no MuJoCo / LeRobot required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from lerobot.faults.wrappers import DropRecoveryEnvWrapper, maybe_wrap_env
from lerobot.faults.factory import make_midair_drop_fault


def _cfg(**kwargs) -> FaultInjectionConfig:
    defaults = dict(
        enabled=True,
        type="midair_drop",
        t_min=2,
        t_max=4,
        object_name="alphabet_soup_1",
        impulse_lin_std=0.1,
        impulse_ang_std=0.1,
        settle_steps=1,
        require_grasp=True,
        min_object_z=0.0,  # unit tests mock poses; height gate covered separately
        seed=42,
        env_ids=None,
        log_path=None,
    )
    defaults.update(kwargs)
    return FaultInjectionConfig(**defaults)


def _action(batch: int, dim: int, fill: float) -> np.ndarray:
    return np.full((batch, dim), fill, dtype=np.float32)


def _mock_rs_env(grasped: bool = True) -> MagicMock:
    rs_env = MagicMock()
    rs_env.objects = {"alphabet_soup_1": MagicMock()}
    rs_env.robots = [MagicMock()]
    rs_env.robots[0]._joint_positions = np.arange(7, dtype=float)
    rs_env.robots[0]._hand_pos = np.array([0.1, 0.2, 0.3], dtype=float)
    rs_env.robots[0]._hand_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    rs_env._check_grasp = MagicMock(return_value=grasped)
    return rs_env


@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_does_not_trigger_before_grasp_when_required(mock_grasped, mock_get_rs):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = False
    inj = MidAirDropFault(_cfg(t_min=0, t_max=10, require_grasp=True), num_envs=1)
    env = MagicMock()
    policy = _action(1, 7, 9.0)

    for _ in range(5):
        out = inj.on_step(env, policy.copy())
        np.testing.assert_array_equal(out, policy)

    mock_grasped.assert_called()
    assert not inj._states[0].triggered


@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_does_not_trigger_when_grasped_but_too_low(mock_grasped, mock_get_rs, mock_pose, mock_eef):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_pose.return_value = {"pos": np.array([0.0, 0.0, 0.04]), "quat_wxyz": np.ones(4)}
    mock_eef.return_value = (np.zeros(3), np.ones(4))
    inj = MidAirDropFault(
        _cfg(t_min=0, t_max=10, require_grasp=True, min_object_z=0.12),
        num_envs=1,
    )
    env = MagicMock()
    for _ in range(5):
        inj.on_step(env, _action(1, 7, 1.0))
    assert not inj._states[0].triggered


@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_does_not_trigger_when_soup_far_from_eef(mock_grasped, mock_get_rs, mock_pose, mock_eef):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_pose.return_value = {"pos": np.array([0.5, 0.0, 0.2]), "quat_wxyz": np.ones(4)}
    mock_eef.return_value = (np.zeros(3), np.ones(4))  # 0.5 m away
    inj = MidAirDropFault(
        _cfg(t_min=0, t_max=10, require_grasp=True, min_object_z=0.12),
        num_envs=1,
    )
    env = MagicMock()
    for _ in range(5):
        inj.on_step(env, _action(1, 7, 1.0))
    assert not inj._states[0].triggered


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_triggers_in_window_when_grasped(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])
    mock_drop.return_value = {
        "object_pose_after": {"pos": [0.0, 0.0, 0.0]},
        "arm_q": [0.0] * 7,
        "impulse": {"lin_vel": [0.1, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
    }

    inj = MidAirDropFault(_cfg(t_min=2, t_max=4, require_grasp=True), num_envs=1)
    env = MagicMock()
    for fill in (1.0, 2.0, 3.0):
        inj.on_step(env, _action(1, 7, fill))

    assert inj._states[0].triggered
    assert inj._states[0].recovery_active
    mock_drop.assert_called_once()
    mock_dest.assert_called_once()


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_after_trigger_returns_planner_actions_not_policy(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])
    mock_drop.return_value = {
        "object_pose_after": {"pos": [0.0, 0.0, 0.0]},
        "arm_q": [0.0] * 7,
        "impulse": {"lin_vel": [0.1, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
    }

    inj = MidAirDropFault(_cfg(t_min=1, t_max=1, require_grasp=False), num_envs=1)
    env = MagicMock()
    inj.on_step(env, _action(1, 7, 1.0))  # step 0: policy
    out = inj.on_step(env, _action(1, 7, 99.0))  # step 1: trigger + recovery

    assert inj._states[0].recovery_active
    assert out[0, 6] in (-1.0, 1.0)
    assert not np.allclose(out, [[99.0] * 7])
    assert inj.loss_mask_for_env(0) == 0.0


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_logs_trigger_event(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
    tmp_path: Path,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_arm_q.return_value = np.ones(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])
    mock_drop.return_value = {
        "object_pose_after": {"pos": [0.1, 0.2, 0.3]},
        "arm_q": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "impulse": {"lin_vel": [0.1, 0.0, 0.0], "ang_vel": [0.0, 0.1, 0.0]},
    }

    log_path = tmp_path / "faults.jsonl"
    logger = FaultEventLogger(log_path)
    inj = MidAirDropFault(
        _cfg(t_min=0, t_max=0, require_grasp=False),
        num_envs=1,
        event_logger=logger,
    )
    env = MagicMock()
    proposed = _action(1, 7, 1.0)
    inj.on_step(env, proposed, episode_ids=[5])
    logger.close()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "midair_drop"
    assert event["status"] == "triggered"
    assert event["evaluation_episode_id"] == 5
    assert event["object_pose"] == {"pos": [0.1, 0.2, 0.3]}
    assert event["arm_q"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert event["impulse"]["lin_vel"] == [0.1, 0.0, 0.0]
    assert event["proposed_action"] == proposed[0].astype(float).tolist()
    assert len(event["executed_recovery_action"]) == 7
    assert event["recovery_destination"] == [0.3, 0.2, 0.9]


def test_disabled_is_noop():
    cfg = _cfg(enabled=False)
    assert make_midair_drop_fault(cfg, num_envs=1) is None
    inj = MidAirDropFault(cfg, num_envs=1)
    proposed = _action(1, 7, 5.0)
    assert inj.apply(proposed) is proposed


def test_maybe_wrap_env_uses_drop_wrapper():
    import gymnasium as gym
    from gymnasium import spaces

    class _Env(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(7,))

        def reset(self, *, seed=None, options=None):
            return np.zeros(2), {}

        def step(self, action):
            return np.zeros(2), 0.0, False, False, {}

    wrapped = maybe_wrap_env(_Env(), _cfg())
    assert isinstance(wrapped, DropRecoveryEnvWrapper)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"t_min": -1}, "t_min"),
        ({"t_max": 1, "t_min": 5}, "t_max"),
        ({"impulse_lin_std": -0.1}, "impulse_lin_std"),
        ({"object_name": ""}, "object_name"),
    ],
)
def test_invalid_config_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FaultInjectionConfig(enabled=True, type="midair_drop", **kwargs)


def test_notify_dones_clears_recovery_state():
    inj = MidAirDropFault(_cfg(), num_envs=1)
    inj._states[0].recovery_active = True
    inj._states[0].triggered = True
    inj.notify_dones(np.array([True]))
    assert inj._states[0].finished
    assert not inj._states[0].recovery_active


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_drop_injection_step_loss_mask(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])
    mock_drop.return_value = {
        "object_pose_after": {"pos": [0.0, 0.0, 0.0]},
        "arm_q": [0.0] * 7,
        "impulse": {"lin_vel": [0.1, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
    }

    inj = MidAirDropFault(_cfg(t_min=0, t_max=0, require_grasp=False), num_envs=1)
    inj.on_step(MagicMock(), _action(1, 7, 1.0))
    assert inj.loss_mask_for_env(0) == 0.0
    inj.on_step(MagicMock(), _action(1, 7, 2.0))
    assert inj.loss_mask_for_env(0) == 1.0


@patch("lerobot.faults.recovery.midair_drop._body_xpos_safe", return_value=np.zeros(3))
@patch("lerobot.faults.recovery.midair_drop.is_object_in_basket")
@patch("lerobot.faults.recovery.midair_drop.seat_object_in_basket_if_above", return_value=True)
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_after_physics_step_seats_when_pending(
    mock_get_rs, mock_pose, mock_seat, mock_in_basket, mock_basket_pos
):
    # First check: not in basket → assist seat; second check: in basket.
    mock_in_basket.side_effect = [False, True]
    mock_get_rs.return_value = _mock_rs_env()
    mock_pose.return_value = {"pos": np.array([0.1, 0.2, 0.05]), "quat_wxyz": np.ones(4)}
    inj = MidAirDropFault(_cfg(), num_envs=1)
    inj._states[0].pending_seat = True
    inj._states[0].planner = MagicMock()
    inj._states[0].planner._done = False
    inj._states[0].planner._phase_name = "open_place"
    inj._states[0].planner._wp_idx = 0
    inj._states[0].planner._waypoints = [np.zeros(3)]

    inj.after_physics_step(MagicMock())

    assert not inj._states[0].pending_seat
    mock_seat.assert_called()
    assert inj._states[0].seat_assisted is True
    assert inj._states[0].place_succeeded
    assert inj._states[0].proof_object_pos is not None
    assert inj._states[0].planner._done is True


def test_finished_planner_holds_still_with_gripper_open():
    """A finished planner must stop commanding motion.

    Replaying the last recovery action re-applies a motion *delta* every step,
    which keeps driving the arm after the release and can shove the just-placed
    can back out of the basket while the physics settles.
    """
    inj = MidAirDropFault(_cfg(), num_envs=1)
    state = inj._states[0]
    state.recovery_active = True
    state.planner = MagicMock()
    state.planner.next_action.return_value = None
    state.planner.just_entered_close = False
    state.planner.just_entered_open = False
    state.last_recovery_action = np.array([0.9, -0.8, 0.7, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    action = inj._next_recovery_action(0)

    assert np.allclose(action[:6], 0.0), f"expected no motion command, got {action[:6]}"
    assert action[6] == pytest.approx(-1.0), "gripper must stay open after release"


@patch("lerobot.faults.recovery.midair_drop._body_xpos_safe", return_value=np.zeros(3))
@patch("lerobot.faults.recovery.midair_drop.is_object_in_basket")
@patch("lerobot.faults.recovery.midair_drop.seat_object_in_basket_if_above", return_value=True)
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
def test_after_physics_step_skips_seat_when_assist_disabled(
    mock_get_rs, mock_pose, mock_seat, mock_in_basket, mock_basket_pos
):
    """seat_assist_enabled=False must leave the miss a miss, not a seated success."""
    mock_in_basket.return_value = False
    mock_get_rs.return_value = _mock_rs_env()
    mock_pose.return_value = {"pos": np.array([0.1, 0.2, 0.05]), "quat_wxyz": np.ones(4)}
    inj = MidAirDropFault(_cfg(seat_assist_enabled=False), num_envs=1)
    inj._states[0].pending_seat = True
    inj._states[0].planner = MagicMock()
    inj._states[0].planner._done = False
    inj._states[0].planner._phase_name = "open_place"
    inj._states[0].planner._wp_idx = 0
    inj._states[0].planner._waypoints = [np.zeros(3)]

    inj.after_physics_step(MagicMock())

    mock_seat.assert_not_called()
    assert inj._states[0].seat_assisted is False
    assert not inj._states[0].place_succeeded


@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_probability_zero_skips_trigger(mock_grasped, mock_get_rs, mock_pose, mock_eef):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_pose.return_value = {"pos": np.array([0.0, 0.0, 0.2]), "quat_wxyz": np.ones(4)}
    mock_eef.return_value = (np.array([0.0, 0.0, 0.2]), np.ones(4))
    inj = MidAirDropFault(
        _cfg(t_min=0, t_max=10, require_grasp=True, probability=0.0, seed=0),
        num_envs=1,
    )
    env = MagicMock()
    for _ in range(5):
        inj.on_step(env, _action(1, 7, 1.0))
    assert not inj._states[0].triggered
    assert inj._states[0].will_activate is False


@patch("lerobot.faults.recovery.midair_drop.get_place_destination")
@patch("lerobot.faults.recovery.midair_drop.midair_drop")
@patch("lerobot.faults.recovery.midair_drop.get_object_pose")
@patch("lerobot.faults.recovery.midair_drop.get_eef_pose")
@patch("lerobot.faults.recovery.midair_drop.get_arm_qpos")
@patch("lerobot.faults.recovery.midair_drop.get_robosuite_env")
@patch("lerobot.faults.recovery.midair_drop.is_object_grasped")
def test_probability_one_triggers_when_eligible(
    mock_grasped,
    mock_get_rs,
    mock_arm_q,
    mock_eef,
    mock_obj_pose,
    mock_drop,
    mock_dest,
):
    mock_get_rs.return_value = _mock_rs_env()
    mock_grasped.return_value = True
    mock_arm_q.return_value = np.zeros(7)
    mock_eef.return_value = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    mock_obj_pose.return_value = {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}
    mock_dest.return_value = np.array([0.3, 0.2, 0.9])
    mock_drop.return_value = {
        "object_pose_after": {"pos": [0.0, 0.0, 0.0]},
        "arm_q": [0.0] * 7,
        "impulse": {"lin_vel": [0.1, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
    }

    inj = MidAirDropFault(
        _cfg(t_min=2, t_max=4, require_grasp=True, probability=1.0),
        num_envs=1,
    )
    env = MagicMock()
    for fill in (1.0, 2.0, 3.0):
        inj.on_step(env, _action(1, 7, fill))

    assert inj._states[0].triggered
    assert inj._states[0].will_activate is True
