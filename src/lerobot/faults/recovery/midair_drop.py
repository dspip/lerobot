# Copyright 2026 Gangelia. All rights reserved.
"""Mid-air object drop fault with recovery planner handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.recovery.fps import resolve_target_fps
from lerobot.faults.sim.libero import (
    _body_xpos,
    force_close_gripper,
    force_open_gripper,
    get_arm_qpos,
    get_eef_pose,
    get_gripper_closing_axis,
    get_object_pose,
    get_place_destination,
    get_robosuite_env,
    hold_gripper_closed,
    is_object_grasped,
    is_object_in_basket,
    midair_drop,
    object_basket_xy_distance,
    object_pose_orientation,
    seat_object_in_basket_if_above,
)
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.recovery.planner import SimpleIKRecoveryPlanner


def _body_xpos_safe(rs_env: Any, name: str) -> np.ndarray | None:
    try:
        return _body_xpos(rs_env, name)
    except Exception:
        return None


@dataclass
class _EnvDropState:
    episode_step: int = 0
    triggered: bool = False
    recovery_active: bool = False
    drop_injection_step: bool = False
    planner: SimpleIKRecoveryPlanner | None = None
    last_recovery_action: np.ndarray | None = None
    episode_id: int | None = None
    finished: bool = False
    grasp_retries: int = 0
    place_retries: int = 0
    lost_grasp_steps: int = 0
    destination_pos: np.ndarray | None = None
    place_succeeded: bool = False
    seat_assisted: bool = False
    pending_seat: bool = False
    proof_object_pos: np.ndarray | None = None
    proof_basket_pos: np.ndarray | None = None
    # First step when grasp+height gates were satisfied (for post_grasp_delay).
    eligible_since: int | None = None
    # Drawn once when all trigger gates pass (honors config.probability).
    will_activate: bool | None = None
    episode_seed: int | None = None
    speed_multiplier: float | None = None
    waypoint_noise_m: float | None = None
    recovery_action_noise_std: float | None = None
    arm_posture_noise_rad: np.ndarray | None = None
    last_impulse_lin: np.ndarray | None = None
    last_impulse_ang: np.ndarray | None = None
    # Planar object→basket distance at the moment of the drop, and why it fired
    # ("delay_elapsed" or "basket_deadline").
    drop_basket_xy_dist: float | None = None
    drop_trigger_reason: str | None = None


def _episode_seed(config_seed: int | None, episode_id: int | None) -> int:
    base = 0 if config_seed is None else int(config_seed)
    ep = 0 if episode_id is None else int(episode_id)
    return base + ep * 10007


class MidAirDropFault:
    """Trigger a physical mid-air drop once, then execute recovery planner actions."""

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ) -> None:
        if config.type != "midair_drop":
            raise ValueError(f"MidAirDropFault requires type='midair_drop', got {config.type!r}.")
        config.validate(num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.event_logger = event_logger
        self._selected = set(range(num_envs)) if config.env_ids is None else set(config.env_ids)
        seed = 0 if config.seed is None else int(config.seed)
        self._rng = np.random.default_rng(seed)
        self._states = [_EnvDropState() for _ in range(num_envs)]

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(
        self,
        env_ids: list[int] | None = None,
        episode_ids: list[int] | dict[int, int] | None = None,
    ) -> None:
        """Clear episode-specific state for the given environments (or all)."""
        indices = range(self.num_envs) if env_ids is None else env_ids
        for i in indices:
            if i < 0 or i >= self.num_envs:
                raise ValueError(f"env_id {i} out of range for num_envs={self.num_envs}.")
            ep_id = None
            if isinstance(episode_ids, dict):
                ep_id = episode_ids.get(i)
            elif isinstance(episode_ids, list):
                index_list = list(indices)
                if len(episode_ids) == len(index_list):
                    ep_id = episode_ids[index_list.index(i)]
                elif i < len(episode_ids):
                    ep_id = episode_ids[i]
            self._states[i] = _EnvDropState(episode_id=ep_id)

    def notify_dones(self, dones: np.ndarray) -> None:
        """Clear recovery state for finished environments."""
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},), got {dones.shape}.")
        for i, done in enumerate(dones):
            if done:
                ep_id = self._states[i].episode_id
                self._states[i] = _EnvDropState(episode_id=ep_id, finished=True)

    def apply(
        self,
        actions: np.ndarray,
        episode_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Return actions to execute; recovery replays planner buffer without env access."""
        if not self.config.enabled:
            return actions

        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected actions with shape ({self.num_envs}, action_dim), got {actions.shape}."
            )

        executed = actions.copy()
        for env_idx in range(self.num_envs):
            if episode_ids is not None:
                self._states[env_idx].episode_id = episode_ids[env_idx]
            if env_idx not in self._selected or self._states[env_idx].finished:
                continue
            if self._states[env_idx].recovery_active:
                executed[env_idx] = self._next_recovery_action(env_idx)
        return executed

    def on_step(
        self,
        env: Any,
        actions: np.ndarray,
        episode_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Step hook with sim access for grasp checks, drop trigger, and recovery."""
        if not self.config.enabled:
            return actions

        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected actions with shape ({self.num_envs}, action_dim), got {actions.shape}."
            )

        executed = actions.copy()
        for env_idx in range(self.num_envs):
            if episode_ids is not None:
                self._states[env_idx].episode_id = episode_ids[env_idx]

            state = self._states[env_idx]
            state.drop_injection_step = False
            if env_idx not in self._selected or state.finished:
                continue

            if state.recovery_active:
                executed[env_idx] = self._next_recovery_action(env_idx, env=env)
                state.episode_step += 1
                continue

            if self._should_trigger(env, env_idx, state):
                proposed = actions[env_idx].copy()
                recovery_action = self._trigger_drop(env, env_idx, state, proposed_action=proposed)
                executed[env_idx] = recovery_action
                state.drop_injection_step = True
                state.episode_step += 1
                continue

            executed[env_idx] = actions[env_idx]
            state.episode_step += 1

        return executed

    def _should_trigger(self, env: Any, env_idx: int, state: _EnvDropState) -> bool:
        if state.triggered:
            return False
        if not (self.config.t_min <= state.episode_step <= self.config.t_max):
            return False
        rs_env = get_robosuite_env(env, env_idx=env_idx)
        if self.config.require_grasp and not is_object_grasped(rs_env, self.config.object_name):
            state.eligible_since = None
            return False
        object_pose = get_object_pose(rs_env, self.config.object_name)
        min_z = float(self.config.min_object_z)
        if min_z > 0.0 and float(object_pose["pos"][2]) < min_z:
            state.eligible_since = None
            return False
        if self.config.require_grasp:
            # Reject false grasps (touching soup while holding another object).
            eef_pos, _ = get_eef_pose(rs_env)
            if float(np.linalg.norm(eef_pos - object_pose["pos"])) > 0.08:
                state.eligible_since = None
                return False
        if state.eligible_since is None:
            state.eligible_since = state.episode_step

        basket_dist = object_basket_xy_distance(
            rs_env, self.config.object_name, basket_name=self.config.basket_name
        )
        min_dist = float(self.config.min_drop_distance_from_basket_m)
        at_deadline = min_dist > 0.0 and basket_dist is not None and basket_dist <= min_dist

        delay = int(self.config.post_grasp_delay_steps)
        if state.episode_step < state.eligible_since + delay and not at_deadline:
            return False
        if state.will_activate is None:
            state.will_activate = bool(self._rng.random() < self.config.probability)
        if not state.will_activate:
            return False
        # Record where/why only for the step that actually drops, so the logs
        # never describe a drop that did not happen.
        state.drop_basket_xy_dist = basket_dist
        state.drop_trigger_reason = "basket_deadline" if at_deadline else "delay_elapsed"
        return True

    def _trigger_drop(
        self,
        env: Any,
        env_idx: int,
        state: _EnvDropState,
        *,
        proposed_action: np.ndarray,
    ) -> np.ndarray:
        rs_env = get_robosuite_env(env, env_idx=env_idx)
        # Bias + Gaussian noise. Keep |v| modest so the can stays in-camera.
        bias = np.asarray(self.config.impulse_lin_bias, dtype=np.float64).reshape(3)
        lin_std = float(self.config.impulse_lin_std)
        ang_std = float(self.config.impulse_ang_std)
        lin_vel = bias + self._rng.normal(0.0, lin_std, size=3)
        # Always ensure a clear downward kick if bias Z was near zero.
        if lin_vel[2] > -0.15:
            lin_vel[2] = -0.45 - abs(float(self._rng.normal(0.0, 0.08)))
        # Soft clamp so extreme draws don't yeet the object out of view.
        lin_vel[:2] = np.clip(lin_vel[:2], -0.25, 0.25)
        lin_vel[2] = float(np.clip(lin_vel[2], -0.9, -0.15))
        ang_vel = self._rng.normal(0.0, ang_std, size=3)
        ang_vel = np.clip(ang_vel, -0.35, 0.35)
        state.last_impulse_lin = lin_vel.copy()
        state.last_impulse_ang = ang_vel.copy()
        telemetry = midair_drop(
            rs_env,
            self.config.object_name,
            lin_vel,
            ang_vel,
            settle_steps=self.config.settle_steps,
            gripper_settle_steps=self.config.gripper_settle_steps,
        )
        state.triggered = True
        state.recovery_active = True

        episode_seed = _episode_seed(self.config.seed, state.episode_id)
        ep_rng = np.random.default_rng(episode_seed)
        speed_multiplier = float(
            ep_rng.uniform(self.config.speed_multiplier_min, self.config.speed_multiplier_max)
        )
        arm_noise_rad = float(np.deg2rad(self.config.arm_posture_noise_deg))
        arm_posture_noise = ep_rng.uniform(-arm_noise_rad, arm_noise_rad, size=3)

        state.episode_seed = episode_seed
        state.speed_multiplier = speed_multiplier
        state.waypoint_noise_m = float(self.config.waypoint_noise_m)
        state.recovery_action_noise_std = float(self.config.recovery_action_noise_std)
        state.arm_posture_noise_rad = arm_posture_noise.copy()

        state.planner = SimpleIKRecoveryPlanner(
            fps=resolve_target_fps(self.config.recovery_fps),
            speed_multiplier=speed_multiplier,
            waypoint_noise_m=self.config.waypoint_noise_m,
            arm_posture_noise_rad=arm_posture_noise,
            side_grasp_enabled=self.config.side_grasp_enabled,
            seed=episode_seed,
        )
        eef_pos, eef_quat = get_eef_pose(rs_env)
        object_pose = get_object_pose(rs_env, self.config.object_name)
        if self.config.recovery_destination is not None:
            destination = np.array(self.config.recovery_destination, dtype=float)
        else:
            destination = get_place_destination(
                rs_env,
                self.config.object_name,
                basket_name=self.config.basket_name,
            )
        state.destination_pos = np.asarray(destination, dtype=np.float64).copy()
        state.grasp_retries = 0
        state.place_retries = 0
        state.lost_grasp_steps = 0
        state.planner.plan(
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            object_pos=object_pose["pos"],
            object_axis=self._object_long_axis(rs_env),
            destination_pos=destination,
            gripper_open=True,
        )
        recovery_action = self._next_recovery_action(env_idx, env=env)
        self._log_event(
            env_idx=env_idx,
            status="triggered",
            telemetry=telemetry,
            arm_q=get_arm_qpos(rs_env),
            proposed_action=proposed_action,
            executed_recovery_action=recovery_action,
            destination_pos=destination,
        )
        return recovery_action

    def _next_recovery_action(self, env_idx: int, *, env: Any | None = None) -> np.ndarray:
        state = self._states[env_idx]
        if state.planner is None:
            raise RuntimeError(f"MidAirDropFault env {env_idx}: recovery_active without planner.")

        eef_pos = None
        object_pos = None
        object_axis = None
        closing_axis = None
        rs_env = None
        if env is not None:
            rs_env = get_robosuite_env(env, env_idx=env_idx)
            eef_pos, _ = get_eef_pose(rs_env)
            object_pose = get_object_pose(rs_env, self.config.object_name)
            object_pos = object_pose["pos"]
            if self.config.side_grasp_enabled:
                object_axis = self._object_long_axis(rs_env)
                closing_axis = get_gripper_closing_axis(rs_env)

        action = state.planner.next_action(
            eef_pos=eef_pos,
            object_pos=object_pos,
            closing_axis=closing_axis,
            object_axis=object_axis,
        )

        # Snap gripper on phase transitions (Panda speed=0.01 never closes in time).
        if rs_env is not None:
            if state.planner.just_entered_close:
                force_close_gripper(rs_env, gripper_settle_steps=self.config.gripper_settle_steps)
            if state.planner.just_entered_open:
                force_open_gripper(rs_env, gripper_settle_steps=self.config.gripper_settle_steps)
                # Seat after the upcoming env.step physics (see after_physics_step).
                state.pending_seat = True

            phase = state.planner.phase_name
            # Refresh basket aim once when entering place phases (not every step — chasing
            # a basket we're brushing would never converge).
            if phase in ("to_basket_hover", "open_place") and state.planner._wp_steps <= 1:
                live_dest = get_place_destination(
                    rs_env,
                    self.config.object_name,
                    basket_name=self.config.basket_name,
                )
                state.destination_pos = np.asarray(live_dest, dtype=np.float64).copy()
                state.planner.retarget_basket(live_dest)

            carrying = phase in ("lift", "to_basket_hover")
            grasped_now = is_object_grasped(rs_env, self.config.object_name)
            obj_z_now = float(object_pos[2]) if object_pos is not None else 0.0

            if carrying and grasped_now:
                # Keep pads clamped — OSC steps otherwise let the speed-ramp loosen.
                hold_gripper_closed(rs_env)
                state.lost_grasp_steps = 0
                if action is not None:
                    action = action.copy()
                    action[6] = 1.0
                    # Slower lateral carry so the can doesn't slip out.
                    action[:3] = np.clip(action[:3] * 0.65, -1.0, 1.0)
            elif carrying and not grasped_now:
                # _check_grasp flickers during OSC; only replan after sustained loss
                # with the object back near the table.
                state.lost_grasp_steps += 1
                hold_gripper_closed(rs_env)
                if action is not None:
                    action = action.copy()
                    action[6] = 1.0
                truly_lost = state.lost_grasp_steps >= 8 and obj_z_now < 0.08
                if truly_lost and state.grasp_retries < 4:
                    state.grasp_retries += 1
                    state.lost_grasp_steps = 0
                    action = self._replan_pick(rs_env, state)

            # After release, if the can missed the basket, pick and place again.
            if (
                phase == "retract_done"
                and state.planner._wp_steps <= 1
                and state.place_retries < 2
                and not is_object_in_basket(
                    rs_env, self.config.object_name, basket_name=self.config.basket_name
                )
            ):
                state.place_retries += 1
                action = self._replan_pick(rs_env, state)

        if action is None:
            # Planner finished. Hold position with the gripper open instead of
            # replaying the last command: that command was a motion delta, so
            # repeating it keeps driving the arm and can shove the can it just
            # released out of the basket while the release physics settles.
            # Returned unjittered — exploration noise past the end of the plan
            # only disturbs the placed can.
            hold = np.zeros(7, dtype=np.float32)
            hold[6] = -1.0
            state.last_recovery_action = hold.copy()
            return hold

        action = self._apply_recovery_action_noise(action, state)
        state.last_recovery_action = action.copy()
        return action

    def _apply_recovery_action_noise(
        self,
        action: np.ndarray,
        state: _EnvDropState,
    ) -> np.ndarray:
        noise_std = float(self.config.recovery_action_noise_std)
        if noise_std <= 0:
            return action
        seed = state.episode_seed
        if seed is None:
            seed = _episode_seed(self.config.seed, state.episode_id)
        noise_rng = np.random.default_rng(seed + state.episode_step * 1009)
        noisy = action.copy()
        noisy[:6] += noise_rng.normal(0.0, noise_std, size=6).astype(np.float32)
        noisy[:6] = np.clip(noisy[:6], -1.0, 1.0)
        return noisy

    def after_physics_step(self, env: Any) -> None:
        """Seat into basket after Gym physics, then freeze recovery.

        Must run *after* ``env.step`` so the control cycle cannot eject a just-seated can.
        """
        for env_idx, state in enumerate(self._states):
            if not state.pending_seat or state.planner is None:
                continue
            state.pending_seat = False
            rs_env = get_robosuite_env(env, env_idx=env_idx)
            # Physics-only check first (before any assistive seat).
            in_basket = bool(
                is_object_in_basket(
                    rs_env,
                    self.config.object_name,
                    basket_name=self.config.basket_name,
                    xy_tol=0.07,
                    z_max=0.14,
                )
            )
            seated = False
            if not in_basket and self.config.seat_assist_enabled:
                # Assist only when already above the rim (no cross-table teleport).
                seated = seat_object_in_basket_if_above(
                    rs_env,
                    self.config.object_name,
                    basket_name=self.config.basket_name,
                )
                if seated:
                    state.seat_assisted = True
                    for _ in range(40):
                        rs_env.sim.step()
            in_basket = bool(
                is_object_in_basket(
                    rs_env,
                    self.config.object_name,
                    basket_name=self.config.basket_name,
                    xy_tol=0.07,
                    z_max=0.14,
                )
            )
            state.place_succeeded = in_basket
            state.proof_object_pos = get_object_pose(rs_env, self.config.object_name)[
                "pos"
            ].astype(float)
            state.proof_basket_pos = _body_xpos_safe(rs_env, self.config.basket_name)
            print(
                f"[drop_fault] place after physics: seated={seated} "
                f"seat_assisted={state.seat_assisted} in_basket={state.place_succeeded} "
                f"obj={state.proof_object_pos.tolist()}",
                flush=True,
            )
            # Freeze planner after the open/place attempt so the episode can end.
            state.planner._done = True
            state.planner._phase_name = "done"
            state.planner._wp_idx = len(state.planner._waypoints)

    def _object_long_axis(self, rs_env: Any) -> np.ndarray | None:
        """World-frame long axis of the carried object, or ``None``.

        Measured from the object's collision geometry, so it is correct for
        assets whose mesh is not elongated along local +Z.
        """
        if not self.config.side_grasp_enabled:
            return None
        try:
            return object_pose_orientation(rs_env, self.config.object_name)["axis"]
        except Exception:
            return None

    def _replan_pick(self, rs_env: Any, state: _EnvDropState) -> np.ndarray:
        """Open gripper and rebuild a pick→place plan from the current poses."""
        force_open_gripper(rs_env, gripper_settle_steps=self.config.gripper_settle_steps)
        eef_pos, eef_quat = get_eef_pose(rs_env)
        object_pose = get_object_pose(rs_env, self.config.object_name)
        dest = state.destination_pos
        if dest is None:
            dest = get_place_destination(
                rs_env,
                self.config.object_name,
                basket_name=self.config.basket_name,
            )
        assert state.planner is not None
        state.planner.replan_from(
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            object_pos=object_pose["pos"],
            object_axis=self._object_long_axis(rs_env),
            destination_pos=dest,
            gripper_open=True,
        )
        action = state.planner.next_action(
            eef_pos=eef_pos,
            object_pos=object_pose["pos"],
            closing_axis=get_gripper_closing_axis(rs_env)
            if self.config.side_grasp_enabled
            else None,
            object_axis=self._object_long_axis(rs_env),
        )
        if action is None:
            action = np.zeros(7, dtype=np.float32)
        return action

    def loss_mask_for_env(self, env_idx: int) -> float:
        """Return ``loss_mask`` for the step that just completed (0.0 on drop injection only)."""
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx={env_idx} out of range for num_envs={self.num_envs}.")
        if self._states[env_idx].drop_injection_step:
            return 0.0
        return 1.0

    def _log_event(
        self,
        *,
        env_idx: int,
        status: str,
        telemetry: dict[str, Any] | None,
        arm_q: np.ndarray | None,
        proposed_action: np.ndarray | None = None,
        executed_recovery_action: np.ndarray | None = None,
        destination_pos: np.ndarray | None = None,
    ) -> None:
        if self.event_logger is None:
            return
        state = self._states[env_idx]
        event: dict[str, Any] = {
            "event": "midair_drop",
            "status": status,
            "fault_type": self.config.type,
            "evaluation_episode_id": state.episode_id,
            "vector_env_id": env_idx,
            "episode_step": state.episode_step,
            "t_min": self.config.t_min,
            "t_max": self.config.t_max,
            "object_name": self.config.object_name,
            "require_grasp": self.config.require_grasp,
            "seed": self.config.seed,
            "recovery_active": state.recovery_active,
        }
        if proposed_action is not None:
            event["proposed_action"] = proposed_action.astype(float).tolist()
        if executed_recovery_action is not None:
            event["executed_recovery_action"] = executed_recovery_action.astype(float).tolist()
        if destination_pos is not None:
            event["recovery_destination"] = destination_pos.astype(float).tolist()
        if state.speed_multiplier is not None:
            event["speed_multiplier"] = state.speed_multiplier
        if state.waypoint_noise_m is not None:
            event["waypoint_noise_m"] = state.waypoint_noise_m
        if state.recovery_action_noise_std is not None:
            event["recovery_action_noise_std"] = state.recovery_action_noise_std
        if state.arm_posture_noise_rad is not None:
            event["arm_posture_noise_rad"] = state.arm_posture_noise_rad.astype(float).tolist()
        if state.episode_seed is not None:
            event["episode_seed"] = state.episode_seed
        if state.last_impulse_lin is not None:
            event["impulse_lin"] = state.last_impulse_lin.astype(float).tolist()
        if state.last_impulse_ang is not None:
            event["impulse_ang"] = state.last_impulse_ang.astype(float).tolist()
        event["post_grasp_delay_steps"] = int(self.config.post_grasp_delay_steps)
        event["min_drop_distance_from_basket_m"] = float(
            self.config.min_drop_distance_from_basket_m
        )
        if state.drop_basket_xy_dist is not None:
            event["drop_basket_xy_dist"] = float(state.drop_basket_xy_dist)
        if state.drop_trigger_reason is not None:
            event["drop_trigger_reason"] = state.drop_trigger_reason
        if telemetry is not None:
            event["object_pose"] = telemetry.get("object_pose_after", telemetry.get("object_pose_before"))
            event["arm_q"] = telemetry.get("arm_q", arm_q.tolist() if arm_q is not None else None)
            event["impulse"] = telemetry.get("impulse")
        self.event_logger.log(event)
