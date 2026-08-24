# Copyright 2026 Gangelia. All rights reserved.
"""Cartesian recovery planner emitting 7D relative OSC actions for SmolVLA datasets.

Closed-loop: each ``next_action`` aims at the current waypoint from the *actual*
EEF pose (open-loop delta chains drift and miss grasps in LIBERO OSC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.faults.recovery.fps import SMOLVLA_LIBERO_TARGET_FPS

# Phases where rotating the wrist is safe (before the fingers hold anything).
_YAW_PHASES = ("retract", "approach_hover", "descend_grasp")
# Phases that must not be left with the fingers crossed over a tipped can.
_YAW_GATED_PHASES = ("approach_hover", "descend_grasp")
# |axis·z| above this means the can is standing; a cylinder is then yaw-invariant.
_UPRIGHT_DOT = 0.6


def _wrap_to_half_pi(angle: float) -> float:
    """Wrap to ``[-pi/2, pi/2]`` — a parallel gripper is symmetric under 180°."""
    return float((angle + np.pi / 2) % np.pi - np.pi / 2)


@dataclass
class _Waypoint:
    pos: np.ndarray
    gripper_open: bool
    hold_steps: int = 1
    track_object: bool = False  # refresh XY from live object pose while approaching
    name: str = ""


class SimpleIKRecoveryPlanner:
    """Plan a pick-place recovery trajectory as normalized 7D delta-OSC actions."""

    _WORKSPACE_XY_LIMIT_M = 0.7

    def __init__(
        self,
        *,
        fps: int = SMOLVLA_LIBERO_TARGET_FPS,
        max_pos_step: float = 0.05,
        max_rot_step: float = 0.5,
        speed_multiplier: float = 1.0,
        # Basket rim tops out ~0.15 m; carry well above so the can clears it.
        lift_height: float = 0.22,
        hover_offset: float = 0.14,
        # EEF site must sit ~2–3 cm above can center; 5 cm closes in air (no grasp).
        grasp_offset: float = 0.025,
        # A tipped-over can has its center about one radius off the table, so the
        # fingers must come down lower than for an upright can.
        grasp_offset_lying: float = 0.012,
        side_grasp_enabled: bool = True,
        # Rotate the wrist at most this fraction of max_rot_step per step.
        yaw_gain: float = 1.0,
        yaw_tol_rad: float = 0.15,
        yaw_stall_multiplier: float = 3.0,
        waypoint_noise_m: float = 0.0,
        arm_posture_noise_rad: np.ndarray | float | None = None,
        arrive_tol: float = 0.02,
        basket_xy_tol: float = 0.012,
        grasp_hold_steps: int = 20,
        place_hold_steps: int = 14,
        max_steps_per_waypoint: int = 100,
        seed: int | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if max_pos_step <= 0 or max_rot_step <= 0:
            raise ValueError("max_pos_step and max_rot_step must be positive.")
        if speed_multiplier <= 0:
            raise ValueError(f"speed_multiplier must be positive, got {speed_multiplier}")

        self.fps = int(fps)
        self.max_pos_step = float(max_pos_step)
        self.max_rot_step = float(max_rot_step)
        self.speed_multiplier = float(speed_multiplier)
        self.lift_height = float(lift_height)
        self.hover_offset = float(hover_offset)
        self.grasp_offset = float(grasp_offset)
        self.grasp_offset_lying = float(grasp_offset_lying)
        self.side_grasp_enabled = bool(side_grasp_enabled)
        self.yaw_gain = float(yaw_gain)
        self.yaw_tol_rad = float(yaw_tol_rad)
        self.yaw_stall_multiplier = float(yaw_stall_multiplier)
        self.yaw_stall_advances = 0
        self.waypoint_noise_m = float(waypoint_noise_m)
        self.arrive_tol = float(arrive_tol)
        self.basket_xy_tol = float(basket_xy_tol)
        self.grasp_hold_steps = int(grasp_hold_steps)
        self.place_hold_steps = int(place_hold_steps)
        self.max_steps_per_waypoint = int(max_steps_per_waypoint)
        self._rng = np.random.default_rng(seed)
        self._arm_posture_bias = self._parse_arm_posture_bias(arm_posture_noise_rad)
        # Rotation (rad) of the posture bias not yet commanded; see _apply_rot_bias.
        self._rot_bias_remaining = (
            None if self._arm_posture_bias is None else self._arm_posture_bias.copy()
        )

        self._waypoints: list[_Waypoint] = []
        self._wp_idx = 0
        self._hold_left = 0
        self._wp_steps = 0
        self._object_pos = np.zeros(3, dtype=np.float64)
        self._destination_pos = np.zeros(3, dtype=np.float64)
        self._actions: np.ndarray | None = None  # open-loop cache for tests / logging
        self._index = 0
        self._closed_loop = True
        self._done = False
        self._phase_name = ""
        self.just_entered_close = False
        self.just_entered_open = False
        # Heading (rad) the finger-closing axis should take before descending.
        # ``None`` for an upright can (a cylinder is grasp-invariant in yaw).
        self._target_closing_yaw: float | None = None
        self._object_lying = False
        self._yaw_error = 0.0

    def reset(self) -> None:
        """Clear the cached plan and action cursor."""
        self._waypoints = []
        self._wp_idx = 0
        self._hold_left = 0
        self._wp_steps = 0
        self._actions = None
        self._index = 0
        self._done = False
        self._phase_name = ""
        self.just_entered_close = False
        self.just_entered_open = False
        self._target_closing_yaw = None
        self._object_lying = False
        self._yaw_error = 0.0
        self.yaw_stall_advances = 0
        self._rot_bias_remaining = (
            None if self._arm_posture_bias is None else self._arm_posture_bias.copy()
        )

    def plan(
        self,
        *,
        eef_pos: np.ndarray,
        eef_quat: np.ndarray,
        object_pos: np.ndarray,
        destination_pos: np.ndarray,
        gripper_open: bool,
        object_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build recovery waypoints and an open-loop action preview ``(T, 7)``.

        ``object_axis`` is the world-frame long axis of the object, when known.

        Keyword-only on purpose: these are five same-shaped pose arrays, and this
        signature has already changed once (an ``object_quat`` argument was
        replaced by the measured ``object_axis``). Positional calls let stale
        callers keep type-checking while silently passing a destination as an
        orientation.
        """
        eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
        eef_quat = np.asarray(eef_quat, dtype=np.float64).reshape(4)
        object_pos = np.asarray(object_pos, dtype=np.float64).reshape(3)
        destination_pos = np.asarray(destination_pos, dtype=np.float64).reshape(3)

        self._set_grasp_orientation(object_axis)
        self._object_pos = object_pos.copy()
        self._destination_pos = destination_pos.copy()
        self._waypoints = self._build_waypoints(eef_pos, object_pos, destination_pos, gripper_open)
        self._wp_idx = 0
        self._hold_left = self._waypoints[0].hold_steps if self._waypoints else 0
        self._wp_steps = 0
        self._done = False
        self._phase_name = self._waypoints[0].name if self._waypoints else ""
        self.just_entered_close = False
        self.just_entered_open = False

        # Keep open-loop preview for unit tests / dataset inspection.
        actions = self._waypoints_to_actions(self._waypoints, start_pos=eef_pos)
        self._actions = actions
        self._index = 0
        self._closed_loop = True
        # Re-arm after the preview: building it spends the one-shot posture
        # offset, and the closed-loop rollout that follows is the real recovery.
        self._rot_bias_remaining = (
            None if self._arm_posture_bias is None else self._arm_posture_bias.copy()
        )
        return actions

    def next_action(
        self,
        eef_pos: np.ndarray | None = None,
        object_pos: np.ndarray | None = None,
        closing_axis: np.ndarray | None = None,
        object_axis: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Return the next planned action, or ``None`` when exhausted.

        When ``eef_pos`` is provided (preferred), track waypoints closed-loop.
        ``closing_axis`` is the measured world-frame finger-closing direction; with
        ``object_axis`` it lets the wrist line up across a tipped-over can.
        """
        self.just_entered_close = False
        self.just_entered_open = False

        if object_axis is not None and self._phase_name in _YAW_PHASES:
            # The can can still roll after landing — keep re-reading its axis
            # until the fingers actually close.
            self._set_grasp_orientation(object_axis)

        if eef_pos is not None and self._closed_loop and self._waypoints:
            return self._next_closed_loop(
                np.asarray(eef_pos, dtype=np.float64).reshape(3),
                None if object_pos is None else np.asarray(object_pos, dtype=np.float64).reshape(3),
                closing_axis,
            )

        if self._actions is None or self._index >= len(self._actions):
            return None
        action = self._actions[self._index]
        self._index += 1
        return action

    @property
    def phase_name(self) -> str:
        return self._phase_name

    @property
    def object_lying(self) -> bool:
        """True when the last-seen object pose was tipped onto its side."""
        return self._object_lying

    @property
    def target_closing_yaw(self) -> float | None:
        """Heading the finger-closing axis should reach before descending."""
        return self._target_closing_yaw

    @property
    def yaw_error(self) -> float:
        """Signed wrist misalignment (rad) from the last closed-loop step."""
        return self._yaw_error

    @property
    def grasp_z_offset(self) -> float:
        """Height above the object center the fingers should close at."""
        return self.grasp_offset_lying if self._object_lying else self.grasp_offset

    def _set_grasp_orientation(self, object_axis: np.ndarray | None) -> None:
        """Decide whether a side grasp is needed and which wrist yaw it wants.

        ``object_axis`` is the world-frame unit vector of the object's long axis
        (see ``libero_sim.object_pose_orientation``). Callers measure it; the
        planner never guesses it from a quaternion.
        """
        self._target_closing_yaw = None
        self._object_lying = False
        if not self.side_grasp_enabled or object_axis is None:
            return
        axis = np.asarray(object_axis, dtype=np.float64).reshape(-1)
        if axis.size != 3 or not np.all(np.isfinite(axis)):
            return
        norm = float(np.linalg.norm(axis))
        if norm < 1e-6:
            return
        axis = axis / norm

        if abs(float(axis[2])) >= _UPRIGHT_DOT:
            return  # standing can: any wrist yaw grasps the same circular profile

        planar = axis[:2]
        if float(np.linalg.norm(planar)) < 1e-6:
            return
        self._object_lying = True
        long_axis_yaw = float(np.arctan2(planar[1], planar[0]))
        # Close across the can's 6.2 cm diameter, never along its 7.6 cm length:
        # the Panda opens 8 cm, so a lengthwise grip has ~2 mm per side.
        self._target_closing_yaw = long_axis_yaw + np.pi / 2

    def _yaw_command(self, closing_axis: np.ndarray | None) -> float:
        """Normalized wrist-Z action driving the fingers onto the can's short axis."""
        if self._target_closing_yaw is None or closing_axis is None:
            self._yaw_error = 0.0
            return 0.0
        axis = np.asarray(closing_axis, dtype=np.float64).reshape(-1)
        if axis.size < 2 or float(np.linalg.norm(axis[:2])) < 1e-6:
            self._yaw_error = 0.0
            return 0.0
        current = float(np.arctan2(axis[1], axis[0]))
        self._yaw_error = _wrap_to_half_pi(self._target_closing_yaw - current)
        return float(
            np.clip(self._yaw_error / self.max_rot_step * self.yaw_gain, -1.0, 1.0)
        )

    @property
    def done(self) -> bool:
        return self._done

    def _parse_arm_posture_bias(
        self,
        arm_posture_noise_rad: np.ndarray | float | None,
    ) -> np.ndarray | None:
        if arm_posture_noise_rad is None:
            return None
        if isinstance(arm_posture_noise_rad, (int, float)):
            return np.full(3, float(arm_posture_noise_rad), dtype=np.float64)
        arr = np.asarray(arm_posture_noise_rad, dtype=np.float64).reshape(-1)
        if arr.size == 3:
            return arr.copy()
        if arr.size == 7:
            return arr[3:6].copy()
        raise ValueError(
            "arm_posture_noise_rad must be scalar, length-3, or length-7 "
            f"(got shape {arr.shape})."
        )

    def _clamp_workspace(self, pos: np.ndarray) -> np.ndarray:
        out = pos.copy()
        out[:2] = np.clip(out[:2], -self._WORKSPACE_XY_LIMIT_M, self._WORKSPACE_XY_LIMIT_M)
        out[2] = max(out[2], 0.0)
        return out

    def _maybe_noise(self, pos: np.ndarray) -> np.ndarray:
        if self.waypoint_noise_m <= 0:
            return pos
        noise = self._rng.normal(0.0, self.waypoint_noise_m, size=3)
        return self._clamp_workspace(pos + noise)

    def _apply_rot_bias(self, action: np.ndarray) -> None:
        """Spend the per-episode posture bias as a bounded, one-shot rotation.

        LIBERO's OSC reads ``action[3:6]`` as a per-step rotation *delta* scaled
        by ``max_rot_step``. Commanding the bias on every step therefore
        integrates without bound — a 2 deg bias winds the wrist hundreds of
        degrees over a long recovery and contorts the arm. Command the offset
        once (in normalized units), then command zero.
        """
        if self._rot_bias_remaining is None:
            return
        cmd = np.clip(self._rot_bias_remaining / self.max_rot_step, -1.0, 1.0)
        self._rot_bias_remaining = self._rot_bias_remaining - cmd * self.max_rot_step
        action[3:6] = cmd.astype(np.float32)

    def _build_waypoints(
        self,
        eef_pos: np.ndarray,
        object_pos: np.ndarray,
        destination_pos: np.ndarray,
        gripper_open: bool,
    ) -> list[_Waypoint]:
        g_open, g_close = True, False
        start_grip = g_open if gripper_open else g_close

        obj_hover = self._maybe_noise(object_pos + np.array([0.0, 0.0, self.hover_offset]))
        obj_grasp = object_pos + np.array([0.0, 0.0, self.grasp_z_offset])
        obj_lift = self._maybe_noise(object_pos + np.array([0.0, 0.0, self.lift_height]))

        # Place: carry HIGH above basket rim (~0.15), then release.
        dest_hover = self._maybe_noise(
            np.array([destination_pos[0], destination_pos[1], 0.28])
        )
        dest_retract = self._maybe_noise(
            np.array([destination_pos[0], destination_pos[1], 0.34])
        )

        retract = self._maybe_noise(eef_pos + np.array([0.0, 0.0, 0.05]))

        return [
            _Waypoint(retract, start_grip, hold_steps=2, name="retract"),
            # Track only while approaching; freeze at grasp so we don't chase/push the can.
            _Waypoint(obj_hover, g_open, hold_steps=3, track_object=True, name="approach_hover"),
            _Waypoint(obj_grasp, g_open, hold_steps=4, track_object=True, name="descend_grasp"),
            _Waypoint(obj_grasp, g_close, hold_steps=self.grasp_hold_steps, track_object=False, name="close_grasp"),
            _Waypoint(obj_lift, g_close, hold_steps=6, track_object=False, name="lift"),
            _Waypoint(dest_hover, g_close, hold_steps=12, name="to_basket_hover"),
            # Open high above the basket — never descend into the rim while holding.
            _Waypoint(dest_hover.copy(), g_open, hold_steps=max(self.place_hold_steps, 20), name="open_place"),
            _Waypoint(dest_retract, g_open, hold_steps=10, name="retract_done"),
        ]

    def retarget_basket(self, destination_pos: np.ndarray) -> None:
        """Update remaining place waypoints to a live basket pose (keep high Z)."""
        destination_pos = np.asarray(destination_pos, dtype=np.float64).reshape(3)
        self._destination_pos = destination_pos.copy()
        hover_z = 0.28
        for wp in self._waypoints[self._wp_idx :]:
            if wp.name in ("to_basket_hover", "open_place"):
                wp.pos = np.array([destination_pos[0], destination_pos[1], hover_z], dtype=np.float64)
            elif wp.name == "retract_done":
                wp.pos = np.array([destination_pos[0], destination_pos[1], hover_z + 0.06], dtype=np.float64)

    def replan_from(
        self,
        *,
        eef_pos: np.ndarray,
        eef_quat: np.ndarray,
        object_pos: np.ndarray,
        destination_pos: np.ndarray | None = None,
        gripper_open: bool = True,
        object_axis: np.ndarray | None = None,
    ) -> np.ndarray:
        """Rebuild waypoints from the current poses (used after a failed regrasp)."""
        dest = self._destination_pos if destination_pos is None else np.asarray(destination_pos, dtype=np.float64)
        return self.plan(
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            object_pos=object_pos,
            object_axis=object_axis,
            destination_pos=dest,
            gripper_open=gripper_open,
        )

    def _current_target(
        self,
        object_pos: np.ndarray | None,
        eef_pos: np.ndarray | None = None,
    ) -> _Waypoint:
        wp = self._waypoints[self._wp_idx]
        if wp.track_object and object_pos is not None:
            # Follow object XY while approaching; keep commanded height offsets.
            refreshed = wp.pos.copy()
            refreshed[0] = object_pos[0]
            refreshed[1] = object_pos[1]
            if wp.name == "descend_grasp":
                refreshed[2] = object_pos[2] + self.grasp_z_offset
            elif wp.name == "approach_hover":
                refreshed[2] = object_pos[2] + self.hover_offset
            return _Waypoint(refreshed, wp.gripper_open, wp.hold_steps, wp.track_object, wp.name)
        # Place phases: put the OBJECT over the basket (EEF may be offset while holding).
        if (
            wp.name in ("to_basket_hover", "open_place")
            and object_pos is not None
            and eef_pos is not None
        ):
            refreshed = wp.pos.copy()
            offset_xy = eef_pos[:2] - object_pos[:2]
            refreshed[0] = self._destination_pos[0] + float(offset_xy[0])
            refreshed[1] = self._destination_pos[1] + float(offset_xy[1])
            return _Waypoint(refreshed, wp.gripper_open, wp.hold_steps, wp.track_object, wp.name)
        return wp

    def _advance_waypoint(self, prev_open: bool, object_pos: np.ndarray | None) -> None:
        self._wp_idx += 1
        self._wp_steps = 0
        if self._wp_idx >= len(self._waypoints):
            self._done = True
            self._phase_name = "done"
            return
        wp = self._waypoints[self._wp_idx]
        # Freeze grasp/lift targets to the live object pose at phase entry.
        if object_pos is not None and wp.name in ("close_grasp", "lift"):
            if wp.name == "close_grasp":
                wp.pos = object_pos + np.array([0.0, 0.0, self.grasp_z_offset])
            else:
                wp.pos = object_pos + np.array([0.0, 0.0, self.lift_height])
            self._object_pos = object_pos.copy()
        self._hold_left = wp.hold_steps
        self._phase_name = wp.name
        if prev_open and not wp.gripper_open:
            self.just_entered_close = True
        if (not prev_open) and wp.gripper_open:
            self.just_entered_open = True

    def _next_closed_loop(
        self,
        eef_pos: np.ndarray,
        object_pos: np.ndarray | None,
        closing_axis: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if self._done or not self._waypoints:
            return None

        yaw_cmd = self._yaw_command(closing_axis)
        # Do not drop onto a tipped can with the fingers still crossed over it.
        yaw_aligned = (
            self._target_closing_yaw is None or abs(self._yaw_error) <= self.yaw_tol_rad
        )

        # Advance through reached (or stuck) waypoints.
        while not self._done:
            wp = self._current_target(object_pos, eef_pos)
            dist, tol = self._phase_distance_tol(wp, eef_pos, object_pos)
            arrived = dist <= tol and (yaw_aligned or wp.name not in _YAW_GATED_PHASES)
            # Never stuck-advance off the basket hover — that causes early releases.
            # open_place may stuck-advance after the hold so recovery can finish.
            allow_stuck = wp.name != "to_basket_hover"
            budget = self.max_steps_per_waypoint
            misaligned_pregrasp = wp.name in _YAW_GATED_PHASES and not yaw_aligned
            if misaligned_pregrasp:
                # Give the wrist servo extra time instead of closing crossed over
                # the can — but stay bounded so a stalled servo cannot hang us.
                budget = int(round(budget * self.yaw_stall_multiplier))
            stuck = allow_stuck and self._wp_steps >= budget
            if stuck and misaligned_pregrasp:
                self.yaw_stall_advances += 1
            if stuck:
                self._hold_left = 0
            if (arrived or stuck) and self._hold_left <= 0:
                prev_open = wp.gripper_open
                self._advance_waypoint(prev_open, object_pos)
                continue
            break

        if self._done:
            return None

        wp = self._current_target(object_pos, eef_pos)
        dist, tol = self._phase_distance_tol(wp, eef_pos, object_pos)
        arrived = dist <= tol and (yaw_aligned or wp.name not in _YAW_GATED_PHASES)
        self._wp_steps += 1

        action = np.zeros(7, dtype=np.float32)
        if not arrived:
            delta = wp.pos - eef_pos
            # Hold height until the wrist lines up, otherwise the fingers hit the
            # can broadside and shove it away instead of grasping.
            if wp.name == "descend_grasp" and not yaw_aligned:
                delta = delta.copy()
                delta[2] = max(float(delta[2]), 0.0)
            # Scale so full max_pos_step → action 1.0 (matches OSC output_max=0.05).
            action[:3] = np.clip(
                delta / self.max_pos_step * self.speed_multiplier,
                -1.0,
                1.0,
            ).astype(np.float32)
        else:
            self._hold_left = max(0, self._hold_left - 1)

        self._apply_rot_bias(action)
        if wp.name in _YAW_PHASES and self._target_closing_yaw is not None:
            action[5] = np.float32(yaw_cmd)
        action[6] = -1.0 if wp.gripper_open else 1.0
        self._phase_name = wp.name
        return action

    def _phase_distance_tol(
        self,
        wp: _Waypoint,
        eef_pos: np.ndarray,
        object_pos: np.ndarray | None,
    ) -> tuple[float, float]:
        if wp.name == "approach_hover":
            return float(np.linalg.norm(wp.pos[:2] - eef_pos[:2])), self.arrive_tol
        if wp.name in ("to_basket_hover", "open_place"):
            # Arrive when the OBJECT (not EEF) is over the basket.
            if object_pos is not None:
                dist = float(np.linalg.norm(object_pos[:2] - self._destination_pos[:2]))
            else:
                dist = float(np.linalg.norm(wp.pos[:2] - eef_pos[:2]))
            return dist, self.basket_xy_tol
        return float(np.linalg.norm(wp.pos - eef_pos)), self.arrive_tol

    def _waypoints_to_actions(
        self,
        waypoints: list[_Waypoint],
        *,
        start_pos: np.ndarray,
    ) -> np.ndarray:
        """Open-loop preview (tests / logging). Not used when closed-loop eef is provided."""
        eff_max_pos = self.max_pos_step * self.speed_multiplier
        segments: list[tuple[np.ndarray, bool]] = [(start_pos, waypoints[0].gripper_open)]
        for wp in waypoints:
            segments.append((wp.pos, wp.gripper_open))
            for _ in range(max(0, wp.hold_steps - 1)):
                segments.append((wp.pos, wp.gripper_open))

        actions: list[np.ndarray] = []
        for i in range(len(segments) - 1):
            cur_pos, _ = segments[i]
            tgt_pos, tgt_grip = segments[i + 1]
            dist = float(np.linalg.norm(tgt_pos - cur_pos))
            grip_val = -1.0 if tgt_grip else 1.0
            if dist < 1e-9:
                actions.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip_val], dtype=np.float32))
                continue

            steps = max(1, int(np.ceil(dist / eff_max_pos)))
            pos_path = np.linspace(cur_pos, tgt_pos, steps + 1)[1:]
            prev = cur_pos
            for p in pos_path:
                delta = p - prev
                prev = p
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(
                    delta / self.max_pos_step * self.speed_multiplier,
                    -1.0,
                    1.0,
                )
                self._apply_rot_bias(action)
                action[6] = grip_val
                actions.append(action)

        if not actions:
            actions.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))
        return np.stack(actions, axis=0)
