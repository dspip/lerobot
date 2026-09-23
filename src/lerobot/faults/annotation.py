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

"""Frame-level failure annotations from live sim GT + injector flags.

``is_failure`` is a **physical** latch (object was held mid-air, then lost,
and is not in the basket). ``injection_active`` is an **injector** pulse or
window. Keeping them separate stops models from learning the injection clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lerobot.faults.sim.libero import (
    DEFAULT_BASKET_NAME,
    DEFAULT_OBJECT_NAME,
    get_object_pose,
    get_robosuite_env,
    is_object_grasped,
    is_object_held_midair,
    is_object_in_basket,
)

# Sticky category ids written as int64 (1,) in Parquet.
FAILURE_TYPE_NONE = 0
FAILURE_TYPE_MIDAIR_DROP = 1
FAILURE_TYPE_OBJECT_SLIP = 2
FAILURE_TYPE_EEF_BUMP = 3
FAILURE_TYPE_ACTION_HOLD = 4
FAILURE_TYPE_ACTION_DELAY = 5
FAILURE_TYPE_ACTION_JITTER = 6
FAILURE_TYPE_SENSOR_DROPOUT = 7
FAILURE_TYPE_VISUAL_OCCLUSION = 8
FAILURE_TYPE_VISUAL_BLUR = 9
FAILURE_TYPE_BRIGHTNESS_DROP = 10
FAILURE_TYPE_OBS_LATENCY = 11

PHASE_NOMINAL = 0
PHASE_INJECTION = 1
PHASE_POST_FAULT = 2
PHASE_RECOVERY = 3

FAILURE_TYPE_FROM_CONFIG: dict[str, int] = {
    "midair_drop": FAILURE_TYPE_MIDAIR_DROP,
    "object_slip": FAILURE_TYPE_OBJECT_SLIP,
    "eef_bump": FAILURE_TYPE_EEF_BUMP,
    "action_hold": FAILURE_TYPE_ACTION_HOLD,
    "action_delay": FAILURE_TYPE_ACTION_DELAY,
    "action_jitter": FAILURE_TYPE_ACTION_JITTER,
    "sensor_dropout": FAILURE_TYPE_SENSOR_DROPOUT,
    "visual_occlusion": FAILURE_TYPE_VISUAL_OCCLUSION,
    "visual_blur": FAILURE_TYPE_VISUAL_BLUR,
    "brightness_drop": FAILURE_TYPE_BRIGHTNESS_DROP,
    "obs_latency": FAILURE_TYPE_OBS_LATENCY,
}

FAILURE_ANNOTATION_FEATURES: dict[str, dict[str, Any]] = {
    "is_failure": {"dtype": "bool", "shape": (1,), "names": None},
    "ever_held_midair": {"dtype": "bool", "shape": (1,), "names": None},
    "failure_onset": {"dtype": "bool", "shape": (1,), "names": None},
    "failure_type": {"dtype": "int64", "shape": (1,), "names": None},
    "injection_active": {"dtype": "bool", "shape": (1,), "names": None},
    "phase": {"dtype": "int64", "shape": (1,), "names": None},
}


def failure_type_id(config_type: str | None) -> int:
    if not config_type:
        return FAILURE_TYPE_NONE
    return int(FAILURE_TYPE_FROM_CONFIG.get(str(config_type), FAILURE_TYPE_NONE))


def default_failure_frame() -> dict[str, np.ndarray]:
    """Successful / unlabeled frame: all flags false, type and phase zero."""
    return {
        "is_failure": np.array([False]),
        "ever_held_midair": np.array([False]),
        "failure_onset": np.array([False]),
        "failure_type": np.array([FAILURE_TYPE_NONE], dtype=np.int64),
        "injection_active": np.array([False]),
        "phase": np.array([PHASE_NOMINAL], dtype=np.int64),
    }


def annotation_to_frame(
    *,
    is_failure: bool,
    ever_held_midair: bool,
    failure_onset: bool,
    failure_type: int,
    injection_active: bool,
    phase: int,
) -> dict[str, np.ndarray]:
    return {
        "is_failure": np.array([bool(is_failure)]),
        "ever_held_midair": np.array([bool(ever_held_midair)]),
        "failure_onset": np.array([bool(failure_onset)]),
        "failure_type": np.array([int(failure_type)], dtype=np.int64),
        "injection_active": np.array([bool(injection_active)]),
        "phase": np.array([int(phase)], dtype=np.int64),
    }


def choose_phase(*, injection_active: bool, is_failure: bool, recovery_active: bool) -> int:
    # Injection pulse wins so the glitch frame is not labeled as recovery-only.
    if injection_active:
        return PHASE_INJECTION
    if recovery_active:
        return PHASE_RECOVERY
    if is_failure:
        return PHASE_POST_FAULT
    return PHASE_NOMINAL


def annotation_from_scripted_phase(phase: str, *, onset: bool = False) -> dict[str, np.ndarray]:
    """Labels for synthetic (no-sim) episodes used by dry-run demos."""
    key = (phase or "vla").strip().lower()
    if key in {"drop", "injection", "fault"}:
        return annotation_to_frame(
            is_failure=True,
            ever_held_midair=False,
            failure_onset=onset,
            failure_type=FAILURE_TYPE_MIDAIR_DROP,
            injection_active=True,
            phase=PHASE_INJECTION,
        )
    if key in {"post", "post_fault"}:
        return annotation_to_frame(
            is_failure=True,
            ever_held_midair=False,
            failure_onset=False,
            failure_type=FAILURE_TYPE_MIDAIR_DROP,
            injection_active=False,
            phase=PHASE_POST_FAULT,
        )
    if key == "recovery":
        return annotation_to_frame(
            is_failure=False,
            ever_held_midair=False,
            failure_onset=False,
            failure_type=FAILURE_TYPE_MIDAIR_DROP,
            injection_active=False,
            phase=PHASE_RECOVERY,
        )
    return default_failure_frame()


@dataclass
class PhysicsSnapshot:
    grasped: bool = False
    held_midair: bool = False
    in_basket: bool = False
    object_z: float | None = None
    available: bool = False


def read_physics_snapshot(
    env: Any,
    env_idx: int,
    *,
    object_name: str = DEFAULT_OBJECT_NAME,
    basket_name: str = DEFAULT_BASKET_NAME,
    min_object_z: float = 0.12,
) -> PhysicsSnapshot:
    """Best-effort MuJoCo/robosuite snapshot. ``available=False`` if not LIBERO."""
    try:
        rs_env = get_robosuite_env(env, env_idx=env_idx)
    except Exception:
        return PhysicsSnapshot()
    try:
        grasped = bool(is_object_grasped(rs_env, object_name))
        held = bool(
            is_object_held_midair(rs_env, object_name, min_object_z=min_object_z)
        )
        in_basket = bool(is_object_in_basket(rs_env, object_name, basket_name=basket_name))
        z = float(get_object_pose(rs_env, object_name)["pos"][2])
        return PhysicsSnapshot(
            grasped=grasped,
            held_midair=held,
            in_basket=in_basket,
            object_z=z,
            available=True,
        )
    except Exception:
        return PhysicsSnapshot()


@dataclass
class FailureAnnotator:
    """Per-env latches for physical drop detection + injector metadata."""

    num_envs: int
    object_name: str = DEFAULT_OBJECT_NAME
    basket_name: str = DEFAULT_BASKET_NAME
    min_object_z: float = 0.12
    injector_type_id: int = FAILURE_TYPE_NONE
    _ever_held: list[bool] = field(init=False)
    _prev_failure: list[bool] = field(init=False)
    _cleared_after_failure: list[bool] = field(init=False)
    _sticky_type: list[int] = field(init=False)
    last_frames: list[dict[str, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self, env_ids: list[int] | None = None) -> None:
        indices = list(range(self.num_envs) if env_ids is None else env_ids)
        if env_ids is None:
            self._ever_held = [False] * self.num_envs
            self._prev_failure = [False] * self.num_envs
            self._cleared_after_failure = [False] * self.num_envs
            self._sticky_type = [FAILURE_TYPE_NONE] * self.num_envs
            self.last_frames = [default_failure_frame() for _ in range(self.num_envs)]
            return
        for i in indices:
            if i < 0 or i >= self.num_envs:
                raise ValueError(f"env_id {i} out of range for num_envs={self.num_envs}.")
            self._ever_held[i] = False
            self._prev_failure[i] = False
            self._cleared_after_failure[i] = False
            self._sticky_type[i] = FAILURE_TYPE_NONE
            self.last_frames[i] = default_failure_frame()

    def update(
        self,
        env: Any,
        *,
        injection_active: np.ndarray | list[bool],
        recovery_active: np.ndarray | list[bool] | None = None,
        physics: list[PhysicsSnapshot] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Advance one env-step. Call **after** physics ``env.step``."""
        inj = np.asarray(injection_active, dtype=bool).reshape(self.num_envs)
        rec = (
            np.zeros(self.num_envs, dtype=bool)
            if recovery_active is None
            else np.asarray(recovery_active, dtype=bool).reshape(self.num_envs)
        )
        frames: list[dict[str, np.ndarray]] = []
        for env_idx in range(self.num_envs):
            snap = (
                physics[env_idx]
                if physics is not None
                else read_physics_snapshot(
                    env,
                    env_idx,
                    object_name=self.object_name,
                    basket_name=self.basket_name,
                    min_object_z=self.min_object_z,
                )
            )
            injection = bool(inj[env_idx])
            recovering = bool(rec[env_idx])
            if injection and self.injector_type_id != FAILURE_TYPE_NONE:
                self._sticky_type[env_idx] = int(self.injector_type_id)

            is_failure = False
            if snap.available:
                if snap.held_midair:
                    self._ever_held[env_idx] = True
                # Physical drop: was held in air this episode, currently free, not placed.
                is_failure = bool(
                    self._ever_held[env_idx] and (not snap.grasped) and (not snap.in_basket)
                )
            # After the first drop is cleared (regrasp / basket), a later ungrasp is
            # usually the recovery release into the basket — not a second failure.
            if is_failure and self._cleared_after_failure[env_idx]:
                is_failure = False

            if is_failure and self._sticky_type[env_idx] == FAILURE_TYPE_NONE:
                self._sticky_type[env_idx] = FAILURE_TYPE_MIDAIR_DROP

            onset = bool(is_failure and not self._prev_failure[env_idx])
            if self._prev_failure[env_idx] and not is_failure:
                self._cleared_after_failure[env_idx] = True
            ftype = int(self._sticky_type[env_idx])

            phase = choose_phase(
                injection_active=injection,
                is_failure=is_failure,
                recovery_active=recovering,
            )
            frame = annotation_to_frame(
                is_failure=is_failure,
                ever_held_midair=bool(self._ever_held[env_idx]),
                failure_onset=onset,
                failure_type=ftype,
                injection_active=injection,
                phase=phase,
            )
            self._prev_failure[env_idx] = is_failure
            self.last_frames[env_idx] = frame
            frames.append(frame)
        return frames


def frames_to_info_arrays(frames: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-env annotation frames into Gymnasium vector-env info arrays."""
    if not frames:
        return {}
    out: dict[str, np.ndarray] = {}
    for key, spec in FAILURE_ANNOTATION_FEATURES.items():
        dtype = np.dtype(spec["dtype"])
        out[key] = np.array([np.asarray(f[key]).reshape(-1)[0] for f in frames], dtype=dtype)
    return out


def clip_failure_to_first_interval(is_failure: list[bool] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the first physical-failure run; recompute onset from that run.

    Used to correct recordings where the recovery release into the basket
    briefly looks like a second drop (ungrasped, not yet ``in_basket``).
    """
    fail = np.asarray([bool(x) for x in np.asarray(is_failure).reshape(-1)], dtype=bool)
    out_fail = np.zeros_like(fail)
    cleared = False
    prev = False
    for i, val in enumerate(fail.tolist()):
        if val and cleared:
            val = False
        out_fail[i] = val
        if prev and not val:
            cleared = True
        prev = val
    out_onset = np.zeros_like(out_fail)
    prev = False
    for i, val in enumerate(out_fail.tolist()):
        out_onset[i] = bool(val and not prev)
        prev = val
    return out_fail, out_onset


def merge_annotation_into_info(info: Any, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(info) if isinstance(info, dict) else {}
    merged.update(arrays)
    return merged


def failure_frame_from_info(info: Any, env_idx: int = 0) -> dict[str, np.ndarray]:
    """Extract one env's annotation from Gym ``info``; zeros if absent."""
    base = default_failure_frame()
    if not isinstance(info, dict):
        return base
    out = dict(base)
    for key, spec in FAILURE_ANNOTATION_FEATURES.items():
        if key not in info:
            continue
        val = np.asarray(info[key])
        if val.ndim == 0:
            item = val.item()
        elif env_idx < val.shape[0]:
            item = val[env_idx]
        else:
            continue
        if spec["dtype"] == "bool":
            out[key] = np.array([bool(item)])
        else:
            out[key] = np.array([int(item)], dtype=np.int64)
    return out


def injector_injection_active(injector: Any, env_idx: int) -> bool:
    """Read a one-step injection pulse / hold window from injector per-env state."""
    if injector is None:
        return False
    states = getattr(injector, "_states", None)
    if states is None or env_idx < 0 or env_idx >= len(states):
        return False
    state = states[env_idx]
    if getattr(state, "drop_injection_step", False):
        return True
    if getattr(state, "just_injected", False):
        return True
    remaining = getattr(state, "remaining", 0)
    try:
        if int(remaining) > 0:
            return True
    except (TypeError, ValueError):
        pass
    # Always-on action faults: delay buffer full, or jitter with non-zero std.
    if getattr(state, "activated", False) and hasattr(injector, "config"):
        ftype = getattr(injector.config, "type", "")
        if ftype in {"action_delay", "action_jitter"}:
            return True
    return False
