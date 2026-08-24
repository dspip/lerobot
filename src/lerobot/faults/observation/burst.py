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

"""Shared burst scheduling for visual observation faults."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.utils import (
    array_to_uint8_hwc,
    copy_obs_tree,
    is_array_like,
    is_image_field,
    iter_image_entries,
    save_png,
)
from lerobot.faults.observation.policy_video import PolicyCameraVideoRecorder, pick_wrist_camera

MAX_DIAG_IMAGES = 20
DIAG_EVERY_N_STEPS = 1


@dataclass
class _EnvBurstState:
    episode_step: int = 0
    remaining: int = 0
    activated: bool = False
    will_activate: bool | None = None
    active_steps: int = 0
    episode_id: int | None = None
    finished: bool = False
    box: tuple[int, int, int, int] | None = None


MutateFn = Callable[[Any, int, int, _EnvBurstState], None]


class VisualBurstFault:
    """Burst visual corruption over image fields for ``duration`` steps at ``trigger_step``."""

    event_name: str = "visual_burst"
    required_type: str = "visual_burst"

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
        *,
        mutate: MutateFn,
        event_name: str | None = None,
        required_type: str | None = None,
        sample_box_on_activate: bool = False,
    ):
        self.required_type = required_type or self.required_type
        self.event_name = event_name or self.event_name
        if config.type != self.required_type:
            raise ValueError(
                f"{type(self).__name__} requires type={self.required_type!r}, got {config.type!r}."
            )
        config.validate(num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.event_logger = event_logger
        self._mutate = mutate
        self._sample_box = sample_box_on_activate
        self.diag_dir = Path(config.diag_dir) if config.diag_dir is not None else None
        if self.diag_dir is not None:
            self.diag_dir.mkdir(parents=True, exist_ok=True)
        self._selected = set(range(num_envs)) if config.env_ids is None else set(config.env_ids)
        seed = 0 if config.seed is None else int(config.seed)
        self._rng = np.random.default_rng(seed)
        self._states = [_EnvBurstState() for _ in range(num_envs)]
        self._diag_saved = 0
        self._video: PolicyCameraVideoRecorder | None = None
        if config.policy_video_dir is not None:
            self._video = PolicyCameraVideoRecorder(
                Path(config.policy_video_dir),
                fault_type=config.type,
                fps=10.0,
            )

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(
        self,
        env_ids: list[int] | None = None,
        episode_ids: list[int] | dict[int, int] | None = None,
    ) -> None:
        if self._video is not None:
            self._video.reset_episode()
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
            self._states[i] = _EnvBurstState(episode_id=ep_id)

    def notify_dones(self, dones: np.ndarray) -> None:
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},), got {dones.shape}.")
        if self._video is not None and bool(np.any(dones)):
            self._video.flush()
        for i, done in enumerate(dones):
            if done:
                ep_id = self._states[i].episode_id
                self._states[i] = _EnvBurstState(episode_id=ep_id, finished=True)

    def close(self) -> None:
        if self._video is not None:
            self._video.flush()
        if self.event_logger is not None:
            self.event_logger.close()

    def apply_obs(
        self,
        obs: Any,
        *,
        episode_ids: list[int] | None = None,
        from_reset: bool = False,
    ) -> Any:
        if not self.config.enabled:
            return obs

        if episode_ids is not None:
            for env_idx, episode_id in enumerate(episode_ids):
                self._states[env_idx].episode_id = episode_id

        if from_reset:
            return obs

        active_envs = self._active_envs(obs, from_reset=False)
        if not active_envs:
            self._maybe_record_video(obs, obs, fault_active=False, env_idx=0)
            self._advance_all_steps()
            return obs

        mutated = copy_obs_tree(obs)
        for env_idx in active_envs:
            self._mutate_tree(mutated, env_idx)
            self._maybe_dump_diag(mutated, env_idx)

        if 0 in active_envs or 0 in self._selected:
            self._maybe_record_video(obs, mutated, fault_active=True, env_idx=0)

        for env_idx in active_envs:
            self._log_event(env_idx=env_idx)
        self._advance_all_steps()
        return mutated

    def _maybe_record_video(
        self,
        clean_obs: Any,
        faulted_obs: Any,
        *,
        fault_active: bool,
        env_idx: int,
    ) -> None:
        if self._video is None:
            return
        state = self._states[env_idx]
        if state.finished:
            return
        clean = pick_wrist_camera(clean_obs, env_idx, self.num_envs)
        faulted = pick_wrist_camera(faulted_obs, env_idx, self.num_envs)
        if clean is None or faulted is None:
            return
        self._video.add(
            clean,
            faulted,
            fault_active=fault_active,
            episode_step=state.episode_step,
        )

    def _sample_occlusion_box(self, obs: Any, env_idx: int) -> tuple[int, int, int, int]:
        h = w = None
        for _key_path, value in iter_image_entries(obs):
            hwc = array_to_uint8_hwc(value, env_idx, self.num_envs)
            if hwc is not None:
                h, w = int(hwc.shape[0]), int(hwc.shape[1])
                break
        if h is None or w is None:
            return (0, 0, 1, 1)

        h_frac = float(
            self._rng.uniform(self.config.occlusion_h_frac_min, self.config.occlusion_h_frac_max)
        )
        w_frac = float(
            self._rng.uniform(self.config.occlusion_w_frac_min, self.config.occlusion_w_frac_max)
        )
        bh = max(1, int(round(h * h_frac)))
        bw = max(1, int(round(w * w_frac)))
        y0 = int(self._rng.integers(0, max(1, h - bh + 1)))
        x0 = int(self._rng.integers(0, max(1, w - bw + 1)))
        return (y0, x0, min(h, y0 + bh), min(w, x0 + bw))

    def _active_envs(self, obs: Any, *, from_reset: bool) -> list[int]:
        if from_reset:
            return []

        active: list[int] = []
        for env_idx in range(self.num_envs):
            if env_idx not in self._selected:
                continue
            state = self._states[env_idx]
            if state.finished:
                continue

            if state.remaining > 0:
                active.append(env_idx)
                continue

            if state.episode_step == self.config.trigger_step and not state.activated:
                state.activated = True
                if state.will_activate is None:
                    state.will_activate = bool(self._rng.random() < self.config.probability)
                if state.will_activate:
                    state.remaining = self.config.duration
                    if self._sample_box:
                        state.box = self._sample_occlusion_box(obs, env_idx)
                    active.append(env_idx)
        return active

    def _advance_all_steps(self) -> None:
        for env_idx in range(self.num_envs):
            state = self._states[env_idx]
            if state.finished:
                continue
            if state.remaining > 0:
                state.active_steps += 1
                state.remaining -= 1
            state.episode_step += 1

    def _mutate_tree(self, node: Any, env_idx: int, key: str | None = None) -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                self._mutate_tree(child, env_idx, str(child_key))
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, np.ndarray)):
            for child in node:
                self._mutate_tree(child, env_idx, key)
            return
        if is_array_like(node) and is_image_field(key, node):
            self._mutate(node, env_idx, self.num_envs, self._states[env_idx])

    def _maybe_dump_diag(self, obs: Any, env_idx: int) -> None:
        if self.diag_dir is None or self._diag_saved >= MAX_DIAG_IMAGES:
            return
        state = self._states[env_idx]
        if state.active_steps != 0 and state.active_steps % DIAG_EVERY_N_STEPS != 0:
            return
        step_label = state.episode_step
        for key_path, value in iter_image_entries(obs):
            if self._diag_saved >= MAX_DIAG_IMAGES:
                break
            hwc = array_to_uint8_hwc(value, env_idx, self.num_envs)
            if hwc is None:
                continue
            safe_key = key_path.replace("/", "_").replace(".", "_")
            path = self.diag_dir / f"step_{step_label:04d}_env{env_idx}_{safe_key}.png"
            if save_png(path, hwc):
                self._diag_saved += 1

    def _log_event(self, *, env_idx: int) -> None:
        if self.event_logger is None:
            return
        state = self._states[env_idx]
        remaining_after = state.remaining - 1
        if state.remaining == self.config.duration:
            status = "activated"
        elif remaining_after > 0:
            status = "active"
        else:
            status = "completed"
        payload: dict[str, Any] = {
            "event": self.event_name,
            "status": status,
            "fault_type": self.config.type,
            "evaluation_episode_id": state.episode_id,
            "vector_env_id": env_idx,
            "episode_step": state.episode_step,
            "trigger_step": self.config.trigger_step,
            "duration": self.config.duration,
            "remaining_after": state.remaining,
            "active_steps": state.active_steps,
            "diag_dir": str(self.diag_dir) if self.diag_dir is not None else None,
            "policy_video_dir": (
                str(self.config.policy_video_dir)
                if self.config.policy_video_dir is not None
                else None
            ),
        }
        if state.box is not None:
            y0, x0, y1, x1 = state.box
            payload["occlusion_box"] = {"y0": y0, "x0": x0, "y1": y1, "x1": x1}
        self.event_logger.log(payload)
