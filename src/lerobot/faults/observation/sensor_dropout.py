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

"""Sensor dropout (camera blackout) fault injection on observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    zero_image_slice,
)

MAX_DIAG_IMAGES = 20
DIAG_EVERY_N_STEPS = 10


@dataclass
class _EnvDropoutState:
    episode_step: int = 0
    remaining: int = 0
    activated: bool = False
    blackout_steps: int = 0
    episode_id: int | None = None
    finished: bool = False


class SensorDropoutFault:
    """Black out camera/image observations for a configured step window."""

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ):
        if config.type != "sensor_dropout":
            raise ValueError(
                f"SensorDropoutFault requires type='sensor_dropout', got {config.type!r}."
            )
        config.validate(num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.event_logger = event_logger
        self.diag_dir = Path(config.diag_dir) if config.diag_dir is not None else None
        if self.diag_dir is not None:
            self.diag_dir.mkdir(parents=True, exist_ok=True)
        self._selected = set(range(num_envs)) if config.env_ids is None else set(config.env_ids)
        self._states = [_EnvDropoutState() for _ in range(num_envs)]
        self._diag_saved = 0

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def reset(
        self,
        env_ids: list[int] | None = None,
        episode_ids: list[int] | dict[int, int] | None = None,
    ) -> None:
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
            self._states[i] = _EnvDropoutState(episode_id=ep_id)

    def notify_dones(self, dones: np.ndarray) -> None:
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},), got {dones.shape}.")
        for i, done in enumerate(dones):
            if done:
                ep_id = self._states[i].episode_id
                self._states[i] = _EnvDropoutState(episode_id=ep_id, finished=True)

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

        active_envs = self._active_envs(from_reset=from_reset)
        if not active_envs:
            if not from_reset:
                self._advance_steps_for_inactive()
            return obs

        mutated = copy_obs_tree(obs)
        for env_idx in active_envs:
            self._blackout_tree(mutated, env_idx)
            self._maybe_dump_diag(mutated, env_idx)

        if not from_reset:
            for env_idx in active_envs:
                self._log_event(env_idx=env_idx)
            self._advance_all_steps()
        return mutated

    def _active_envs(self, *, from_reset: bool) -> list[int]:
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

            if (
                state.episode_step == self.config.trigger_step
                and not state.activated
            ):
                state.activated = True
                state.remaining = self.config.duration
                active.append(env_idx)
        return active

    def _advance_all_steps(self) -> None:
        for env_idx in range(self.num_envs):
            state = self._states[env_idx]
            if state.finished:
                continue
            if state.remaining > 0:
                state.blackout_steps += 1
                state.remaining -= 1
            state.episode_step += 1

    def _advance_steps_for_inactive(self) -> None:
        self._advance_all_steps()

    def _blackout_tree(self, node: Any, env_idx: int, key: str | None = None) -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                self._blackout_tree(child, env_idx, str(child_key))
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, np.ndarray)):
            for child in node:
                self._blackout_tree(child, env_idx, key)
            return
        if is_array_like(node) and is_image_field(key, node):
            zero_image_slice(node, env_idx, self.num_envs)

    def _maybe_dump_diag(self, obs: Any, env_idx: int) -> None:
        if self.diag_dir is None or self._diag_saved >= MAX_DIAG_IMAGES:
            return

        state = self._states[env_idx]
        if state.blackout_steps != 0 and state.blackout_steps % DIAG_EVERY_N_STEPS != 0:
            return

        step_label = state.episode_step
        for cam_idx, (key_path, value) in enumerate(iter_image_entries(obs)):
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
        self.event_logger.log(
            {
                "event": "sensor_dropout",
                "status": status,
                "fault_type": self.config.type,
                "evaluation_episode_id": state.episode_id,
                "vector_env_id": env_idx,
                "episode_step": state.episode_step,
                "trigger_step": self.config.trigger_step,
                "duration": self.config.duration,
                "remaining_after": state.remaining,
                "blackout_steps": state.blackout_steps,
                "diag_dir": str(self.diag_dir) if self.diag_dir is not None else None,
            }
        )
