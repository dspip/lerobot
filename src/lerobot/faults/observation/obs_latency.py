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

"""Observation latency (stale-frame) fault injection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.utils import copy_obs_tree
from lerobot.faults.observation.policy_video import PolicyCameraVideoRecorder, pick_wrist_camera


@dataclass
class _EnvLatencyState:
    buffer: deque[Any] = field(default_factory=deque)
    episode_step: int = 0
    activated: bool = False
    episode_id: int | None = None
    finished: bool = False


class ObsLatencyFault:
    """Serve observations from ``latency_steps`` ago while physics advances.

    Warm-up: until the FIFO has ``latency_steps`` frames, return the current obs
    and enqueue a copy. Once full, enqueue current and return the oldest (stale).
    """

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ):
        if config.type != "obs_latency":
            raise ValueError(f"ObsLatencyFault requires type='obs_latency', got {config.type!r}.")
        config.validate(num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.event_logger = event_logger
        self._selected = set(range(num_envs)) if config.env_ids is None else set(config.env_ids)
        self._states = [_EnvLatencyState() for _ in range(num_envs)]
        # Shared FIFO for vectorized all-env mode (one stale tree for the batch).
        self._batch_buffer: deque[Any] = deque()
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
        if env_ids is None:
            self._batch_buffer = deque()
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
            self._states[i] = _EnvLatencyState(episode_id=ep_id)

    def notify_dones(self, dones) -> None:
        import numpy as np

        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},), got {dones.shape}.")
        if self._video is not None and bool(np.any(dones)):
            self._video.flush()
        for i, done in enumerate(dones):
            if done:
                ep_id = self._states[i].episode_id
                self._states[i] = _EnvLatencyState(episode_id=ep_id, finished=True)

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
        if from_reset:
            return obs

        if self.num_envs == 1:
            return self._apply_single(obs, env_idx=0, episode_ids=episode_ids)

        return self._apply_vector(obs, episode_ids=episode_ids)

    def _maybe_record_video(
        self,
        clean_obs: Any,
        served_obs: Any,
        *,
        lag_active: bool,
        env_idx: int,
    ) -> None:
        if self._video is None:
            return
        state = self._states[env_idx]
        if state.finished:
            return
        clean = pick_wrist_camera(clean_obs, env_idx, self.num_envs)
        faulted = pick_wrist_camera(served_obs, env_idx, self.num_envs)
        if clean is None or faulted is None:
            return
        self._video.add(
            clean,
            faulted,
            fault_active=lag_active,
            episode_step=state.episode_step,
        )

    def _apply_latency(
        self,
        obs: Any,
        *,
        env_idx: int,
        buffer: deque[Any],
        episode_ids: list[int] | None,
    ) -> Any:
        if episode_ids is not None and env_idx < len(episode_ids):
            self._states[env_idx].episode_id = episode_ids[env_idx]
        state = self._states[env_idx]
        if env_idx not in self._selected or state.finished:
            return obs

        k = int(self.config.latency_steps)
        current = copy_obs_tree(obs)
        if len(buffer) < k:
            buffer.append(current)
            served = obs
            lag_active = False
        else:
            buffer.append(current)
            served = buffer.popleft()
            lag_active = True
            state.activated = True

        self._maybe_record_video(obs, served, lag_active=lag_active, env_idx=env_idx)
        self._log(env_idx=env_idx, lag_active=lag_active, buffer_len=len(buffer))
        state.episode_step += 1
        return served

    def _apply_single(
        self,
        obs: Any,
        *,
        env_idx: int,
        episode_ids: list[int] | None,
    ) -> Any:
        return self._apply_latency(
            obs,
            env_idx=env_idx,
            buffer=self._states[env_idx].buffer,
            episode_ids=episode_ids,
        )

    def _apply_vector(self, obs: Any, *, episode_ids: list[int] | None) -> Any:
        """Apply a shared stale tree to the full batch.

        Multi-env selective ``env_ids`` (subset of the batch) is not supported:
        image trees are delayed as one unit. Use ``env_ids=None`` (all envs) or
        ``num_envs=1``.
        """
        if episode_ids is not None:
            for env_idx, episode_id in enumerate(episode_ids):
                self._states[env_idx].episode_id = episode_id

        if self._selected != set(range(self.num_envs)):
            raise ValueError(
                "obs_latency with num_envs>1 requires env_ids=None (all environments). "
                f"Got env_ids={sorted(self._selected)} for num_envs={self.num_envs}."
            )

        schedule_idx = None
        for env_idx in range(self.num_envs):
            if not self._states[env_idx].finished:
                schedule_idx = env_idx
                break
        if schedule_idx is None:
            return obs

        served = self._apply_latency(
            obs,
            env_idx=schedule_idx,
            buffer=self._batch_buffer,
            episode_ids=None,
        )
        sched_step = self._states[schedule_idx].episode_step
        for env_idx in range(self.num_envs):
            if env_idx != schedule_idx and not self._states[env_idx].finished:
                self._states[env_idx].episode_step = sched_step
                self._states[env_idx].activated = self._states[schedule_idx].activated
        return served

    def _log(self, *, env_idx: int, lag_active: bool, buffer_len: int) -> None:
        if self.event_logger is None or not lag_active:
            return
        state = self._states[env_idx]
        status = "activated" if state.episode_step == int(self.config.latency_steps) else "active"
        self.event_logger.log(
            {
                "event": "obs_latency",
                "status": status,
                "fault_type": self.config.type,
                "evaluation_episode_id": state.episode_id,
                "vector_env_id": env_idx,
                "episode_step": state.episode_step,
                "latency_steps": int(self.config.latency_steps),
                "buffer_len": buffer_len,
                "stale": True,
                "policy_video_dir": (
                    str(self.config.policy_video_dir)
                    if self.config.policy_video_dir is not None
                    else None
                ),
            }
        )
