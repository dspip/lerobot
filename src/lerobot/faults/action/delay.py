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

"""Action-delay fault injection (control/network latency via FIFO buffer)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger


@dataclass
class _EnvDelayState:
    buffer: deque[np.ndarray] = field(default_factory=deque)
    episode_step: int = 0
    activated: bool = False
    episode_id: int | None = None
    finished: bool = False


class ActionDelayFault:
    """Execute actions from ``delay_steps`` ago via a per-env FIFO buffer."""

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ):
        if config.type != "action_delay":
            raise ValueError(f"ActionDelayFault requires type='action_delay', got {config.type!r}.")
        config.validate(num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.event_logger = event_logger
        self._selected = set(range(num_envs)) if config.env_ids is None else set(config.env_ids)
        self._states = [_EnvDelayState() for _ in range(num_envs)]

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
            self._states[i] = _EnvDelayState(episode_id=ep_id)

    def notify_dones(self, dones: np.ndarray) -> None:
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},), got {dones.shape}.")
        for i, done in enumerate(dones):
            if done:
                ep_id = self._states[i].episode_id
                self._states[i] = _EnvDelayState(episode_id=ep_id, finished=True)

    def apply(
        self,
        actions: np.ndarray,
        episode_ids: list[int] | None = None,
    ) -> np.ndarray:
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

            if env_idx not in self._selected:
                executed[env_idx] = actions[env_idx]
                continue

            state = self._states[env_idx]
            if state.finished:
                executed[env_idx] = actions[env_idx]
                continue

            proposed = actions[env_idx].copy()
            buf = state.buffer

            if len(buf) < self.config.delay_steps:
                executed[env_idx] = proposed
                buf.append(proposed.copy())
            else:
                delayed = buf.popleft()
                buf.append(proposed.copy())
                executed[env_idx] = delayed
                if not np.array_equal(proposed, delayed):
                    status = "activated" if not state.activated else "active"
                    state.activated = True
                    self._log_event(
                        env_idx=env_idx,
                        status=status,
                        proposed=proposed,
                        delayed=delayed,
                    )

            state.episode_step += 1

        return executed

    def _log_event(
        self,
        *,
        env_idx: int,
        status: str,
        proposed: np.ndarray,
        delayed: np.ndarray,
    ) -> None:
        if self.event_logger is None:
            return
        state = self._states[env_idx]
        self.event_logger.log(
            {
                "event": "action_delay",
                "status": status,
                "fault_type": self.config.type,
                "evaluation_episode_id": state.episode_id,
                "vector_env_id": env_idx,
                "episode_step": state.episode_step,
                "delay_steps": self.config.delay_steps,
                "proposed_action": proposed.astype(float).tolist(),
                "executed_delayed_action": delayed.astype(float).tolist(),
            }
        )
