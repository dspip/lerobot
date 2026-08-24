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

"""End-effector external impulse bump (inject-only, no recovery)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.sim.libero import apply_eef_bump, get_robosuite_env
from lerobot.faults.logging import FaultEventLogger


@dataclass
class _EnvBumpState:
    episode_step: int = 0
    triggered: bool = False
    will_activate: bool | None = None
    episode_id: int | None = None
    finished: bool = False


class EefBumpFault:
    """Once in ``[t_min, t_max]``, apply a short force kick to the EEF; keep VLA in control."""

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ) -> None:
        if config.type != "eef_bump":
            raise ValueError(f"EefBumpFault requires type='eef_bump', got {config.type!r}.")
        config.validate(num_envs=num_envs)
        self.config = config
        self.num_envs = num_envs
        self.event_logger = event_logger
        self._selected = set(range(num_envs)) if config.env_ids is None else set(config.env_ids)
        seed = 0 if config.seed is None else int(config.seed)
        self._rng = np.random.default_rng(seed)
        self._states = [_EnvBumpState() for _ in range(num_envs)]

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
            self._states[i] = _EnvBumpState(episode_id=ep_id)

    def notify_dones(self, dones: np.ndarray) -> None:
        dones = np.asarray(dones, dtype=bool)
        if dones.shape != (self.num_envs,):
            raise ValueError(f"dones must have shape ({self.num_envs},), got {dones.shape}.")
        for i, done in enumerate(dones):
            if done:
                ep_id = self._states[i].episode_id
                self._states[i] = _EnvBumpState(episode_id=ep_id, finished=True)

    def on_step(
        self,
        env: Any,
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
            state = self._states[env_idx]
            if env_idx not in self._selected or state.finished:
                continue
            if self._should_trigger(state):
                self._trigger(env, env_idx, state)
            state.episode_step += 1
        return executed

    def _should_trigger(self, state: _EnvBumpState) -> bool:
        if state.triggered:
            return False
        if not (self.config.t_min <= state.episode_step <= self.config.t_max):
            return False
        if state.will_activate is None:
            state.will_activate = bool(self._rng.random() < self.config.probability)
        return bool(state.will_activate)

    def _trigger(self, env: Any, env_idx: int, state: _EnvBumpState) -> None:
        rs_env = get_robosuite_env(env, env_idx=env_idx)
        force = self._rng.normal(0.0, float(self.config.bump_force_std), size=3)
        telemetry = apply_eef_bump(
            rs_env,
            force,
            settle_steps=max(int(self.config.settle_steps), 5),
        )
        state.triggered = True
        if self.event_logger is not None:
            self.event_logger.log(
                {
                    "event": "eef_bump",
                    "status": "activated",
                    "fault_type": self.config.type,
                    "evaluation_episode_id": state.episode_id,
                    "vector_env_id": env_idx,
                    "episode_step": state.episode_step,
                    "t_min": self.config.t_min,
                    "t_max": self.config.t_max,
                    "force": np.asarray(telemetry["force"], dtype=float).tolist(),
                    "eef_pre": np.asarray(telemetry["eef_pre"], dtype=float).tolist(),
                    "eef_post": np.asarray(telemetry["eef_post"], dtype=float).tolist(),
                }
            )
