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

"""Gymnasium wrappers that apply fault injection around ``reset`` / ``step``."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.factory import ActionFaultInjector, make_action_fault_injector


def _num_envs(env: Any) -> int:
    return int(getattr(env, "num_envs", 1))


def _is_wrappable_env(obj: Any) -> bool:
    """LeRobot uses VectorEnv; in Gymnasium 1.x VectorEnv is not a subclass of Env."""
    return isinstance(obj, (gym.Env, VectorEnv)) or (
        hasattr(obj, "step") and hasattr(obj, "reset") and hasattr(obj, "action_space")
    )


def _as_batch(action: np.ndarray, num_envs: int) -> tuple[np.ndarray, bool]:
    """Return (batch, was_single). Batch shape is ``(num_envs, action_dim)``."""
    action = np.asarray(action)
    if num_envs == 1 and action.ndim == 1:
        return action[None, ...], True
    return action, False


def _from_batch(action: np.ndarray, was_single: bool) -> np.ndarray:
    return action[0] if was_single else action


def _extract_dones(result: tuple) -> np.ndarray | None:
    if not isinstance(result, tuple) or len(result) != 5:
        return None
    _, _, terminated, truncated, _ = result
    dones = np.asarray(terminated) | np.asarray(truncated)
    if dones.ndim == 0:
        dones = np.asarray([bool(dones)])
    return dones


class FaultEnvWrapper:
    """Apply optional action fault injectors around the env API."""

    def __init__(self, env: Any, config: FaultInjectionConfig):
        num_envs = _num_envs(env)
        action_injector = make_action_fault_injector(config, num_envs=num_envs)
        if action_injector is None:
            raise ValueError("FaultEnvWrapper requires an enabled action fault injector.")
        self.env = env
        self.fault_config = config
        self.action_injector: ActionFaultInjector = action_injector
        self.num_envs = num_envs

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _notify_dones(self, dones: np.ndarray) -> None:
        self.action_injector.notify_dones(dones.reshape(self.num_envs))

    def _close_loggers(self) -> None:
        closer = getattr(self.action_injector, "close", None)
        if callable(closer):
            closer()
            return
        if self.action_injector.event_logger is not None:
            self.action_injector.event_logger.close()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.action_injector.reset()
        return result

    def step(self, action):
        batch, was_single = _as_batch(np.asarray(action), self.num_envs)
        executed = self.action_injector.apply(batch)
        step_action = _from_batch(executed, was_single)

        result = self.env.step(step_action)

        dones = _extract_dones(result) if isinstance(result, tuple) else None
        if dones is not None:
            self._notify_dones(dones)
        return result

    def close(self):
        self._close_loggers()
        return self.env.close()


def maybe_wrap_env(env: Any, config: FaultInjectionConfig | None) -> Any:
    """Wrap ``env`` when fault config is enabled; otherwise return ``env`` unchanged."""
    if config is None or not config.enabled:
        return env
    if not _is_wrappable_env(env):
        return env
    return FaultEnvWrapper(env, config)


def maybe_wrap_env_tree(envs: Any, config: FaultInjectionConfig | None) -> Any:
    """Wrap Gym envs inside LeRobot's nested ``make_env`` return structure.

    LeRobot returns ``dict[task_group, dict[task_id, vec_env]]``. Non-env leaves
    are returned unchanged.
    """
    if config is None or not config.enabled:
        return envs
    if isinstance(envs, dict):
        return {key: maybe_wrap_env_tree(value, config) for key, value in envs.items()}
    if _is_wrappable_env(envs):
        return maybe_wrap_env(envs, config)
    return envs
