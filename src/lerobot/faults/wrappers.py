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
from lerobot.faults.factory import (
    ActionFaultInjector,
    ObsFaultInjector,
    SimInjectFaultInjector,
    make_action_fault_injector,
    make_obs_fault_injector,
    make_sim_inject_fault,
)


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
    """Apply optional action and/or observation fault injectors around the env API."""

    def __init__(self, env: Any, config: FaultInjectionConfig):
        num_envs = _num_envs(env)
        action_injector = make_action_fault_injector(config, num_envs=num_envs)
        obs_injector = make_obs_fault_injector(config, num_envs=num_envs)
        if action_injector is None and obs_injector is None:
            raise ValueError("FaultEnvWrapper requires at least one enabled fault injector.")
        self.env = env
        self.fault_config = config
        self.action_injector: ActionFaultInjector | None = action_injector
        self.obs_injector: ObsFaultInjector | None = obs_injector
        self.num_envs = num_envs

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _apply_obs(self, obs: Any, *, from_reset: bool) -> Any:
        if self.obs_injector is None:
            return obs
        return self.obs_injector.apply_obs(obs, from_reset=from_reset)

    def _apply_obs_to_result(self, result: Any, *, from_reset: bool) -> Any:
        if isinstance(result, tuple) and len(result) >= 1:
            obs = self._apply_obs(result[0], from_reset=from_reset)
            return (obs, *result[1:])
        return self._apply_obs(result, from_reset=from_reset)

    def _notify_dones(self, dones: np.ndarray) -> None:
        if self.action_injector is not None:
            self.action_injector.notify_dones(dones.reshape(self.num_envs))
        if self.obs_injector is not None:
            self.obs_injector.notify_dones(dones.reshape(self.num_envs))

    def _close_loggers(self) -> None:
        for injector in (self.action_injector, self.obs_injector):
            if injector is None:
                continue
            closer = getattr(injector, "close", None)
            if callable(closer):
                closer()
                continue
            if injector.event_logger is not None:
                injector.event_logger.close()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if self.action_injector is not None:
            self.action_injector.reset()
        if self.obs_injector is not None:
            self.obs_injector.reset()
        return self._apply_obs_to_result(result, from_reset=True)

    def step(self, action):
        step_action = action
        if self.action_injector is not None:
            batch, was_single = _as_batch(np.asarray(action), self.num_envs)
            executed = self.action_injector.apply(batch)
            step_action = _from_batch(executed, was_single)

        result = self.env.step(step_action)
        result = self._apply_obs_to_result(result, from_reset=False)

        dones = _extract_dones(result) if isinstance(result, tuple) else None
        if dones is not None:
            self._notify_dones(dones)
        return result

    def close(self):
        self._close_loggers()
        return self.env.close()


class SimFaultEnvWrapper:
    """Wrap env for inject-only sim-state faults (object slip, EEF bump)."""

    def __init__(self, env: Any, config: FaultInjectionConfig):
        if type(env).__name__ == "AsyncVectorEnv":
            raise TypeError(
                "SimFaultEnvWrapper does not support AsyncVectorEnv "
                "(no per-sub-env sim access). Use SyncVectorEnv or a single LiberoEnv."
            )
        num_envs = _num_envs(env)
        fault = make_sim_inject_fault(config, num_envs=num_envs)
        if fault is None:
            raise ValueError(
                "SimFaultEnvWrapper requires enabled type in {'object_slip', 'eef_bump'}."
            )
        self.env = env
        self.fault_config = config
        self.fault: SimInjectFaultInjector = fault
        self.num_envs = num_envs

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def _notify_dones(self, dones: np.ndarray) -> None:
        self.fault.notify_dones(dones.reshape(self.num_envs))

    def _close_logger(self) -> None:
        if self.fault.event_logger is not None:
            self.fault.event_logger.close()

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self.fault.reset()
        return result

    def step(self, action):
        batch, was_single = _as_batch(np.asarray(action), self.num_envs)
        executed = self.fault.on_step(self.env, batch)
        step_action = _from_batch(executed, was_single)
        result = self.env.step(step_action)
        dones = _extract_dones(result) if isinstance(result, tuple) else None
        if dones is not None:
            self._notify_dones(dones)
        return result

    def close(self):
        self._close_logger()
        return self.env.close()


def maybe_wrap_env(env: Any, config: FaultInjectionConfig | None) -> Any:
    """Wrap ``env`` when fault config is enabled; otherwise return ``env`` unchanged."""
    if config is None or not config.enabled:
        return env
    if not _is_wrappable_env(env):
        return env
    if config.type in {"object_slip", "eef_bump"}:
        return SimFaultEnvWrapper(env, config)
    return FaultEnvWrapper(env, config)


def maybe_wrap_env_tree(envs: Any, config: FaultInjectionConfig | None) -> Any:
    """Wrap Gym envs inside LeRobot's nested ``make_env`` return structure."""
    if config is None or not config.enabled:
        return envs
    if isinstance(envs, dict):
        return {key: maybe_wrap_env_tree(value, config) for key, value in envs.items()}
    if _is_wrappable_env(envs):
        return maybe_wrap_env(envs, config)
    return envs
