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

import contextlib
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv

from lerobot.faults.annotation import (
    FailureAnnotator,
    failure_type_id,
    frames_to_info_arrays,
    injector_injection_active,
    merge_annotation_into_info,
)
from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.factory import (
    ActionFaultInjector,
    ObsFaultInjector,
    RecoveryFaultInjector,
    SimInjectFaultInjector,
    make_action_fault_injector,
    make_midair_drop_fault,
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


def _make_annotator(config: FaultInjectionConfig, num_envs: int) -> FailureAnnotator:
    return FailureAnnotator(
        num_envs=num_envs,
        object_name=str(getattr(config, "object_name", "alphabet_soup_1") or "alphabet_soup_1"),
        basket_name=str(getattr(config, "basket_name", "basket_1") or "basket_1"),
        min_object_z=float(getattr(config, "min_object_z", 0.12) or 0.0),
        injector_type_id=failure_type_id(config.type),
    )


def _annotate_result(
    annotator: FailureAnnotator,
    env: Any,
    result: Any,
    *,
    injection_active: np.ndarray,
    recovery_active: np.ndarray | None = None,
) -> Any:
    """Stamp Gym ``info`` with per-env annotation arrays after physics."""
    annotator.update(env, injection_active=injection_active, recovery_active=recovery_active)
    arrays = frames_to_info_arrays(annotator.last_frames)
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return (obs, reward, terminated, truncated, merge_annotation_into_info(info, arrays))
    return result


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
        """Wire action and/or observation injectors and failure annotation."""
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
        self._annotator = _make_annotator(config, num_envs)

    def failure_annotation(self, env_idx: int = 0) -> dict:
        """Latest failure-annotation arrays for one vectorized sub-env."""
        return self._annotator.last_frames[env_idx]

    @property
    def unwrapped(self) -> Any:
        """Inner env without fault wrappers."""
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped env."""
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
        """Reset injectors, annotator, and the inner env; optionally corrupt observations."""
        result = self.env.reset(**kwargs)
        if self.action_injector is not None:
            self.action_injector.reset()
        if self.obs_injector is not None:
            self.obs_injector.reset()
        self._annotator.reset()
        return self._apply_obs_to_result(result, from_reset=True)

    def step(self, action):
        """Apply action faults, step the env, annotate failures, and notify on dones."""
        step_action = action
        if self.action_injector is not None:
            batch, was_single = _as_batch(np.asarray(action), self.num_envs)
            executed = self.action_injector.apply(batch)
            step_action = _from_batch(executed, was_single)

        result = self.env.step(step_action)
        result = self._apply_obs_to_result(result, from_reset=False)

        injection = np.array(
            [
                injector_injection_active(self.action_injector, i)
                or injector_injection_active(self.obs_injector, i)
                for i in range(self.num_envs)
            ],
            dtype=bool,
        )
        result = _annotate_result(self._annotator, self.env, result, injection_active=injection)

        dones = _extract_dones(result) if isinstance(result, tuple) else None
        if dones is not None:
            self._notify_dones(dones)
        return result

    def close(self):
        """Close fault loggers and the inner env."""
        self._close_loggers()
        return self.env.close()


class SimFaultEnvWrapper:
    """Wrap env for inject-only sim-state faults (object slip, EEF bump)."""

    def __init__(self, env: Any, config: FaultInjectionConfig):
        """Attach a sim-state fault injector (slip or EEF bump) to ``env``."""
        if type(env).__name__ == "AsyncVectorEnv":
            raise TypeError(
                "SimFaultEnvWrapper does not support AsyncVectorEnv "
                "(no per-sub-env sim access). Use SyncVectorEnv or a single LiberoEnv."
            )
        num_envs = _num_envs(env)
        fault = make_sim_inject_fault(config, num_envs=num_envs)
        if fault is None:
            raise ValueError("SimFaultEnvWrapper requires enabled type in {'object_slip', 'eef_bump'}.")
        self.env = env
        self.fault_config = config
        self.fault: SimInjectFaultInjector = fault
        self.num_envs = num_envs
        self._annotator = _make_annotator(config, num_envs)

    def failure_annotation(self, env_idx: int = 0) -> dict:
        """Latest failure-annotation arrays for one vectorized sub-env."""
        return self._annotator.last_frames[env_idx]

    @property
    def unwrapped(self) -> Any:
        """Inner env without fault wrappers."""
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped env."""
        return getattr(self.env, name)

    def _notify_dones(self, dones: np.ndarray) -> None:
        self.fault.notify_dones(dones.reshape(self.num_envs))

    def _close_logger(self) -> None:
        if self.fault.event_logger is not None:
            self.fault.event_logger.close()

    def reset(self, **kwargs):
        """Reset the sim fault and annotator, then the inner env."""
        result = self.env.reset(**kwargs)
        self.fault.reset()
        self._annotator.reset()
        return result

    def step(self, action):
        """Run sim injectors before physics, annotate, and notify on dones."""
        batch, was_single = _as_batch(np.asarray(action), self.num_envs)
        executed = self.fault.on_step(self.env, batch)
        step_action = _from_batch(executed, was_single)
        result = self.env.step(step_action)
        injection = np.array(
            [injector_injection_active(self.fault, i) for i in range(self.num_envs)],
            dtype=bool,
        )
        result = _annotate_result(self._annotator, self.env, result, injection_active=injection)
        dones = _extract_dones(result) if isinstance(result, tuple) else None
        if dones is not None:
            self._notify_dones(dones)
        return result

    def close(self):
        """Close the fault event logger and the inner env."""
        self._close_logger()
        return self.env.close()


class DropRecoveryEnvWrapper:
    """Wrap env for mid-air drop faults that need sim access and recovery control."""

    def __init__(self, env: Any, config: FaultInjectionConfig):
        """Attach mid-air drop + recovery control and failure annotation."""
        if type(env).__name__ == "AsyncVectorEnv":
            raise TypeError(
                "DropRecoveryEnvWrapper does not support AsyncVectorEnv "
                "(no per-sub-env sim access). Use SyncVectorEnv or a single LiberoEnv."
            )
        num_envs = _num_envs(env)
        fault = make_midair_drop_fault(config, num_envs=num_envs)
        if fault is None:
            raise ValueError("DropRecoveryEnvWrapper requires enabled type='midair_drop'.")
        self.env = env
        self.fault_config = config
        self.fault: RecoveryFaultInjector = fault
        self.num_envs = num_envs
        self.last_executed_action: np.ndarray | None = None
        self._annotator = _make_annotator(config, num_envs)

    def failure_annotation(self, env_idx: int = 0) -> dict:
        """Latest failure-annotation arrays for one vectorized sub-env."""
        return self._annotator.last_frames[env_idx]

    @property
    def unwrapped(self) -> Any:
        """Inner env without fault wrappers."""
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the wrapped env."""
        return getattr(self.env, name)

    def _notify_dones(self, dones: np.ndarray) -> None:
        self.fault.notify_dones(dones.reshape(self.num_envs))

    def _close_logger(self) -> None:
        if self.fault.event_logger is not None:
            self.fault.event_logger.close()

    def reset(self, **kwargs):
        """Reset drop/recovery state, install LIBERO hooks, and reset the inner env."""
        result = self.env.reset(**kwargs)
        self.fault.reset()
        self._annotator.reset()
        self._install_libero_no_reset_hook()
        return result

    def _install_libero_no_reset_hook(self) -> None:
        """Prevent LeRobot LiberoEnv from reset()-on-success during recovery."""
        try:
            from lerobot.faults.sim.libero import unwrap_libero_env

            target = self.env
            if hasattr(target, "envs"):
                target = target.envs[0]
            libero = unwrap_libero_env(target)
        except Exception:
            return
        if getattr(libero, "_faults_no_reset_hook", False):
            return
        fault = self.fault

        def step_without_success_reset(action):
            libero._ensure_env()
            assert libero._env is not None
            raw_obs, reward, done, info = libero._env.step(action)
            is_success = libero._env.check_success()
            terminated = bool(done or is_success)
            info = dict(info) if info is not None else {}
            info.update(
                {
                    "task": getattr(libero, "task", None),
                    "task_id": getattr(libero, "task_id", None),
                    "done": done,
                    "is_success": is_success,
                }
            )
            observation = libero._format_raw_obs(raw_obs)
            recovering = any(s.recovery_active for s in fault._states)
            if terminated and not recovering:
                libero.reset()
            truncated = False
            return observation, reward, terminated, truncated, info

        libero.step = step_without_success_reset  # type: ignore[method-assign]
        libero._faults_no_reset_hook = True

    def step(self, action):
        """Run drop/recovery control, suppress autoreset while recovering, and annotate."""
        batch, was_single = _as_batch(np.asarray(action), self.num_envs)
        executed = self.fault.on_step(self.env, batch)
        step_action = _from_batch(executed, was_single)
        self.last_executed_action = np.asarray(step_action, dtype=np.float32).copy()
        result = self.env.step(step_action)
        self.fault.after_physics_step(self.env)

        recovery_mask = [bool(s.recovery_active) for s in self.fault._states]
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
            term = np.atleast_1d(np.asarray(terminated)).copy()
            trunc = np.atleast_1d(np.asarray(truncated)).copy()
            for i, active in enumerate(recovery_mask):
                if i < len(term) and active:
                    term[i] = False
                    trunc[i] = False
            if np.asarray(terminated).ndim == 0:
                terminated = bool(term[0])
                truncated = bool(trunc[0])
            else:
                terminated = term.reshape(np.asarray(terminated).shape)
                truncated = trunc.reshape(np.asarray(truncated).shape)
            result = (obs, reward, terminated, truncated, info)

        self._clear_autoreset_latch(recovery_mask)

        injection = np.array(
            [injector_injection_active(self.fault, i) for i in range(self.num_envs)],
            dtype=bool,
        )
        recovery = np.array(recovery_mask, dtype=bool)
        result = _annotate_result(
            self._annotator,
            self.env,
            result,
            injection_active=injection,
            recovery_active=recovery,
        )

        dones = _extract_dones(result) if isinstance(result, tuple) else None
        if dones is not None:
            self._notify_dones(dones)
        return result

    def _clear_autoreset_latch(self, recovery_mask: list[bool]) -> None:
        env: Any = self.env
        seen: set[int] = set()
        while env is not None and id(env) not in seen:
            seen.add(id(env))
            flags = getattr(env, "_autoreset_envs", None)
            if flags is not None:
                arr = np.asarray(flags)
                for i, active in enumerate(recovery_mask):
                    if active and i < arr.shape[0]:
                        arr[i] = False
                with contextlib.suppress(Exception):
                    env._autoreset_envs = arr
                for attr in ("_terminations", "_truncations"):
                    buf = getattr(env, attr, None)
                    if buf is not None:
                        b = np.asarray(buf)
                        for i, active in enumerate(recovery_mask):
                            if active and i < b.shape[0]:
                                b[i] = False
                        with contextlib.suppress(Exception):
                            setattr(env, attr, b)
                return
            env = getattr(env, "env", None)

    def loss_mask(self, env_idx: int = 0) -> float:
        """Training loss weight for the current step (0 during injection)."""
        return self.fault.loss_mask_for_env(env_idx)

    def consume_policy_reset(self, env_idx: int = 0) -> bool:
        """Return and clear a one-shot policy hidden-state reset request."""
        return self.fault.consume_policy_reset(env_idx)

    def close(self):
        """Close the fault event logger and the inner env."""
        self._close_logger()
        return self.env.close()


def maybe_wrap_env(env: Any, config: FaultInjectionConfig | None) -> Any:
    """Wrap ``env`` when fault config is enabled; otherwise return ``env`` unchanged."""
    if config is None or not config.enabled:
        return env
    if not _is_wrappable_env(env):
        return env
    if config.type == "midair_drop":
        return DropRecoveryEnvWrapper(env, config)
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
