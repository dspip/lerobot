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

"""Shared frame logging helpers for drop-recovery dataset rows."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from lerobot.faults.annotation import default_failure_frame
from lerobot.faults.recovery.loss_mask import loss_mask_from_fault

__all__ = [
    "DatasetStepLogger",
    "annotation_for_datagen_env",
    "log_fault_recovery_step",
    "log_step_from_recovery_env",
    "loss_mask_for_datagen_env",
    "loss_mask_for_recovery_env",
    "should_log_sim_step",
]


def should_log_sim_step(
    sim_step: int,
    *,
    recording_stride: int,
    force_drop_injection: bool = False,
) -> bool:
    if force_drop_injection:
        return True
    stride = max(int(recording_stride), 1)
    return sim_step % stride == 0


def loss_mask_for_datagen_env(env: Any, *, is_drop_episode: bool, env_idx: int = 0) -> float:
    if not is_drop_episode:
        return 1.0
    return loss_mask_for_recovery_env(env, env_idx)


def annotation_for_datagen_env(env: Any, *, is_drop_episode: bool, env_idx: int = 0) -> dict[str, Any]:
    if not is_drop_episode:
        return default_failure_frame()
    return env.failure_annotation(env_idx)


class DatasetStepLogger(Protocol):
    def log_step(
        self,
        observation_dict: dict[str, Any],
        action: np.ndarray,
        task: str,
        loss_mask: float,
        phase: str | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> None: ...


def log_fault_recovery_step(
    logger: DatasetStepLogger,
    *,
    observation_dict: dict[str, Any],
    executed_action: np.ndarray | list[float],
    task: str,
    loss_mask: float,
    phase: str | None = None,
    annotation: dict[str, Any] | None = None,
) -> None:
    logger.log_step(
        observation_dict,
        executed_action,
        task,
        loss_mask,
        phase=phase,
        annotation=annotation,
    )


def loss_mask_for_recovery_env(env: Any, env_idx: int = 0) -> float:
    fault = env.fault
    state = fault._states[env_idx]
    return loss_mask_from_fault(
        triggered=bool(state.triggered),
        drop_injection_step=bool(state.drop_injection_step),
        recovery_active=bool(state.recovery_active),
        post_drop_dwell_step=bool(getattr(state, "post_drop_dwell_step", False)),
    )


def log_step_from_recovery_env(
    logger: DatasetStepLogger,
    env: Any,
    *,
    observation: dict[str, Any] | None,
    executed_action: np.ndarray | list[float],
    task: str,
    phase: str | None,
    sim_step: int,
    recording_stride: int,
    env_idx: int = 0,
) -> None:
    state = env.fault._states[env_idx]
    is_drop_frame = bool(state.drop_injection_step)
    if sim_step % recording_stride != 0 and not is_drop_frame:
        return
    if observation is None:
        return
    executed = executed_action
    if np.asarray(executed).ndim == 2:
        executed = np.asarray(executed)[0]
    mask = loss_mask_for_recovery_env(env, env_idx)
    annotation = env.failure_annotation(env_idx)
    log_fault_recovery_step(
        logger,
        observation_dict=observation,
        executed_action=executed,
        task=task,
        loss_mask=mask,
        phase=phase,
        annotation=annotation,
    )
