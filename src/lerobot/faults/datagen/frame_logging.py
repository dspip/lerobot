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

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from lerobot.faults.datagen.dataset_writer import DatagenEpisodeSession

__all__ = [
    "POST_STEP_LOGGING_CONTRACT",
    "DatasetStepLogger",
    "log_fault_recovery_step",
    "log_post_step_to_session",
    "log_step_from_recovery_env",
    "loss_mask_for_datagen_env",
    "should_log_sim_step",
]

POST_STEP_LOGGING_CONTRACT = (
    "Datagen frames use POST env.step() state: post_step_observation is the observation "
    "returned by env.step, and executed_action is env.last_executed_action when present "
    "(otherwise the action tensor passed into env.step)."
)


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
    return float(env.loss_mask(env_idx))


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


def log_post_step_to_session(
    session: DatagenEpisodeSession,
    *,
    env: Any,
    post_step_observation: dict[str, Any],
    executed_action: np.ndarray | list[float],
    task: str,
    phase: str,
    is_drop_episode: bool,
    env_idx: int = 0,
    observation_to_frame: Any | None = None,
) -> None:
    """Log one frame using POST-step env state (see POST_STEP_LOGGING_CONTRACT)."""
    if observation_to_frame is None:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.faults.recovery.dataset_logger import libero_obs_to_frame

        def observation_to_frame(obs: dict[str, Any]) -> dict[str, Any]:
            return libero_obs_to_frame(preprocess_observation(obs))

    executed = executed_action
    if np.asarray(executed).ndim == 2:
        executed = np.asarray(executed)[0]
    frame = observation_to_frame(post_step_observation)
    mask = loss_mask_for_datagen_env(env, is_drop_episode=is_drop_episode, env_idx=env_idx)
    annotation = env.failure_annotation(env_idx)
    session.log_step(
        frame,
        executed,
        task,
        mask,
        phase=phase,
        annotation=annotation,
    )


def log_step_from_recovery_env(
    session: DatagenEpisodeSession,
    env: Any,
    *,
    post_step_observation: dict[str, Any] | None,
    executed_action: np.ndarray | list[float],
    task: str,
    phase: str | None,
    sim_step: int,
    recording_stride: int,
    is_drop_episode: bool,
    env_idx: int = 0,
) -> None:
    state = env.fault._states[env_idx]
    is_drop_frame = bool(is_drop_episode and state.drop_injection_step)
    if sim_step % recording_stride != 0 and not is_drop_frame:
        return
    if post_step_observation is None:
        return
    log_post_step_to_session(
        session,
        env=env,
        post_step_observation=post_step_observation,
        executed_action=executed_action,
        task=task,
        phase=phase or "",
        is_drop_episode=is_drop_episode,
        env_idx=env_idx,
    )
