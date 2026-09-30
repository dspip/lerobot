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

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from fault_system.datagen.dataset_writer import DatagenEpisodeSession

__all__ = [
    "POST_STEP_LOGGING_CONTRACT",
    "DatagenPostStepLogContext",
    "DatasetStepLogger",
    "build_datagen_post_step_log_context",
    "log_fault_recovery_step",
    "log_post_step_to_session",
    "loss_mask_for_datagen_env",
    "should_log_sim_step",
    "should_log_view_tick",
]

from fault_system.datagen.recording_views import (  # noqa: E402
    build_datagen_frame_labels,
    mujoco_sim_time_s,
    offer_tick_to_views,
    should_log_view_tick,
)

DATAGEN_POST_STEP_ANNOTATION_KEY = "datagen_post_step"


@dataclass(frozen=True)
class DatagenPostStepLogContext:
    """POST env.step() metadata attached to each datagen dataset row."""

    sim_step: int
    phase: str
    is_drop_episode: bool
    drop_injection_step: bool
    post_drop_dwell_step: bool
    recovery_active: bool
    loss_mask: float


def build_datagen_post_step_log_context(
    env: Any,
    *,
    sim_step: int,
    phase: str,
    is_drop_episode: bool,
    env_idx: int = 0,
) -> DatagenPostStepLogContext:
    """Summarize fault phase and loss mask after one env step."""
    state = env.fault.post_step_state(env_idx)
    drop_injection_step = bool(is_drop_episode and state.drop_injection_step)
    post_drop_dwell_step = bool(
        is_drop_episode and state.triggered and not state.recovery_active and not state.drop_injection_step
    )
    recovery_active = bool(is_drop_episode and state.recovery_active)
    loss_mask = loss_mask_for_datagen_env(env, is_drop_episode=is_drop_episode, env_idx=env_idx)
    return DatagenPostStepLogContext(
        sim_step=int(sim_step),
        phase=str(phase),
        is_drop_episode=bool(is_drop_episode),
        drop_injection_step=drop_injection_step,
        post_drop_dwell_step=post_drop_dwell_step,
        recovery_active=recovery_active,
        loss_mask=float(loss_mask),
    )


POST_STEP_LOGGING_CONTRACT = (
    "Datagen frames use POST env.step() state: post_step_observation is the observation "
    "returned by env.step, and executed_action is env.last_executed_action when present "
    "(otherwise the action tensor passed into env.step)."
)


def should_log_sim_step(
    sim_step: int,
    *,
    recording_stride: int,
) -> bool:
    """Return whether this sim step should be written at the configured stride."""
    return should_log_view_tick(sim_step, stride=recording_stride)


def loss_mask_for_datagen_env(env: Any, *, is_drop_episode: bool, env_idx: int = 0) -> float:
    """Return ``1.0`` for nominal episodes or the wrapper loss mask for drop episodes."""
    if not is_drop_episode:
        return 1.0
    return float(env.loss_mask(env_idx))


class DatasetStepLogger(Protocol):
    """Minimal logger surface used by datagen frame helpers."""

    def log_step(
        self,
        observation_dict: dict[str, Any],
        action: np.ndarray,
        task: str,
        loss_mask: float,
        phase: str | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> None:
        """Append one dataset frame with optional phase and annotation metadata."""
        ...


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
    """Forward one POST-step frame to any ``DatasetStepLogger`` implementation."""
    logger.log_step(
        observation_dict,
        executed_action,
        task,
        loss_mask,
        phase=phase,
        annotation=annotation,
    )


def _object_z_for_logging(rs_env: Any, object_name: str | None) -> float | None:
    if rs_env is None or not object_name:
        return None
    try:
        from fault_system.sim.libero import get_object_pose

        return float(get_object_pose(rs_env, object_name)["pos"][2])
    except Exception:
        return None


def log_post_step_to_session(
    session: DatagenEpisodeSession,
    *,
    env: Any,
    post_step_observation: dict[str, Any],
    executed_action: np.ndarray | list[float],
    task: str,
    phase: str,
    is_drop_episode: bool,
    sim_step: int,
    env_idx: int = 0,
    observation_to_frame: Any | None = None,
    rs_env: Any | None = None,
    object_name: str | None = None,
) -> None:
    """Log one control tick across all recording views (POST-step contract)."""
    if observation_to_frame is None:
        from lerobot.envs.utils import preprocess_observation
        from fault_system.recovery.dataset_logger import libero_obs_to_frame

        def observation_to_frame(obs: dict[str, Any]) -> dict[str, Any]:
            return libero_obs_to_frame(preprocess_observation(obs))

    executed = executed_action
    if np.asarray(executed).ndim == 2:
        executed = np.asarray(executed)[0]
    log_ctx = build_datagen_post_step_log_context(
        env,
        sim_step=sim_step,
        phase=phase,
        is_drop_episode=is_drop_episode,
        env_idx=env_idx,
    )
    mask = log_ctx.loss_mask
    state = env.fault.post_step_state(env_idx)
    object_z = _object_z_for_logging(rs_env, object_name)
    raw_labels = build_datagen_frame_labels(
        state,
        tick=sim_step,
        is_drop_episode=is_drop_episode,
        prev_triggered=session.last_tick_triggered,
        object_z=object_z,
    )
    session.last_tick_triggered = bool(is_drop_episode and getattr(state, "triggered", False))
    session.record_tick_metadata(
        tick=sim_step,
        drop_release=bool(raw_labels["drop_release"][0]),
        drop_event=bool(raw_labels["drop_event"][0]),
        attempt_index=int(raw_labels["attempt_index"][0]),
        object_z=object_z,
    )
    base_annotation = dict(env.failure_annotation(env_idx))
    base_annotation[DATAGEN_POST_STEP_ANNOTATION_KEY] = asdict(log_ctx)

    view_states = tuple(v.view_state for v in session.recording_views)
    if rs_env is None:
        rs_env = getattr(env, "rs_env", None)
    if rs_env is None:
        try:
            from fault_system.sim.libero import get_robosuite_env

            rs_env = get_robosuite_env(env, env_idx)
        except Exception:
            rs_env = None
    sim_time_s = mujoco_sim_time_s(rs_env) if rs_env is not None else float(sim_step)

    decisions = offer_tick_to_views(
        view_states,
        tick=sim_step,
        sim_time_s=sim_time_s,
        annotation=base_annotation,
        raw_labels=raw_labels,
    )
    any_log = any(d.log for d in decisions)
    frame = observation_to_frame(post_step_observation) if any_log else None

    for binding, decision in zip(session.recording_views, decisions, strict=True):
        if not decision.log or frame is None or decision.merged_annotation is None:
            continue
        session.log_step(
            frame,
            executed,
            task,
            mask,
            phase=phase,
            annotation=decision.merged_annotation,
            view_name=binding.name,
        )
