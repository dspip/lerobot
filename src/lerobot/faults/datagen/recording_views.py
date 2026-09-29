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

"""Multi-view datagen recording: stride sampling and pulse OR-accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "PULSE_ANNOTATION_KEYS",
    "ViewRecordingState",
    "build_datagen_frame_labels",
    "master_dataset_directory",
    "master_dataset_repo_id",
    "mujoco_sim_time_s",
    "offer_tick_to_views",
    "should_log_view_tick",
    "tick_index_at_or_after_frame",
]

PULSE_ANNOTATION_KEYS: frozenset[str] = frozenset({"failure_onset", "drop_release"})


def master_dataset_directory(dataset_root: Any) -> Any:
    """Sibling path for the 20 Hz master view (``<dataset>_20hz``)."""
    from pathlib import Path

    root = Path(dataset_root)
    return root.parent / f"{root.name}_20hz"


def master_dataset_repo_id(repo_id: str) -> str:
    return f"{repo_id}_20hz"


def should_log_view_tick(tick: int, *, stride: int) -> bool:
    """Return whether ``tick`` is recorded at ``stride`` (uniform spacing, no forced frames)."""
    step = max(int(stride), 1)
    return int(tick) % step == 0


def mujoco_sim_time_s(rs_env: Any) -> float:
    """MuJoCo simulation time in seconds."""
    sim = getattr(rs_env, "sim", None)
    if sim is None:
        raise RuntimeError("mujoco sim time unavailable: rs_env has no sim")
    data = getattr(sim, "data", None)
    if data is None:
        raise RuntimeError("mujoco sim time unavailable: sim has no data")
    return float(data.time)


def build_datagen_frame_labels(
    state: Any,
    *,
    tick: int,
    is_drop_episode: bool,
    prev_triggered: bool = False,
    object_z: float | None = None,
) -> dict[str, np.ndarray]:
    """Per-control-tick datagen label scalars (before view pulse merging).

    ``drop_injection_step`` stays set for the whole fall, so the release pulse is
    the rising edge of ``triggered`` (``prev_triggered`` is the previous tick's value).
    """
    triggered = bool(is_drop_episode and getattr(state, "triggered", False))
    drop_release = bool(triggered and not prev_triggered)
    falling = bool(getattr(state, "falling", False))
    drop_event = bool(
        triggered and (falling or getattr(state, "drop_injection_step", False))
    )
    if not is_drop_episode:
        attempt_index = 0
    elif not triggered:
        attempt_index = 0
    elif drop_event:
        attempt_index = 0
    else:
        attempt_index = 1

    return {
        "tick_index": np.array([int(tick)], dtype=np.int64),
        "drop_release": np.array([drop_release], dtype=bool),
        "drop_event": np.array([drop_event], dtype=bool),
        "attempt_index": np.array([int(attempt_index)], dtype=np.int64),
        "_object_z": object_z,
    }


def tick_index_at_or_after_frame(tick: int, *, stride: int) -> int:
    """First logged frame index (0-based) whose ``tick_index`` is at or after ``tick``."""
    step = max(int(stride), 1)
    if tick % step == 0:
        return tick // step
    return (tick // step) + 1


@dataclass
class ViewRecordingState:
    """One dataset view (stride + pulse state between logged frames)."""

    name: str
    stride: int
    pulse_or: dict[str, bool] = field(default_factory=dict)
    last_logged_sim_time_s: float | None = None
    frames_logged: int = 0

    def reset_episode(self) -> None:
        self.pulse_or = {}
        self.last_logged_sim_time_s = None
        self.frames_logged = 0

    def accumulate_pulses(self, annotation: dict[str, Any], raw_labels: dict[str, np.ndarray]) -> None:
        for key in PULSE_ANNOTATION_KEYS:
            val = False
            if key in annotation:
                val = bool(np.asarray(annotation[key]).reshape(-1)[0])
            elif key in raw_labels:
                val = bool(np.asarray(raw_labels[key]).reshape(-1)[0])
            if val:
                self.pulse_or[key] = True

    def merge_pulses_into(self, annotation: dict[str, Any], raw_labels: dict[str, np.ndarray]) -> dict[str, Any]:
        merged = dict(annotation)
        for key in PULSE_ANNOTATION_KEYS:
            val = bool(self.pulse_or.get(key, False))
            if key in merged:
                spec_shape = np.asarray(merged[key]).shape
                merged[key] = np.array([val]).reshape(spec_shape)
            elif key in raw_labels:
                merged[key] = np.array([val], dtype=raw_labels[key].dtype)
        for label_key, label_val in raw_labels.items():
            if label_key.startswith("_"):
                continue
            if label_key in PULSE_ANNOTATION_KEYS:
                continue
            merged[label_key] = label_val
        return merged

    def clear_pulses(self) -> None:
        self.pulse_or = {}


@dataclass
class _ViewLogDecision:
    view: ViewRecordingState
    log: bool
    merged_annotation: dict[str, Any] | None = None


def offer_tick_to_views(
    views: tuple[ViewRecordingState, ...],
    *,
    tick: int,
    sim_time_s: float,
    annotation: dict[str, Any],
    raw_labels: dict[str, np.ndarray],
) -> tuple[_ViewLogDecision, ...]:
    """Update pulse accumulators; return which views log this tick."""
    for view in views:
        view.accumulate_pulses(annotation, raw_labels)

    decisions: list[_ViewLogDecision] = []
    for view in views:
        if not should_log_view_tick(tick, stride=view.stride):
            decisions.append(_ViewLogDecision(view=view, log=False))
            continue
        if view.last_logged_sim_time_s is not None and sim_time_s < view.last_logged_sim_time_s - 1e-9:
            raise RuntimeError(
                "MuJoCo sim time decreased between logged frames in the same episode "
                f"(view={view.name!r}, prev={view.last_logged_sim_time_s}, now={sim_time_s}); "
                "refusing to record a post-autoreset frame"
            )
        merged = view.merge_pulses_into(annotation, raw_labels)
        view.last_logged_sim_time_s = sim_time_s
        view.frames_logged += 1
        view.clear_pulses()
        decisions.append(_ViewLogDecision(view=view, log=True, merged_annotation=merged))
    return tuple(decisions)
