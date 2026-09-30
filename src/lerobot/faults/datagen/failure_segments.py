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

"""Sidecar ``meta/failure_segments.parquet`` for datagen attempt metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.faults.datagen.recording_views import release_and_landing_ticks, tick_index_at_or_after_frame

__all__ = [
    "AttemptTickSnapshot",
    "FailureSegmentRow",
    "FailureSegmentsWriter",
    "attempt_rows_from_tick_snapshots",
    "merge_failure_segment_rows",
]

FAILURE_SEGMENTS_REL_PATH = Path("meta") / "failure_segments.parquet"


@dataclass(frozen=True)
class AttemptTickSnapshot:
    tick: int
    drop_release: bool
    drop_event: bool
    attempt_index: int
    object_z: float | None = None


@dataclass(frozen=True)
class FailureSegmentRow:
    episode_index: int
    logical_episode_index: int
    object_name: str
    task_id: int
    controller: str
    post_drop_mode: str
    is_drop_episode: bool
    attempt_index: int
    attempt_start_frame: int
    attempt_end_frame: int
    has_drop: bool
    release_tick: int | None
    landing_tick: int | None
    release_frame: int | None
    landing_frame: int | None
    release_time_s: float | None
    landing_time_s: float | None
    object_z_at_release: float | None
    episode_seed: int
    layout_seed: int


def attempt_rows_from_tick_snapshots(
    snapshots: list[AttemptTickSnapshot],
    *,
    stride: int,
    control_hz: int,
    episode_index: int,
    logical_episode_index: int,
    object_name: str,
    task_id: int,
    controller: str,
    post_drop_mode: str,
    is_drop_episode: bool,
    episode_seed: int,
    layout_seed: int,
) -> list[FailureSegmentRow]:
    """Build sidecar rows from per-tick labels captured during logging."""
    if not snapshots:
        return []

    ticks = [s.tick for s in snapshots]
    release_tick, landing_tick, _last_event_tick = release_and_landing_ticks(snapshots)
    object_z_at_release: float | None = None
    for snap in snapshots:
        if snap.drop_release and release_tick is not None and snap.tick == release_tick:
            object_z_at_release = snap.object_z
            break

    has_drop = release_tick is not None
    if not is_drop_episode or not has_drop:
        end_tick = ticks[-1]
        end_frame = tick_index_at_or_after_frame(end_tick, stride=stride)
        return [
            FailureSegmentRow(
                episode_index=episode_index,
                logical_episode_index=logical_episode_index,
                object_name=object_name,
                task_id=task_id,
                controller=controller,
                post_drop_mode=post_drop_mode,
                is_drop_episode=is_drop_episode,
                attempt_index=0,
                attempt_start_frame=0,
                attempt_end_frame=end_frame,
                has_drop=False,
                release_tick=None,
                landing_tick=None,
                release_frame=None,
                landing_frame=None,
                release_time_s=None,
                landing_time_s=None,
                object_z_at_release=None,
                episode_seed=episode_seed,
                layout_seed=layout_seed,
            )
        ]

    assert release_tick is not None
    landing_boundary = landing_tick if landing_tick is not None else ticks[-1]
    attempt0_end_tick = landing_boundary
    attempt0_end_frame = tick_index_at_or_after_frame(attempt0_end_tick, stride=stride)
    attempt1_start_tick = landing_tick if landing_tick is not None else release_tick + 1
    attempt1_start_frame = tick_index_at_or_after_frame(attempt1_start_tick, stride=stride)
    end_frame = tick_index_at_or_after_frame(ticks[-1], stride=stride)

    release_frame = tick_index_at_or_after_frame(release_tick, stride=stride)
    landing_frame = (
        tick_index_at_or_after_frame(landing_tick, stride=stride) if landing_tick is not None else None
    )
    hz = float(control_hz)

    rows = [
        FailureSegmentRow(
            episode_index=episode_index,
            logical_episode_index=logical_episode_index,
            object_name=object_name,
            task_id=task_id,
            controller=controller,
            post_drop_mode=post_drop_mode,
            is_drop_episode=True,
            attempt_index=0,
            attempt_start_frame=0,
            attempt_end_frame=attempt0_end_frame,
            has_drop=True,
            release_tick=release_tick,
            landing_tick=landing_tick,
            release_frame=release_frame,
            landing_frame=landing_frame,
            release_time_s=release_tick / hz,
            landing_time_s=(landing_tick / hz) if landing_tick is not None else None,
            object_z_at_release=object_z_at_release,
            episode_seed=episode_seed,
            layout_seed=layout_seed,
        ),
        FailureSegmentRow(
            episode_index=episode_index,
            logical_episode_index=logical_episode_index,
            object_name=object_name,
            task_id=task_id,
            controller=controller,
            post_drop_mode=post_drop_mode,
            is_drop_episode=True,
            attempt_index=1,
            attempt_start_frame=attempt1_start_frame,
            attempt_end_frame=end_frame,
            has_drop=False,
            release_tick=None,
            landing_tick=None,
            release_frame=None,
            landing_frame=None,
            release_time_s=None,
            landing_time_s=None,
            object_z_at_release=None,
            episode_seed=episode_seed,
            layout_seed=layout_seed,
        ),
    ]
    if landing_tick is None:
        return [rows[0]]
    return rows


def merge_failure_segment_rows(existing: list[FailureSegmentRow], new: list[FailureSegmentRow]) -> list[FailureSegmentRow]:
    return list(existing) + list(new)


@dataclass
class FailureSegmentsWriter:
    """Accumulates sidecar rows per dataset root and writes parquet on finalize."""

    dataset_root: Path
    stride: int
    control_hz: int
    rows: list[FailureSegmentRow] = field(default_factory=list)
    _episode_snapshots: list[AttemptTickSnapshot] = field(default_factory=list)

    def reset_episode_snapshots(self) -> None:
        self._episode_snapshots = []

    @property
    def episode_snapshots(self) -> tuple[AttemptTickSnapshot, ...]:
        return tuple(self._episode_snapshots)

    def record_tick(
        self,
        *,
        tick: int,
        drop_release: bool,
        drop_event: bool,
        attempt_index: int,
        object_z: float | None,
    ) -> None:
        self._episode_snapshots.append(
            AttemptTickSnapshot(
                tick=int(tick),
                drop_release=bool(drop_release),
                drop_event=bool(drop_event),
                attempt_index=int(attempt_index),
                object_z=object_z,
            )
        )

    def commit_episode(
        self,
        *,
        episode_index: int,
        logical_episode_index: int,
        object_name: str,
        task_id: int,
        controller: str,
        post_drop_mode: str,
        is_drop_episode: bool,
        episode_seed: int,
        layout_seed: int,
    ) -> None:
        new_rows = attempt_rows_from_tick_snapshots(
            self._episode_snapshots,
            stride=self.stride,
            control_hz=self.control_hz,
            episode_index=episode_index,
            logical_episode_index=logical_episode_index,
            object_name=object_name,
            task_id=task_id,
            controller=controller,
            post_drop_mode=post_drop_mode,
            is_drop_episode=is_drop_episode,
            episode_seed=episode_seed,
            layout_seed=layout_seed,
        )
        self.rows = merge_failure_segment_rows(self.rows, new_rows)
        self.reset_episode_snapshots()

    def discard_episode(self) -> None:
        self.reset_episode_snapshots()

    def load_existing(self) -> None:
        """Replace in-memory rows with the parquet already on disk, if any."""
        path = Path(self.dataset_root) / FAILURE_SEGMENTS_REL_PATH
        if not path.is_file():
            return
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("failure_segments load requires pyarrow") from exc
        table = pq.read_table(path)
        self.rows = [FailureSegmentRow(**row) for row in table.to_pylist()]

    def finalize(self) -> Path | None:
        if not self.rows:
            return None
        path = Path(self.dataset_root) / FAILURE_SEGMENTS_REL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("failure_segments finalize requires pyarrow") from exc

        table = pa.Table.from_pylist([row.__dict__ for row in self.rows])
        pq.write_table(table, path)
        return path
