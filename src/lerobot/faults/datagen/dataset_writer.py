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

"""Unified dataset ownership for drop datagen matrix runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.manifest import (
    EpisodeMetadataRow,
    RunManifest,
    RunStatus,
    build_episode_metadata_row,
    write_run_manifest_atomic,
)
from lerobot.faults.datagen.failure_segments import FailureSegmentsWriter
from lerobot.faults.datagen.recipe import DropDatagenRecipe, EpisodeSeedManifest, PostDropMode
from lerobot.faults.datagen.recording_views import (
    ViewRecordingState,
    master_dataset_directory,
    master_dataset_repo_id,
)
from lerobot.faults.recovery.fps import recording_stride

__all__ = [
    "DatagenEpisodeSession",
    "LoggerFactory",
    "RunDatasetFinalizeError",
    "RunDatasetWriter",
    "StaleRunOutputError",
    "evaluate_datagen_keep",
    "variant_dataset_directory",
    "variant_repo_id",
]

LoggerFactory = Callable[..., Any]

VariantKey = str
_SINGLE_DATASET_KEY = "dataset"


class StaleRunOutputError(FileExistsError):
    """Raised when a new run targets a non-empty output directory."""


class RunDatasetFinalizeError(RuntimeError):
    """Raised when one or more variant loggers fail to finalize."""

    def __init__(self, errors: list[BaseException]) -> None:
        """Attach the logger finalize exceptions that caused the failure."""
        self.errors = list(errors)
        msg = "; ".join(str(exc) for exc in errors)
        super().__init__(f"dataset logger finalize failed: {msg}")


def variant_dataset_directory(recipe: DropDatagenRecipe, manifest: EpisodeSeedManifest) -> Path:
    """Return the single LeRobot dataset root for the recording job."""
    del manifest
    return Path(recipe.recording.output_dir) / "dataset"


def variant_repo_id(recipe: DropDatagenRecipe, manifest: EpisodeSeedManifest) -> str:
    """Hub-style repo id for the single recorded dataset."""
    del manifest
    return str(recipe.name)


def _variant_key(manifest: EpisodeSeedManifest) -> VariantKey:
    del manifest
    return _SINGLE_DATASET_KEY


def assert_fresh_run_output_dir(recipe: DropDatagenRecipe) -> None:
    root = Path(recipe.recording.output_dir)
    if not root.exists():
        return
    if any(root.iterdir()):
        raise StaleRunOutputError(
            f"Recording output {root} is not empty. "
            "Use a fresh output_dir for a new run (resume is not supported)."
        )


def evaluate_datagen_keep(request: EpisodeRequest, result: EpisodeResult) -> tuple[bool, str | None]:
    """Decide whether logged frames should be committed for this matrix outcome."""
    if result.outcome in ("nominal_grasp_missed", "nominal_carry_stalled"):
        return False, result.outcome
    plan = request.paired_plan
    mode = request.manifest.post_drop_mode
    if not plan.drop_decision.drop:
        if result.success:
            return True, "nominal_placement_success"
        return False, result.outcome or "nominal_failed"
    if mode is PostDropMode.IMMEDIATE_SMOLVLA:
        if result.outcome in {
            "no_eligible_path",
            "nominal_completed_without_drop",
            "runtime_keepout",
        }:
            return False, result.outcome
        if result.success:
            return True, "smolvla_placed_after_drop"
        return True, "planned_smolvla_fail"
    if result.success:
        return True, "recovery_success"
    return False, result.outcome or "recovery_failed"


@dataclass
class _BoundRecordingView:
    name: str
    dataset_root: Path
    logger: Any
    policy_fps: int
    stride: int
    view_state: ViewRecordingState
    segments: FailureSegmentsWriter


@dataclass
class DatagenEpisodeSession:
    """One open episode buffer on a shared variant logger (optional 20 Hz master sibling)."""

    manifest: EpisodeSeedManifest
    dataset_root: Path
    repo_id: str
    logger: Any
    policy_fps: int
    control_hz: int = 20
    recipe: DropDatagenRecipe | None = None
    _views: tuple[_BoundRecordingView, ...] = ()
    _open: bool = False
    last_tick_triggered: bool = False

    @property
    def is_open(self) -> bool:
        """Whether this session has buffered at least one frame since open or commit."""
        return self._open

    @property
    def recording_views(self) -> tuple[_BoundRecordingView, ...]:
        if self._views:
            return self._views
        return (
            _BoundRecordingView(
                name="dataset",
                dataset_root=self.dataset_root,
                logger=self.logger,
                policy_fps=self.policy_fps,
                stride=1,
                view_state=ViewRecordingState(name="dataset", stride=1),
                segments=FailureSegmentsWriter(
                    self.dataset_root,
                    stride=1,
                    control_hz=self.control_hz,
                ),
            ),
        )

    def reset_view_episode_state(self) -> None:
        self.last_tick_triggered = False
        for view in self.recording_views:
            view.view_state.reset_episode()
            view.segments.reset_episode_snapshots()

    def log_step(
        self,
        observation_dict: dict[str, Any],
        action: np.ndarray | list[float],
        task: str,
        loss_mask: float,
        *,
        phase: str | None = None,
        annotation: dict[str, Any] | None = None,
        view_name: str = "dataset",
    ) -> None:
        """Append one frame to a named recording view."""
        self._open = True
        view = next(v for v in self.recording_views if v.name == view_name)
        view.logger.log_step(
            observation_dict,
            action,
            task,
            loss_mask,
            phase=phase,
            annotation=annotation,
        )

    def record_tick_metadata(
        self,
        *,
        tick: int,
        drop_release: bool,
        drop_event: bool,
        attempt_index: int,
        object_z: float | None,
    ) -> None:
        for view in self.recording_views:
            view.segments.record_tick(
                tick=tick,
                drop_release=drop_release,
                drop_event=drop_event,
                attempt_index=attempt_index,
                object_z=object_z,
            )

    def discard(self) -> None:
        """Drop buffered frames without committing an episode."""
        if self._open:
            for view in self.recording_views:
                view.logger.clear_open_episode()
                view.segments.discard_episode()
        self._open = False
        self.reset_view_episode_state()

    def commit(self, *, success: bool) -> int:
        """Persist buffered frames; return the dataset episode index written."""
        if not self._open:
            raise RuntimeError("cannot commit datagen episode: no open frame buffer")
        index = int(self.logger.dataset_episode_index_on_commit())
        recipe = self.recipe
        object_name = recipe.object_names[0] if recipe is not None else ""
        task_id = int(recipe.task_id) if recipe is not None else 0
        for view in self.recording_views:
            if view.name != "dataset":
                _ = int(view.logger.dataset_episode_index_on_commit())
            view.logger.end_episode(episode_metadata={"success": bool(success)})
            view.segments.commit_episode(
                episode_index=index,
                logical_episode_index=int(self.manifest.logical_episode_index),
                object_name=object_name,
                task_id=task_id,
                controller=self.manifest.controller.value,
                post_drop_mode=self.manifest.post_drop_mode.value,
                is_drop_episode=bool(self.manifest.drop),
                episode_seed=int(self.manifest.episode_seed),
                layout_seed=int(self.manifest.layout_seed),
            )
        self._open = False
        self.reset_view_episode_state()
        return index


class RunDatasetWriter:
    """One logger per controller/mode variant; run-level manifest after successful finalize."""

    def __init__(
        self,
        recipe: DropDatagenRecipe,
        *,
        logger_factory: LoggerFactory | None = None,
        skip_fresh_output_check: bool = False,
        logical_range: tuple[int, int] | None = None,
        recipe_content_hash: str | None = None,
    ) -> None:
        """Create per-variant loggers and enforce a fresh recording output directory."""
        if not skip_fresh_output_check:
            assert_fresh_run_output_dir(recipe)
        self._recipe = recipe
        self._logical_range = logical_range
        self._recipe_content_hash = recipe_content_hash
        self._logger_factory = logger_factory or _default_logger_factory
        self._loggers: dict[VariantKey, Any] = {}
        self._master_loggers: dict[VariantKey, Any] = {}
        self._segment_writers: dict[tuple[VariantKey, str], FailureSegmentsWriter] = {}
        self._episode_rows: list[EpisodeMetadataRow] = []
        self._manifest_written = False
        self._run_started = False

    @property
    def episode_rows(self) -> tuple[EpisodeMetadataRow, ...]:
        """Metadata rows accumulated for episodes processed so far."""
        return tuple(self._episode_rows)

    def mark_run_started(self) -> None:
        """Record that matrix orchestration has begun (manifest required on abort)."""
        self._run_started = True

    @property
    def run_started(self) -> bool:
        """Whether the matrix loop has started at least one logical episode."""
        return self._run_started

    @property
    def manifest_written(self) -> bool:
        """Whether a terminal or in-progress manifest has been persisted."""
        return self._manifest_written

    def _manifest_path(self) -> Path:
        return Path(self._recipe.recording.output_dir) / "run_manifest.json"

    def _persist_manifest(self, *, status: RunStatus, error_summary: str | None) -> Path:
        path = self._manifest_path()
        write_run_manifest_atomic(
            path,
            RunManifest(
                recipe_name=self._recipe.name,
                base_seed=int(self._recipe.recording.base_seed),
                output_dir=str(self._recipe.recording.output_dir),
                episodes=list(self._episode_rows),
                run_status=status,
                error_summary=error_summary,
                logical_range=self._logical_range,
                recipe_content_hash=self._recipe_content_hash,
            ),
        )
        return path

    def open_episode_session(self, manifest: EpisodeSeedManifest) -> DatagenEpisodeSession:
        """Return a session bound to the logger for this controller/mode variant."""
        key = _variant_key(manifest)
        root = variant_dataset_directory(self._recipe, manifest)
        repo_id = variant_repo_id(self._recipe, manifest)
        control_hz = int(self._recipe.control_hz)
        dataset_stride = recording_stride(control_hz, int(self._recipe.recording.dataset_fps))
        if key not in self._loggers:
            info_path = root / "meta" / "info.json"
            if info_path.is_file():
                raise StaleRunOutputError(
                    f"Variant dataset already exists at {root}. Refusing to append to a prior run."
                )
            root.parent.mkdir(parents=True, exist_ok=True)
            self._loggers[key] = self._logger_factory(
                root,
                repo_id,
                policy_fps=self._recipe.recording.dataset_fps,
                append=False,
            )
        views: list[_BoundRecordingView] = [
            _BoundRecordingView(
                name="dataset",
                dataset_root=root,
                logger=self._loggers[key],
                policy_fps=int(self._recipe.recording.dataset_fps),
                stride=dataset_stride,
                view_state=ViewRecordingState(name="dataset", stride=dataset_stride),
                segments=self._segment_writer_for(key, "dataset", root, dataset_stride, control_hz),
            )
        ]
        master_fps = self._recipe.recording.master_fps
        if master_fps is not None:
            master_root = master_dataset_directory(root)
            master_repo = master_dataset_repo_id(repo_id)
            if key not in self._master_loggers:
                master_info = master_root / "meta" / "info.json"
                if master_info.is_file():
                    raise StaleRunOutputError(
                        f"Master dataset already exists at {master_root}. Refusing to append."
                    )
                master_root.parent.mkdir(parents=True, exist_ok=True)
                self._master_loggers[key] = self._logger_factory(
                    master_root,
                    master_repo,
                    policy_fps=int(master_fps),
                    append=False,
                )
            views.append(
                _BoundRecordingView(
                    name="master_20hz",
                    dataset_root=master_root,
                    logger=self._master_loggers[key],
                    policy_fps=int(master_fps),
                    stride=1,
                    view_state=ViewRecordingState(name="master_20hz", stride=1),
                    segments=self._segment_writer_for(key, "master_20hz", master_root, 1, control_hz),
                )
            )
        session = DatagenEpisodeSession(
            manifest=manifest,
            dataset_root=root,
            repo_id=repo_id,
            logger=self._loggers[key],
            policy_fps=int(self._recipe.recording.dataset_fps),
            control_hz=control_hz,
            recipe=self._recipe,
            _views=tuple(views),
        )
        session.reset_view_episode_state()
        return session

    def _segment_writer_for(
        self,
        key: VariantKey,
        view_name: str,
        root: Path,
        stride: int,
        control_hz: int,
    ) -> FailureSegmentsWriter:
        slot = (key, view_name)
        if slot not in self._segment_writers:
            self._segment_writers[slot] = FailureSegmentsWriter(root, stride=stride, control_hz=control_hz)
        return self._segment_writers[slot]

    def record_episode_outcome(
        self,
        request: EpisodeRequest,
        result: EpisodeResult,
        session: DatagenEpisodeSession,
    ) -> EpisodeMetadataRow:
        """Commit or discard frames, append manifest metadata, and update ``result`` keep flags."""
        keep, reason = evaluate_datagen_keep(request, result)
        dataset_episode_index: int | None = None
        if keep:
            if not session.is_open:
                manifest = session.manifest
                raise ValueError(
                    "cannot keep datagen episode with no logged frames "
                    f"({manifest.controller.value} × {manifest.post_drop_mode.value})"
                )
            dataset_episode_index = session.commit(success=result.success)
        elif session.is_open:
            session.discard()
        row = build_episode_metadata_row(
            request,
            result,
            keep=keep,
            keep_reason=reason,
            dataset_episode_index=dataset_episode_index,
        )
        self._episode_rows.append(row)
        result.keep = keep
        result.keep_reason = reason if keep else None
        result.reject_reason = reason if not keep else None
        self._persist_manifest(status=RunStatus.IN_PROGRESS, error_summary=None)
        return row

    def write_aborted_manifest(self, error_summary: str) -> Path:
        """Persist partial episode rows and mark the run aborted."""
        self._finalize_loggers()
        path = self._persist_manifest(status=RunStatus.ABORTED, error_summary=error_summary)
        self._manifest_written = True
        return path

    def finalize(self) -> Path:
        """Flush all variant datasets and write ``run_manifest.json`` once."""
        if self._manifest_written:
            return self._manifest_path()
        self._finalize_loggers()
        path = self._persist_manifest(status=RunStatus.COMPLETE, error_summary=None)
        self._manifest_written = True
        return path

    def finalize_loggers_only(self) -> None:
        """Flush variant datasets without writing run_manifest.json (failed matrix run)."""
        self._finalize_loggers()

    def _finalize_loggers(self) -> None:
        errors: list[BaseException] = []
        for logger in self._loggers.values():
            try:
                logger.finalize()
            except Exception as exc:
                errors.append(exc)
        for logger in self._master_loggers.values():
            try:
                logger.finalize()
            except Exception as exc:
                errors.append(exc)
        for segments in self._segment_writers.values():
            try:
                segments.finalize()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RunDatasetFinalizeError(errors)


def _default_logger_factory(
    root: Path,
    repo_id: str,
    *,
    policy_fps: int,
    append: bool = False,
) -> Any:
    from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger

    return FaultRecoveryDatasetLogger(
        root=root,
        repo_id=repo_id,
        policy_fps=policy_fps,
        append=append,
    )
