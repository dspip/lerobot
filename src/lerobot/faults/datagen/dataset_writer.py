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

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.manifest import (
    EpisodeMetadataRow,
    RunManifest,
    build_episode_metadata_row,
    write_run_manifest_atomic,
)
from lerobot.faults.datagen.recipe import DropDatagenRecipe, EpisodeSeedManifest

__all__ = [
    "DatagenEpisodeSession",
    "LoggerFactory",
    "RunDatasetWriter",
    "evaluate_datagen_keep",
    "variant_dataset_directory",
    "variant_repo_id",
]

LoggerFactory = Callable[..., Any]

VariantKey = tuple[str, str]


def variant_dataset_directory(recipe: DropDatagenRecipe, manifest: EpisodeSeedManifest) -> Path:
    base = Path(recipe.recording.output_dir)
    return base / manifest.controller.value / manifest.post_drop_mode.value / "dataset"


def variant_repo_id(recipe: DropDatagenRecipe, manifest: EpisodeSeedManifest) -> str:
    return f"{recipe.name}/{manifest.controller.value}/{manifest.post_drop_mode.value}"


def _variant_key(manifest: EpisodeSeedManifest) -> VariantKey:
    return (manifest.controller.value, manifest.post_drop_mode.value)


def evaluate_datagen_keep(request: EpisodeRequest, result: EpisodeResult) -> tuple[bool, str | None]:
    plan = request.paired_plan
    if not plan.drop_decision.drop:
        if result.success:
            return True, "nominal_placement_success"
        return False, result.outcome or "nominal_failed"
    if result.success:
        return True, "recovery_success"
    return False, result.outcome or "recovery_failed"


@dataclass
class DatagenEpisodeSession:
    manifest: EpisodeSeedManifest
    dataset_root: Path
    repo_id: str
    logger: Any
    policy_fps: int
    _open: bool = False

    def log_step(
        self,
        observation_dict: dict[str, Any],
        action: np.ndarray | list[float],
        task: str,
        loss_mask: float,
        *,
        phase: str | None = None,
        annotation: dict[str, Any] | None = None,
    ) -> None:
        self._open = True
        self.logger.log_step(
            observation_dict,
            action,
            task,
            loss_mask,
            phase=phase,
            annotation=annotation,
        )

    def discard(self) -> None:
        if self._open:
            self.logger.clear_open_episode()
        self._open = False

    def discard(self) -> None:
        self.logger.clear_open_episode()


class RunDatasetWriter:
    """One logger per controller/mode variant; run-level manifest on finalize."""

    def __init__(
        self,
        recipe: DropDatagenRecipe,
        *,
        logger_factory: LoggerFactory | None = None,
    ) -> None:
        self._recipe = recipe
        self._logger_factory = logger_factory or _default_logger_factory
        self._loggers: dict[VariantKey, Any] = {}
        self._episode_rows: list[EpisodeMetadataRow] = []

    @property
    def episode_rows(self) -> tuple[EpisodeMetadataRow, ...]:
        return tuple(self._episode_rows)

    def open_episode_session(self, manifest: EpisodeSeedManifest) -> DatagenEpisodeSession:
        key = _variant_key(manifest)
        root = variant_dataset_directory(self._recipe, manifest)
        repo_id = variant_repo_id(self._recipe, manifest)
        if key not in self._loggers:
            info_path = root / "meta" / "info.json"
            if root.exists() and not info_path.is_file():
                shutil.rmtree(root)
            append = info_path.is_file()
            self._loggers[key] = self._logger_factory(
                root,
                repo_id,
                policy_fps=self._recipe.recording.dataset_fps,
                append=append,
            )
        return DatagenEpisodeSession(
            manifest=manifest,
            dataset_root=root,
            repo_id=repo_id,
            logger=self._loggers[key],
            policy_fps=int(self._recipe.recording.dataset_fps),
        )

    def commit_episode(
        self,
        session: DatagenEpisodeSession,
        *,
        keep: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        episode_data = metadata if keep else None
        _commit_logger_episode(session.logger, keep=keep, episode_data=episode_data)
        if metadata is not None and "controller" in metadata:
            self._episode_rows.append(_row_from_metadata_dict(metadata))

    def record_episode(
        self,
        request: EpisodeRequest,
        result: EpisodeResult,
        session: DatagenEpisodeSession,
    ) -> EpisodeMetadataRow:
        return self.record_episode_outcome(request, result, session)

    def record_episode_outcome(
        self,
        request: EpisodeRequest,
        result: EpisodeResult,
        session: DatagenEpisodeSession,
    ) -> EpisodeMetadataRow:
        keep, reason = evaluate_datagen_keep(request, result)
        row = build_episode_metadata_row(request, result, keep=keep, keep_reason=reason)
        if session._open:
            _commit_logger_episode(session.logger, keep=keep, episode_data=row.to_dict() if keep else None)
            session._open = False
        self._episode_rows.append(row)
        result.keep = keep
        result.keep_reason = reason if keep else None
        result.reject_reason = reason if not keep else None
        return row

    def finalize(self) -> Path:
        for logger in self._loggers.values():
            logger.finalize()
        manifest_path = Path(self._recipe.recording.output_dir) / "run_manifest.json"
        write_run_manifest_atomic(
            manifest_path,
            RunManifest(
                recipe_name=self._recipe.name,
                base_seed=int(self._recipe.recording.base_seed),
                output_dir=str(self._recipe.recording.output_dir),
                episodes=list(self._episode_rows),
            ),
        )
        return manifest_path

    def __enter__(self) -> RunDatasetWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            for logger in self._loggers.values():
                if hasattr(logger, "clear_open_episode"):
                    try:
                        logger.clear_open_episode()
                    except Exception:
                        pass
        self.finalize()


def _row_from_metadata_dict(metadata: dict[str, Any]) -> EpisodeMetadataRow:
    from lerobot.faults.datagen.manifest import EPISODE_METADATA_FIELDS

    kwargs = {field: metadata.get(field) for field in EPISODE_METADATA_FIELDS}
    return EpisodeMetadataRow(**kwargs)  # type: ignore[arg-type]


def _commit_logger_episode(logger: Any, *, keep: bool, episode_data: dict[str, Any] | None) -> None:
    if keep:
        try:
            logger.end_episode(episode_data=episode_data)
        except TypeError:
            logger.end_episode()
    else:
        logger.clear_open_episode()


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
