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
    build_episode_metadata_row,
    write_run_manifest_atomic,
)
from lerobot.faults.datagen.recipe import DropDatagenRecipe, EpisodeSeedManifest

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

VariantKey = tuple[str, str]


class StaleRunOutputError(FileExistsError):
    """Raised when a new run targets a non-empty output directory."""


class RunDatasetFinalizeError(RuntimeError):
    """Raised when one or more variant loggers fail to finalize."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        msg = "; ".join(str(exc) for exc in errors)
        super().__init__(f"dataset logger finalize failed: {msg}")


def variant_dataset_directory(recipe: DropDatagenRecipe, manifest: EpisodeSeedManifest) -> Path:
    base = Path(recipe.recording.output_dir)
    return base / manifest.controller.value / manifest.post_drop_mode.value / "dataset"


def variant_repo_id(recipe: DropDatagenRecipe, manifest: EpisodeSeedManifest) -> str:
    return f"{recipe.name}/{manifest.controller.value}/{manifest.post_drop_mode.value}"


def _variant_key(manifest: EpisodeSeedManifest) -> VariantKey:
    return (manifest.controller.value, manifest.post_drop_mode.value)


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
    """One open episode buffer on a shared variant logger."""

    manifest: EpisodeSeedManifest
    dataset_root: Path
    repo_id: str
    logger: Any
    policy_fps: int
    _open: bool = False

    @property
    def is_open(self) -> bool:
        return self._open

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

    def commit(self) -> int:
        """Persist buffered frames; return the dataset episode index written."""
        if not self._open:
            raise RuntimeError("cannot commit datagen episode: no open frame buffer")
        index = int(self.logger.dataset_episode_index_on_commit())
        self.logger.end_episode()
        self._open = False
        return index


class RunDatasetWriter:
    """One logger per controller/mode variant; run-level manifest after successful finalize."""

    def __init__(
        self,
        recipe: DropDatagenRecipe,
        *,
        logger_factory: LoggerFactory | None = None,
        skip_fresh_output_check: bool = False,
    ) -> None:
        if not skip_fresh_output_check:
            assert_fresh_run_output_dir(recipe)
        self._recipe = recipe
        self._logger_factory = logger_factory or _default_logger_factory
        self._loggers: dict[VariantKey, Any] = {}
        self._episode_rows: list[EpisodeMetadataRow] = []
        self._manifest_written = False

    @property
    def episode_rows(self) -> tuple[EpisodeMetadataRow, ...]:
        return tuple(self._episode_rows)

    def open_episode_session(self, manifest: EpisodeSeedManifest) -> DatagenEpisodeSession:
        key = _variant_key(manifest)
        root = variant_dataset_directory(self._recipe, manifest)
        repo_id = variant_repo_id(self._recipe, manifest)
        if key not in self._loggers:
            info_path = root / "meta" / "info.json"
            if info_path.is_file():
                raise StaleRunOutputError(
                    f"Variant dataset already exists at {root}. "
                    "Refusing to append to a prior run."
                )
            root.parent.mkdir(parents=True, exist_ok=True)
            self._loggers[key] = self._logger_factory(
                root,
                repo_id,
                policy_fps=self._recipe.recording.dataset_fps,
                append=False,
            )
        return DatagenEpisodeSession(
            manifest=manifest,
            dataset_root=root,
            repo_id=repo_id,
            logger=self._loggers[key],
            policy_fps=int(self._recipe.recording.dataset_fps),
        )

    def record_episode_outcome(
        self,
        request: EpisodeRequest,
        result: EpisodeResult,
        session: DatagenEpisodeSession,
    ) -> EpisodeMetadataRow:
        keep, reason = evaluate_datagen_keep(request, result)
        dataset_episode_index: int | None = None
        if keep:
            if not session.is_open:
                manifest = session.manifest
                raise ValueError(
                    "cannot keep datagen episode with no logged frames "
                    f"({manifest.controller.value} × {manifest.post_drop_mode.value})"
                )
            dataset_episode_index = session.commit()
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
        return row

    def finalize(self) -> Path:
        if self._manifest_written:
            return Path(self._recipe.recording.output_dir) / "run_manifest.json"
        self._finalize_loggers()
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
        self._manifest_written = True
        return manifest_path

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
