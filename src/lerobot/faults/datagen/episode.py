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

"""Shared episode request/result types and scene-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from lerobot.faults.datagen.dataset_writer import DatagenEpisodeSession

from lerobot.faults.datagen.paired_context import PairedEpisodePlan
from lerobot.faults.datagen.recipe import (
    DatagenController,
    DropDatagenRecipe,
    EpisodeSeedManifest,
    PostDropMode,
    effective_post_drop_dwell_steps,
)

__all__ = [
    "EpisodeRequest",
    "EpisodeResult",
    "select_episode_object",
]


def select_episode_object(drop_seed: int, object_names: tuple[str, ...]) -> str:
    if not object_names:
        raise ValueError("object_names must be non-empty")
    rng = np.random.default_rng(int(drop_seed))
    return object_names[int(rng.integers(0, len(object_names)))]


@dataclass(frozen=True)
class EpisodeRequest:
    recipe: DropDatagenRecipe
    manifest: EpisodeSeedManifest
    object_name: str
    output_dir: Path
    paired_plan: PairedEpisodePlan
    shared_layout: dict[str, dict[str, list[float]]]
    headless: bool = True
    device: str = "cuda"
    episode_session: DatagenEpisodeSession | None = None


@dataclass
class EpisodeResult:
    controller: DatagenController
    post_drop_mode: PostDropMode
    logical_episode_index: int
    episode_index: int
    output_dir: Path
    object_name: str
    episode_seed: int
    layout_seed: int
    drop_seed: int
    controller_seed: int
    success: bool
    outcome: str
    dwell_steps: int = 0
    drop_trigger: dict[str, Any] | None = None
    trigger_pose: list[float] | None = None
    actual_dwell_steps: int | None = None
    error: str | None = None
    keep: bool | None = None
    keep_reason: str | None = None
    reject_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run(
        cls,
        request: EpisodeRequest,
        *,
        success: bool,
        outcome: str,
        drop_trigger: dict[str, Any] | None = None,
        trigger_pose: list[float] | None = None,
        actual_dwell_steps: int | None = None,
        error: str | None = None,
        **details: Any,
    ) -> EpisodeResult:
        dwell = effective_post_drop_dwell_steps(request.recipe, request.manifest.post_drop_mode)
        return cls(
            controller=request.manifest.controller,
            post_drop_mode=request.manifest.post_drop_mode,
            logical_episode_index=request.manifest.logical_episode_index,
            episode_index=request.manifest.episode_index,
            output_dir=request.output_dir,
            object_name=request.object_name,
            episode_seed=request.manifest.episode_seed,
            layout_seed=request.manifest.layout_seed,
            drop_seed=request.manifest.drop_seed,
            controller_seed=request.manifest.controller_seed,
            success=success,
            outcome=outcome,
            dwell_steps=dwell,
            drop_trigger=drop_trigger,
            trigger_pose=trigger_pose,
            actual_dwell_steps=actual_dwell_steps,
            error=error,
            details=dict(details),
        )

    @classmethod
    def ok(cls, request: EpisodeRequest, *, outcome: str = "completed", **details: Any) -> EpisodeResult:
        return cls.from_run(request, success=True, outcome=outcome, **details)

    @classmethod
    def failed(cls, request: EpisodeRequest, *, outcome: str, error: str) -> EpisodeResult:
        return cls.from_run(request, success=False, outcome=outcome, error=error)
