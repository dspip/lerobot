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

"""Experiment-matrix orchestration for unified drop datagen."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from lerobot.faults.datagen.controllers.base import DatagenControllerAdapter
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult, select_episode_object
from lerobot.faults.datagen.recipe import (
    DatagenController,
    DropDatagenRecipe,
    EpisodeSeedManifest,
    paired_episode_seed_manifests,
)

__all__ = [
    "DropDatagenRunnerError",
    "default_adapter_factories",
    "run_drop_datagen_matrix",
    "variant_output_directory",
]

AdapterFactory = Callable[[DropDatagenRecipe], DatagenControllerAdapter]


class DropDatagenRunnerError(RuntimeError):
    """Raised when matrix orchestration cannot run a requested variant."""


def variant_output_directory(recipe: DropDatagenRecipe, manifest: EpisodeSeedManifest) -> Path:
    base = Path(recipe.recording.output_dir)
    return (
        base
        / manifest.controller.value
        / manifest.post_drop_mode.value
        / f"episode_{manifest.logical_episode_index:04d}"
    )


def _adapter_for(
    controller: DatagenController,
    recipe: DropDatagenRecipe,
    factories: dict[DatagenController, AdapterFactory],
) -> DatagenControllerAdapter:
    factory = factories.get(controller)
    if factory is None:
        raise DropDatagenRunnerError(
            f"Missing adapter factory for controller {controller.value!r}"
        )
    return factory(recipe)


def run_drop_datagen_matrix(
    recipe: DropDatagenRecipe,
    *,
    logical_episode_indices: Sequence[int] | None = None,
    adapter_factories: dict[DatagenController, AdapterFactory] | None = None,
    headless: bool = True,
) -> tuple[EpisodeResult, ...]:
    factories = default_adapter_factories() if adapter_factories is None else adapter_factories
    if logical_episode_indices is None:
        episodes_per_variant = recipe.experiment_matrix[0].episodes
        logical_episode_indices = tuple(range(int(episodes_per_variant)))
    results: list[EpisodeResult] = []
    for logical_index in logical_episode_indices:
        manifests = paired_episode_seed_manifests(recipe, logical_episode_index=int(logical_index))
        if not manifests:
            raise DropDatagenRunnerError(
                f"No manifests for logical episode index {logical_index}"
            )
        object_name = select_episode_object(manifests[0].drop_seed, recipe.object_names)
        for manifest in manifests:
            adapter = _adapter_for(manifest.controller, recipe, factories)
            output_dir = variant_output_directory(recipe, manifest)
            output_dir.mkdir(parents=True, exist_ok=True)
            request = EpisodeRequest(
                recipe=recipe,
                manifest=manifest,
                object_name=object_name,
                output_dir=output_dir,
                headless=headless,
            )
            try:
                results.append(adapter.run_episode(request))
            except Exception as exc:  # noqa: BLE001 — surface variant context
                raise DropDatagenRunnerError(
                    f"{manifest.controller.value} × {manifest.post_drop_mode.value} "
                    f"episode {manifest.logical_episode_index} failed: {exc}"
                ) from exc
    return tuple(results)


def default_adapter_factories() -> dict[DatagenController, AdapterFactory]:
    from lerobot.faults.datagen.controllers.simple_ik import SimpleIKDatagenAdapter
    from lerobot.faults.datagen.controllers.smolvla import SmolVLADatagenAdapter

    return {
        DatagenController.SIMPLE_IK: lambda recipe: SimpleIKDatagenAdapter(recipe),
        DatagenController.SMOLVLA: lambda recipe: SmolVLADatagenAdapter(recipe),
    }
