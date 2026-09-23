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

from lerobot.faults.datagen.controllers.base import DatagenControllerAdapter
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult, select_episode_object
from lerobot.faults.datagen.layout_provider import (
    LayoutProviderContext,
    SharedLayoutProvider,
    libero_init_state_count,
    libero_shared_layout_provider,
)
from lerobot.faults.datagen.dataset_writer import RunDatasetWriter
from lerobot.faults.datagen.paired_context import build_paired_episode_plan
from lerobot.faults.datagen.recipe import (
    DatagenController,
    DropDatagenRecipe,
    EpisodeSeedManifest,
    paired_episode_seed_manifests,
)

__all__ = [
    "DropDatagenRunnerError",
    "InitStateCountProvider",
    "default_adapter_factories",
    "run_drop_datagen_matrix",
    "variant_output_directory",
]

AdapterFactory = Callable[[DropDatagenRecipe], DatagenControllerAdapter]
InitStateCountProvider = Callable[[DropDatagenRecipe], int]


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
    device: str = "cuda",
    layout_provider: SharedLayoutProvider | None = None,
    init_state_count_provider: InitStateCountProvider | None = None,
    dataset_writer: RunDatasetWriter | None = None,
) -> tuple[EpisodeResult, ...]:
    factories = default_adapter_factories() if adapter_factories is None else adapter_factories
    layout_fn = layout_provider or libero_shared_layout_provider
    init_count_fn = init_state_count_provider or libero_init_state_count
    if logical_episode_indices is None:
        episodes_per_variant = recipe.experiment_matrix[0].episodes
        logical_episode_indices = tuple(range(int(episodes_per_variant)))
    try:
        num_init_states = int(init_count_fn(recipe))
    except Exception as exc:
        raise DropDatagenRunnerError(
            f"could not resolve LIBERO init state count: {exc}"
        ) from exc
    writer = dataset_writer if dataset_writer is not None else RunDatasetWriter(recipe)
    results: list[EpisodeResult] = []
    try:
        for logical_index in logical_episode_indices:
            manifests = paired_episode_seed_manifests(recipe, logical_episode_index=int(logical_index))
            if not manifests:
                raise DropDatagenRunnerError(
                    f"No manifests for logical episode index {logical_index}"
                )
            object_name = select_episode_object(manifests[0].drop_seed, recipe.object_names)
            paired_plan = build_paired_episode_plan(
                recipe,
                manifest=manifests[0],
                object_name=object_name,
                num_init_states=num_init_states,
            )
            layout_ctx = LayoutProviderContext(
                recipe=recipe,
                plan=paired_plan,
                object_name=object_name,
            )
            try:
                shared_layout = layout_fn(layout_ctx)
            except Exception as exc:
                raise DropDatagenRunnerError(
                    f"logical episode {logical_index}: shared layout failed: {exc}"
                ) from exc
            for manifest in manifests:
                adapter = _adapter_for(manifest.controller, recipe, factories)
                output_dir = variant_output_directory(recipe, manifest)
                output_dir.mkdir(parents=True, exist_ok=True)
                session = writer.open_episode_session(manifest)
                request = EpisodeRequest(
                    recipe=recipe,
                    manifest=manifest,
                    object_name=object_name,
                    output_dir=output_dir,
                    paired_plan=paired_plan,
                    shared_layout=shared_layout,
                    headless=headless,
                    device=device,
                    episode_session=session,
                )
                try:
                    result = adapter.run_episode(request)
                except Exception as exc:
                    session.discard()
                    raise DropDatagenRunnerError(
                        f"{manifest.controller.value} × {manifest.post_drop_mode.value} "
                        f"episode {manifest.logical_episode_index} failed: {exc}"
                    ) from exc
                writer.record_episode_outcome(request, result, session)
                results.append(result)
    finally:
        writer.finalize()
    return tuple(results)


def default_adapter_factories() -> dict[DatagenController, AdapterFactory]:
    from lerobot.faults.datagen.controllers.simple_ik import SimpleIKDatagenAdapter
    from lerobot.faults.datagen.controllers.smolvla import SmolVLADatagenAdapter

    return {
        DatagenController.SIMPLE_IK: lambda recipe: SimpleIKDatagenAdapter(recipe),
        DatagenController.SMOLVLA: lambda recipe: SmolVLADatagenAdapter(recipe),
    }
