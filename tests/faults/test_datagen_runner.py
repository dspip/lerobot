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

from __future__ import annotations

from pathlib import Path

import pytest

from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.paired_context import build_paired_episode_plan
from lerobot.faults.datagen.recipe import (
    DatagenController,
    PostDropMode,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)
from lerobot.faults.datagen.runner import (
    DropDatagenRunnerError,
    run_drop_datagen_matrix,
    variant_output_directory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def test_variant_output_directory_is_deterministic(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    recipe = recipe.__class__(
        **{
            **recipe.__dict__,
            "recording": recipe.recording.__class__(
                base_seed=recipe.recording.base_seed,
                output_dir=str(tmp_path / "out"),
                dataset_fps=recipe.recording.dataset_fps,
            ),
        }
    )
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    simple = next(m for m in manifests if m.controller is DatagenController.SIMPLE_IK)

    path = variant_output_directory(recipe, simple)
    assert path == tmp_path / "out" / "simple_ik" / "immediate_ik" / "episode_0000"


def test_run_matrix_invokes_all_variants_with_injected_adapters(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    recipe = recipe.__class__(
        **{
            **recipe.__dict__,
            "recording": recipe.recording.__class__(
                base_seed=9000,
                output_dir=str(tmp_path / "datagen"),
                dataset_fps=10,
            ),
        }
    )
    seen: list[tuple[DatagenController, PostDropMode]] = []

    class _FakeAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            seen.append((request.manifest.controller, request.manifest.post_drop_mode))
            return EpisodeResult.from_run(request, success=True, outcome="ok")

    factories = {
        DatagenController.SIMPLE_IK: lambda _recipe: _FakeAdapter(),
        DatagenController.SMOLVLA: lambda _recipe: _FakeAdapter(),
    }
    results = run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=(0,),
        adapter_factories=factories,
    )
    assert len(results) == 5
    assert all(r.success for r in results)
    assert len(seen) == 5


def test_run_matrix_fails_fast_on_missing_adapter_factory() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    with pytest.raises(DropDatagenRunnerError, match="adapter factory"):
        run_drop_datagen_matrix(
            recipe,
            logical_episode_indices=(0,),
            adapter_factories={},
        )


def test_run_matrix_wraps_adapter_exceptions_with_context(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)

    class _BoomAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            raise RuntimeError("boom")

    factories = {
        DatagenController.SIMPLE_IK: lambda _recipe: _BoomAdapter(),
        DatagenController.SMOLVLA: lambda _recipe: _BoomAdapter(),
    }
    with pytest.raises(DropDatagenRunnerError, match="simple_ik × immediate_ik"):
        run_drop_datagen_matrix(
            recipe,
            logical_episode_indices=(0,),
            adapter_factories=factories,
        )


def test_paired_plan_attached_to_requests(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    captured: list = []

    class _SpyAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            captured.append(request.paired_plan)
            return EpisodeResult.from_run(request, success=True, outcome="ok")

    factories = {
        DatagenController.SIMPLE_IK: lambda _recipe: _SpyAdapter(),
        DatagenController.SMOLVLA: lambda _recipe: _SpyAdapter(),
    }
    run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=(0,),
        adapter_factories=factories,
    )
    assert len(captured) == 5
    assert all(p.motion_profile == captured[0].motion_profile for p in captured)
