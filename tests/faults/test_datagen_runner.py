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
from lerobot.faults.datagen.layout_provider import LayoutProviderContext
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

_FAKE_LAYOUT = {
    "alphabet_soup_1": {
        "pos": [0.1, 0.2, 0.03],
        "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
}


def _fake_layout_provider(_ctx: LayoutProviderContext) -> dict:
    return _FAKE_LAYOUT


def _fake_init_count(_recipe) -> int:
    return 50


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
    layouts: list[dict] = []

    class _FakeAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            seen.append((request.manifest.controller, request.manifest.post_drop_mode))
            layouts.append(request.shared_layout)
            return EpisodeResult.from_run(request, success=True, outcome="ok")

    factories = {
        DatagenController.SIMPLE_IK: lambda _recipe: _FakeAdapter(),
        DatagenController.SMOLVLA: lambda _recipe: _FakeAdapter(),
    }
    results = run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=(0,),
        adapter_factories=factories,
        layout_provider=_fake_layout_provider,
        init_state_count_provider=_fake_init_count,
    )
    assert len(results) == 5
    assert all(r.success for r in results)
    assert len(seen) == 5
    assert all(layout == _FAKE_LAYOUT for layout in layouts)


def test_run_matrix_fails_fast_on_missing_adapter_factory() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    with pytest.raises(DropDatagenRunnerError, match="adapter factory"):
        run_drop_datagen_matrix(
            recipe,
            logical_episode_indices=(0,),
            adapter_factories={},
            layout_provider=_fake_layout_provider,
            init_state_count_provider=_fake_init_count,
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
            layout_provider=_fake_layout_provider,
            init_state_count_provider=_fake_init_count,
        )


def test_run_matrix_raises_on_layout_provider_failure(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)

    def _fail(_ctx: LayoutProviderContext) -> dict:
        raise RuntimeError("layout broke")

    with pytest.raises(DropDatagenRunnerError, match="shared layout failed"):
        run_drop_datagen_matrix(
            recipe,
            logical_episode_indices=(0,),
            adapter_factories={
                DatagenController.SIMPLE_IK: lambda _r: _FakeAdapter(),
            },
            layout_provider=_fail,
            init_state_count_provider=_fake_init_count,
        )


class _FakeAdapter:
    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        return EpisodeResult.from_run(request, success=True, outcome="ok")


def test_init_state_count_provider_called_once_per_matrix_run(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    matrix_two_episodes = tuple(
        variant.__class__(**{**variant.__dict__, "episodes": 2})
        for variant in recipe.experiment_matrix
    )
    recipe = recipe.__class__(
        **{
            **recipe.__dict__,
            "recording": recipe.recording.__class__(
                base_seed=9000,
                output_dir=str(tmp_path / "datagen"),
                dataset_fps=10,
            ),
            "experiment_matrix": matrix_two_episodes,
        }
    )
    calls: list[int] = []

    def _count_once(_recipe) -> int:
        calls.append(1)
        return 50

    run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=(0, 1),
        adapter_factories={
            DatagenController.SIMPLE_IK: lambda _recipe: _FakeAdapter(),
            DatagenController.SMOLVLA: lambda _recipe: _FakeAdapter(),
        },
        layout_provider=_fake_layout_provider,
        init_state_count_provider=_count_once,
    )
    assert len(calls) == 1


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
        layout_provider=_fake_layout_provider,
        init_state_count_provider=_fake_init_count,
    )
    assert len(captured) == 5
    assert all(p.drop_u == captured[0].drop_u for p in captured)
