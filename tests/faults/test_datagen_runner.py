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
from typing import Any

import numpy as np
import pytest

from lerobot.faults.datagen.dataset_writer import DatagenEpisodeSession, RunDatasetWriter
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


class _NoopDatasetLogger:
    def __init__(self, root: Path, repo_id: str, **_kwargs) -> None:
        self.root = root
        self._total_episodes = 0

    def log_step(self, *args, **kwargs) -> None:
        del args, kwargs

    def dataset_episode_index_on_commit(self) -> int:
        return self._total_episodes

    def end_episode(self, *args, **kwargs) -> int:
        del args, kwargs
        index = self._total_episodes
        self._total_episodes += 1
        return index

    def clear_open_episode(self) -> None:
        return None

    def finalize(self) -> None:
        return None


class _TrackingDatasetLogger:
    """Records commits/discards for runner failure-path tests."""

    def __init__(self, root: Path, repo_id: str, **_kwargs) -> None:
        self.root = root
        self._open = False
        self.committed = 0
        self.discarded = 0
        self._total = 0

    def log_step(self, *args, **kwargs) -> None:
        del args, kwargs
        self._open = True

    def dataset_episode_index_on_commit(self) -> int:
        return self._total

    def end_episode(self, *args, **kwargs) -> int:
        del args, kwargs
        index = self._total
        self._total += 1
        self.committed += 1
        self._open = False
        return index

    def clear_open_episode(self) -> None:
        self.discarded += 1
        self._open = False

    def finalize(self) -> None:
        if self._open:
            self.end_episode()


def _minimal_frame() -> dict[str, np.ndarray]:
    return {
        "observation.state": np.zeros(8, dtype=np.float32),
        "observation.images.image": np.zeros((4, 4, 3), dtype=np.uint8),
        "observation.images.image2": np.zeros((4, 4, 3), dtype=np.uint8),
    }


def _recipe_with_output(tmp_path: Path, *, base_seed: int = 9000):
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    return recipe.__class__(
        **{
            **recipe.__dict__,
            "recording": recipe.recording.__class__(
                base_seed=base_seed,
                output_dir=str(tmp_path / "datagen"),
                dataset_fps=10,
            ),
        }
    )


def _writer_for(recipe) -> RunDatasetWriter:
    return RunDatasetWriter(
        recipe,
        logger_factory=lambda root, repo_id, **_kw: _NoopDatasetLogger(root, repo_id),
    )


def _tracking_writer_for(recipe) -> RunDatasetWriter:
    loggers: list[_TrackingDatasetLogger] = []

    def factory(root: Path, repo_id: str, **_kw: Any) -> _TrackingDatasetLogger:
        logger = _TrackingDatasetLogger(root, repo_id)
        loggers.append(logger)
        return logger

    writer = RunDatasetWriter(recipe, logger_factory=factory)
    writer._test_loggers = loggers  # type: ignore[attr-defined]
    return writer


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
    recipe = _recipe_with_output(tmp_path)
    seen: list[tuple[DatagenController, PostDropMode]] = []
    layouts: list[dict] = []

    class _FakeAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            seen.append((request.manifest.controller, request.manifest.post_drop_mode))
            layouts.append(request.shared_layout)
            if request.episode_session is not None:
                request.episode_session.log_step(_minimal_frame(), np.zeros(7), "task", 1.0)
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
        dataset_writer=_writer_for(recipe),
    )
    assert len(results) == 5
    assert all(r.success for r in results)
    assert len(seen) == 5
    assert all(layout == _FAKE_LAYOUT for layout in layouts)


def test_run_matrix_fails_fast_on_missing_adapter_factory(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)
    with pytest.raises(DropDatagenRunnerError, match="adapter factory"):
        run_drop_datagen_matrix(
            recipe,
            logical_episode_indices=(0,),
            adapter_factories={},
            layout_provider=_fake_layout_provider,
            init_state_count_provider=_fake_init_count,
            dataset_writer=_writer_for(recipe),
        )


def test_run_matrix_wraps_adapter_exceptions_with_context(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)

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
            dataset_writer=_writer_for(recipe),
        )


def test_run_matrix_raises_on_layout_provider_failure(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)

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
            dataset_writer=_writer_for(recipe),
        )


class _FakeAdapter:
    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        if request.episode_session is not None:
            request.episode_session.log_step(_minimal_frame(), np.zeros(7), "task", 1.0)
        return EpisodeResult.from_run(request, success=True, outcome="ok")


def test_init_state_count_provider_called_once_per_matrix_run(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)
    matrix_two_episodes = tuple(
        variant.__class__(**{**variant.__dict__, "episodes": 2}) for variant in recipe.experiment_matrix
    )
    recipe = recipe.__class__(**{**recipe.__dict__, "experiment_matrix": matrix_two_episodes})
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
        dataset_writer=_writer_for(recipe),
    )
    assert len(calls) == 1


def test_keyboard_interrupt_discards_partial_episode_keeps_prior_commits(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)
    writer = _tracking_writer_for(recipe)
    variant_index = {"n": 0}

    class _InterruptOnThirdAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            assert request.episode_session is not None
            request.episode_session.log_step(_minimal_frame(), np.zeros(7), "task", 1.0)
            n = variant_index["n"]
            variant_index["n"] += 1
            if n < 2:
                return EpisodeResult.from_run(request, success=True, outcome="ok")
            raise KeyboardInterrupt()

    factories = {
        DatagenController.SIMPLE_IK: lambda _recipe: _InterruptOnThirdAdapter(),
        DatagenController.SMOLVLA: lambda _recipe: _InterruptOnThirdAdapter(),
    }
    with pytest.raises(KeyboardInterrupt):
        run_drop_datagen_matrix(
            recipe,
            logical_episode_indices=(0,),
            adapter_factories=factories,
            layout_provider=_fake_layout_provider,
            init_state_count_provider=_fake_init_count,
            dataset_writer=writer,
        )
    assert len(writer.episode_rows) == 2
    manifest_path = Path(recipe.recording.output_dir) / "run_manifest.json"
    assert manifest_path.is_file()
    from lerobot.faults.datagen.manifest import read_run_manifest

    loaded = read_run_manifest(manifest_path)
    assert loaded.run_status.value == "aborted"
    assert len(loaded.episodes) == 2
    loggers: list[_TrackingDatasetLogger] = writer._test_loggers  # type: ignore[attr-defined]
    assert sum(logger.committed for logger in loggers) == 2
    assert sum(logger.discarded for logger in loggers) == 1


def test_cleanup_failure_adds_note_without_replacing_primary_cause(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)
    writer = _writer_for(recipe)

    class _FailAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            if request.episode_session is not None:
                request.episode_session.log_step(_minimal_frame(), np.zeros(7), "task", 1.0)
            raise ValueError("primary failure")

    def _boom_finalize_loggers(self: RunDatasetWriter) -> None:
        raise RuntimeError("cleanup failed")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RunDatasetWriter, "_finalize_loggers", _boom_finalize_loggers)
        with pytest.raises(DropDatagenRunnerError, match="primary failure") as exc_info:
            run_drop_datagen_matrix(
                recipe,
                logical_episode_indices=(0,),
                adapter_factories={
                    DatagenController.SIMPLE_IK: lambda _r: _FailAdapter(),
                    DatagenController.SMOLVLA: lambda _r: _FailAdapter(),
                },
                layout_provider=_fake_layout_provider,
                init_state_count_provider=_fake_init_count,
                dataset_writer=writer,
            )
    err = exc_info.value
    assert isinstance(err.__cause__, ValueError)
    assert str(err.__cause__) == "primary failure"
    assert err.__cause__.__cause__ is None
    assert any("datagen cleanup failed" in note and "cleanup failed" in note for note in err.__notes__)


def test_record_episode_outcome_commit_failure_leaves_no_manifest_row(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)
    writer = _tracking_writer_for(recipe)
    commit_attempts = {"n": 0}
    original_commit = DatagenEpisodeSession.commit

    def commit_fail_on_third(self: DatagenEpisodeSession) -> int:
        commit_attempts["n"] += 1
        if commit_attempts["n"] == 3:
            raise RuntimeError("commit failed")
        return original_commit(self)

    class _AlwaysLogAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            assert request.episode_session is not None
            request.episode_session.log_step(_minimal_frame(), np.zeros(7), "task", 1.0)
            return EpisodeResult.from_run(request, success=True, outcome="ok")

    factories = {
        DatagenController.SIMPLE_IK: lambda _recipe: _AlwaysLogAdapter(),
        DatagenController.SMOLVLA: lambda _recipe: _AlwaysLogAdapter(),
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DatagenEpisodeSession, "commit", commit_fail_on_third)
        with pytest.raises(RuntimeError, match="commit failed"):
            run_drop_datagen_matrix(
                recipe,
                logical_episode_indices=(0,),
                adapter_factories=factories,
                layout_provider=_fake_layout_provider,
                init_state_count_provider=_fake_init_count,
                dataset_writer=writer,
            )
    assert len(writer.episode_rows) == 2
    manifest_path = Path(recipe.recording.output_dir) / "run_manifest.json"
    assert manifest_path.is_file()
    from lerobot.faults.datagen.manifest import read_run_manifest

    loaded = read_run_manifest(manifest_path)
    assert loaded.run_status.value == "aborted"
    assert len(loaded.episodes) == 2
    loggers: list[_TrackingDatasetLogger] = writer._test_loggers  # type: ignore[attr-defined]
    assert sum(logger.committed for logger in loggers) == 2


def test_paired_plan_attached_to_requests(tmp_path: Path) -> None:
    recipe = _recipe_with_output(tmp_path)
    captured: list = []

    class _SpyAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            captured.append(request.paired_plan)
            if request.episode_session is not None:
                request.episode_session.log_step(_minimal_frame(), np.zeros(7), "task", 1.0)
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
        dataset_writer=_writer_for(recipe),
    )
    assert len(captured) == 5
    assert all(p.drop_u == captured[0].drop_u for p in captured)
