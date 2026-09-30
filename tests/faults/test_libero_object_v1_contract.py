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

"""Contracts for the two LIBERO-Object v1 recordings: labels, held-out, resume."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lerobot_faults.datagen.dataset_writer import RunDatasetWriter, StaleRunOutputError
from lerobot_faults.datagen.episode import EpisodeRequest, EpisodeResult, select_episode_object_round_robin
from lerobot_faults.datagen.failure_segments import FailureSegmentsWriter
from lerobot_faults.datagen.layout_provider import LayoutProviderContext
from lerobot_faults.datagen.libero_object_tasks import official_task_id
from lerobot_faults.datagen.recipe import (
    DatagenController,
    RecipeError,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)
from lerobot_faults.datagen.recording_views import drop_window_bounds, tick_in_drop_window
from lerobot_faults.datagen.runner import run_drop_datagen_matrix
from lerobot_faults.datagen.scene import hide_scene_objects, sample_object_layout
from lerobot_faults.recovery.midair_drop import MidAirDropFault

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES = REPO_ROOT / "examples" / "faults" / "recipes"
SUCCESS_RECIPE = RECIPES / "libero_object_success.json"
DROP_RECIPE = RECIPES / "libero_object_drop.json"
CAN_DROP_RECIPE = RECIPES / "can_drop_datagen.json"
RECORDED = (
    "alphabet_soup_1",
    "cream_cheese_1",
    "salad_dressing_1",
    "bbq_sauce_1",
    "ketchup_1",
    "butter_1",
    "milk_1",
    "chocolate_pudding_1",
)
HELD_OUT = ("tomato_sauce_1", "orange_juice_1")


def _snap(tick: int, *, release: bool = False, event: bool = False) -> SimpleNamespace:
    return SimpleNamespace(tick=tick, drop_release=release, drop_event=event)


def test_drop_window_is_two_stored_frames_around_the_fall() -> None:
    snapshots = [
        _snap(8),
        _snap(9),
        _snap(10, release=True, event=True),
        _snap(11, event=True),
        _snap(12, event=True),
        _snap(13, event=True),
        _snap(14, event=True),
        _snap(15),
        _snap(16),
    ]
    bounds = drop_window_bounds(snapshots, pad_frames=2, dataset_stride=2)
    assert bounds == (6, 19)
    logged = list(range(0, 22, 2))
    inside = [tick for tick in logged if tick_in_drop_window(tick, bounds)]
    assert inside == [6, 8, 10, 12, 14, 16, 18]


def test_drop_window_is_empty_without_a_release() -> None:
    bounds = drop_window_bounds([_snap(0), _snap(2)], pad_frames=2, dataset_stride=2)
    assert bounds is None
    assert tick_in_drop_window(2, bounds) is False


def test_cli_fps_override_writes_one_view() -> None:
    from examples.faults.run_drop_datagen import _apply_overrides

    recipe = load_drop_datagen_recipe(DROP_RECIPE)
    ten = _apply_overrides(recipe, base_seed=None, output=None, episodes=None, fps=None)
    twenty = _apply_overrides(recipe, base_seed=None, output=None, episodes=None, fps=20)
    assert ten.recording.dataset_fps == 10
    assert ten.recording.master_fps is None
    assert twenty.recording.dataset_fps == 20
    assert twenty.recording.master_fps is None


def test_success_and_drop_recipes_match_the_episode_table() -> None:
    success = load_drop_datagen_recipe(SUCCESS_RECIPE)
    drop = load_drop_datagen_recipe(DROP_RECIPE)
    assert success.object_names == RECORDED
    assert drop.object_names == RECORDED
    assert success.held_out_object_names == HELD_OUT
    assert drop.held_out_object_names == HELD_OUT
    assert success.recording.dataset_fps == 10
    assert success.recording.master_fps is None
    assert success.control_hz == 20
    assert [(row.post_drop_mode.value, row.drop, row.episodes) for row in success.experiment_matrix] == [
        ("immediate_ik", False, 400)
    ]
    assert [(row.post_drop_mode.value, row.drop, row.episodes) for row in drop.experiment_matrix] == [
        ("immediate_ik", True, 80),
        ("continue_then_ik", True, 80),
        ("reset_then_ik", True, 80),
        ("immediate_ik", False, 80),
        ("immediate_smolvla", True, 80),
    ]
    success_counts = Counter(
        select_episode_object_round_robin(index, success.object_names) for index in range(400)
    )
    drop_counts = Counter(select_episode_object_round_robin(index, drop.object_names) for index in range(80))
    assert success_counts == dict.fromkeys(RECORDED, 50)
    assert drop_counts == dict.fromkeys(RECORDED, 10)


def test_held_out_cannot_overlap_recorded_pick_targets(tmp_path: Path) -> None:
    raw = json.loads(SUCCESS_RECIPE.read_text(encoding="utf-8"))
    raw["held_out_object_names"] = ["alphabet_soup_1"]
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RecipeError, match="overlaps"):
        load_drop_datagen_recipe(path)


def test_sample_layout_omits_held_out_objects() -> None:
    recipe = load_drop_datagen_recipe(SUCCESS_RECIPE)
    captured: dict[str, list[str]] = {}

    def movable(_rs, *, basket_name: str, required_object_name: str) -> list[str]:
        del basket_name, required_object_name
        return ["alphabet_soup_1", "tomato_sauce_1", "orange_juice_1"]

    def pose(_rs, name: str) -> dict[str, np.ndarray]:
        del name
        return {"pos": np.zeros(3), "quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0])}

    def destination(_rs, _name: str, *, basket_name: str) -> np.ndarray:
        del basket_name
        return np.zeros(3)

    def sample(*, objects, **_kwargs):
        captured["names"] = [obj.name for obj in objects]
        return objects

    with (
        patch("lerobot_faults.datagen.scene.movable_object_names", movable),
        patch("lerobot_faults.datagen.scene.get_object_pose", pose),
        patch("lerobot_faults.datagen.scene.get_place_destination", destination),
        patch("lerobot_faults.datagen.scene.sample_layout", sample),
    ):
        layout = sample_object_layout(object(), recipe, "alphabet_soup_1", np.random.default_rng(0))
    assert captured["names"] == ["alphabet_soup_1"]
    assert layout is not None
    assert "tomato_sauce_1" not in layout
    assert "orange_juice_1" not in layout


def test_hide_scene_objects_clears_collision_and_skips_missing() -> None:
    class _Model:
        nbody = 2
        body_parentid = np.array([0, 0])
        geom_bodyid = np.array([1])
        geom_rgba = np.ones((1, 4))
        geom_contype = np.ones(1, dtype=int)
        geom_conaffinity = np.ones(1, dtype=int)

        def body_name2id(self, name: str) -> int:
            if name == "tomato_sauce_1_main":
                return 1
            raise KeyError(name)

    class _Env:
        def __init__(self) -> None:
            self.sim = SimpleNamespace(model=_Model())

        def get_object(self, name: str):
            if name == "tomato_sauce_1":
                return SimpleNamespace(root_body="tomato_sauce_1_main")
            raise KeyError(name)

    env = _Env()
    with patch("lerobot_faults.datagen.scene.set_object_pose") as moved:
        hide_scene_objects(env, ("tomato_sauce_1", "not_in_scene"))
    assert env.sim.model.geom_rgba[0, 3] == 0.0
    assert int(env.sim.model.geom_contype[0]) == 0
    assert int(env.sim.model.geom_conaffinity[0]) == 0
    moved.assert_called_once()


def test_immediate_smolvla_loss_mask_stays_off_after_recovery_starts() -> None:
    from tests.faults.test_midair_drop_fault import _cfg

    smolvla = MidAirDropFault(_cfg(post_drop_mode="immediate_smolvla"), num_envs=1)
    state = smolvla._states[0]
    state.triggered = True
    state.recovery_active = True
    state.drop_injection_step = False
    assert smolvla.loss_mask_for_env(0) == 0.0

    ik = MidAirDropFault(_cfg(post_drop_mode="immediate_ik"), num_envs=1)
    ik_state = ik._states[0]
    ik_state.triggered = True
    ik_state.recovery_active = True
    ik_state.drop_injection_step = False
    assert ik.loss_mask_for_env(0) == 1.0


class _ColumnLogger:
    def __init__(self, root: Path, repo_id: str, **_kwargs) -> None:
        del repo_id
        self.root = root
        self._total = 0
        self.dataset = SimpleNamespace(
            writer=SimpleNamespace(
                episode_buffer={
                    "size": 0,
                    "tick_index": [],
                    "drop_window": [],
                }
            )
        )

    def log_step(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_open_episode_bool_column(self, key: str, values: list[bool]) -> None:
        buffer = self.dataset.writer.episode_buffer
        if int(buffer["size"]) != len(values):
            raise RuntimeError("size mismatch")
        buffer[key] = values

    def dataset_episode_index_on_commit(self) -> int:
        return self._total

    def end_episode(self, *args, **kwargs) -> int:
        del args, kwargs
        index = self._total
        self._total += 1
        return index

    def clear_open_episode(self) -> None:
        return None

    def finalize(self) -> None:
        return None


def _recipe_at(tmp_path: Path):
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    return recipe.__class__(
        **{
            **recipe.__dict__,
            "recording": recipe.recording.__class__(
                base_seed=9000,
                output_dir=str(tmp_path / "datagen"),
                dataset_fps=10,
                episodes=recipe.recording.episodes,
            ),
        }
    )


def test_commit_backfills_drop_window_and_sidecar_object(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    writer = RunDatasetWriter(
        recipe,
        logger_factory=lambda root, repo_id, **kw: _ColumnLogger(root, repo_id, **kw),
    )
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=3)[0]
    session = writer.open_episode_session(manifest)
    session.log_step({}, np.zeros(7), "task", 1.0)
    for tick, release, event in (
        (8, False, False),
        (10, True, True),
        (12, False, True),
        (14, False, True),
        (16, False, False),
    ):
        session.record_tick_metadata(
            tick=tick,
            drop_release=release,
            drop_event=event,
            attempt_index=0,
            object_z=0.2,
        )
    logged_ticks = [4, 8, 10, 12, 14, 16]
    buffer = session.logger.dataset.writer.episode_buffer
    buffer["size"] = len(logged_ticks)
    buffer["tick_index"] = [np.array([tick], dtype=np.int64) for tick in logged_ticks]
    buffer["drop_window"] = [np.array([False]) for _ in logged_ticks]
    session.commit(success=True, object_name="butter_1", task_id=official_task_id("butter_1"))
    flags = buffer["drop_window"]
    # release 10, landing 16, stride 2, pad 4 ticks -> [6, 20]
    assert flags == [False, True, True, True, True, True]
    row = session.recording_views[0].segments.rows[0]
    assert row.object_name == "butter_1"
    assert row.task_id == official_task_id("butter_1")
    assert row.object_name != recipe.object_names[0]
    reloaded = FailureSegmentsWriter(session.dataset_root, stride=2, control_hz=20)
    reloaded.load_existing()
    assert reloaded.rows[0].object_name == "butter_1"
    assert reloaded.rows[0].task_id == official_task_id("butter_1")


def _fake_layout(_ctx: LayoutProviderContext) -> dict:
    return {
        "alphabet_soup_1": {
            "pos": [0.1, 0.2, 0.03],
            "quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    }


class _FakeAdapter:
    def __init__(self) -> None:
        self.seen: list[int] = []

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        self.seen.append(int(request.manifest.logical_episode_index))
        if request.episode_session is not None:
            request.episode_session.log_step({}, np.zeros(7), "task", 1.0)
        return EpisodeResult.from_run(request, success=True, outcome="ok")


def test_resume_skips_completed_variants_including_rejects(tmp_path: Path) -> None:
    recipe = _recipe_at(tmp_path)
    adapter = _FakeAdapter()
    writer = RunDatasetWriter(
        recipe,
        logger_factory=lambda root, repo_id, **kw: _ColumnLogger(root, repo_id, **kw),
    )
    run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=(0,),
        adapter_factories={DatagenController.SIMPLE_IK: lambda _recipe: adapter},
        layout_provider=_fake_layout,
        init_state_count_provider=lambda _recipe: 50,
        dataset_writer=writer,
    )
    assert adapter.seen == [0, 0, 0, 0, 0]
    kept = sum(1 for row in writer.episode_rows if row.keep)
    info = Path(recipe.recording.output_dir) / "dataset" / "meta"
    info.mkdir(parents=True, exist_ok=True)
    (info / "info.json").write_text(json.dumps({"total_episodes": kept}), encoding="utf-8")

    resumed_adapter = _FakeAdapter()
    resumed = RunDatasetWriter(
        recipe,
        logger_factory=lambda root, repo_id, **kw: _ColumnLogger(root, repo_id, **kw),
        resume=True,
    )
    assert len(resumed.completed_variant_keys()) == 5
    run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=(0, 1),
        adapter_factories={DatagenController.SIMPLE_IK: lambda _recipe: resumed_adapter},
        layout_provider=_fake_layout,
        init_state_count_provider=lambda _recipe: 50,
        dataset_writer=resumed,
    )
    assert resumed_adapter.seen == [1, 1, 1, 1, 1]

    (info / "info.json").write_text(json.dumps({"total_episodes": kept - 1}), encoding="utf-8")
    with pytest.raises(StaleRunOutputError, match="manifest kept"):
        RunDatasetWriter(recipe, resume=True)
