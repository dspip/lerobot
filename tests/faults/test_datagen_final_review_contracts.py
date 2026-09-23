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

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lerobot.faults.datagen.controllers.smolvla import (
    SmolVLADatagenAdapter,
    smolvla_behavioral_outcome,
)
from lerobot.faults.datagen.dataset_writer import RunDatasetWriter
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.manifest import read_run_manifest
from lerobot.faults.datagen.paired_context import build_paired_episode_plan, resolve_init_state_id
from lerobot.faults.datagen.path_drop import EligiblePath, eligible_path
from lerobot.faults.datagen.recipe import (
    DatagenController,
    RecipeError,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)
from lerobot.faults.datagen.runner import DropDatagenRunnerError, run_drop_datagen_matrix
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment
from tests.faults.test_datagen_recipe import _valid_unified_recipe, _write_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"

LIBERO_TASK_DESCRIPTION = "pick up the alphabet soup and place it in the basket"


def test_post_drop_dwell_steps_must_be_at_least_one(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["post_drop"] = {"dwell_steps": 0}
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="dwell_steps"):
        load_drop_datagen_recipe(path)


def test_base_seed_must_be_non_negative(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["recording"] = {
        **payload["recording"],
        "base_seed": -1,
    }
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="base_seed"):
        load_drop_datagen_recipe(path)


def test_rejects_unsupported_task_or_object(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe(task="libero_spatial")
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="task"):
        load_drop_datagen_recipe(path)

    path2 = tmp_path / "recipe2.json"
    payload2 = _valid_unified_recipe(object_names=["milk_1"])
    _write_recipe(path2, payload2)
    with pytest.raises(RecipeError, match="object_names"):
        load_drop_datagen_recipe(path2)


def test_rejects_unknown_recipe_section_keys(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["placement"]["typo_field"] = 1
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="unknown"):
        load_drop_datagen_recipe(path)


def test_eligible_phases_filters_path_pieces() -> None:
    carry = CarryPath(
        segments=(
            PathSegment("lift", (0.55, 0.05, 0.20), (0.10, 0.05, 0.25)),
            PathSegment("to_basket_via", (0.10, 0.05, 0.25), (0.06, 0.04, 0.24)),
            PathSegment("to_basket_hover", (0.06, 0.04, 0.24), (0.02, 0.02, 0.22)),
        ),
        requested_transport_offset_m=0.06,
        resolved_transport_offset_m=0.06,
        fallback=False,
    )
    import numpy as np

    basket_xy = np.array([0.0, 0.0])
    keepout = 0.05
    via_only = eligible_path(
        carry,
        basket_xy=basket_xy,
        keepout_m=keepout,
        eligible_phases=("to_basket_via",),
    )
    lift_only = eligible_path(
        carry,
        basket_xy=basket_xy,
        keepout_m=keepout,
        eligible_phases=("lift",),
    )
    assert isinstance(via_only, EligiblePath)
    assert via_only.pieces
    assert all(piece.segment_name == "to_basket_via" for piece in via_only.pieces)
    assert all(piece.segment_name == "lift" for piece in lift_only.pieces)
    assert {p.segment_name for p in via_only.pieces}.isdisjoint({p.segment_name for p in lift_only.pieces})


def test_resolve_init_state_id_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError, match="num_init_states"):
        resolve_init_state_id(9000, 0)


def test_smolvla_behavioral_outcome_not_pipeline_failed() -> None:
    outcome = smolvla_behavioral_outcome(
        {
            "behavioral_success": False,
            "checks": {
                "fault_triggered": False,
                "grasped_before_drop": True,
            },
        }
    )
    assert outcome == "drop_not_triggered"
    assert outcome != "pipeline_failed"


@pytest.mark.parametrize("manifest_index", range(5))
def test_smolvla_adapter_uses_episode_and_drop_seeds(tmp_path: Path, manifest_index: int) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    manifest = manifests[manifest_index]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    captured: dict = {}

    def _pipe(_output_dir: Path, **kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {
            "behavioral_success": True,
            "checks": {"object_in_basket": True},
            "motion_profile": {"speed_multiplier": plan.motion_profile.speed_multiplier},
        }

    adapter = SmolVLADatagenAdapter(recipe, pipeline_runner=_pipe)
    request = EpisodeRequest(
        recipe=recipe,
        manifest=manifest,
        object_name="alphabet_soup_1",
        output_dir=tmp_path / "ep",
        paired_plan=plan,
        shared_layout={"alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        device="cpu",
    )
    adapter.run_episode(request)
    assert captured["episode_seed"] == manifest.episode_seed
    assert captured["drop_seed"] == manifest.drop_seed
    assert captured.get("seed") == manifest.episode_seed
    assert captured.get("task_description") is None
    assert captured["motion_profile"] == plan.motion_profile


@pytest.mark.parametrize("manifest_index", (0, 1))
def test_simple_ik_adapter_uses_episode_and_drop_seeds(tmp_path: Path, manifest_index: int) -> None:
    from lerobot.faults.datagen.controllers.simple_ik import SimpleIKDatagenAdapter

    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    manifest = manifests[manifest_index]
    assert manifest.controller == DatagenController.SIMPLE_IK
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    captured: dict = {}

    def _fake_loop(*_args, **kwargs):  # noqa: ANN003
        captured.update(kwargs)
        from lerobot.faults.datagen.controllers import simple_ik as mod

        return mod.SimpleIKEpisodeFacts(True, "recovery_completed_in_basket", None, None, 0)

    import lerobot.faults.datagen.controllers.simple_ik as simple_ik_mod

    adapter = SimpleIKDatagenAdapter(recipe)
    mock_env = MagicMock()
    mock_env.fault = MagicMock()
    mock_env.fault.set_recovery_motion_profile = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simple_ik_mod, "run_simple_ik_episode_loop", _fake_loop)
        mp.setattr(simple_ik_mod, "make_env", lambda *a, **k: {"libero_object": {0: MagicMock()}})
        mp.setattr(simple_ik_mod, "DropRecoveryEnvWrapper", lambda *a, **k: mock_env)
        mp.setattr(
            simple_ik_mod,
            "unwrap_libero_env",
            lambda v: MagicMock(_init_states=[0], init_state_id=0),
        )
        mp.setattr(simple_ik_mod, "get_robosuite_env", lambda e, i=0: MagicMock())
        mp.setattr(simple_ik_mod, "apply_serializable_layout", lambda *a, **k: None)
        mp.setattr(simple_ik_mod, "read_control_freq", lambda rs: recipe.control_hz)
        mp.setattr(simple_ik_mod, "read_libero_task_description", lambda v: LIBERO_TASK_DESCRIPTION)
        mp.setattr(
            simple_ik_mod, "_new_planner", lambda *a, **k: MagicMock(phase_name="lift", carry_path=None)
        )
        mock_env.reset = MagicMock(return_value=({}, {}))
        request = EpisodeRequest(
            recipe=recipe,
            manifest=manifest,
            object_name="alphabet_soup_1",
            output_dir=tmp_path / "ep",
            paired_plan=plan,
            shared_layout={"alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        )
        adapter.run_episode(request)
    mock_env.fault.set_recovery_motion_profile.assert_called_once()
    profile_call = mock_env.fault.set_recovery_motion_profile.call_args
    assert profile_call.kwargs["speed_multiplier"] == plan.motion_profile.speed_multiplier
    mock_env.reset.assert_called_once_with(seed=plan.episode_seed)
    import numpy as np

    drop_rng = captured["drop_rng"]
    assert isinstance(drop_rng, np.random.Generator)
    np.testing.assert_array_equal(
        drop_rng.integers(0, 100, size=3),
        np.random.default_rng(plan.drop_seed).integers(0, 100, size=3),
    )
    assert captured["task"] == LIBERO_TASK_DESCRIPTION


def test_simple_ik_rejects_control_hz_mismatch(tmp_path: Path) -> None:
    from lerobot.faults.datagen.controllers.simple_ik import SimpleIKDatagenAdapter

    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    import lerobot.faults.datagen.controllers.simple_ik as simple_ik_mod

    mock_env = MagicMock()
    mock_env.fault = MagicMock()
    adapter = SimpleIKDatagenAdapter(recipe)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(simple_ik_mod, "make_env", lambda *a, **k: {"libero_object": {0: MagicMock()}})
        mp.setattr(simple_ik_mod, "DropRecoveryEnvWrapper", lambda *a, **k: mock_env)
        mp.setattr(
            simple_ik_mod,
            "unwrap_libero_env",
            lambda v: MagicMock(_init_states=[0], init_state_id=0),
        )
        mp.setattr(simple_ik_mod, "get_robosuite_env", lambda e, i=0: MagicMock())
        mp.setattr(simple_ik_mod, "apply_serializable_layout", lambda *a, **k: None)
        mp.setattr(simple_ik_mod, "read_control_freq", lambda rs: 30)
        mp.setattr(simple_ik_mod, "_new_planner", lambda *a, **k: MagicMock())
        mock_env.reset = MagicMock(return_value=({}, {}))
        request = EpisodeRequest(
            recipe=recipe,
            manifest=manifest,
            object_name="alphabet_soup_1",
            output_dir=tmp_path / "ep",
            paired_plan=plan,
            shared_layout={"alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        )
        with pytest.raises(ValueError, match="control_hz"):
            adapter.run_episode(request)


def test_run_manifest_tracks_in_progress_and_aborted(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    recipe = recipe.__class__(
        **{
            **recipe.__dict__,
            "recording": recipe.recording.__class__(
                base_seed=9000,
                output_dir=str(tmp_path / "out"),
                dataset_fps=10,
            ),
        }
    )
    writer = RunDatasetWriter(
        recipe,
        logger_factory=lambda root, repo_id, **_kw: MagicMock(
            log_step=MagicMock(),
            clear_open_episode=MagicMock(),
            dataset_episode_index_on_commit=lambda: 0,
            end_episode=MagicMock(),
            finalize=MagicMock(),
        ),
    )

    class _FailAdapter:
        def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
            raise RuntimeError("boom")

    from lerobot.faults.datagen.layout_provider import LayoutProviderContext

    def _layout(_ctx: LayoutProviderContext) -> dict:
        return {"alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}

    with pytest.raises(DropDatagenRunnerError):
        run_drop_datagen_matrix(
            recipe,
            logical_episode_indices=(0,),
            adapter_factories={
                DatagenController.SIMPLE_IK: lambda _r: _FailAdapter(),
                DatagenController.SMOLVLA: lambda _r: _FailAdapter(),
            },
            layout_provider=_layout,
            init_state_count_provider=lambda _r: 50,
            dataset_writer=writer,
        )
    manifest_path = tmp_path / "out" / "run_manifest.json"
    assert manifest_path.is_file()
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert raw["run_status"] == "aborted"
    assert raw.get("error_summary")


def test_smolvla_adapter_reuses_policy_resources(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    from types import SimpleNamespace

    load_calls = {"n": 0}
    bundle = SimpleNamespace(
        policy_path=recipe.smolvla.policy_path,
        device="cpu",
    )

    def _fake_load(**_kwargs):  # noqa: ANN003
        load_calls["n"] += 1
        return bundle

    def _pipe(_output_dir: Path, **kwargs):  # noqa: ANN003
        assert kwargs["policy_resources"] is bundle
        return {"behavioral_success": True, "checks": {}}

    import lerobot.faults.datagen.controllers.smolvla as smolvla_mod

    adapter = SmolVLADatagenAdapter(recipe, pipeline_runner=_pipe)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(smolvla_mod, "load_smolvla_policy_resources", _fake_load)
        for manifest in manifests[:2]:
            plan = build_paired_episode_plan(
                recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
            )
            request = EpisodeRequest(
                recipe=recipe,
                manifest=manifest,
                object_name="alphabet_soup_1",
                output_dir=tmp_path / manifest.post_drop_mode.value,
                paired_plan=plan,
                shared_layout={
                    "alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}
                },
                device="cpu",
            )
            adapter.run_episode(request)
    assert load_calls["n"] == 1


def test_read_run_manifest_includes_run_status(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_text(
        json.dumps(
            {
                "recipe_name": "x",
                "base_seed": 1,
                "output_dir": str(tmp_path),
                "run_status": "complete",
                "error_summary": None,
                "episodes": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = read_run_manifest(path)
    assert loaded.run_status == "complete"
