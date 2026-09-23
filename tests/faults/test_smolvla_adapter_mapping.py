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

from lerobot.faults.datagen.controllers.smolvla import SmolVLADatagenAdapter
from lerobot.faults.datagen.episode import EpisodeRequest
from lerobot.faults.datagen.paired_context import build_paired_episode_plan
from lerobot.faults.datagen.recipe import load_drop_datagen_recipe, paired_episode_seed_manifests

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def test_smolvla_adapter_passes_sampled_target_to_pipeline(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[2]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    captured: dict = {}

    def _fake_pipeline(output_dir: Path, **kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return {"behavioral_success": True, "fault_config": {"post_drop_dwell_steps": 80}}

    adapter = SmolVLADatagenAdapter(recipe, pipeline_runner=_fake_pipeline)
    layout = {"alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}}
    request = EpisodeRequest(
        recipe=recipe,
        manifest=manifest,
        object_name="alphabet_soup_1",
        output_dir=tmp_path,
        paired_plan=plan,
        shared_layout=layout,
        device="cpu",
    )
    result = adapter.run_episode(request)
    assert result.success is True
    assert captured["drop_xy_target_m"] == pytest.approx(plan.smolvla_target.target_m)
    assert captured["init_state_id"] == plan.init_state_id
    assert captured["min_drop_distance_from_basket_m"] == recipe.smolvla.min_drop_distance_from_basket_m
    assert captured["shared_layout"] == layout


def test_smolvla_adapter_maps_unsuccessful_summary_without_raising(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[2]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )

    def _fail_summary(output_dir: Path, **kwargs):  # noqa: ANN003
        return {"behavioral_success": False, "success": False}

    adapter = SmolVLADatagenAdapter(recipe, pipeline_runner=_fail_summary)
    request = EpisodeRequest(
        recipe=recipe,
        manifest=manifest,
        object_name="alphabet_soup_1",
        output_dir=tmp_path,
        paired_plan=plan,
        shared_layout={"alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        device="cpu",
    )
    result = adapter.run_episode(request)
    assert result.success is False
    assert result.outcome == "pipeline_failed"


def test_smolvla_adapter_propagates_real_pipeline_exceptions(tmp_path: Path) -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[2]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )

    def _boom(_output_dir: Path, **_kwargs):  # noqa: ANN003
        raise RuntimeError("pipeline exploded")

    adapter = SmolVLADatagenAdapter(recipe, pipeline_runner=_boom)
    request = EpisodeRequest(
        recipe=recipe,
        manifest=manifest,
        object_name="alphabet_soup_1",
        output_dir=tmp_path,
        paired_plan=plan,
        shared_layout={"alphabet_soup_1": {"pos": [0.0, 0.0, 0.0], "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        device="cpu",
    )
    import pytest

    with pytest.raises(RuntimeError, match="pipeline exploded"):
        adapter.run_episode(request)
