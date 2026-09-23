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

import pytest

from lerobot.faults.datagen.recipe import (
    DatagenController,
    PostDropMode,
    RecipeError,
    effective_post_drop_dwell_steps,
    expand_experiment_matrix,
    legacy_drop_recipe,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def _valid_unified_recipe(**overrides: object) -> dict[str, object]:
    recipe: dict[str, object] = {
        "name": "can_drop_datagen",
        "task": "libero_object",
        "task_id": 0,
        "control_hz": 20,
        "q": 1.0,
        "object_names": ["alphabet_soup_1"],
        "basket_name": "basket_1",
        "placement": {
            "xy_range_m": 0.12,
            "min_basket_clearance_m": 0.35,
            "distractor_basket_clearance_m": 0.15,
            "min_pairwise_clearance_m": 0.08,
            "yaw_range_deg": [-180.0, 180.0],
            "max_attempts": 200,
        },
        "simple_ik": {
            "trajectory_randomization_enabled": True,
            "pickup_via_offset_m": 0.03,
            "transport_via_offset_m": 0.06,
            "arm_posture_noise_deg": 3.0,
            "speed_multiplier_range": [0.9, 1.1],
            "waypoint_blend_radius_m": 0.02,
            "path_drop": {
                "eligible_phases": ["lift", "to_container"],
                "min_drop_distance_from_basket_m": 0.30,
                "hard_keepout_floor_m": 0.22,
            },
        },
        "smolvla": {
            "policy_path": "lerobot/smolvla_libero",
            "post_grasp_delay_steps": 0,
            "min_drop_distance_from_basket_m": 0.30,
            "drop_xy_bands": [
                {"name": "lift", "min_m": 0.48, "max_m": 0.52},
                {"name": "early", "min_m": 0.42, "max_m": 0.46},
                {"name": "mid", "min_m": 0.34, "max_m": 0.38},
                {"name": "late", "min_m": 0.30, "max_m": 0.33},
            ],
        },
        "post_drop": {"dwell_steps": 80},
        "recording": {
            "base_seed": 9000,
            "output_dir": "outputs/can_drop_datagen",
            "dataset_fps": 10,
        },
        "experiment_matrix": [
            {"controller": "simple_ik", "post_drop_mode": "immediate_ik", "episodes": 1},
            {"controller": "simple_ik", "post_drop_mode": "continue_then_ik", "episodes": 1},
            {"controller": "smolvla", "post_drop_mode": "immediate_ik", "episodes": 1},
            {"controller": "smolvla", "post_drop_mode": "continue_then_ik", "episodes": 1},
            {"controller": "smolvla", "post_drop_mode": "reset_then_ik", "episodes": 1},
        ],
    }
    recipe.update(overrides)
    return recipe


def _write_recipe(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_can_drop_recipe_file_loads() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    assert recipe.object_names == ("alphabet_soup_1",)
    assert recipe.smolvla.policy_path == "lerobot/smolvla_libero"
    assert recipe.post_drop.dwell_steps == 80
    assert len(recipe.experiment_matrix) == 5
    path_drop = recipe.simple_ik.path_drop
    assert path_drop is not None
    assert path_drop.eligible_phases == ("lift", "to_container")
    assert legacy_drop_recipe(recipe).hard_keepout_floor_m == 0.22


def test_expand_matrix_stable_order_and_episode_count() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    runs = expand_experiment_matrix(recipe)
    assert len(runs) == 5
    assert [(r.controller, r.post_drop_mode) for r in runs] == [
        (DatagenController.SIMPLE_IK, PostDropMode.IMMEDIATE_IK),
        (DatagenController.SIMPLE_IK, PostDropMode.CONTINUE_THEN_IK),
        (DatagenController.SMOLVLA, PostDropMode.IMMEDIATE_IK),
        (DatagenController.SMOLVLA, PostDropMode.CONTINUE_THEN_IK),
        (DatagenController.SMOLVLA, PostDropMode.RESET_THEN_IK),
    ]
    assert all(r.episode_index == 0 for r in runs)
    assert all(r.logical_episode_index == 0 for r in runs)


def test_rejects_simple_ik_reset_then_ik(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["experiment_matrix"] = [
        {"controller": "simple_ik", "post_drop_mode": "reset_then_ik", "episodes": 1},
    ]
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="reset_then_ik"):
        load_drop_datagen_recipe(path)


@pytest.mark.parametrize(
    ("controller", "mode"),
    [
        ("hover_wander", "continue_then_ik"),
        ("simple_ik", "hover_wander"),
    ],
)
def test_rejects_unknown_controller_or_mode(tmp_path: Path, controller: str, mode: str) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["experiment_matrix"] = [
        {"controller": controller, "post_drop_mode": mode, "episodes": 1},
    ]
    _write_recipe(path, payload)
    with pytest.raises(RecipeError):
        load_drop_datagen_recipe(path)


def test_effective_dwell_steps() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    assert effective_post_drop_dwell_steps(recipe, PostDropMode.IMMEDIATE_IK) == 0
    assert effective_post_drop_dwell_steps(recipe, PostDropMode.CONTINUE_THEN_IK) == 80
    assert effective_post_drop_dwell_steps(recipe, PostDropMode.RESET_THEN_IK) == 80


def test_paired_seed_manifests_share_layout_and_drop(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    _write_recipe(path, _valid_unified_recipe())
    recipe = load_drop_datagen_recipe(path)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    assert len(manifests) == 5
    layout = {m.layout_seed for m in manifests}
    drop = {m.drop_seed for m in manifests}
    assert len(layout) == 1
    assert len(drop) == 1
    controller = {m.controller_seed for m in manifests}
    assert len(controller) == 5


def test_paired_seed_manifests_deterministic_and_vary_by_episode(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["experiment_matrix"] = [{**entry, "episodes": 2} for entry in payload["experiment_matrix"]]
    _write_recipe(path, payload)
    recipe = load_drop_datagen_recipe(path)
    a = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    b = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    c = paired_episode_seed_manifests(recipe, logical_episode_index=1)
    assert [(m.layout_seed, m.drop_seed, m.controller_seed) for m in a] == [
        (m.layout_seed, m.drop_seed, m.controller_seed) for m in b
    ]
    assert a[0].layout_seed != c[0].layout_seed
    assert a[0].drop_seed != c[0].drop_seed


def test_object_names_rejects_non_poc_values(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe(object_names=["alphabet_soup_1", "milk_1"])
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="object_names"):
        load_drop_datagen_recipe(path)


def test_rejects_missing_experiment_matrix_pair(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["experiment_matrix"] = payload["experiment_matrix"][:4]
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="exactly 5"):
        load_drop_datagen_recipe(path)


def test_rejects_duplicate_experiment_matrix_pair(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    matrix = list(payload["experiment_matrix"])
    matrix[1] = dict(matrix[0])
    payload["experiment_matrix"] = matrix
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="duplicate"):
        load_drop_datagen_recipe(path)


def test_rejects_unequal_matrix_episode_counts(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    matrix = list(payload["experiment_matrix"])
    matrix[0] = {**matrix[0], "episodes": 2}
    payload["experiment_matrix"] = matrix
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="same episodes"):
        load_drop_datagen_recipe(path)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    [
        ("simple_ik.trajectory_randomization_enabled", "false"),
        ("simple_ik.trajectory_randomization_enabled", 1),
        ("name", 42),
        ("q", "1.0"),
        ("placement.xy_range_m", "0.12"),
    ],
)
def test_unified_recipe_rejects_malformed_json_types(
    tmp_path: Path,
    field_path: str,
    bad_value: object,
) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    parts = field_path.split(".")
    target: dict[str, object] = payload
    for key in parts[:-1]:
        target = target[key]  # type: ignore[index]
    target[parts[-1]] = bad_value
    _write_recipe(path, payload)
    with pytest.raises(RecipeError):
        load_drop_datagen_recipe(path)


def test_unified_simple_ik_requires_path_drop_object(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    del payload["simple_ik"]["path_drop"]
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="path_drop"):
        load_drop_datagen_recipe(path)


def test_unified_rejects_top_level_drop_section(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["drop"] = {
        "eligible_phases": ["lift"],
        "min_drop_distance_from_basket_m": 0.3,
        "hard_keepout_floor_m": 0.22,
    }
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="drop"):
        load_drop_datagen_recipe(path)
