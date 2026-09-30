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

from lerobot_faults.datagen.paired_context import build_paired_episode_plan
from lerobot_faults.datagen.recipe import (
    DatagenController,
    PostDropMode,
    RecipeError,
    allocate_episode_counts,
    effective_post_drop_dwell_steps,
    expand_experiment_matrix,
    legacy_drop_recipe,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"
IK_RANDOM_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "alphabet_soup_ik_random.json"


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
            "use_stock_layout": False,
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
            "episodes": 5,
        },
        "experiment_matrix": [
            {"controller": "simple_ik", "post_drop_mode": "immediate_ik", "drop": True, "weight": 1},
            {"controller": "simple_ik", "post_drop_mode": "continue_then_ik", "drop": True, "weight": 1},
            {"controller": "simple_ik", "post_drop_mode": "reset_then_ik", "drop": True, "weight": 1},
            {"controller": "simple_ik", "post_drop_mode": "immediate_ik", "drop": False, "weight": 1},
            {"controller": "simple_ik", "post_drop_mode": "immediate_smolvla", "drop": True, "weight": 1},
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
    assert recipe.recording.episodes == 100
    assert all(variant.weight == 1 for variant in recipe.experiment_matrix)
    assert all(variant.episodes == 20 for variant in recipe.experiment_matrix)
    path_drop = recipe.simple_ik.path_drop
    assert path_drop is not None
    assert path_drop.eligible_phases == ("lift", "to_container")
    assert legacy_drop_recipe(recipe).hard_keepout_floor_m == 0.22


def test_legacy_recipe_uses_row_episode_counts_as_dataset_total(tmp_path: Path) -> None:
    payload = _valid_unified_recipe()
    del payload["recording"]["episodes"]
    legacy_matrix = []
    for entry in payload["experiment_matrix"]:
        legacy_entry = dict(entry)
        legacy_entry["episodes"] = legacy_entry.pop("weight")
        legacy_matrix.append(legacy_entry)
    payload["experiment_matrix"] = legacy_matrix
    path = tmp_path / "legacy_recipe.json"
    _write_recipe(path, payload)

    recipe = load_drop_datagen_recipe(path)

    assert recipe.recording.episodes == 5
    assert [variant.episodes for variant in recipe.experiment_matrix] == [1, 1, 1, 1, 1]


def test_weighted_recipe_requires_recording_episode_total(tmp_path: Path) -> None:
    payload = _valid_unified_recipe()
    del payload["recording"]["episodes"]
    path = tmp_path / "weighted_recipe.json"
    _write_recipe(path, payload)

    with pytest.raises(RecipeError, match="recording.episodes is required"):
        load_drop_datagen_recipe(path)


def test_alphabet_soup_ik_random_recipe_is_nominal_only() -> None:
    recipe = load_drop_datagen_recipe(IK_RANDOM_RECIPE)
    assert recipe.object_names == ("alphabet_soup_1",)
    assert recipe.simple_ik.trajectory_randomization_enabled is True
    assert recipe.recording.output_dir == "outputs/alphabet_soup_success_ik_random"
    assert recipe.recording.base_seed != 9000
    assert len(recipe.experiment_matrix) == 1
    row = recipe.experiment_matrix[0]
    assert row.controller is DatagenController.SIMPLE_IK
    assert row.drop is False
    assert row.weight == 1
    assert row.episodes == 100
    assert recipe.recording.episodes == 100
    runs = expand_experiment_matrix(recipe)
    assert len(runs) == 100
    assert all(not run.drop for run in runs)


def test_expand_matrix_stable_order_and_episode_count() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    runs = expand_experiment_matrix(recipe)
    assert len(runs) == 100
    assert sum(r.episodes for r in recipe.experiment_matrix) == 100
    triples = [(r.controller, r.post_drop_mode, r.drop) for r in runs]
    expected_triple = [
        (DatagenController.SIMPLE_IK, PostDropMode.IMMEDIATE_IK, True),
        (DatagenController.SIMPLE_IK, PostDropMode.CONTINUE_THEN_IK, True),
        (DatagenController.SIMPLE_IK, PostDropMode.RESET_THEN_IK, True),
        (DatagenController.SIMPLE_IK, PostDropMode.IMMEDIATE_IK, False),
        (DatagenController.SIMPLE_IK, PostDropMode.IMMEDIATE_SMOLVLA, True),
    ]
    for triple in expected_triple:
        assert triples.count(triple) == 20
    assert triples[:20] == [expected_triple[0]] * 20
    assert [r.episode_index for r in runs[:5]] == [0, 1, 2, 3, 4]


def test_accepts_simple_ik_reset_then_ik_and_immediate_smolvla(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["experiment_matrix"] = [
        {"controller": "simple_ik", "post_drop_mode": "reset_then_ik", "drop": True, "weight": 1},
        {"controller": "simple_ik", "post_drop_mode": "immediate_smolvla", "drop": True, "weight": 1},
    ]
    payload["recording"]["episodes"] = 2
    _write_recipe(path, payload)
    recipe = load_drop_datagen_recipe(path)
    assert [v.post_drop_mode for v in recipe.experiment_matrix] == [
        PostDropMode.RESET_THEN_IK,
        PostDropMode.IMMEDIATE_SMOLVLA,
    ]
    assert all(v.controller is DatagenController.SIMPLE_IK for v in recipe.experiment_matrix)


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
        {"controller": controller, "post_drop_mode": mode, "drop": True, "weight": 1},
    ]
    _write_recipe(path, payload)
    with pytest.raises(RecipeError):
        load_drop_datagen_recipe(path)


def test_effective_dwell_steps() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    assert effective_post_drop_dwell_steps(recipe, PostDropMode.IMMEDIATE_IK) == 0
    assert effective_post_drop_dwell_steps(recipe, PostDropMode.CONTINUE_THEN_IK) == 80
    assert effective_post_drop_dwell_steps(recipe, PostDropMode.RESET_THEN_IK) == 80
    assert effective_post_drop_dwell_steps(recipe, PostDropMode.IMMEDIATE_SMOLVLA) == 0


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
    assert {m.drop for m in manifests} == {True, False}
    drop_true_u = {m.drop for m in manifests if m.post_drop_mode is PostDropMode.IMMEDIATE_IK}
    assert drop_true_u == {True, False}


def test_paired_seed_manifests_deterministic_and_vary_by_episode(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["recording"]["episodes"] = 10
    payload["experiment_matrix"] = [{**entry, "weight": 2} for entry in payload["experiment_matrix"]]
    _write_recipe(path, payload)
    recipe = load_drop_datagen_recipe(path)
    assert all(variant.episodes == 2 for variant in recipe.experiment_matrix)
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
    payload = _valid_unified_recipe(object_names=["not_a_libero_object_1"])
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="object_names"):
        load_drop_datagen_recipe(path)

    path = tmp_path / "cream_ok.json"
    payload = _valid_unified_recipe(object_names=["cream_cheese_1"], task_id=1)
    _write_recipe(path, payload)
    recipe = load_drop_datagen_recipe(path)
    assert recipe.object_names == ("cream_cheese_1",)
    assert recipe.task_id == 1

    path = tmp_path / "cream_bad_task.json"
    payload = _valid_unified_recipe(object_names=["cream_cheese_1"], task_id=0)
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="task_id"):
        load_drop_datagen_recipe(path)

    from lerobot_faults.datagen.libero_object_tasks import supported_object_names

    path = tmp_path / "all_objects.json"
    payload = _valid_unified_recipe(
        object_names=list(supported_object_names()),
        task_id=0,
    )
    _write_recipe(path, payload)
    multi = load_drop_datagen_recipe(path)
    assert multi.task_id == 0
    assert len(multi.object_names) == 10


def test_allows_shorter_matrix_if_triples_unique(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["experiment_matrix"] = payload["experiment_matrix"][:2]
    _write_recipe(path, payload)
    recipe = load_drop_datagen_recipe(path)
    assert len(recipe.experiment_matrix) == 2


def test_rejects_duplicate_experiment_matrix_pair(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    matrix = list(payload["experiment_matrix"])
    matrix[1] = dict(matrix[0])
    payload["experiment_matrix"] = matrix
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="duplicate"):
        load_drop_datagen_recipe(path)


def test_unequal_weights_allocate_by_share(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    matrix = list(payload["experiment_matrix"])
    matrix[0] = {**matrix[0], "weight": 2}
    payload["experiment_matrix"] = matrix
    payload["recording"]["episodes"] = 6
    _write_recipe(path, payload)
    recipe = load_drop_datagen_recipe(path)
    counts = [variant.episodes for variant in recipe.experiment_matrix]
    assert counts == [2, 1, 1, 1, 1]
    assert sum(counts) == 6


def test_allocate_episode_counts_equal_weights() -> None:
    assert allocate_episode_counts(100, (1, 1, 1, 1, 1)) == (20, 20, 20, 20, 20)


def test_allocate_episode_counts_remainder() -> None:
    assert allocate_episode_counts(11, (1, 1, 1, 1, 1)) == (3, 2, 2, 2, 2)
    assert sum(allocate_episode_counts(11, (1, 1, 1, 1, 1))) == 11


def test_allocate_episode_counts_rejects_too_small_total() -> None:
    with pytest.raises(RecipeError, match="too small to give every experiment_matrix row"):
        allocate_episode_counts(3, (1, 1, 1, 1, 1))


def test_matrix_rejects_weight_and_episodes_together(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    matrix = list(payload["experiment_matrix"])
    matrix[0] = {**matrix[0], "weight": 1, "episodes": 1}
    payload["experiment_matrix"] = matrix
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="weight or episodes, not both"):
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


def test_matrix_drop_flag_required(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["experiment_matrix"] = [
        {"controller": "simple_ik", "post_drop_mode": "immediate_ik", "weight": 1},
    ]
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="drop"):
        load_drop_datagen_recipe(path)


def test_paired_plan_honors_matrix_drop_not_q(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe(q=0.0)
    _write_recipe(path, payload)
    recipe = load_drop_datagen_recipe(path)
    drop_manifest, *_, no_drop_manifest, type5 = paired_episode_seed_manifests(
        recipe, logical_episode_index=0
    )
    del type5
    drop_plan = build_paired_episode_plan(
        recipe, manifest=drop_manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    no_drop_plan = build_paired_episode_plan(
        recipe, manifest=no_drop_manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    assert drop_plan.drop_decision.drop is True
    assert drop_plan.drop_u is not None
    assert no_drop_plan.drop_decision.drop is False
    assert drop_plan.drop_seed == no_drop_plan.drop_seed


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


# ---------------------------------------------------------------------------
# Task 3: use_stock_layout
# ---------------------------------------------------------------------------


def test_can_drop_recipe_has_use_stock_layout_false() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    assert recipe.smolvla.use_stock_layout is False
    assert [v.drop for v in recipe.experiment_matrix] == [True, True, True, False, True]
    assert recipe.experiment_matrix[2].post_drop_mode is PostDropMode.RESET_THEN_IK
    assert recipe.experiment_matrix[4].post_drop_mode is PostDropMode.IMMEDIATE_SMOLVLA


def test_smolvla_rejects_non_bool_use_stock_layout(tmp_path: Path) -> None:
    """use_stock_layout must be a JSON boolean, not a string or number."""
    path = tmp_path / "recipe.json"
    payload = _valid_unified_recipe()
    payload["smolvla"]["use_stock_layout"] = "true"  # type: ignore[index]
    _write_recipe(path, payload)
    with pytest.raises(RecipeError, match="use_stock_layout"):
        load_drop_datagen_recipe(path)
