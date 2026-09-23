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

from lerobot.faults.datagen.recipe import RecipeError, load_recipe


def _valid_recipe(**overrides: object) -> dict[str, object]:
    recipe: dict[str, object] = {
        "q": 0.5,
        "object_name": "alphabet_soup_1",
        "basket_name": "basket_1",
        "placement": {
            "xy_range_m": 0.12,
            "min_basket_clearance_m": 0.35,
            "distractor_basket_clearance_m": 0.15,
            "min_pairwise_clearance_m": 0.08,
            "yaw_range_deg": [-180, 180],
            "max_attempts": 200,
        },
        "drop": {
            "eligible_phases": ["lift", "to_container"],
            "min_drop_distance_from_basket_m": 0.30,
            "hard_keepout_floor_m": 0.22,
        },
        "simple_ik": {
            "trajectory_randomization_enabled": False,
            "pickup_via_offset_m": 0.03,
            "transport_via_offset_m": 0.06,
            "arm_posture_noise_deg": 3.0,
            "speed_multiplier_range": [0.8, 1.2],
            "waypoint_blend_radius_m": 0.04,
        },
    }
    recipe.update(overrides)
    return recipe


def _write_recipe(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_recipe_reads_q_and_object(tmp_path: Path):
    path = tmp_path / "recipe.json"
    _write_recipe(path, _valid_recipe())

    recipe = load_recipe(path)

    assert recipe.q == 0.5
    assert recipe.object_name == "alphabet_soup_1"
    assert recipe.drop.eligible_phases == ("lift", "to_container")


def test_load_recipe_missing_file():
    with pytest.raises(RecipeError, match="not found"):
        load_recipe(Path("/no/such/recipe.json"))


@pytest.mark.parametrize("q", [-0.01, 1.01])
def test_load_recipe_rejects_q_out_of_range(tmp_path: Path, q: float):
    path = tmp_path / "recipe.json"
    _write_recipe(path, _valid_recipe(q=q))

    with pytest.raises(RecipeError, match="q"):
        load_recipe(path)


def test_load_recipe_requires_object_name(tmp_path: Path):
    path = tmp_path / "recipe.json"
    payload = _valid_recipe()
    del payload["object_name"]
    _write_recipe(path, payload)

    with pytest.raises(RecipeError, match="object_name"):
        load_recipe(path)


def test_load_recipe_reads_trajectory_randomization(tmp_path: Path):
    path = tmp_path / "recipe.json"
    _write_recipe(path, _valid_recipe())

    recipe = load_recipe(path)

    assert recipe.simple_ik.trajectory_randomization_enabled is False
    assert recipe.simple_ik.pickup_via_offset_m == 0.03
    assert recipe.simple_ik.transport_via_offset_m == 0.06
    assert recipe.simple_ik.speed_multiplier_range == (0.8, 1.2)
    assert recipe.simple_ik.waypoint_blend_radius_m == 0.04


def test_load_recipe_requires_randomization_enabled(tmp_path: Path):
    path = tmp_path / "recipe.json"
    payload = _valid_recipe()
    del payload["simple_ik"]["trajectory_randomization_enabled"]
    _write_recipe(path, payload)

    with pytest.raises(RecipeError, match="trajectory_randomization_enabled"):
        load_recipe(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pickup_via_offset_m", -0.01),
        ("transport_via_offset_m", -0.01),
        ("arm_posture_noise_deg", -1.0),
        ("waypoint_blend_radius_m", -0.01),
    ],
)
def test_load_recipe_rejects_negative_motion_values(
    tmp_path: Path,
    field: str,
    value: float,
):
    path = tmp_path / "recipe.json"
    payload = _valid_recipe()
    payload["simple_ik"][field] = value
    _write_recipe(path, payload)

    with pytest.raises(RecipeError, match=field):
        load_recipe(path)


@pytest.mark.parametrize(
    "speed_range",
    [[], [0.8], [1.2, 0.8], [0.0, 1.2]],
)
def test_load_recipe_rejects_invalid_speed_range(
    tmp_path: Path,
    speed_range: list[float],
):
    path = tmp_path / "recipe.json"
    payload = _valid_recipe()
    payload["simple_ik"]["speed_multiplier_range"] = speed_range
    _write_recipe(path, payload)

    with pytest.raises(RecipeError, match="speed_multiplier_range"):
        load_recipe(path)
