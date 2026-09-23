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

"""Datagen recipe JSON load and post-drop mode sampling (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lerobot.faults.recovery.recording_recipe import (
    load_datagen_recipe,
    sample_post_drop_mode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_simpleik_datagen.json"


def test_load_default_recipe_dwell_and_continue_mode() -> None:
    recipe = load_datagen_recipe(DEFAULT_RECIPE)
    assert recipe["post_drop"]["dwell_steps"] == 80
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert sample_post_drop_mode(rng, recipe["post_drop"]["mode_weights"]) == "continue_then_ik"


def test_sample_reset_only() -> None:
    weights = {"continue_then_ik": 0.0, "reset_then_ik": 1.0, "immediate_ik": 0.0}
    rng = np.random.default_rng(1)
    for _ in range(10):
        assert sample_post_drop_mode(rng, weights) == "reset_then_ik"


def test_invalid_weights_raise() -> None:
    base = {"continue_then_ik": 1.0, "reset_then_ik": 0.0, "immediate_ik": 0.0}
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Weight"):
        sample_post_drop_mode(rng, {**base, "reset_then_ik": -0.1})
    with pytest.raises(ValueError, match="sum"):
        sample_post_drop_mode(rng, {"continue_then_ik": 0.0, "reset_then_ik": 0.0, "immediate_ik": 0.0})
    with pytest.raises(ValueError, match="Unknown"):
        sample_post_drop_mode(rng, {**base, "hover_wander": 1.0})


def test_extra_top_level_key_still_loads(tmp_path: Path) -> None:
    data = json.loads(DEFAULT_RECIPE.read_text(encoding="utf-8"))
    data["teammate_pose_block"] = {"note": "future merge"}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_datagen_recipe(path)
    assert loaded["post_drop"]["dwell_steps"] == 80
    assert loaded["teammate_pose_block"]["note"] == "future merge"
