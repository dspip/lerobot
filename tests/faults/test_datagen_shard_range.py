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

from fault_system.datagen.recipe import RecipeError, load_drop_datagen_recipe
from fault_system.datagen.shard_range import max_logical_episodes, validate_logical_shard_range

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def test_validate_logical_shard_range_accepts_valid_slice() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    max_ep = max_logical_episodes(recipe)
    assert validate_logical_shard_range(recipe, 0, min(2, max_ep)) == (0, min(2, max_ep))


def test_validate_logical_shard_range_rejects_inverted() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    with pytest.raises(RecipeError, match="logical-end must be >"):
        validate_logical_shard_range(recipe, 3, 3)


def test_validate_logical_shard_range_rejects_past_max() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    max_ep = max_logical_episodes(recipe)
    with pytest.raises(RecipeError, match="logical-end must be <="):
        validate_logical_shard_range(recipe, 0, max_ep + 1)
