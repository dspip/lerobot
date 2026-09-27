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

from examples.faults import run_drop_datagen as cli
from lerobot.faults.datagen.recipe import RecipeError, load_drop_datagen_recipe

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def test_cli_rejects_non_positive_episodes_override() -> None:
    code = cli.main(
        ["--recipe", str(CAN_DROP_RECIPE), "--episodes", "0"],
    )
    assert code == 2


def test_apply_overrides_episodes_splits_can_drop_total() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    updated = cli._apply_overrides(recipe, base_seed=None, output=None, episodes=5)
    assert updated.recording.episodes == 5
    assert [variant.episodes for variant in updated.experiment_matrix] == [1, 1, 1, 1, 1]


def test_apply_overrides_episodes_too_small_for_matrix_raises() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    with pytest.raises(RecipeError, match="too small to give every experiment_matrix row"):
        cli._apply_overrides(recipe, base_seed=None, output=None, episodes=1)


def test_cli_episodes_one_returns_error_code() -> None:
    code = cli.main(["--recipe", str(CAN_DROP_RECIPE), "--episodes", "1"])
    assert code == 2


def test_smolvla_pipeline_run_pipeline_is_in_package() -> None:
    from lerobot.faults.datagen.smolvla_pipeline import run_pipeline

    assert callable(run_pipeline)
