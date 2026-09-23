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

"""Public drop-datagen entry points (recipe + CLI only)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPES_DIR = REPO_ROOT / "examples" / "faults" / "recipes"
CAN_DROP_RECIPE = RECIPES_DIR / "can_drop_datagen.json"


def test_sole_checked_in_drop_datagen_recipe() -> None:
    json_files = sorted(p.name for p in RECIPES_DIR.glob("*.json"))
    assert json_files == ["can_drop_datagen.json"]
    assert CAN_DROP_RECIPE.is_file()


def test_unified_cli_entry_point() -> None:
    from examples.faults import run_drop_datagen as cli

    assert callable(cli.main)
    from lerobot.faults.datagen.recipe import load_drop_datagen_recipe

    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    assert recipe.name == "can_drop_datagen"


def test_legacy_datagen_compatibility_apis_removed() -> None:
    import lerobot.faults.datagen as datagen_pkg
    import lerobot.faults.datagen.recipe as recipe_mod
    import lerobot.faults.recovery.recording_recipe as recording_mod

    for name in (
        "load_datagen_recipe",
        "load_legacy_post_drop_recipe",
        "load_legacy_simple_ik_recipe",
        "load_recipe",
        "sample_post_drop_mode",
        "LegacyPostDropConfig",
        "DatagenRecipe",
    ):
        assert not hasattr(recording_mod, name), name
        assert not hasattr(datagen_pkg, name), name
        assert not hasattr(recipe_mod, name), name

    with pytest.raises(ImportError):
        from lerobot.faults.recovery import load_datagen_recipe  # noqa: F401

    with pytest.raises(ImportError):
        from lerobot.faults.recovery import sample_post_drop_mode  # noqa: F401
