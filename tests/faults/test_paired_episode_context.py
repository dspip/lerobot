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

from lerobot.faults.datagen.paired_context import build_paired_episode_plan
from lerobot.faults.datagen.recipe import load_drop_datagen_recipe, paired_episode_seed_manifests

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def test_paired_plan_identical_for_all_variants_same_logical_episode() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    plans = [
        build_paired_episode_plan(recipe, manifest=m, object_name="alphabet_soup_1")
        for m in manifests
    ]
    keys = (
        "init_state_id",
        "motion_profile",
        "path_trigger",
        "smolvla_target",
        "drop_decision",
    )
    for key in keys:
        ref = getattr(plans[0], key) if key != "motion_profile" else plans[0].motion_profile
        for plan in plans[1:]:
            other = getattr(plan, key) if key != "motion_profile" else plan.motion_profile
            assert other == ref


def test_controller_seed_does_not_change_pre_drop_plan() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    simple = next(m for m in manifests if m.controller.value == "simple_ik")
    smol = next(m for m in manifests if m.controller.value == "smolvla")
    assert simple.controller_seed != smol.controller_seed
    plan_a = build_paired_episode_plan(recipe, manifest=simple, object_name="alphabet_soup_1")
    plan_b = build_paired_episode_plan(recipe, manifest=smol, object_name="alphabet_soup_1")
    assert plan_a.path_trigger == plan_b.path_trigger
    assert plan_a.smolvla_target == plan_b.smolvla_target
    assert plan_a.motion_profile == plan_b.motion_profile
