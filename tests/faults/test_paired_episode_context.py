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

import numpy as np

from lerobot.faults.datagen.drop_timing import keepout_m
from lerobot.faults.datagen.paired_context import build_paired_episode_plan, resolve_path_drop_trigger
from lerobot.faults.datagen.path_drop import eligible_path
from lerobot.faults.datagen.recipe import (
    legacy_drop_recipe,
    load_drop_datagen_recipe,
    paired_episode_seed_manifests,
)
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment

REPO_ROOT = Path(__file__).resolve().parents[2]
CAN_DROP_RECIPE = REPO_ROOT / "examples" / "faults" / "recipes" / "can_drop_datagen.json"


def test_paired_plan_identical_for_all_variants_same_logical_episode() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifests = paired_episode_seed_manifests(recipe, logical_episode_index=0)
    plans = [
        build_paired_episode_plan(recipe, manifest=m, object_name="alphabet_soup_1", num_init_states=50)
        for m in manifests
    ]
    keys = (
        "init_state_id",
        "motion_profile",
        "drop_u",
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
    plan_a = build_paired_episode_plan(
        recipe, manifest=simple, object_name="alphabet_soup_1", num_init_states=50
    )
    plan_b = build_paired_episode_plan(
        recipe, manifest=smol, object_name="alphabet_soup_1", num_init_states=50
    )
    assert plan_a.drop_u == plan_b.drop_u
    assert plan_a.smolvla_target == plan_b.smolvla_target
    assert plan_a.motion_profile == plan_b.motion_profile


def test_drop_u_maps_to_real_planner_path_with_keepout() -> None:
    recipe = load_drop_datagen_recipe(CAN_DROP_RECIPE)
    manifest = paired_episode_seed_manifests(recipe, logical_episode_index=0)[0]
    plan = build_paired_episode_plan(
        recipe, manifest=manifest, object_name="alphabet_soup_1", num_init_states=50
    )
    assert plan.drop_u is not None
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
    segment_names = [segment.name for segment in carry.segments]
    assert segment_names == ["lift", "to_basket_via", "to_basket_hover"]
    basket_xy = np.array([0.0, 0.0])
    drop_recipe = legacy_drop_recipe(recipe)
    keepout = keepout_m(
        drop_recipe.min_drop_distance_from_basket_m,
        drop_recipe.hard_keepout_floor_m,
    )
    trigger = resolve_path_drop_trigger(
        plan,
        carry,
        basket_xy=basket_xy,
        keepout_m=keepout,
    )
    assert trigger is not None
    path = eligible_path(carry, basket_xy=basket_xy, keepout_m=keepout)
    assert path.total > 0.0
    assert trigger.segment_name == carry.segments[trigger.segment_order].name
    assert segment_names.index(trigger.segment_name) == trigger.segment_order
    segment = carry.segments[trigger.segment_order]
    start = np.asarray(segment.start_xyz, dtype=np.float64)
    end = np.asarray(segment.end_xyz, dtype=np.float64)
    point = start + trigger.target_t * (end - start)
    dist_xy = float(np.linalg.norm(point[:2] - basket_xy))
    assert dist_xy >= keepout - 1e-6
    np.testing.assert_allclose(point, start + trigger.target_t * (end - start))
    seg_vec = end - start
    denom = float(np.dot(seg_vec, seg_vec))
    if denom > 1e-15:
        t_on_segment = float(np.clip(np.dot(point - start, seg_vec) / denom, 0.0, 1.0))
        assert abs(t_on_segment - trigger.target_t) <= 1e-9
