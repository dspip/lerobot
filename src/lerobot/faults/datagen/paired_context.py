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

"""Deterministic paired pre-drop plans shared across controller variants."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.faults.datagen.drop_timing import DropDecision
from lerobot.faults.datagen.drop_trigger import BandDistanceTarget, sample_smolvla_band_target
from lerobot.faults.datagen.motion_profile import EpisodeMotionProfile, sample_episode_motion_profile
from lerobot.faults.datagen.path_drop import PathTrigger, eligible_path, sample_path_drop
from lerobot.faults.datagen.recipe import DropDatagenRecipe, EpisodeSeedManifest, legacy_drop_recipe
from lerobot.faults.datagen.drop_timing import keepout_m
from lerobot.faults.recovery.trajectory import CarryPath, PathSegment

__all__ = ["PairedEpisodePlan", "build_paired_episode_plan", "resolve_init_state_id"]


@dataclass(frozen=True)
class PairedEpisodePlan:
    episode_seed: int
    layout_seed: int
    drop_seed: int
    init_state_id: int
    object_name: str
    motion_profile: EpisodeMotionProfile
    drop_decision: DropDecision
    path_trigger: PathTrigger | None
    smolvla_target: BandDistanceTarget


def resolve_init_state_id(episode_seed: int, num_init_states: int) -> int:
    if num_init_states <= 0:
        return 0
    return int(episode_seed) % int(num_init_states)


def _reference_carry_path(recipe: DropDatagenRecipe) -> CarryPath:
    """Minimal carry polyline for paired path-drop sampling (seed-only, no sim)."""
    drop = legacy_drop_recipe(recipe)
    keepout = keepout_m(
        drop.min_drop_distance_from_basket_m,
        drop.hard_keepout_floor_m,
    )
    _ = keepout
    return CarryPath(
        segments=(
            PathSegment("lift", (0.5, 0.0, 0.03), (0.5, 0.0, 0.25)),
            PathSegment("to_basket_hover", (0.5, 0.0, 0.25), (0.0, 0.0, 0.25)),
        ),
        requested_transport_offset_m=recipe.simple_ik.transport_via_offset_m,
        resolved_transport_offset_m=recipe.simple_ik.transport_via_offset_m,
        fallback=False,
    )


def build_paired_episode_plan(
    recipe: DropDatagenRecipe,
    *,
    manifest: EpisodeSeedManifest,
    object_name: str,
    num_init_states: int = 10,
) -> PairedEpisodePlan:
    drop_recipe = legacy_drop_recipe(recipe)
    drop_rng = np.random.default_rng(manifest.drop_seed)
    motion_profile = sample_episode_motion_profile(recipe.simple_ik, manifest.drop_seed)
    carry = _reference_carry_path(recipe)
    path = eligible_path(carry, basket_xy=np.zeros(2), keepout_m=keepout_m(
        drop_recipe.min_drop_distance_from_basket_m,
        drop_recipe.hard_keepout_floor_m,
    ))
    decision, trigger = sample_path_drop(recipe.q, path, drop_rng)
    smolvla_target = sample_smolvla_band_target(drop_rng, recipe.smolvla.drop_xy_bands)
    return PairedEpisodePlan(
        episode_seed=manifest.episode_seed,
        layout_seed=manifest.layout_seed,
        drop_seed=manifest.drop_seed,
        init_state_id=resolve_init_state_id(manifest.episode_seed, num_init_states),
        object_name=object_name,
        motion_profile=motion_profile,
        drop_decision=decision,
        path_trigger=trigger,
        smolvla_target=smolvla_target,
    )
