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
from lerobot.faults.datagen.path_drop import PathTrigger, eligible_path, path_trigger_at_drop_u
from lerobot.faults.datagen.recipe import DropDatagenRecipe, EpisodeSeedManifest
from lerobot.faults.recovery.trajectory import CarryPath

__all__ = [
    "PairedEpisodePlan",
    "build_paired_episode_plan",
    "resolve_init_state_id",
    "resolve_path_drop_trigger",
]

_DROP_Q_STREAM = 0x4451
_DROP_U_STREAM = 0x4452
_SMOLVLA_STREAM = 0x534D4F4C


@dataclass(frozen=True)
class PairedEpisodePlan:
    episode_seed: int
    layout_seed: int
    drop_seed: int
    init_state_id: int
    object_name: str
    motion_profile: EpisodeMotionProfile
    drop_decision: DropDecision
    drop_u: float | None
    smolvla_target: BandDistanceTarget


def resolve_init_state_id(episode_seed: int, num_init_states: int) -> int:
    if num_init_states <= 0:
        return 0
    return int(episode_seed) % int(num_init_states)


def _paired_drop_draws(
    drop_seed: int,
    q: float,
    bands: tuple,
) -> tuple[DropDecision, float | None, BandDistanceTarget]:
    q_rng = np.random.default_rng(np.random.SeedSequence([int(drop_seed), _DROP_Q_STREAM]))
    u_rng = np.random.default_rng(np.random.SeedSequence([int(drop_seed), _DROP_U_STREAM]))
    smol_rng = np.random.default_rng(np.random.SeedSequence([int(drop_seed), _SMOLVLA_STREAM]))
    if float(q_rng.random()) >= float(q):
        decision = DropDecision(drop=False, step=None, reason="skipped_q")
        drop_u = None
    else:
        decision = DropDecision(drop=True, step=None, reason="injected")
        drop_u = float(u_rng.uniform(0.0, 1.0))
    smolvla_target = sample_smolvla_band_target(smol_rng, bands)
    return decision, drop_u, smolvla_target


def build_paired_episode_plan(
    recipe: DropDatagenRecipe,
    *,
    manifest: EpisodeSeedManifest,
    object_name: str,
    num_init_states: int,
) -> PairedEpisodePlan:
    motion_profile = sample_episode_motion_profile(recipe.simple_ik, manifest.drop_seed)
    decision, drop_u, smolvla_target = _paired_drop_draws(
        manifest.drop_seed,
        recipe.q,
        recipe.smolvla.drop_xy_bands,
    )
    return PairedEpisodePlan(
        episode_seed=manifest.episode_seed,
        layout_seed=manifest.layout_seed,
        drop_seed=manifest.drop_seed,
        init_state_id=resolve_init_state_id(manifest.episode_seed, num_init_states),
        object_name=object_name,
        motion_profile=motion_profile,
        drop_decision=decision,
        drop_u=drop_u,
        smolvla_target=smolvla_target,
    )


def resolve_path_drop_trigger(
    plan: PairedEpisodePlan,
    carry_path: CarryPath,
    *,
    basket_xy: np.ndarray,
    keepout_m: float,
) -> PathTrigger | None:
    """Map paired ``drop_u`` onto the controller's real eligible carry path."""
    if plan.drop_u is None:
        return None
    path = eligible_path(carry_path, basket_xy=basket_xy, keepout_m=float(keepout_m))
    if path.total <= 0.0:
        return None
    return path_trigger_at_drop_u(plan.drop_u, path)
