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

"""Seeded episode-level trajectory shape and speed profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lerobot.faults.datagen.recipe import SimpleIKRecipe

_MOTION_STREAM = 0x4D4F544E


@dataclass(frozen=True)
class MotionLegProfile:
    """Coherent shape and posture values for one nominal/recovery motion."""

    pickup_offset_xy_m: tuple[float, float]
    transport_offset_m: float
    posture_bias_rad: tuple[float, float, float]


@dataclass(frozen=True)
class EpisodeMotionProfile:
    """One speed plus independent nominal and recovery path shapes."""

    speed_multiplier: float
    nominal: MotionLegProfile
    recovery: MotionLegProfile


def _identity_leg() -> MotionLegProfile:
    return MotionLegProfile(
        pickup_offset_xy_m=(0.0, 0.0),
        transport_offset_m=0.0,
        posture_bias_rad=(0.0, 0.0, 0.0),
    )


def _sample_leg(
    rng: np.random.Generator,
    *,
    pickup_radius_m: float,
    transport_offset_m: float,
    posture_rad: float,
) -> MotionLegProfile:
    radius = float(pickup_radius_m) * float(np.sqrt(rng.random()))
    angle = float(rng.uniform(-np.pi, np.pi))
    pickup = (radius * float(np.cos(angle)), radius * float(np.sin(angle)))
    transport = float(rng.uniform(-transport_offset_m, transport_offset_m))
    posture = tuple(float(v) for v in rng.uniform(-posture_rad, posture_rad, size=3))
    return MotionLegProfile(
        pickup_offset_xy_m=pickup,
        transport_offset_m=transport,
        posture_bias_rad=posture,
    )


def sample_episode_motion_profile(
    recipe: SimpleIKRecipe,
    episode_seed: int,
) -> EpisodeMotionProfile:
    """Sample a reproducible profile without consuming any caller-owned RNG."""
    if not recipe.trajectory_randomization_enabled:
        identity = _identity_leg()
        return EpisodeMotionProfile(
            speed_multiplier=1.0,
            nominal=identity,
            recovery=identity,
        )

    rng = np.random.default_rng(
        np.random.SeedSequence([int(episode_seed), _MOTION_STREAM])
    )
    speed = float(rng.uniform(*recipe.speed_multiplier_range))
    posture_rad = float(np.deg2rad(recipe.arm_posture_noise_deg))
    kwargs = {
        "pickup_radius_m": recipe.pickup_via_offset_m,
        "transport_offset_m": recipe.transport_via_offset_m,
        "posture_rad": posture_rad,
    }
    return EpisodeMotionProfile(
        speed_multiplier=speed,
        nominal=_sample_leg(rng, **kwargs),
        recovery=_sample_leg(rng, **kwargs),
    )
