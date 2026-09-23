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

import numpy as np

from lerobot.faults.datagen.motion_profile import sample_episode_motion_profile
from lerobot.faults.datagen.recipe import SimpleIKRecipe


def _recipe(*, enabled: bool = True) -> SimpleIKRecipe:
    return SimpleIKRecipe(
        trajectory_randomization_enabled=enabled,
        pickup_via_offset_m=0.03,
        transport_via_offset_m=0.06,
        arm_posture_noise_deg=3.0,
        speed_multiplier_range=(0.8, 1.2),
        waypoint_blend_radius_m=0.04,
    )


def test_disabled_profile_is_identity():
    profile = sample_episode_motion_profile(_recipe(enabled=False), 12)

    assert profile.speed_multiplier == 1.0
    assert profile.nominal.pickup_offset_xy_m == (0.0, 0.0)
    assert profile.nominal.transport_offset_m == 0.0
    assert profile.nominal.posture_bias_rad == (0.0, 0.0, 0.0)
    assert profile.recovery == profile.nominal


def test_profile_is_reproducible_from_episode_seed():
    first = sample_episode_motion_profile(_recipe(), 91)
    second = sample_episode_motion_profile(_recipe(), 91)

    assert first == second


def test_different_seeds_change_the_profile():
    assert sample_episode_motion_profile(_recipe(), 91) != sample_episode_motion_profile(
        _recipe(), 92
    )


def test_offsets_stay_inside_configured_bounds():
    for seed in range(200):
        profile = sample_episode_motion_profile(_recipe(), seed)
        for leg in (profile.nominal, profile.recovery):
            assert np.linalg.norm(leg.pickup_offset_xy_m) <= 0.03 + 1e-12
            assert abs(leg.transport_offset_m) <= 0.06
            assert max(abs(v) for v in leg.posture_bias_rad) <= np.deg2rad(3.0)


def test_speed_stays_inside_configured_range():
    speeds = [
        sample_episode_motion_profile(_recipe(), seed).speed_multiplier
        for seed in range(200)
    ]

    assert min(speeds) >= 0.8
    assert max(speeds) <= 1.2
    assert min(speeds) < 0.9
    assert max(speeds) > 1.1


def test_nominal_and_recovery_have_distinct_offsets():
    profile = sample_episode_motion_profile(_recipe(), 3)

    assert profile.nominal != profile.recovery


def test_speed_is_shared_by_nominal_and_recovery():
    profile = sample_episode_motion_profile(_recipe(), 4)

    assert isinstance(profile.speed_multiplier, float)
    assert not hasattr(profile.nominal, "speed_multiplier")
    assert not hasattr(profile.recovery, "speed_multiplier")


def test_profile_sampling_does_not_consume_caller_rng():
    caller = np.random.default_rng(77)
    before = caller.random()
    sample_episode_motion_profile(_recipe(), 77)
    after = caller.random()

    replay = np.random.default_rng(77)
    assert before == replay.random()
    assert after == replay.random()
