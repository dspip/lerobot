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
import pytest

from lerobot.faults.datagen.drop_trigger import (
    BandDistanceTrigger,
    sample_smolvla_band_target,
)
from lerobot.faults.datagen.recipe import DropXYBand


def _bands() -> tuple[DropXYBand, ...]:
    return (
        DropXYBand(name="lift", min_m=0.48, max_m=0.52),
        DropXYBand(name="mid", min_m=0.34, max_m=0.38),
    )


def test_sample_smolvla_band_target_is_deterministic_and_in_band() -> None:
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    target_a = sample_smolvla_band_target(rng_a, _bands())
    target_b = sample_smolvla_band_target(rng_b, _bands())

    assert target_a == target_b
    assert target_a.band.min_m <= target_a.target_m <= target_a.band.max_m
    assert target_a.band.name in {"lift", "mid"}


def test_band_trigger_requires_held_midair() -> None:
    rng = np.random.default_rng(0)
    sampled = sample_smolvla_band_target(rng, _bands())
    trigger = BandDistanceTrigger(sampled)

    assert not trigger.evaluate(
        held_midair=False,
        basket_distance_m=sampled.target_m,
        prev_basket_distance_m=sampled.target_m + 0.05,
    )


def test_band_trigger_fires_on_downward_crossing() -> None:
    rng = np.random.default_rng(1)
    sampled = sample_smolvla_band_target(rng, _bands())
    trigger = BandDistanceTrigger(sampled)
    target = sampled.target_m

    assert not trigger.evaluate(
        held_midair=True,
        basket_distance_m=target + 0.02,
        prev_basket_distance_m=target + 0.05,
    )
    assert trigger.evaluate(
        held_midair=True,
        basket_distance_m=target - 0.001,
        prev_basket_distance_m=target + 0.01,
    )


def test_band_trigger_does_not_fire_outside_sampled_band() -> None:
    rng = np.random.default_rng(2)
    sampled = sample_smolvla_band_target(rng, _bands())
    trigger = BandDistanceTrigger(sampled)
    outside = sampled.band.max_m + 0.05

    assert not trigger.evaluate(
        held_midair=True,
        basket_distance_m=outside,
        prev_basket_distance_m=outside + 0.02,
    )


def test_sample_smolvla_band_target_rejects_empty_bands() -> None:
    with pytest.raises(ValueError, match="bands"):
        sample_smolvla_band_target(np.random.default_rng(0), ())
