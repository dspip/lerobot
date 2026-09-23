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

import pytest

from lerobot.faults.datagen.drop_trigger import (
    smolvla_fault_drop_fields,
    sample_smolvla_band_target,
)
from lerobot.faults.recovery.basket_drop_target import basket_distance_target_reached
from lerobot.faults.datagen.recipe import DropXYBand


def test_target_crossing_detects_single_step_jump() -> None:
    target = 0.36
    assert basket_distance_target_reached(
        prev_m=0.50,
        curr_m=0.30,
        target_m=target,
        band_min_m=0.34,
        band_max_m=0.38,
        held_midair=True,
    )


def test_target_requires_held_midair() -> None:
    assert not basket_distance_target_reached(
        prev_m=0.50,
        curr_m=0.35,
        target_m=0.36,
        band_min_m=0.34,
        band_max_m=0.38,
        held_midair=False,
    )


def test_smolvla_fault_fields_carry_target_and_band() -> None:
    sampled = sample_smolvla_band_target(
        __import__("numpy").random.default_rng(3),
        (DropXYBand(name="mid", min_m=0.34, max_m=0.38),),
    )
    fields = smolvla_fault_drop_fields(sampled)
    assert fields["drop_xy_target_m"] == pytest.approx(sampled.target_m)
    assert fields["drop_xy_band_min"] == sampled.band.min_m
    assert fields["drop_xy_band_max"] == sampled.band.max_m
