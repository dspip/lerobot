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

"""Training-grade mid-air-drop recording recipe (no simulator)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.recovery.recording_recipe import (
    DEFAULT_POST_GRASP_DELAY_MAX,
    DEFAULT_POST_GRASP_DELAY_MIN,
    sample_post_grasp_delay_steps,
    training_midair_drop_kwargs,
)


def test_sample_delay_is_inclusive_and_seeded() -> None:
    a = sample_post_grasp_delay_steps(np.random.default_rng(7), 20, 60)
    b = sample_post_grasp_delay_steps(np.random.default_rng(7), 20, 60)
    assert a == b
    assert DEFAULT_POST_GRASP_DELAY_MIN <= a <= DEFAULT_POST_GRASP_DELAY_MAX


def test_sample_delay_fixed_when_min_equals_max() -> None:
    assert sample_post_grasp_delay_steps(np.random.default_rng(0), 40, 40) == 40


def test_sample_delay_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="delay_max"):
        sample_post_grasp_delay_steps(np.random.default_rng(0), 10, 5)


def test_training_kwargs_disable_seat_assist_and_require_carry() -> None:
    kw = training_midair_drop_kwargs(
        t_min=40,
        t_max=400,
        seed=1000,
        post_grasp_delay_steps=32,
        log_path=Path("/tmp/fault_events.jsonl"),
        policy_fps=10,
    )
    assert kw["seat_assist_enabled"] is False
    assert kw["post_grasp_delay_steps"] == 32
    assert kw["min_drop_distance_from_basket_m"] == 0.30
    assert kw["require_grasp"] is True
    cfg = FaultInjectionConfig(**kw)
    assert cfg.seat_assist_enabled is False
    assert cfg.post_grasp_delay_steps == 32
