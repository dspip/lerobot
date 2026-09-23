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

from lerobot.faults.datagen.demo_randomization import RandomizationConfig, sample_episode_params


def test_sample_episode_params_seeded() -> None:
    cfg = RandomizationConfig(seed=0, t_min=20, t_max=20, drop_duration=1)
    a = sample_episode_params(np.random.default_rng(7), cfg)
    b = sample_episode_params(np.random.default_rng(7), cfg)
    assert a["t_fault"] == b["t_fault"] == 20
    assert a["t_recovery_start"] == b["t_recovery_start"] == 21
    assert np.allclose(a["impulse_lin_vel"], b["impulse_lin_vel"])
    assert a["speed_multiplier"] == b["speed_multiplier"]


def test_randomization_config_rejects_inverted_t_range() -> None:
    with pytest.raises(ValueError, match="t_max"):
        RandomizationConfig(t_min=10, t_max=5)


def test_randomization_config_rejects_invalid_drop_duration() -> None:
    with pytest.raises(ValueError, match="drop_duration"):
        RandomizationConfig(drop_duration=0)
