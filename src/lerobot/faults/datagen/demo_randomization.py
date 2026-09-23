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

"""Synthetic demo randomization for drop-recovery dry runs (not dataset recording)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RandomizationConfig:
    """Randomization axes for scripted drop-recovery demo episodes."""

    t_min: int = 10
    t_max: int = 30
    impulse_lin_std: float = 0.5
    impulse_ang_std: float = 0.2
    impulse_lin_bias: tuple[float, float, float] = (0.0, 0.0, -0.5)
    recovery_action_noise_std: float = 0.02
    arm_posture_noise_deg: float = 3.0
    waypoint_noise_m: float = 0.015
    speed_multiplier_min: float = 0.8
    speed_multiplier_max: float = 1.2
    drop_duration: int = 3
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.t_min < 0:
            raise ValueError(f"t_min must be >= 0, got {self.t_min}.")
        if self.t_max < self.t_min:
            raise ValueError(f"t_max must be >= t_min (got {self.t_max} < {self.t_min}).")
        if self.impulse_lin_std < 0 or self.impulse_ang_std < 0:
            raise ValueError("impulse std values must be non-negative.")
        if self.recovery_action_noise_std < 0:
            raise ValueError("recovery_action_noise_std must be non-negative.")
        if self.waypoint_noise_m < 0:
            raise ValueError("waypoint_noise_m must be non-negative.")
        if self.arm_posture_noise_deg < 0:
            raise ValueError("arm_posture_noise_deg must be non-negative.")
        if self.speed_multiplier_min <= 0 or self.speed_multiplier_max <= 0:
            raise ValueError("speed_multiplier bounds must be positive.")
        if self.speed_multiplier_max < self.speed_multiplier_min:
            raise ValueError("speed_multiplier_max must be >= speed_multiplier_min.")
        if self.drop_duration < 1:
            raise ValueError("drop_duration must be >= 1.")


def sample_episode_params(
    rng: np.random.Generator,
    config: RandomizationConfig | None = None,
) -> dict[str, Any]:
    """Sample one episode's randomized parameters for demo / dry-run use."""
    cfg = config or RandomizationConfig()
    t_fault = int(rng.integers(cfg.t_min, cfg.t_max + 1))
    lin = np.asarray(cfg.impulse_lin_bias, dtype=np.float64) + rng.normal(
        0.0, cfg.impulse_lin_std, size=3
    )
    ang = rng.normal(0.0, cfg.impulse_ang_std, size=3)
    speed_multiplier = float(rng.uniform(cfg.speed_multiplier_min, cfg.speed_multiplier_max))
    arm_noise_rad = float(np.deg2rad(cfg.arm_posture_noise_deg))
    arm_delta = rng.uniform(-arm_noise_rad, arm_noise_rad, size=3)

    return {
        "t_fault": t_fault,
        "t_recovery_start": t_fault + cfg.drop_duration,
        "drop_duration": cfg.drop_duration,
        "impulse_lin_vel": lin.astype(np.float64),
        "impulse_ang_vel": ang.astype(np.float64),
        "recovery_action_noise_std": cfg.recovery_action_noise_std,
        "arm_posture_noise_rad": arm_delta.astype(np.float64),
        "waypoint_noise_m": cfg.waypoint_noise_m,
        "speed_multiplier": speed_multiplier,
        "seed": cfg.seed,
    }
