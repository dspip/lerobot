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

"""Training-grade defaults for mid-air-drop dataset recording.

The library ``FaultInjectionConfig`` still defaults to ``post_grasp_delay_steps=0``
and ``seat_assist_enabled=True`` so existing injector unit tests stay valid.
This module is the recipe for **datasets we might train on**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# 20 Hz LIBERO control: 20 steps = 1.0 s of carry after first mid-air grasp.
DEFAULT_POST_GRASP_DELAY_MIN = 20
DEFAULT_POST_GRASP_DELAY_MAX = 60
DEFAULT_MIN_DROP_DISTANCE_FROM_BASKET_M = 0.30
DEFAULT_POST_DROP_DWELL_STEPS = 80


def sample_post_grasp_delay_steps(
    rng: np.random.Generator,
    delay_min: int = DEFAULT_POST_GRASP_DELAY_MIN,
    delay_max: int = DEFAULT_POST_GRASP_DELAY_MAX,
) -> int:
    """Inclusive integer delay (env steps) between first eligible grasp and drop."""
    lo = int(delay_min)
    hi = int(delay_max)
    if lo < 0:
        raise ValueError(f"delay_min must be >= 0 (got {lo}).")
    if hi < lo:
        raise ValueError(f"delay_max must be >= delay_min (got {hi} < {lo}).")
    return int(rng.integers(lo, hi + 1))


def training_midair_drop_kwargs(
    *,
    t_min: int,
    t_max: int,
    seed: int,
    post_grasp_delay_steps: int,
    log_path: Path | str,
    policy_fps: int,
    seat_assist_enabled: bool = False,
    min_drop_distance_from_basket_m: float | None = None,
    drop_xy_band_min: float | None = None,
    drop_xy_band_max: float | None = None,
    drop_xy_target_m: float | None = None,
    post_drop_dwell_steps: int | None = None,
    post_drop_mode: str = "continue_then_ik",
) -> dict[str, Any]:
    """Fault kwargs for an episode that is allowed into a training mix."""
    return {
        "enabled": True,
        "type": "midair_drop",
        "t_min": int(t_min),
        "t_max": int(t_max),
        "require_grasp": True,
        "min_object_z": 0.12,
        "impulse_lin_std": 0.05,
        "impulse_ang_std": 0.05,
        "impulse_lin_bias": (0.0, 0.0, -0.55),
        "post_grasp_delay_steps": int(post_grasp_delay_steps),
        "min_drop_distance_from_basket_m": (
            float(DEFAULT_MIN_DROP_DISTANCE_FROM_BASKET_M)
            if min_drop_distance_from_basket_m is None
            else float(min_drop_distance_from_basket_m)
        ),
        "seat_assist_enabled": bool(seat_assist_enabled),
        "settle_steps": 100,
        "gripper_settle_steps": 25,
        "recovery_fps": int(policy_fps),
        "seed": int(seed),
        "log_path": Path(log_path),
        "drop_xy_band_min": (None if drop_xy_band_min is None else float(drop_xy_band_min)),
        "drop_xy_band_max": (None if drop_xy_band_max is None else float(drop_xy_band_max)),
        "drop_xy_target_m": (None if drop_xy_target_m is None else float(drop_xy_target_m)),
        "post_drop_dwell_steps": (
            DEFAULT_POST_DROP_DWELL_STEPS if post_drop_dwell_steps is None else int(post_drop_dwell_steps)
        ),
        "post_drop_mode": str(post_drop_mode),
    }
