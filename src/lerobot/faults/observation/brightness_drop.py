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

"""Brightness-drop fault on camera observations."""

from __future__ import annotations

from typing import Any

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.utils import apply_brightness_scale
from lerobot.faults.observation.burst import VisualBurstFault, _EnvBurstState


def _make_brightness_mutate(scale: float):
    def _mutate(value: Any, env_idx: int, num_envs: int, state: _EnvBurstState) -> None:
        del state
        apply_brightness_scale(value, env_idx, num_envs, scale=scale)

    return _mutate


class BrightnessDropFault(VisualBurstFault):
    """Scale camera image brightness down for a burst window."""

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ):
        super().__init__(
            config,
            num_envs,
            event_logger,
            mutate=_make_brightness_mutate(float(config.brightness_scale)),
            event_name="brightness_drop",
            required_type="brightness_drop",
            sample_box_on_activate=False,
        )
