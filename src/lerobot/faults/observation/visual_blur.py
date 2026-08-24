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

"""Visual blur fault on camera observations."""

from __future__ import annotations

from typing import Any

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.utils import apply_box_blur
from lerobot.faults.observation.burst import VisualBurstFault, _EnvBurstState


def _make_blur_mutate(radius: int):
    def _mutate(value: Any, env_idx: int, num_envs: int, state: _EnvBurstState) -> None:
        del state
        apply_box_blur(value, env_idx, num_envs, radius=radius)

    return _mutate


class VisualBlurFault(VisualBurstFault):
    """Apply a mild box blur to camera images for a burst window."""

    def __init__(
        self,
        config: FaultInjectionConfig,
        num_envs: int,
        event_logger: FaultEventLogger | None = None,
    ):
        radius = max(1, int(round(float(config.blur_sigma))))
        super().__init__(
            config,
            num_envs,
            event_logger,
            mutate=_make_blur_mutate(radius),
            event_name="visual_blur",
            required_type="visual_blur",
            sample_box_on_activate=False,
        )
