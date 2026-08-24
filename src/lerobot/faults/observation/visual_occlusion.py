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

"""Visual occlusion (black bounding-box) fault on camera observations."""

from __future__ import annotations

from typing import Any

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.utils import apply_occlusion_box
from lerobot.faults.observation.burst import VisualBurstFault, _EnvBurstState


def _mutate_occlusion(value: Any, env_idx: int, num_envs: int, state: _EnvBurstState) -> None:
    if state.box is None:
        return
    y0, x0, y1, x1 = state.box
    apply_occlusion_box(value, env_idx, num_envs, y0=y0, x0=x0, y1=y1, x1=x1)


class VisualOcclusionFault(VisualBurstFault):
    """Paint a solid black rectangle over camera images for a burst window."""

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
            mutate=_mutate_occlusion,
            event_name="visual_occlusion",
            required_type="visual_occlusion",
            sample_box_on_activate=True,
        )
