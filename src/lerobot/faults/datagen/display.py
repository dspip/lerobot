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

"""Display-only geometry for the live viewer.

Scaling happens at draw time so the camera panel can be enlarged without
touching ``observation_height`` / ``observation_width``, which decide what a
recorded dataset actually contains.
"""

from __future__ import annotations

_MIN_HUD_FONT_PX = 11
_HUD_FONT_DIVISOR = 45


def fit_display_size(src_w: int, src_h: int, target_px: int) -> tuple[int, int]:
    """Scale a frame so its longest edge is ``target_px``, keeping the aspect."""
    src_w = int(src_w)
    src_h = int(src_h)
    target_px = int(target_px)
    if src_w < 1 or src_h < 1:
        raise ValueError(f"source size must be positive, got {src_w}x{src_h}")
    if target_px < 1:
        raise ValueError(f"target_px must be >= 1, got {target_px}")
    scale = target_px / max(src_w, src_h)
    return max(1, round(src_w * scale)), max(1, round(src_h * scale))


def hud_font_size(panel_px: int) -> int:
    """Pick a HUD font that stays legible as the panel grows."""
    return max(_MIN_HUD_FONT_PX, round(int(panel_px) / _HUD_FONT_DIVISOR))
