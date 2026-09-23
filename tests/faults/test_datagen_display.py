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

from lerobot.faults.datagen.display import fit_display_size, hud_font_size


def test_square_frame_scales_up_to_the_target():
    assert fit_display_size(256, 256, 720) == (720, 720)


def test_landscape_frame_keeps_aspect_ratio():
    assert fit_display_size(320, 240, 640) == (640, 480)


def test_portrait_frame_fits_the_long_edge():
    assert fit_display_size(240, 320, 640) == (480, 640)


def test_frame_larger_than_target_scales_down():
    assert fit_display_size(512, 512, 256) == (256, 256)


def test_short_edge_never_rounds_to_zero():
    width, height = fit_display_size(1000, 1, 100)

    assert (width, height) == (100, 1)


def test_rejects_non_positive_target():
    with pytest.raises(ValueError):
        fit_display_size(256, 256, 0)


def test_rejects_non_positive_source():
    with pytest.raises(ValueError):
        fit_display_size(0, 256, 720)


def test_hud_font_grows_with_the_panel():
    assert hud_font_size(720) > hud_font_size(256)


def test_hud_font_has_a_readable_floor():
    assert hud_font_size(64) >= 11
