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

import math

from lerobot.faults.recovery.xy_band_offset_probe import (
    compass_offsets,
    first_band_hits_after_grasp,
    soup_basket_xy_series,
    summarize_nominal_offset_probe,
)


def test_compass_offsets_have_3cm_and_5cm() -> None:
    specs = compass_offsets((0.03, 0.05))
    assert len(specs) == 16
    r03 = [s for s in specs if s["radius_m"] == 0.03]
    r05 = [s for s in specs if s["radius_m"] == 0.05]
    assert len(r03) == 8
    assert len(r05) == 8
    for s in r03:
        assert math.isclose(math.hypot(s["dx"], s["dy"]), 0.03, rel_tol=1e-9, abs_tol=1e-9)


def test_band_hits_after_grasp_skips_pre_grasp() -> None:
    # Distances fall through lift/early/mid/late after step 2.
    dists = [0.60, 0.60, 0.50, 0.44, 0.36, 0.31]
    hits = first_band_hits_after_grasp(dists, first_grasp_step=2)
    assert hits["lift"]["step"] == 2
    assert hits["early"]["step"] == 3
    assert hits["mid"]["step"] == 4
    assert hits["late"]["step"] == 5


def test_summarize_nominal_uses_basket_and_grasp() -> None:
    summary = {
        "success": True,
        "first_grasp_step": 1,
        "basket_destination": [0.0, 0.0, 0.0],
        "object_traj": [[0.5, 0.0, 0.04], [0.40, 0.0, 0.15], [0.32, 0.0, 0.20]],
    }
    out = summarize_nominal_offset_probe(summary)
    assert out["pick_success"] is True
    assert out["xy_at_grasp_m"] == 0.40
    assert soup_basket_xy_series([[3.0, 4.0]], (0.0, 0.0)) == [5.0]
