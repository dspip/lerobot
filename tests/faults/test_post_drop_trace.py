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

from lerobot.faults.recovery.post_drop_trace import summarize_post_drop_trace


def _frame(
    t: int,
    *,
    grasped: bool = False,
    eef_obj_xy_m: float = 0.05,
    eef_above_obj_m: float = 0.03,
    gripper_open: bool = True,
    in_basket_z014: bool = False,
    in_basket_z020: bool = False,
    obj_z: float = 0.15,
    wrist_visible: bool | None = None,
) -> dict:
    return {
        "t_since_drop": t,
        "grasped": grasped,
        "eef_obj_xy_m": eef_obj_xy_m,
        "eef_above_obj_m": eef_above_obj_m,
        "gripper_open": gripper_open,
        "in_basket_z014": in_basket_z014,
        "in_basket_z020": in_basket_z020,
        "obj_z": obj_z,
        "wrist_visible": wrist_visible,
    }


def test_summarize_empty():
    out = summarize_post_drop_trace([])
    assert out["n_frames"] == 0
    assert out["regrasp_t"] is None
    assert out["median_eef_obj_xy_m"] is None
    assert out["frac_wrist_visible"] is None


def test_summarize_hover_like_trace():
    frames = [_frame(t, eef_above_obj_m=0.04, gripper_open=True, grasped=False) for t in range(5)]
    out = summarize_post_drop_trace(frames)
    assert out["n_frames"] == 5
    assert out["n_seconds"] == 0.25
    assert out["regrasped"] is False
    assert out["frac_gripper_open"] == 1.0
    assert out["frac_grasped"] == 0.0
    assert out["median_eef_above_obj_m"] == 0.04
    assert out["frac_eef_above_0.02_0.10"] == 1.0


def test_summarize_regrasp_at_t10():
    frames = [_frame(t, grasped=(t >= 10)) for t in range(20)]
    out = summarize_post_drop_trace(frames)
    assert out["regrasped"] is True
    assert out["regrasp_t"] == 10
    assert out["frac_grasped"] == 0.5


def test_wrist_visible_all_none():
    frames = [_frame(t, wrist_visible=None) for t in range(3)]
    out = summarize_post_drop_trace(frames)
    assert out["frac_wrist_visible"] is None
