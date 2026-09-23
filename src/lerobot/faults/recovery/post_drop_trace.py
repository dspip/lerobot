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

"""Summarize post–mid-air-drop policy motion traces (diagnostic, not training)."""

from __future__ import annotations

from statistics import median
from typing import Any


def _empty_summary() -> dict[str, Any]:
    return {
        "n_frames": 0,
        "n_seconds": 0.0,
        "regrasped": False,
        "regrasp_t": None,
        "frac_grasped": 0.0,
        "frac_gripper_open": 0.0,
        "median_eef_obj_xy_m": None,
        "median_eef_above_obj_m": None,
        "frac_eef_xy_gt_0.15": 0.0,
        "frac_eef_above_0.02_0.10": 0.0,
        "frac_eef_above_0.10_0.25": 0.0,
        "frac_in_basket_z014": 0.0,
        "frac_in_basket_z020": 0.0,
        "frac_wrist_visible": None,
        "final_obj_z": None,
        "min_obj_z": None,
    }


def summarize_post_drop_trace(frames: list[dict]) -> dict[str, Any]:
    """Aggregate post-drop control frames (20 Hz) into scalar diagnostics."""
    if not frames:
        return _empty_summary()

    n_frames = len(frames)
    grasped_flags = [bool(f["grasped"]) for f in frames]
    gripper_open_flags = [bool(f["gripper_open"]) for f in frames]
    xy_vals = [float(f["eef_obj_xy_m"]) for f in frames]
    above_vals = [float(f["eef_above_obj_m"]) for f in frames]
    obj_z_vals = [float(f["obj_z"]) for f in frames]
    wrist_vals = [f["wrist_visible"] for f in frames if f.get("wrist_visible") is not None]

    regrasp_t: int | None = None
    for f in frames:
        t = int(f["t_since_drop"])
        if t > 0 and bool(f["grasped"]):
            regrasp_t = t
            break

    def _frac(pred) -> float:
        if n_frames == 0:
            return 0.0
        return sum(1 for f in frames if pred(f)) / n_frames

    wrist_frac: float | None = None if not wrist_vals else sum(1 for v in wrist_vals if v) / len(wrist_vals)

    return {
        "n_frames": n_frames,
        "n_seconds": n_frames / 20.0,
        "regrasped": regrasp_t is not None,
        "regrasp_t": regrasp_t,
        "frac_grasped": sum(grasped_flags) / n_frames,
        "frac_gripper_open": sum(gripper_open_flags) / n_frames,
        "median_eef_obj_xy_m": float(median(xy_vals)),
        "median_eef_above_obj_m": float(median(above_vals)),
        "frac_eef_xy_gt_0.15": _frac(lambda f: float(f["eef_obj_xy_m"]) > 0.15),
        "frac_eef_above_0.02_0.10": _frac(lambda f: 0.02 <= float(f["eef_above_obj_m"]) < 0.10),
        "frac_eef_above_0.10_0.25": _frac(lambda f: 0.10 <= float(f["eef_above_obj_m"]) < 0.25),
        "frac_in_basket_z014": _frac(lambda f: bool(f["in_basket_z014"])),
        "frac_in_basket_z020": _frac(lambda f: bool(f["in_basket_z020"])),
        "frac_wrist_visible": wrist_frac,
        "final_obj_z": float(obj_z_vals[-1]),
        "min_obj_z": float(min(obj_z_vals)),
    }


def aggregate_seed_summaries(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mean numeric fractions across seeds plus regrasp / no-drop counts."""
    if not per_seed:
        return {
            "n_seeds": 0,
            "n_regrasped": 0,
            "n_never_dropped": 0,
        }

    frac_keys = (
        "frac_grasped",
        "frac_gripper_open",
        "frac_eef_xy_gt_0.15",
        "frac_eef_above_0.02_0.10",
        "frac_eef_above_0.10_0.25",
        "frac_in_basket_z014",
        "frac_in_basket_z020",
    )
    agg: dict[str, Any] = {"n_seeds": len(per_seed), "n_regrasped": 0, "n_never_dropped": 0}
    for key in frac_keys:
        vals = [float(s[key]) for s in per_seed.values() if s.get("n_frames", 0) > 0]
        agg[f"mean_{key}"] = float(sum(vals) / len(vals)) if vals else 0.0

    wrist_vals = [
        s["frac_wrist_visible"] for s in per_seed.values() if s.get("frac_wrist_visible") is not None
    ]
    agg["mean_frac_wrist_visible"] = float(sum(wrist_vals) / len(wrist_vals)) if wrist_vals else None

    for s in per_seed.values():
        if s.get("dropped") is False:
            agg["n_never_dropped"] += 1
        if s.get("regrasped"):
            agg["n_regrasped"] += 1

    return agg
