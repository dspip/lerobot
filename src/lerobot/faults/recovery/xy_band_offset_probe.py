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

"""Nominal pick probes at fixed soup XY offsets (no GPU)."""

from __future__ import annotations

import math
from typing import Any

from .xy_band_checkpoint import XY_BANDS

COMPASS_DIRS: list[tuple[str, float, float]] = [
    ("+x", 1.0, 0.0),
    ("-x", -1.0, 0.0),
    ("+y", 0.0, 1.0),
    ("-y", 0.0, -1.0),
    ("+x+y", 1.0, 1.0),
    ("+x-y", 1.0, -1.0),
    ("-x+y", -1.0, 1.0),
    ("-x-y", -1.0, -1.0),
]


def compass_offsets(radii_m: tuple[float, ...] = (0.03, 0.05)) -> list[dict[str, Any]]:
    """Return labeled (dx, dy) offsets at each radius along 8 compass headings."""
    out: list[dict[str, Any]] = []
    for radius in radii_m:
        r = float(radius)
        for name, ux, uy in COMPASS_DIRS:
            n = math.hypot(ux, uy)
            dx = r * ux / n
            dy = r * uy / n
            out.append(
                {
                    "label": f"r{int(round(r * 100)):02d}_{name}",
                    "radius_m": r,
                    "dir": name,
                    "dx": dx,
                    "dy": dy,
                }
            )
    return out


def soup_basket_xy_series(
    object_traj: list[list[float]],
    basket_xy: tuple[float, float] | list[float],
) -> list[float]:
    bx, by = float(basket_xy[0]), float(basket_xy[1])
    dists: list[float] = []
    for pose in object_traj:
        if not pose or len(pose) < 2:
            continue
        dists.append(math.hypot(float(pose[0]) - bx, float(pose[1]) - by))
    return dists


def first_band_hits_after_grasp(
    xy_dists: list[float],
    *,
    first_grasp_step: int | None,
    bands: list[tuple[str, float, float]] | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """First time after lift that soup-basket XY falls inside each drop band."""
    band_list = bands if bands is not None else XY_BANDS
    start = 0 if first_grasp_step is None else max(0, int(first_grasp_step))
    hits: dict[str, dict[str, float | int | None]] = {}
    for name, lo, hi in band_list:
        step = None
        dist = None
        for i in range(start, len(xy_dists)):
            d = float(xy_dists[i])
            if lo <= d <= hi:
                step = i
                dist = d
                break
        hits[name] = {"step": step, "dist_m": dist, "band_lo": lo, "band_hi": hi}
    return hits


def summarize_nominal_offset_probe(summary: dict[str, Any]) -> dict[str, Any]:
    traj = summary.get("object_traj") or []
    basket = summary.get("basket_destination")
    basket_xy = None
    if isinstance(basket, (list, tuple)) and len(basket) >= 2:
        basket_xy = (float(basket[0]), float(basket[1]))
    dists = soup_basket_xy_series(traj, basket_xy) if basket_xy is not None else []
    grasp = summary.get("first_grasp_step")
    grasp_dist = None
    if grasp is not None and dists and 0 <= int(grasp) < len(dists):
        grasp_dist = float(dists[int(grasp)])
    return {
        "pick_success": bool(summary.get("success")),
        "first_grasp_step": grasp,
        "xy_at_grasp_m": grasp_dist,
        "xy_start_m": float(dists[0]) if dists else None,
        "xy_min_m": min(dists) if dists else None,
        "band_hits": first_band_hits_after_grasp(dists, first_grasp_step=grasp),
        "n_traj": len(dists),
    }
