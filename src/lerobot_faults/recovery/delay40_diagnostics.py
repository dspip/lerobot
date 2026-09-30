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

"""Offline reconstruction of midair_drop Option B gates from pipeline logs."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OPTION_B_MIN_XY_M = 0.30
HARD_BASKET_KEEPOUT_M = 0.22
MIN_OBJECT_Z_M = 0.12
PLACING_LIKE_MAX_XY_M = 0.40
PLACING_LIKE_MIN_HEIGHT_M = 0.10

BLOCK_REASONS = (
    "not_in_t_window",
    "not_grasped",
    "height_below_0.12",
    "hard_pocket_lt_0.22",
    "optionB_keepout_lt_0.30",
    "delay_not_elapsed",
    "would_trigger",
)


@dataclass(frozen=True)
class FrameRow:
    episode_frame: int
    steps_since_grasp: int | None
    object_x: float
    object_y: float
    object_z: float
    basket_x: float
    basket_y: float
    basket_z: float
    xy_dist_m: float
    grasped: bool
    height_m: float
    delay_ok: bool
    dist_ok: bool
    hard_ok: bool
    t_window_ok: bool
    height_ok: bool
    would_trigger: bool
    eligible_lost: bool
    placing_like: bool
    block_reason: str


def xy_distance(object_xyz: list[float] | tuple[float, ...], basket_xyz: list[float] | tuple[float, ...]) -> float:
    ox, oy = float(object_xyz[0]), float(object_xyz[1])
    bx, by = float(basket_xyz[0]), float(basket_xyz[1])
    return math.hypot(ox - bx, oy - by)


def compute_block_reason(
    *,
    t_window_ok: bool,
    grasped: bool,
    height_ok: bool,
    hard_ok: bool,
    dist_ok: bool,
    delay_ok: bool,
) -> str:
    if not t_window_ok:
        return "not_in_t_window"
    if not grasped:
        return "not_grasped"
    if not height_ok:
        return "height_below_0.12"
    if not hard_ok:
        return "hard_pocket_lt_0.22"
    if not dist_ok:
        return "optionB_keepout_lt_0.30"
    if not delay_ok:
        return "delay_not_elapsed"
    return "would_trigger"


def _first_crossing_below(
    frames: list[FrameRow],
    threshold: float,
    *,
    start_index: int,
) -> int | None:
    prev_above = True
    for row in frames:
        if row.episode_frame < start_index:
            continue
        above = row.xy_dist_m >= threshold
        if prev_above and row.xy_dist_m < threshold:
            return row.episode_frame
        prev_above = above
    return None


def reconstruct_frames(pipeline_log: dict[str, Any]) -> list[FrameRow]:
    """Build per-step rows from a pipeline_log dict (CPU-only, no sim)."""
    traj = pipeline_log.get("object_traj") or []
    grasp_flags = pipeline_log.get("grasp_flags") or []
    basket = pipeline_log.get("basket_destination")
    if basket is None:
        raise ValueError("pipeline_log missing basket_destination")
    basket_xyz = [float(basket[0]), float(basket[1]), float(basket[2])]

    first_grasp = pipeline_log.get("first_grasp_step")
    first_grasp_int: int | None = None if first_grasp is None else int(first_grasp)

    fault_cfg = pipeline_log.get("fault_config") or {}
    delay_steps = int(fault_cfg.get("post_grasp_delay_steps", 40))
    t_min = int(pipeline_log.get("t_min", 0))
    t_max = int(pipeline_log.get("t_max", 10**9))

    rows: list[FrameRow] = []
    prev_xy: float | None = None
    for i, obj in enumerate(traj):
        obj_xyz = [float(obj[0]), float(obj[1]), float(obj[2])]
        xy_dist = xy_distance(obj_xyz, basket_xyz)
        grasped = bool(grasp_flags[i]) if i < len(grasp_flags) else False
        height = obj_xyz[2]
        steps_since = None if first_grasp_int is None else i - first_grasp_int
        t_window_ok = t_min <= i <= t_max
        height_ok = height >= MIN_OBJECT_Z_M
        delay_ok = first_grasp_int is not None and i >= first_grasp_int + delay_steps
        dist_ok = xy_dist >= OPTION_B_MIN_XY_M
        hard_ok = xy_dist >= HARD_BASKET_KEEPOUT_M
        eligible_lost = (
            first_grasp_int is not None
            and i >= first_grasp_int
            and (not grasped or not height_ok)
        )
        would_trigger = (
            t_window_ok and grasped and height_ok and delay_ok and dist_ok and hard_ok
        )
        block_reason = compute_block_reason(
            t_window_ok=t_window_ok,
            grasped=grasped,
            height_ok=height_ok,
            hard_ok=hard_ok,
            dist_ok=dist_ok,
            delay_ok=delay_ok,
        )
        placing_like = False
        if prev_xy is not None:
            placing_like = (
                xy_dist < prev_xy
                and xy_dist < PLACING_LIKE_MAX_XY_M
                and height >= PLACING_LIKE_MIN_HEIGHT_M
            )
        prev_xy = xy_dist
        rows.append(
            FrameRow(
                episode_frame=i,
                steps_since_grasp=steps_since,
                object_x=obj_xyz[0],
                object_y=obj_xyz[1],
                object_z=obj_xyz[2],
                basket_x=basket_xyz[0],
                basket_y=basket_xyz[1],
                basket_z=basket_xyz[2],
                xy_dist_m=xy_dist,
                grasped=grasped,
                height_m=height,
                delay_ok=delay_ok,
                dist_ok=dist_ok,
                hard_ok=hard_ok,
                t_window_ok=t_window_ok,
                height_ok=height_ok,
                would_trigger=would_trigger,
                eligible_lost=eligible_lost,
                placing_like=placing_like,
                block_reason=block_reason,
            )
        )
    return rows


def frames_from_first_grasp(rows: list[FrameRow], first_grasp_step: int | None) -> list[FrameRow]:
    if first_grasp_step is None:
        return []
    start = int(first_grasp_step)
    return [r for r in rows if r.episode_frame >= start]


def summarize_episode(pipeline_log: dict[str, Any], rows: list[FrameRow]) -> dict[str, Any]:
    first_grasp = pipeline_log.get("first_grasp_step")
    first_grasp_int: int | None = None if first_grasp is None else int(first_grasp)
    fault_cfg = pipeline_log.get("fault_config") or {}
    delay_steps = int(fault_cfg.get("post_grasp_delay_steps", 40))
    basket = pipeline_log.get("basket_destination") or [0.0, 0.0, 0.0]

    grasp_rows = frames_from_first_grasp(rows, first_grasp_int)
    delay_target: int | None = None if first_grasp_int is None else first_grasp_int + delay_steps
    delay_frame_exists = (
        delay_target is not None and delay_target < len(pipeline_log.get("object_traj") or [])
    )

    def _snapshot(step: int | None) -> dict[str, Any] | None:
        if step is None:
            return None
        match = [r for r in rows if r.episode_frame == step]
        if not match:
            return None
        r = match[0]
        return {
            "episode_frame": r.episode_frame,
            "object_xyz": [r.object_x, r.object_y, r.object_z],
            "xy_dist_m": r.xy_dist_m,
            "block_reason": r.block_reason,
            "grasped": r.grasped,
            "height_m": r.height_m,
        }

    at_grasp = _snapshot(first_grasp_int)
    at_delay = _snapshot(delay_target) if delay_frame_exists else _snapshot(
        len(rows) - 1 if rows else None
    )
    if not delay_frame_exists and at_delay is not None:
        at_delay = {**at_delay, "note": "episode ended before first_grasp+delay; using last frame"}

    latest_safe: int | None = None
    for r in reversed(grasp_rows):
        if r.xy_dist_m >= OPTION_B_MIN_XY_M and r.grasped and r.height_ok:
            latest_safe = r.episode_frame
            break

    start_idx = first_grasp_int if first_grasp_int is not None else 0
    return {
        "seed": pipeline_log.get("seed"),
        "post_grasp_delay_steps": delay_steps,
        "triggered_at": pipeline_log.get("triggered_at"),
        "first_grasp_step": first_grasp_int,
        "basket_destination_xyz": [float(x) for x in basket[:3]],
        "basket_body_z_note": "pipeline basket_destination Z is basket body Z + 0.10 m (place hover)",
        "grasp_flags_off_by_one": (
            "grasp_flags[i] is grasped_now BEFORE env.step at index i; object_traj[i] is POST-step"
        ),
        "at_first_grasp": at_grasp,
        "at_delay_target": at_delay,
        "delay_target_step": delay_target,
        "delay_target_frame_exists": delay_frame_exists,
        "first_cross_below_0.50": _first_crossing_below(grasp_rows, 0.50, start_index=start_idx),
        "first_cross_below_0.40": _first_crossing_below(grasp_rows, 0.40, start_index=start_idx),
        "first_cross_below_0.30": _first_crossing_below(grasp_rows, 0.30, start_index=start_idx),
        "latest_safe_drop_step": latest_safe,
        "episode_length_steps": len(rows),
    }


def write_frames_csv(path: Path, rows: list[FrameRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [f.name for f in FrameRow.__dataclass_fields__.values()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_pipeline_log(episode_dir: Path) -> dict[str, Any]:
    log_path = episode_dir / "pipeline_log.json"
    return json.loads(log_path.read_text(encoding="utf-8"))


def build_q1_q8(cross: dict[str, Any]) -> dict[str, Any]:
    """Standard report questions (numeric answers from frame CSVs)."""
    failed = cross.get("delay40_failed_seeds") or {}
    success = cross.get("delay20_success_seeds") or {}

    def _xy_at_delay(seed: int) -> float | None:
        s = failed.get(str(seed)) or failed.get(seed)
        if not s:
            return None
        snap = s.get("at_delay_target") or {}
        return snap.get("xy_dist_m")

    def _xy_at_success_delay(seed: int) -> float | None:
        s = success.get(str(seed)) or success.get(seed)
        if not s:
            return None
        snap = s.get("at_delay_target") or {}
        return snap.get("xy_dist_m")

    def _first_cross_30(seed: int) -> int | None:
        s = failed.get(str(seed)) or failed.get(seed)
        if not s:
            return None
        return s.get("first_cross_below_0.30")

    return {
        "Q1_xy_dist_at_grasp_plus_delay_seed_4008": _xy_at_delay(4008),
        "Q2_xy_dist_at_grasp_plus_delay_seed_4012": _xy_at_delay(4012),
        "Q3_xy_dist_at_grasp_plus_delay_seed_4020": _xy_at_delay(4020),
        "Q4_xy_dist_at_grasp_plus_delay_seed_4030": _xy_at_delay(4030),
        "Q5_xy_dist_at_grasp_plus_delay_seed_4040": _xy_at_delay(4040),
        "Q6_xy_dist_at_grasp_plus_20_success_seed_4004": _xy_at_success_delay(4004),
        "Q7_xy_dist_at_grasp_plus_20_success_seed_4007": _xy_at_success_delay(4007),
        "Q8_first_frame_xy_below_0.30_after_grasp_seed_4008": _first_cross_30(4008),
    }
