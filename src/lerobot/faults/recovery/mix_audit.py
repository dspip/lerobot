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

"""Audit helpers for failure-mix dataset outputs (no GPU/sim)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.faults.recovery.midair_drop import HARD_BASKET_KEEPOUT_M
from lerobot.faults.recovery.mix_recording import episode_dir_for_seed, parse_int_list

HARD_POCKET_XY_M = float(HARD_BASKET_KEEPOUT_M)


@dataclass
class AuditIssue:
    level: str  # "fail" | "warn"
    message: str


@dataclass
class DropEpisodeRow:
    seed: int
    delay_steps: int | None
    triggered_at: int | None
    carry_steps: int | None
    drop_basket_xy_dist: float | None
    drop_trigger_reason: str | None
    pre_drop_xy: tuple[float, float] | None


@dataclass
class MixAuditResult:
    rows: list[DropEpisodeRow] = field(default_factory=list)
    issues: list[AuditIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def has_failures(self) -> bool:
        return any(i.level == "fail" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warn" for i in self.issues)

    def exit_code(self) -> int:
        if self.has_failures:
            return 1
        if self.has_warnings:
            return 2
        return 0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_midair_drop_event(fault_events_path: Path) -> dict[str, Any] | None:
    if not fault_events_path.is_file():
        return None
    for line in fault_events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "midair_drop" and event.get("status") == "triggered":
            return event
    return None


def _pre_drop_xy(pipeline: dict[str, Any], event: dict[str, Any] | None) -> tuple[float, float] | None:
    pose = pipeline.get("pre_drop_pose")
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        return float(pose[0]), float(pose[1])
    if event is not None:
        obj = event.get("object_pose") or event.get("pos")
        if isinstance(obj, dict):
            pos = obj.get("pos") or obj.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                return float(pos[0]), float(pos[1])
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            return float(obj[0]), float(obj[1])
    return None


def min_pairwise_xy_distance(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    best = math.inf
    for i in range(len(points)):
        x0, y0 = points[i]
        for j in range(i + 1, len(points)):
            x1, y1 = points[j]
            d = math.hypot(x1 - x0, y1 - y0)
            best = min(best, d)
    return float(best) if math.isfinite(best) else None


def default_min_unique_triggered_at(n_drops: int) -> int:
    return 5 if n_drops >= 8 else 2


def _dataset_has_videos(dataset_root: Path) -> bool:
    videos_dir = dataset_root / "videos"
    if not videos_dir.is_dir():
        return False
    return any(videos_dir.rglob("*.mp4")) or any(videos_dir.rglob("*.avi"))


def audit_failure_mix(
    output_dir: Path,
    *,
    min_xy_m: float = 0.30,
    min_pairwise_xy_m: float = 0.05,
    min_unique_triggered_at: int | None = None,
    expect_delay_grid: list[int] | None = None,
    verify_parquet_fn: Any | None = None,
) -> MixAuditResult:
    """Inspect mix outputs; populate rows, issues, and stats."""
    output_dir = Path(output_dir)
    mix_log_path = output_dir / "mix_log.json"
    if not mix_log_path.is_file():
        raise FileNotFoundError(f"Missing {mix_log_path}")

    mix_log = _load_json(mix_log_path)
    dataset_root = Path(mix_log.get("dataset_root") or output_dir / "dataset")
    kept_episodes: list[dict[str, Any]] = list(mix_log.get("kept_episodes") or [])
    if not kept_episodes:
        for seed in mix_log.get("kept_drop_seeds") or []:
            kept_episodes.append({"seed": seed, "kind": "drop"})
        for seed in mix_log.get("kept_nominal_seeds") or []:
            kept_episodes.append({"seed": seed, "kind": "nominal"})

    result = MixAuditResult()
    drop_rows: list[DropEpisodeRow] = []
    triggered_at_values: set[int] = set()
    delay_values: set[int] = set()
    pre_drop_points: list[tuple[float, float]] = []

    for entry in kept_episodes:
        seed = int(entry["seed"])
        kind = str(entry.get("kind", "drop"))
        ep_dir = episode_dir_for_seed(output_dir, seed)
        pipeline_path = ep_dir / "pipeline_log.json"
        if not pipeline_path.is_file():
            result.issues.append(AuditIssue("fail", f"seed {seed}: missing pipeline_log.json"))
            continue
        pipeline = _load_json(pipeline_path)
        event = _read_midair_drop_event(ep_dir / "fault_events.jsonl")

        if kind == "drop":
            fault_cfg = pipeline.get("fault_config") or {}
            delay = entry.get("delay_steps")
            if delay is None:
                delay = fault_cfg.get("post_grasp_delay_steps")
            delay_int = int(delay) if delay is not None else None
            if delay_int is not None:
                delay_values.add(delay_int)

            triggered_at = pipeline.get("triggered_at")
            if triggered_at is not None:
                triggered_at_values.add(int(triggered_at))

            drop_dist = None if event is None else event.get("drop_basket_xy_dist")
            if drop_dist is not None:
                drop_dist = float(drop_dist)
            trigger_reason = None if event is None else event.get("drop_trigger_reason")
            pre_xy = _pre_drop_xy(pipeline, event)
            if pre_xy is not None:
                pre_drop_points.append(pre_xy)

            row = DropEpisodeRow(
                seed=seed,
                delay_steps=delay_int,
                triggered_at=int(triggered_at) if triggered_at is not None else None,
                carry_steps=pipeline.get("carry_steps"),
                drop_basket_xy_dist=drop_dist,
                drop_trigger_reason=trigger_reason,
                pre_drop_xy=pre_xy,
            )
            drop_rows.append(row)

            if not pipeline.get("success"):
                result.issues.append(AuditIssue("fail", f"seed {seed}: drop episode not successful"))
            if triggered_at is None:
                result.issues.append(AuditIssue("fail", f"seed {seed}: triggered_at is null"))
            if pipeline.get("seat_assisted"):
                result.issues.append(AuditIssue("fail", f"seed {seed}: seat_assisted=true"))
            if drop_dist is None:
                result.issues.append(AuditIssue("fail", f"seed {seed}: missing drop_basket_xy_dist"))
            elif drop_dist < HARD_POCKET_XY_M:
                result.issues.append(
                    AuditIssue(
                        "fail",
                        f"seed {seed}: drop_basket_xy_dist {drop_dist:.3f} < hard pocket {HARD_POCKET_XY_M}",
                    )
                )
            elif drop_dist < float(min_xy_m):
                result.issues.append(
                    AuditIssue(
                        "fail",
                        f"seed {seed}: drop_basket_xy_dist {drop_dist:.3f} < min_xy_m {min_xy_m}",
                    )
                )
            if trigger_reason == "basket_keepout_edge":
                result.issues.append(
                    AuditIssue(
                        "fail",
                        f"seed {seed}: drop_trigger_reason=basket_keepout_edge (Option B forbids)",
                    )
                )
            episode_mp4 = ep_dir / "videos" / "full_pipeline.mp4"
            if not episode_mp4.is_file() and not _dataset_has_videos(dataset_root):
                result.issues.append(
                    AuditIssue("fail", f"seed {seed}: no episode video and dataset/videos empty")
                )
        else:
            checks = pipeline.get("checks") or {}
            if not pipeline.get("success"):
                result.issues.append(AuditIssue("fail", f"seed {seed}: nominal episode not successful"))
            if pipeline.get("triggered_at") is not None or checks.get("fault_triggered"):
                result.issues.append(AuditIssue("fail", f"seed {seed}: nominal episode fault triggered"))
            in_basket = checks.get("object_in_basket")
            if in_basket is False:
                result.issues.append(AuditIssue("fail", f"seed {seed}: nominal object not in basket"))

    n_kept = len(kept_episodes)
    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        info = _load_json(info_path)
        total_eps = int(info.get("total_episodes", -1))
        if total_eps != n_kept:
            result.issues.append(
                AuditIssue(
                    "fail",
                    f"dataset total_episodes={total_eps} != mix kept count {n_kept}",
                )
            )
    else:
        result.issues.append(AuditIssue("fail", f"missing dataset meta/info.json under {dataset_root}"))

    n_drops = len(drop_rows)
    min_unique = (
        default_min_unique_triggered_at(n_drops)
        if min_unique_triggered_at is None
        else int(min_unique_triggered_at)
    )
    if n_drops > 0 and len(triggered_at_values) < min_unique:
        result.issues.append(
            AuditIssue(
                "fail",
                f"unique triggered_at={len(triggered_at_values)} < required {min_unique}",
            )
        )

    if expect_delay_grid is not None:
        expected = {int(d) for d in expect_delay_grid}
        if not expected.issubset(delay_values):
            missing = sorted(expected - delay_values)
            result.issues.append(
                AuditIssue("fail", f"kept drops missing delay bins: {missing}")
            )

    pairwise_min = min_pairwise_xy_distance(pre_drop_points)
    if pairwise_min is not None and pairwise_min < float(min_pairwise_xy_m):
        result.issues.append(
            AuditIssue(
                "fail",
                f"min pairwise pre_drop XY {pairwise_min:.3f}m < {min_pairwise_xy_m}m (clustered drops)",
            )
        )

    if verify_parquet_fn is not None and dataset_root.is_dir():
        try:
            result.stats["parquet"] = verify_parquet_fn(dataset_root, require_failure=True)
        except Exception as exc:
            result.issues.append(AuditIssue("fail", f"parquet verify failed: {exc}"))

    result.rows = drop_rows
    result.stats.update(
        {
            "n_kept_episodes": n_kept,
            "n_kept_drops": n_drops,
            "unique_delay_steps": sorted(delay_values),
            "unique_triggered_at": sorted(triggered_at_values),
            "min_pairwise_pre_drop_xy_m": pairwise_min,
            "hard_pocket_xy_m": HARD_POCKET_XY_M,
            "min_xy_m": float(min_xy_m),
        }
    )
    return result


def format_drop_table(rows: list[DropEpisodeRow]) -> str:
    header = (
        f"{'seed':>6} {'delay':>6} {'trig':>5} {'carry':>5} "
        f"{'xy_dist':>8} {'reason':>20} {'pre_drop_xy':>18}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        xy = "" if r.pre_drop_xy is None else f"({r.pre_drop_xy[0]:.3f},{r.pre_drop_xy[1]:.3f})"
        delay_s = "-" if r.delay_steps is None else str(r.delay_steps)
        trig_s = "-" if r.triggered_at is None else str(r.triggered_at)
        carry_s = "-" if r.carry_steps is None else str(r.carry_steps)
        dist_s = "-" if r.drop_basket_xy_dist is None else f"{r.drop_basket_xy_dist:.3f}"
        reason_s = r.drop_trigger_reason or "-"
        lines.append(
            f"{r.seed:>6} {delay_s:>6} {trig_s:>5} {carry_s:>5} "
            f"{dist_s:>8} {reason_s:>20} {xy:>18}"
        )
    return "\n".join(lines)


def audit_report_dict(result: MixAuditResult) -> dict[str, Any]:
    return {
        "stats": result.stats,
        "issues": [{"level": i.level, "message": i.message} for i in result.issues],
        "drop_rows": [
            {
                "seed": r.seed,
                "delay_steps": r.delay_steps,
                "triggered_at": r.triggered_at,
                "carry_steps": r.carry_steps,
                "drop_basket_xy_dist": r.drop_basket_xy_dist,
                "drop_trigger_reason": r.drop_trigger_reason,
                "pre_drop_xy": list(r.pre_drop_xy) if r.pre_drop_xy else None,
            }
            for r in result.rows
        ],
        "exit_code": result.exit_code(),
    }


def parse_expect_delay_grid(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return parse_int_list(value)
