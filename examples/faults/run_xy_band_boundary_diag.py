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

#!/usr/bin/env python3
"""Diagnostic runs near the basket XY boundary (not training-grade)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# (label, target_dist, min_drop, band_lo, band_hi)
BOUNDARY_TARGETS: list[tuple[str, float, float, float, float]] = [
    ("d030", 0.30, 0.30, 0.30, 0.32),
    ("d027", 0.27, 0.27, 0.27, 0.29),
    ("d024", 0.24, 0.24, 0.24, 0.26),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "xy_band_boundary_diag",
    )
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed-start", type=int, default=6200)
    parser.add_argument("--attempts-per-target", type=int, default=2)
    parser.add_argument("--t-min", type=int, default=0)
    parser.add_argument("--t-max", type=int, default=400)
    args = parser.parse_args(argv)

    if "xy_band_pilot" in str(args.output_dir) and "boundary" not in str(args.output_dir):
        print("Refusing to write boundary diag into xy_band_pilot output.", flush=True)
        return 2

    from lerobot.faults.datagen.smolvla_pipeline import run_pipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes_root = args.output_dir / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    # Per-episode dataset under the wiped episode dir. A shared dataset_diag
    # without append=True makes FaultRecoveryDatasetLogger fail on attempt 2+.
    records: list[dict[str, Any]] = []
    seed = int(args.seed_start)

    for label, target_dist, min_drop, band_lo, band_hi in BOUNDARY_TARGETS:
        for attempt in range(int(args.attempts_per_target)):
            ep_dir = episodes_root / f"{label}_seed_{seed}"
            print(
                f"[boundary] target={target_dist} band=[{band_lo},{band_hi}] min_drop={min_drop} seed={seed}",
                flush=True,
            )
            try:
                summary = run_pipeline(
                    ep_dir,
                    policy_path=args.policy_path,
                    device=args.device,
                    seed=seed,
                    t_min=int(args.t_min),
                    t_max=int(args.t_max),
                    post_grasp_delay_steps=0,
                    seat_assist_enabled=False,
                    forbid_seat_assist=True,
                    wipe_output_dir=True,
                    dataset_root=None,
                    ds_logger=None,
                    raise_on_failure=False,
                    copy_demo_gif=False,
                    episode_kind="drop",
                    repo_id="local/xy_band_boundary_diag",
                    min_drop_distance_from_basket_m=float(min_drop),
                    drop_xy_band_min=float(band_lo),
                    drop_xy_band_max=float(band_hi),
                    fault_overrides={"object_name": "alphabet_soup_1"},
                )
            except Exception as exc:
                print(f"[boundary] seed={seed} crashed: {exc}", flush=True)
                summary = {"success": False, "checks": {"exception": str(exc)}}

            checks = summary.get("checks") or {}
            records.append(
                {
                    "label": label,
                    "requested_dist_m": target_dist,
                    "band_lo": band_lo,
                    "band_hi": band_hi,
                    "min_drop_distance_from_basket_m": min_drop,
                    "seed": seed,
                    "attempt": attempt,
                    "success": bool(summary.get("success")),
                    "triggered_at": summary.get("triggered_at"),
                    "drop_basket_xy_dist": summary.get("drop_basket_xy_dist"),
                    "pre_drop_xy": (
                        summary.get("pre_drop_pose")[:2] if summary.get("pre_drop_pose") else None
                    ),
                    "landing_pos": summary.get("final_object_pos"),
                    "in_basket": checks.get("object_in_basket"),
                    "object_visible_after_drop": checks.get("object_in_view_after_drop"),
                    "regrasped_after_drop": summary.get("regrasped_after_drop"),
                    "final_place_ok": checks.get("object_in_basket"),
                    "seat_assisted": summary.get("seat_assisted"),
                    "video_mp4": summary.get("video_mp4"),
                    "pipeline_log": str(ep_dir / "pipeline_log.json"),
                }
            )
            seed += 1

    log_path = args.output_dir / "boundary_log.json"
    payload = {"attempts": records, "note": "diagnostic only; not a training dataset"}
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
