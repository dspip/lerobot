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
"""Record training-grade XY-band mid-air drops (Option B, no seat assist)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BANDS: list[tuple[str, float, float]] = [
    ("lift", 0.48, 0.52),
    ("early", 0.42, 0.46),
    ("mid", 0.34, 0.38),
    ("late", 0.30, 0.33),
]


def _load_run_pipeline():
    path = REPO_ROOT / "examples" / "faults" / "run_full_drop_recovery_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_full_drop_recovery_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_pipeline


def _episode_dir(output_dir: Path, seed: int) -> Path:
    from lerobot.faults.recovery.mix_recording import episode_dir_for_seed

    return episode_dir_for_seed(output_dir, seed)


def main(argv: list[str] | None = None) -> int:
    from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger
    from lerobot.faults.recovery.fps import SMOLVLA_LIBERO_TARGET_FPS
    from lerobot.faults.recovery.mix_recording import (
        default_seed_sequence,
        mix_output_layout,
        should_keep_episode,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "xy_band_pilot",
    )
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--drops-per-band", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--seed-start", type=int, default=5100)
    parser.add_argument("--repo-id", default="local/xy_band_pilot")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--t-min", type=int, default=0)
    parser.add_argument("--t-max", type=int, default=400)
    parser.add_argument(
        "--min-drop-distance-m",
        type=float,
        default=0.30,
        help="Training recipe min_drop_distance_from_basket_m (Option B)",
    )
    args = parser.parse_args(argv)

    run_pipeline = _load_run_pipeline()
    layout = mix_output_layout(args.output_dir)
    dataset_root = layout["dataset"]
    pilot_log_path = args.output_dir / "pilot_log.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout["episodes"].mkdir(parents=True, exist_ok=True)

    if dataset_root.exists() and not args.append:
        print(
            f"Refusing to overwrite existing dataset at {dataset_root}. "
            "Pass --append or remove the directory.",
            flush=True,
        )
        return 2

    logger = FaultRecoveryDatasetLogger(
        root=dataset_root,
        repo_id=args.repo_id,
        policy_fps=SMOLVLA_LIBERO_TARGET_FPS,
        append=bool(args.append and dataset_root.exists()),
    )

    seed_pool = default_seed_sequence(args.seed_start, int(args.max_attempts))
    seed_index = 0
    kept_per_band: dict[str, int] = {name: 0 for name, _, _ in DEFAULT_BANDS}
    attempts_log: list[dict[str, Any]] = []
    kept_episodes: list[dict[str, Any]] = []

    for band_name, band_lo, band_hi in DEFAULT_BANDS:
        while kept_per_band[band_name] < int(args.drops_per_band):
            if seed_index >= len(seed_pool):
                print("[pilot] seed pool exhausted before band quotas filled.", flush=True)
                break
            seed = seed_pool[seed_index]
            seed_index += 1
            ep_dir = _episode_dir(args.output_dir, seed)
            print(
                f"[pilot] band={band_name} [{band_lo},{band_hi}] seed={seed} "
                f"kept={kept_per_band[band_name]}/{args.drops_per_band}",
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
                    dataset_root=dataset_root,
                    ds_logger=logger,
                    raise_on_failure=False,
                    copy_demo_gif=False,
                    episode_kind="drop",
                    repo_id=args.repo_id,
                    min_drop_distance_from_basket_m=float(args.min_drop_distance_m),
                    drop_xy_band_min=float(band_lo),
                    drop_xy_band_max=float(band_hi),
                    fault_overrides={"object_name": "alphabet_soup_1"},
                )
            except Exception as exc:
                try:
                    logger.clear_open_episode()
                except Exception:
                    pass
                summary = {
                    "success": False,
                    "seed": seed,
                    "checks": {"exception": str(exc)},
                }

            kept = should_keep_episode(summary, "drop")
            attempts_log.append(
                {
                    "seed": seed,
                    "band": band_name,
                    "band_lo": band_lo,
                    "band_hi": band_hi,
                    "kept": kept,
                    "success": bool(summary.get("success")),
                    "checks": summary.get("checks"),
                }
            )
            if kept:
                kept_per_band[band_name] += 1
                kept_episodes.append(
                    {
                        "seed": seed,
                        "band": band_name,
                        "band_lo": band_lo,
                        "band_hi": band_hi,
                        "drop_basket_xy_dist": summary.get("drop_basket_xy_dist"),
                        "drop_trigger_reason": summary.get("drop_trigger_reason"),
                        "triggered_at": summary.get("triggered_at"),
                        "seat_assisted": summary.get("seat_assisted"),
                        "success": summary.get("success"),
                        "checks": summary.get("checks"),
                        "video_mp4": summary.get("video_mp4"),
                        "pipeline_log": str(ep_dir / "pipeline_log.json"),
                    }
                )
        if kept_per_band[band_name] < int(args.drops_per_band):
            break

    try:
        logger.clear_open_episode()
    except Exception:
        pass
    if any(kept_per_band.values()):
        logger.finalize()

    payload = {
        "bands": [
            {"name": n, "min": lo, "max": hi, "kept": kept_per_band[n]}
            for n, lo, hi in DEFAULT_BANDS
        ],
        "kept_episodes": kept_episodes,
        "attempts": attempts_log,
        "dataset_dir": str(dataset_root),
    }
    pilot_log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {pilot_log_path}", flush=True)

    target = len(DEFAULT_BANDS) * int(args.drops_per_band)
    kept_total = sum(kept_per_band.values())
    if kept_total < target:
        print(f"[pilot] incomplete: kept {kept_total}/{target} drops", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
