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
"""Checkpoint 1: copy XY-band pilot + record 4 drops + 8 nominal episodes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    from lerobot.faults.recovery.mix_recording import commit_or_discard, mix_output_layout
    from lerobot.faults.recovery.xy_band_checkpoint import (
        XY_BANDS,
        copy_pilot_into_checkpoint,
        filter_checkpoint_seeds,
        is_blocked_checkpoint_seed,
        rejection_reason,
        should_keep_checkpoint_drop,
        should_keep_checkpoint_nominal,
        summary_episode_fields,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "xy_band_checkpoint1",
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "xy_band_pilot",
    )
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-new-drops-per-band", type=int, default=1)
    parser.add_argument("--n-new-nominal", type=int, default=8)
    parser.add_argument("--drop-seed-start", type=int, default=7200)
    parser.add_argument("--nominal-seed-start", type=int, default=8200)
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--repo-id", default="local/xy_band_checkpoint1")
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout["episodes"].mkdir(parents=True, exist_ok=True)

    registry_path = args.output_dir / "kept_registry.json"
    if not dataset_root.exists():
        pilot_kept = copy_pilot_into_checkpoint(
            pilot_dir=args.pilot_dir,
            output_dir=args.output_dir,
        )
    else:
        if registry_path.is_file():
            pilot_kept = json.loads(registry_path.read_text(encoding="utf-8")).get("kept") or []
        else:
            pilot_kept = []

    kept_registry: list[dict[str, Any]] = list(pilot_kept)
    kept_per_band: dict[str, int] = {name: 0 for name, _, _ in XY_BANDS}
    for row in kept_registry:
        if row.get("kind") == "drop" and row.get("band") in kept_per_band:
            kept_per_band[str(row["band"])] += 1

    logger = FaultRecoveryDatasetLogger(
        root=dataset_root,
        repo_id=args.repo_id,
        policy_fps=SMOLVLA_LIBERO_TARGET_FPS,
        append=True,
    )

    drop_seed_pool = filter_checkpoint_seeds(
        [args.drop_seed_start + i for i in range(int(args.max_attempts))]
    )
    nominal_seed_pool = filter_checkpoint_seeds(
        [args.nominal_seed_start + i for i in range(int(args.max_attempts))]
    )
    drop_seed_index = 0
    nominal_seed_index = 0

    attempts_log: list[dict[str, Any]] = []
    checkpoint_log_path = args.output_dir / "checkpoint_log.json"

    def _kept_in_band(band_name: str) -> list[dict[str, Any]]:
        return [
            k
            for k in kept_registry
            if k.get("kind") == "drop" and k.get("band") == band_name
        ]

    for band_name, band_lo, band_hi in XY_BANDS:
        target_new = int(args.n_new_drops_per_band)
        new_kept = 0
        while new_kept < target_new:
            if drop_seed_index >= len(drop_seed_pool):
                print("[checkpoint] drop seed pool exhausted.", flush=True)
                break
            seed = drop_seed_pool[drop_seed_index]
            drop_seed_index += 1
            if is_blocked_checkpoint_seed(seed):
                continue
            ep_dir = _episode_dir(args.output_dir, seed)
            print(
                f"[checkpoint] drop band={band_name} seed={seed} "
                f"new_kept={new_kept}/{target_new}",
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
                    defer_dataset_commit=True,
                )
            except Exception as exc:
                try:
                    logger.clear_open_episode()
                except Exception:
                    pass
                summary = {"success": False, "seed": seed, "checks": {"exception": str(exc)}}

            existing = _kept_in_band(band_name)
            kept = should_keep_checkpoint_drop(
                summary,
                band_lo=band_lo,
                band_hi=band_hi,
                existing_kept=existing,
            )
            reason = rejection_reason(
                summary,
                kind="drop",
                band_lo=band_lo,
                band_hi=band_hi,
                existing_kept=existing,
            )
            fields = summary_episode_fields(summary)
            attempts_log.append(
                {
                    "seed": seed,
                    "kind": "drop",
                    "band": band_name,
                    "band_lo": band_lo,
                    "band_hi": band_hi,
                    "kept": kept,
                    "reason": reason,
                    **fields,
                    "pipeline_log": str(ep_dir / "pipeline_log.json"),
                }
            )
            if kept:
                commit_or_discard(logger, keep=True)
                new_kept += 1
                entry = {
                    "seed": seed,
                    "kind": "drop",
                    "band": band_name,
                    "band_lo": band_lo,
                    "band_hi": band_hi,
                    "source": "checkpoint",
                    **fields,
                }
                kept_registry.append(entry)
            else:
                try:
                    logger.clear_open_episode()
                except Exception:
                    commit_or_discard(logger, keep=False)

    nominal_kept = 0
    while nominal_kept < int(args.n_new_nominal):
        if nominal_seed_index >= len(nominal_seed_pool):
            print("[checkpoint] nominal seed pool exhausted.", flush=True)
            break
        seed = nominal_seed_pool[nominal_seed_index]
        nominal_seed_index += 1
        if is_blocked_checkpoint_seed(seed):
            continue
        ep_dir = _episode_dir(args.output_dir, seed)
        print(
            f"[checkpoint] nominal seed={seed} kept={nominal_kept}/{args.n_new_nominal}",
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
                seat_assist_enabled=False,
                forbid_seat_assist=True,
                wipe_output_dir=True,
                dataset_root=dataset_root,
                ds_logger=logger,
                raise_on_failure=False,
                copy_demo_gif=False,
                episode_kind="nominal",
                repo_id=args.repo_id,
                defer_dataset_commit=True,
            )
        except Exception as exc:
            try:
                logger.clear_open_episode()
            except Exception:
                pass
            summary = {"success": False, "seed": seed, "checks": {"exception": str(exc)}}

        kept = should_keep_checkpoint_nominal(summary)
        reason = rejection_reason(summary, kind="nominal")
        fields = summary_episode_fields(summary)
        attempts_log.append(
            {
                "seed": seed,
                "kind": "nominal",
                "kept": kept,
                "reason": reason,
                **fields,
                "pipeline_log": str(ep_dir / "pipeline_log.json"),
            }
        )
        if kept:
            commit_or_discard(logger, keep=True)
            nominal_kept += 1
            kept_registry.append(
                {
                    "seed": seed,
                    "kind": "nominal",
                    "source": "checkpoint",
                    **fields,
                }
            )
        else:
            try:
                logger.clear_open_episode()
            except Exception:
                commit_or_discard(logger, keep=False)

    try:
        logger.clear_open_episode()
    except Exception:
        pass
    if kept_registry:
        logger.finalize()

    registry_path.write_text(json.dumps({"kept": kept_registry}, indent=2), encoding="utf-8")

    drop_kept_total = sum(1 for k in kept_registry if k.get("kind") == "drop")
    nominal_kept_total = sum(1 for k in kept_registry if k.get("kind") == "nominal")
    payload = {
        "bands": [{"name": n, "min": lo, "max": hi} for n, lo, hi in XY_BANDS],
        "kept_registry": kept_registry,
        "attempts": attempts_log,
        "dataset_dir": str(dataset_root),
        "quotas": {
            "drops_target": 12,
            "nominal_target": 8,
            "drops_kept": drop_kept_total,
            "nominal_kept": nominal_kept_total,
        },
    }
    checkpoint_log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {checkpoint_log_path}", flush=True)

    drops_ok = drop_kept_total >= 12
    per_band_ok = all(
        sum(1 for k in kept_registry if k.get("kind") == "drop" and k.get("band") == name) >= 3
        for name, _, _ in XY_BANDS
    )
    nominal_ok = nominal_kept_total >= int(args.n_new_nominal) and nominal_kept_total >= 8
    if not (drops_ok and per_band_ok and nominal_ok):
        print(
            f"[checkpoint] incomplete: drops={drop_kept_total}/12 nominal={nominal_kept_total}/8",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
