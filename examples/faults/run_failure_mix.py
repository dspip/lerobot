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
"""Record many drop-recovery and nominal successes into one LeRobot dataset root."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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


def _cli_flag_present(flag: str, argv: list[str] | None) -> bool:
    args = argv if argv is not None else sys.argv[1:]
    for token in args:
        if token == flag or token.startswith(f"{flag}="):
            return True
    return False


def _run_one_attempt(
    *,
    run_pipeline: Any,
    args: argparse.Namespace,
    logger: Any,
    dataset_root: Path,
    output_dir: Path,
    seed: int,
    kind: str,
    post_grasp_delay_steps: int | None,
) -> dict[str, Any]:
    from lerobot.faults.recovery.mix_recording import episode_dir_for_seed

    episode_dir = episode_dir_for_seed(output_dir, seed)
    pipeline_kwargs: dict[str, Any] = {
        "policy_path": args.policy_path,
        "device": args.device,
        "seed": seed,
        "post_grasp_delay_min": args.post_grasp_delay_min,
        "post_grasp_delay_max": args.post_grasp_delay_max,
        "seat_assist_enabled": bool(args.allow_seat_assist),
        "forbid_seat_assist": not bool(args.allow_seat_assist),
        "wipe_output_dir": True,
        "dataset_root": dataset_root,
        "ds_logger": logger,
        "raise_on_failure": False,
        "copy_demo_gif": False,
        "episode_kind": kind,
        "repo_id": args.repo_id,
    }
    if post_grasp_delay_steps is not None:
        pipeline_kwargs["post_grasp_delay_steps"] = int(post_grasp_delay_steps)
    elif args.post_grasp_delay_steps is not None:
        pipeline_kwargs["post_grasp_delay_steps"] = int(args.post_grasp_delay_steps)
    if args.min_drop_distance_m is not None:
        pipeline_kwargs["min_drop_distance_from_basket_m"] = float(args.min_drop_distance_m)
    if args.t_min is not None:
        pipeline_kwargs["t_min"] = int(args.t_min)
    if args.t_max is not None:
        pipeline_kwargs["t_max"] = int(args.t_max)

    try:
        return run_pipeline(episode_dir, **pipeline_kwargs)
    except Exception as exc:
        try:
            logger.clear_open_episode()
        except Exception as clear_exc:
            print(f"[mix] could not discard open episode: {clear_exc}", flush=True)
        print(f"[mix] seed={seed} kind={kind} crashed: {exc}", flush=True)
        return {
            "success": False,
            "seed": seed,
            "episode_kind": kind,
            "episode_committed": False,
            "checks": {"exception": str(exc)},
        }


def main(argv: list[str] | None = None) -> int:
    from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger
    from lerobot.faults.recovery.fps import SMOLVLA_LIBERO_TARGET_FPS
    from lerobot.faults.recovery.mix_recording import (
        MixCounters,
        compute_mix_max_attempts,
        default_seed_sequence,
        episode_dir_for_seed,
        mix_output_layout,
        next_open_delay_grid_bin,
        parse_drops_per_delay,
        parse_int_list,
        parse_seed_list,
        resolve_mix_drop_target,
        should_keep_episode,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "failure_mix_v1",
    )
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-drop", type=int, default=None)
    parser.add_argument("--n-nominal", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; overrides seed-start sequence length",
    )
    parser.add_argument("--repo-id", default="local/failure_mix_v1")
    parser.add_argument("--append", action="store_true", help="Append to existing dataset root")
    parser.add_argument("--post-grasp-delay-min", type=int, default=20)
    parser.add_argument("--post-grasp-delay-max", type=int, default=60)
    parser.add_argument("--post-grasp-delay-steps", type=int, default=None)
    parser.add_argument(
        "--delay-grid",
        type=str,
        default=None,
        help="Comma-separated post-grasp delay steps per bin (e.g. 20,30,40,50,60)",
    )
    parser.add_argument(
        "--drops-per-delay",
        type=str,
        default="3",
        help="Kept drops per delay bin: single int (broadcast) or comma list matching --delay-grid",
    )
    parser.add_argument(
        "--t-min",
        type=int,
        default=None,
        help="Override pipeline fault t_min (default 40 when omitted)",
    )
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="Override pipeline fault t_max (default 400 when omitted)",
    )
    parser.add_argument(
        "--min-drop-distance-m",
        type=float,
        default=None,
        help="Override training recipe min_drop_distance_from_basket_m (default 0.30)",
    )
    parser.add_argument(
        "--allow-seat-assist",
        action="store_true",
        help="Enable rim teleport (not for training-grade mixes)",
    )
    args = parser.parse_args(argv)

    n_drop_explicit = _cli_flag_present("--n-drop", argv)
    delay_grid = parse_int_list(args.delay_grid) if args.delay_grid else None
    drops_per_bin: list[int] | None = None
    if delay_grid is not None:
        try:
            drops_per_bin = parse_drops_per_delay(delay_grid, args.drops_per_delay)
        except ValueError as exc:
            print(f"[mix] {exc}", flush=True)
            return 2
    try:
        n_drop = resolve_mix_drop_target(
            delay_grid=delay_grid,
            drops_per_delay=args.drops_per_delay if delay_grid else int(args.drops_per_delay),
            n_drop=args.n_drop,
            n_drop_explicit=n_drop_explicit,
        )
    except ValueError as exc:
        print(f"[mix] {exc}", flush=True)
        return 2
    n_nominal = int(args.n_nominal)
    if n_nominal < 0:
        print(f"[mix] n-nominal must be >= 0 (got {n_nominal})", flush=True)
        return 2

    run_pipeline = _load_run_pipeline()

    layout = mix_output_layout(args.output_dir)
    dataset_root = layout["dataset"]
    mix_log_path = layout["mix_log"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layout["episodes"].mkdir(parents=True, exist_ok=True)

    if dataset_root.exists() and not args.append:
        print(
            f"Refusing to overwrite existing dataset at {dataset_root}. "
            "Pass --append to add episodes or remove the directory.",
            flush=True,
        )
        return 2

    counters = MixCounters(n_drop_target=n_drop, n_nominal_target=n_nominal)
    max_attempts = compute_mix_max_attempts(
        n_drop=n_drop,
        n_nominal=n_nominal,
        delay_grid=delay_grid,
        max_attempts_arg=int(args.max_attempts),
    )
    if args.seeds:
        seed_pool = parse_seed_list(args.seeds)
    else:
        seed_pool = default_seed_sequence(args.seed_start, max_attempts)

    logger = FaultRecoveryDatasetLogger(
        root=dataset_root,
        repo_id=args.repo_id,
        policy_fps=SMOLVLA_LIBERO_TARGET_FPS,
        append=bool(args.append and dataset_root.exists()),
    )

    attempts_log: list[dict[str, Any]] = []
    kept_episodes: list[dict[str, Any]] = []
    seed_index = 0
    kept_per_delay: dict[int, int] = {int(d): 0 for d in delay_grid} if delay_grid else {}

    def _log_attempt(seed: int, kind: str, summary: dict[str, Any], kept: bool, delay_steps: int | None) -> None:
        attempts_log.append(
            {
                "seed": seed,
                "kind": kind,
                "kept": kept,
                "success": bool(summary.get("success")),
                "episode_committed": bool(summary.get("episode_committed")),
                "delay_steps": delay_steps,
                "checks": summary.get("checks"),
            }
        )
        if kept:
            record: dict[str, Any] = {"seed": seed, "kind": kind}
            if kind == "drop":
                fault_cfg = summary.get("fault_config") or {}
                record["delay_steps"] = int(
                    delay_steps
                    if delay_steps is not None
                    else fault_cfg.get("post_grasp_delay_steps", 0)
                )
            kept_episodes.append(record)

    try:
        if delay_grid is not None:
            while counters.attempts < max_attempts:
                assert drops_per_bin is not None
                delay = next_open_delay_grid_bin(
                    delay_grid,
                    drops_per_bin=drops_per_bin,
                    kept_per_delay=kept_per_delay,
                )
                if delay is None:
                    break
                if seed_index >= len(seed_pool):
                    print(
                        f"[mix] seed pool exhausted ({len(seed_pool)} seeds) before drop grid filled; "
                        "pass more --seeds or raise --max-attempts",
                        flush=True,
                    )
                    break
                seed = seed_pool[seed_index]
                seed_index += 1
                print(
                    f"[mix] attempt {counters.attempts + 1}/{max_attempts} "
                    f"kind=drop delay={delay} seed={seed} "
                    f"(kept drop={counters.drop_kept}/{n_drop})",
                    flush=True,
                )
                summary = _run_one_attempt(
                    run_pipeline=run_pipeline,
                    args=args,
                    logger=logger,
                    dataset_root=dataset_root,
                    output_dir=args.output_dir,
                    seed=seed,
                    kind="drop",
                    post_grasp_delay_steps=delay,
                )
                kept = should_keep_episode(summary, "drop")
                counters.record_attempt("drop", kept)
                if kept:
                    kept_per_delay[delay] = kept_per_delay.get(delay, 0) + 1
                _log_attempt(seed, "drop", summary, kept, delay)
                print(
                    f"[mix] seed={seed} kind=drop delay={delay} kept={kept} "
                    f"totals drop={counters.drop_kept}/{n_drop}",
                    flush=True,
                )

            while counters.nominal_kept < n_nominal and counters.attempts < max_attempts:
                if seed_index >= len(seed_pool):
                    print(
                        f"[mix] seed pool exhausted ({len(seed_pool)} seeds) before nominal quota; "
                        "pass more --seeds or raise --max-attempts",
                        flush=True,
                    )
                    break
                seed = seed_pool[seed_index]
                seed_index += 1
                print(
                    f"[mix] attempt {counters.attempts + 1}/{max_attempts} "
                    f"kind=nominal seed={seed} "
                    f"(kept nominal={counters.nominal_kept}/{n_nominal})",
                    flush=True,
                )
                summary = _run_one_attempt(
                    run_pipeline=run_pipeline,
                    args=args,
                    logger=logger,
                    dataset_root=dataset_root,
                    output_dir=args.output_dir,
                    seed=seed,
                    kind="nominal",
                    post_grasp_delay_steps=None,
                )
                kept = should_keep_episode(summary, "nominal")
                counters.record_attempt("nominal", kept)
                _log_attempt(seed, "nominal", summary, kept, None)
                print(
                    f"[mix] seed={seed} kind=nominal kept={kept} "
                    f"totals nominal={counters.nominal_kept}/{n_nominal}",
                    flush=True,
                )
        else:
            while counters.attempts < max_attempts and not counters.quotas_met:
                kind = counters.next_kind_interleave()
                if kind is None:
                    break
                if seed_index >= len(seed_pool):
                    print(
                        f"[mix] seed pool exhausted ({len(seed_pool)} seeds) before quotas; "
                        "pass more --seeds or raise --max-attempts",
                        flush=True,
                    )
                    break
                seed = seed_pool[seed_index]
                seed_index += 1
                print(
                    f"[mix] attempt {counters.attempts + 1}/{max_attempts} "
                    f"kind={kind} seed={seed} "
                    f"(kept drop={counters.drop_kept}/{n_drop} "
                    f"nominal={counters.nominal_kept}/{n_nominal})",
                    flush=True,
                )
                summary = _run_one_attempt(
                    run_pipeline=run_pipeline,
                    args=args,
                    logger=logger,
                    dataset_root=dataset_root,
                    output_dir=args.output_dir,
                    seed=seed,
                    kind=kind,
                    post_grasp_delay_steps=None,
                )
                kept = should_keep_episode(summary, kind)
                counters.record_attempt(kind, kept)
                delay_steps = None
                if kind == "drop":
                    fault_cfg = summary.get("fault_config") or {}
                    if "post_grasp_delay_steps" in fault_cfg:
                        delay_steps = int(fault_cfg["post_grasp_delay_steps"])
                _log_attempt(seed, kind, summary, kept, delay_steps)
                print(
                    f"[mix] seed={seed} kind={kind} kept={kept} "
                    f"totals drop={counters.drop_kept}/{n_drop} "
                    f"nominal={counters.nominal_kept}/{n_nominal}",
                    flush=True,
                )
    finally:
        if getattr(logger, "_episode_open", False):
            try:
                logger.clear_open_episode()
            except Exception as clear_exc:
                print(f"[mix] leftover episode discard failed: {clear_exc}", flush=True)
        logger.finalize()

    mix_record = {
        "output_dir": str(args.output_dir),
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "delay_grid": delay_grid,
        "drops_per_delay": drops_per_bin if delay_grid else None,
        "counters": counters.to_dict(),
        "attempts": attempts_log,
        "kept_episodes": kept_episodes,
        "kept_drop_seeds": [a["seed"] for a in kept_episodes if a["kind"] == "drop"],
        "kept_nominal_seeds": [a["seed"] for a in kept_episodes if a["kind"] == "nominal"],
    }
    mix_log_path.write_text(json.dumps(mix_record, indent=2))
    print(json.dumps(mix_record["counters"], indent=2), flush=True)

    total_kept = counters.drop_kept + counters.nominal_kept
    if total_kept < 1:
        print("[mix] FAIL: zero episodes committed to dataset", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
