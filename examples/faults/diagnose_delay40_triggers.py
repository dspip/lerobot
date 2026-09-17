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
"""Diagnose why delay=40 midair_drop episodes never trigger (offline logs only).

Example::

    uv run python examples/faults/diagnose_delay40_triggers.py \\
        --output-dir outputs/failure_mix_v1 \\
        --seeds 4008,4012,4020,4030,4040
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

from lerobot.faults.recovery.delay40_diagnostics import (
    build_q1_q8,
    frames_from_first_grasp,
    load_pipeline_log,
    reconstruct_frames,
    summarize_episode,
    write_frames_csv,
)

DEFAULT_FAILED_SEEDS = [4008, 4012, 4020, 4030, 4040]
SUCCESS_CONTRAST_SEEDS = [4004, 4007]
VIDEO_STRIDE = 2
VIDEO_FPS = 10.0


def _parse_seeds(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def _is_delay40_failure(log: dict[str, Any]) -> bool:
    delay = int((log.get("fault_config") or {}).get("post_grasp_delay_steps", -1))
    return delay == 40 and log.get("triggered_at") is None


def _annotate_video(
    *,
    src_mp4: Path,
    dst_mp4: Path,
    rows: list[Any],
    first_grasp_step: int | None,
    delay_steps: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(src_mp4))
    if not cap.isOpened():
        return {"ok": False, "error": f"could not open {src_mp4}"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst_mp4), fourcc, VIDEO_FPS, (width, height))

    row_by_step = {r.episode_frame: r for r in rows}
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        step = frame_idx * VIDEO_STRIDE
        row = row_by_step.get(step)
        lines = [f"video_frame={frame_idx}  ctrl_step~={step}"]
        if row is not None:
            ssg = row.steps_since_grasp
            ssg_s = "n/a" if ssg is None else str(ssg)
            lines.extend(
                [
                    f"xy_dist={row.xy_dist_m:.3f} m",
                    f"grasped={'yes' if row.grasped else 'no'}",
                    f"steps_since_grasp={ssg_s}",
                    f"block={row.block_reason}",
                ]
            )
            if ssg == delay_steps:
                cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 255), 8)
                lines.append(f"DELAY_ELAPSED ({delay_steps} steps since grasp)")
        else:
            lines.append("(no log row for this step)")

        y = 28
        for line in lines:
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += 26
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return {
        "ok": True,
        "source": str(src_mp4),
        "output": str(dst_mp4),
        "video_frames": frame_idx,
        "expected_control_steps": frame_idx * VIDEO_STRIDE,
        "log_steps": len(rows),
        "first_grasp_step": first_grasp_step,
    }


def _write_report_md(
    path: Path,
    *,
    failed_summaries: dict[int, dict[str, Any]],
    success_summaries: dict[int, dict[str, Any]],
    q1_q8: dict[str, Any],
    video_meta: dict[str, Any],
) -> None:
    lines = [
        "# Delay-40 midair_drop trigger diagnosis",
        "",
        "Reconstructed from `pipeline_log.json` (CPU-only). Option B keep-out: XY ≥ 0.30 m.",
        "",
        "## grasp_flags timing",
        "",
        "`grasp_flags[i]` is `grasped_now` **before** `env.step` at control index `i`; "
        "`object_traj[i]` is **after** that step (one-step offset vs post-step pose).",
        "",
        "## Q1–Q8 (numeric, from frame CSVs)",
        "",
    ]
    for key, val in q1_q8.items():
        lines.append(f"- **{key}**: {val}")
    lines.extend(["", "## Delay-40 failed seeds (triggered_at null)", ""])
    for seed, summary in sorted(failed_summaries.items()):
        lines.append(f"### seed_{seed}")
        lines.append(f"- first_grasp_step: {summary.get('first_grasp_step')}")
        lines.append(f"- delay target step: {summary.get('delay_target_step')}")
        lines.append(f"- delay target exists: {summary.get('delay_target_frame_exists')}")
        at_d = summary.get("at_delay_target") or {}
        lines.append(f"- at delay: xy_dist={at_d.get('xy_dist_m')} block={at_d.get('block_reason')}")
        lines.append(
            f"- crossings (from above, after grasp): "
            f"0.50@{summary.get('first_cross_below_0.50')} "
            f"0.40@{summary.get('first_cross_below_0.40')} "
            f"0.30@{summary.get('first_cross_below_0.30')}"
        )
        lines.append(f"- latest_safe_drop_step: {summary.get('latest_safe_drop_step')}")
        lines.append("")
    lines.extend(["## Delay-20 success contrast", ""])
    for seed, summary in sorted(success_summaries.items()):
        at_d = summary.get("at_delay_target") or {}
        lines.append(
            f"- seed_{seed}: first_grasp={summary.get('first_grasp_step')} "
            f"xy@grasp+{summary.get('post_grasp_delay_steps')}={at_d.get('xy_dist_m')} "
            f"triggered_at={summary.get('triggered_at')}"
        )
    lines.extend(["", "## Diagnostic video", "", json.dumps(video_meta, indent=2)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_FAILED_SEEDS),
        help="Comma-separated delay-40 failed seeds to analyze",
    )
    parser.add_argument(
        "--video-seed",
        type=int,
        default=4008,
        help="Failed delay-40 seed with full_pipeline.mp4 for overlay video",
    )
    args = parser.parse_args(argv)

    out_root = Path(args.output_dir)
    diag_dir = out_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    episodes = out_root / "episodes"

    failed_seeds = _parse_seeds(args.seeds)
    failed_summaries: dict[int, dict[str, Any]] = {}
    paths_written: list[str] = []

    for seed in failed_seeds:
        ep_dir = episodes / f"seed_{seed}"
        log = load_pipeline_log(ep_dir)
        if not _is_delay40_failure(log):
            print(f"[warn] seed_{seed} is not delay=40 with triggered_at=null; skipping", file=sys.stderr)
            continue
        all_rows = reconstruct_frames(log)
        grasp_rows = frames_from_first_grasp(all_rows, log.get("first_grasp_step"))
        csv_path = diag_dir / f"seed_{seed}_frames.csv"
        write_frames_csv(csv_path, grasp_rows)
        summary = summarize_episode(log, all_rows)
        summary_path = diag_dir / f"seed_{seed}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        failed_summaries[seed] = summary
        paths_written.extend([str(csv_path), str(summary_path)])
        print(f"seed_{seed}: {len(grasp_rows)} grasp+ frames, delay xy={summary.get('at_delay_target')}")

    success_summaries: dict[int, dict[str, Any]] = {}
    for seed in SUCCESS_CONTRAST_SEEDS:
        ep_dir = episodes / f"seed_{seed}"
        if not (ep_dir / "pipeline_log.json").is_file():
            continue
        log = load_pipeline_log(ep_dir)
        if int((log.get("fault_config") or {}).get("post_grasp_delay_steps", 0)) != 20:
            continue
        if log.get("triggered_at") is None:
            continue
        all_rows = reconstruct_frames(log)
        success_summaries[seed] = summarize_episode(log, all_rows)

    video_meta: dict[str, Any] = {"attempted_seed": args.video_seed}
    video_src = episodes / f"seed_{args.video_seed}" / "videos" / "full_pipeline.mp4"
    video_meta["source_exists"] = video_src.is_file()
    if video_src.is_file():
        vlog = load_pipeline_log(episodes / f"seed_{args.video_seed}")
        vrows = reconstruct_frames(vlog)
        delay = int((vlog.get("fault_config") or {}).get("post_grasp_delay_steps", 40))
        video_out = diag_dir / "delay40_diagnostic.mp4"
        video_meta.update(
            _annotate_video(
                src_mp4=video_src,
                dst_mp4=video_out,
                rows=vrows,
                first_grasp_step=vlog.get("first_grasp_step"),
                delay_steps=delay,
            )
        )
        paths_written.append(str(video_out))
    else:
        video_meta["error"] = "full_pipeline.mp4 missing"

    cross = {
        "delay40_failed_seeds": {str(k): v for k, v in failed_summaries.items()},
        "delay20_success_seeds": {str(k): v for k, v in success_summaries.items()},
        "videos": {
            seed: (episodes / f"seed_{seed}" / "videos" / "full_pipeline.mp4").is_file()
            for seed in list(failed_seeds) + SUCCESS_CONTRAST_SEEDS
        },
    }
    cross["Q1_Q8"] = build_q1_q8(cross)
    cross_path = diag_dir / "cross_seed_summary.json"
    cross_path.write_text(json.dumps(cross, indent=2), encoding="utf-8")
    paths_written.append(str(cross_path))

    report_path = diag_dir / "report.md"
    _write_report_md(
        report_path,
        failed_summaries=failed_summaries,
        success_summaries=success_summaries,
        q1_q8=cross["Q1_Q8"],
        video_meta=video_meta,
    )
    paths_written.append(str(report_path))

    print(json.dumps({"paths": paths_written, "Q1_Q8": cross["Q1_Q8"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
