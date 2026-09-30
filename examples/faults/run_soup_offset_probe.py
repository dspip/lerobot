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
"""Nominal pick-and-place probes at ±3 cm and ±5 cm soup offsets (no injected drop)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    from lerobot.faults.recovery.xy_band_offset_probe import (
        compass_offsets,
        summarize_nominal_offset_probe,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "xy_band_offset_probe",
    )
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seed",
        type=int,
        default=8301,
        help="Fixed seed so only soup XY changes (a known-good nominal seed).",
    )
    args = parser.parse_args(argv)

    if "xy_band_checkpoint1" in str(args.output_dir):
        print("Refusing to write offset probe into the checkpoint dataset.", flush=True)
        return 2

    from lerobot.faults.datagen.smolvla_pipeline import run_pipeline

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for spec in compass_offsets((0.03, 0.05)):
        label = str(spec["label"])
        ep_dir = args.output_dir / "episodes" / label
        print(
            f"[offset-probe] {label} dx={spec['dx']:.4f} dy={spec['dy']:.4f} seed={args.seed}",
            flush=True,
        )
        try:
            summary = run_pipeline(
                ep_dir,
                policy_path=args.policy_path,
                device=args.device,
                seed=int(args.seed),
                wipe_output_dir=True,
                dataset_root=ep_dir / "dataset_unused",
                ds_logger=None,
                raise_on_failure=False,
                copy_demo_gif=False,
                episode_kind="nominal",
                soup_xy_offset=(float(spec["dx"]), float(spec["dy"])),
            )
        except Exception as exc:
            print(f"[offset-probe] {label} crashed: {exc}", flush=True)
            summary = {"success": False, "checks": {"exception": str(exc)}}

        analysis = summarize_nominal_offset_probe(summary)
        row = {
            **spec,
            "seed": int(args.seed),
            "initial_soup_xy": summary.get("initial_soup_xy"),
            "offset_applied": summary.get("soup_xy_offset_applied"),
            "video_mp4": summary.get("video_mp4"),
            **analysis,
        }
        records.append(row)
        print(
            f"[offset-probe] {label} pick={row['pick_success']} "
            f"grasp_step={row['first_grasp_step']} xy_at_grasp={row['xy_at_grasp_m']}",
            flush=True,
        )

    report = {"seed": int(args.seed), "attempts": records}
    out_path = args.output_dir / "offset_probe.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)

    n_ok = sum(1 for r in records if r.get("pick_success"))
    print(f"[offset-probe] pick success {n_ok}/{len(records)}", flush=True)
    return 0 if n_ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
