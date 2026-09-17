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
"""Audit kept episodes from run_xy_band_pilot.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-3


def _parquet_for_seed(dataset_root: Path, seed: int) -> Path | None:
    meta = dataset_root / "meta" / "episodes.jsonl"
    if not meta.exists():
        return None
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("seed") == seed or row.get("episode_index") is not None:
            # Fall back: any parquet under data/
            break
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        return None
    parquets = sorted(data_dir.rglob("*.parquet"))
    return parquets[0] if parquets else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "xy_band_pilot",
    )
    parser.add_argument("--epsilon", type=float, default=EPS)
    args = parser.parse_args(argv)

    log_path = args.output_dir / "pilot_log.json"
    if not log_path.exists():
        print(f"Missing {log_path}", file=sys.stderr)
        return 1

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    kept = payload.get("kept_episodes") or []
    dataset_root = Path(payload.get("dataset_dir") or args.output_dir / "dataset")

    issues: list[str] = []
    rows: list[tuple[str, ...]] = []

    for ep in kept:
        seed = ep.get("seed")
        band = ep.get("band", "?")
        lo = float(ep["band_lo"])
        hi = float(ep["band_hi"])
        dist = ep.get("drop_basket_xy_dist")
        reason = ep.get("drop_trigger_reason")
        seat = ep.get("seat_assisted")
        ok = ep.get("success")
        video = ep.get("video_mp4")
        triggered = ep.get("triggered_at")

        row_issues: list[str] = []
        if triggered is None:
            row_issues.append("not triggered")
        if reason != "xy_band":
            row_issues.append(f"reason={reason!r}")
        if dist is None:
            row_issues.append("missing drop_basket_xy_dist")
        elif not (lo - args.epsilon <= float(dist) <= hi + args.epsilon):
            row_issues.append(f"dist {dist} outside [{lo},{hi}]")
        if seat:
            row_issues.append("seat_assisted")
        if not ok:
            row_issues.append("success false")
        if video and not Path(video).is_file():
            row_issues.append("missing mp4")
        pq = _parquet_for_seed(dataset_root, int(seed)) if seed is not None else None
        if dataset_root.exists() and (pq is None or not pq.is_file()):
            row_issues.append("missing parquet")

        status = "PASS" if not row_issues else "FAIL"
        if row_issues:
            issues.append(f"seed={seed}: " + "; ".join(row_issues))
        rows.append(
            (
                status,
                str(seed),
                band,
                f"[{lo:.2f},{hi:.2f}]",
                f"{dist:.4f}" if dist is not None else "—",
                reason or "—",
                str(seat),
                str(ok),
                str(video or "—"),
            )
        )

    headers = (
        "status",
        "seed",
        "band",
        "requested",
        "drop_xy",
        "reason",
        "seat",
        "success",
        "video_mp4",
    )
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")

    if issues:
        print("\nIssues:", file=sys.stderr)
        for msg in issues:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    print("\nAUDIT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
