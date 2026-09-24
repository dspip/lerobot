#!/usr/bin/env python3
"""Live board of per-logical-episode variant outcomes from run_manifest.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

VARIANT_ORDER = (
    ("simple_ik", "immediate_ik"),
    ("simple_ik", "continue_then_ik"),
    ("smolvla", "immediate_ik"),
    ("smolvla", "continue_then_ik"),
    ("smolvla", "reset_then_ik"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/can_5_datasets_full_no_fails"),
        help="Datagen recording.output_dir that contains run_manifest.json",
    )
    parser.add_argument("--interval", type=float, default=1.0)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    episodes = raw.get("episodes")
    if not isinstance(episodes, list):
        return []
    return [row for row in episodes if isinstance(row, dict)]


def _status(row: dict) -> str:
    if bool(row.get("success")):
        return "success"
    reason = row.get("reject_reason") or row.get("outcome") or "failed"
    return f"failed ({reason})"


def _render(rows: list[dict], *, waiting: bool) -> str:
    by_logical: dict[int, dict[tuple[str, str], dict]] = defaultdict(dict)
    for row in rows:
        try:
            logical = int(row["logical_episode_index"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(row.get("controller", "")), str(row.get("post_drop_mode", "")))
        by_logical[logical][key] = row

    lines = [
        "Drop datagen results (Ctrl+C to stop watching)",
        "",
    ]
    if waiting and not by_logical:
        lines.append("Waiting for run_manifest.json …")
        return "\n".join(lines)

    for logical in sorted(by_logical):
        lines.append("-------------------")
        lines.append(f"episode #{logical + 1}:")
        variants = by_logical[logical]
        for index, key in enumerate(VARIANT_ORDER, start=1):
            row = variants.get(key)
            if row is None:
                lines.append(f"{index}. pending")
            else:
                lines.append(f"{index}. {_status(row)}")
        extra = [k for k in variants if k not in VARIANT_ORDER]
        for key in extra:
            ctrl, mode = key
            lines.append(f"   extra {ctrl}/{mode}: {_status(variants[key])}")
        lines.append("-------------------")
    if waiting:
        lines.append("")
        lines.append("(run still writing; this board refreshes)")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    manifest = args.output_dir / "run_manifest.json"
    last = None
    try:
        while True:
            rows = _load_rows(manifest)
            text = _render(rows, waiting=True)
            if text != last:
                os.system("clear")
                print(text, flush=True)
                last = text
            time.sleep(max(0.2, float(args.interval)))
    except KeyboardInterrupt:
        rows = _load_rows(manifest)
        os.system("clear")
        print(_render(rows, waiting=False), flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
