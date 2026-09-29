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

"""Audit a finalized drop datagen run (single shard or merged)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lerobot.faults.datagen.audit import AuditError, audit_drop_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Recording output directory (contains run_manifest.json and dataset/).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write JSON audit report to this path (default: <run_dir>/audit_report.json).",
    )
    parser.add_argument(
        "--heldout-objects",
        type=str,
        default="",
        help="Comma-separated object instance names that must not appear as episode targets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    heldout = frozenset(x.strip() for x in args.heldout_objects.split(",") if x.strip())
    try:
        report = audit_drop_run(args.run_dir, heldout_objects=heldout)
    except AuditError as exc:
        print(f"Audit error: {exc}", file=sys.stderr)
        return 2
    report_path = args.report or (Path(args.run_dir) / "audit_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report.ok:
        print(f"Audit OK — report {report_path}", flush=True)
        return 0
    print(f"Audit FAILED — {report.error_counts}", file=sys.stderr)
    for note in report.notes:
        print(f"  {note}", file=sys.stderr)
    print(f"Full report: {report_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
