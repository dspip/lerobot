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
"""Audit checkpoint 1 XY-band dataset (checkpoint_log, parquet, pipeline logs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    from lerobot.faults.recovery.xy_band_checkpoint_audit import (
        audit_xy_band_checkpoint,
        format_checkpoint_table,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "xy_band_checkpoint1",
    )
    args = parser.parse_args(argv)

    result = audit_xy_band_checkpoint(args.output_dir)
    print(format_checkpoint_table(result.rows))
    print()
    print("## Discarded attempts")
    for att in result.discarded:
        print(
            f"seed={att.get('seed')} kind={att.get('kind')} band={att.get('band')} "
            f"reason={att.get('reason')}"
        )
    print()
    print("## Action stats")
    print(json.dumps(result.action_stats, indent=2))
    print()
    for issue in result.issues:
        print(f"[{issue.level.upper()}] {issue.message}")

    report_path = args.output_dir / "audit_report.json"
    report_path.write_text(
        json.dumps(
            {
                "rows": result.rows,
                "issues": [{"level": i.level, "message": i.message} for i in result.issues],
                "action_stats": result.action_stats,
                "discarded": result.discarded,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {report_path}")

    code = result.exit_code()
    print("AUDIT PASS" if code == 0 else "AUDIT FAIL")
    return code


if __name__ == "__main__":
    sys.exit(main())
