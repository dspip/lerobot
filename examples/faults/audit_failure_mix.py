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
"""Audit a failure-mix output directory (mix_log, pipeline logs, parquet, videos).

Example::

    uv run python examples/faults/audit_failure_mix.py \\
        --output-dir outputs/failure_mix_v1 \\
        --expect-delay-grid 20,30,40,50,60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_verify_parquet():
    import importlib.util

    path = REPO_ROOT / "examples" / "faults" / "verify_failure_parquet.py"
    spec = importlib.util.spec_from_file_location("verify_failure_parquet", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load verify module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_failure_parquet


def main(argv: list[str] | None = None) -> int:
    from lerobot.faults.recovery.mix_audit import (
        audit_failure_mix,
        audit_report_dict,
        format_drop_table,
        parse_expect_delay_grid,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--min-xy-m",
        type=float,
        default=0.30,
        help="Fail if drop_basket_xy_dist is below this (recipe default 0.30)",
    )
    parser.add_argument(
        "--min-pairwise-xy-m",
        type=float,
        default=0.05,
        help="Fail if min pairwise pre-drop XY distance is below this",
    )
    parser.add_argument(
        "--min-unique-triggered-at",
        type=int,
        default=None,
        help="Require at least this many distinct triggered_at values among kept drops",
    )
    parser.add_argument(
        "--expect-delay-grid",
        type=str,
        default=None,
        help="Fail unless kept drops cover every delay in this comma-separated grid",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip verify_failure_parquet on the dataset root",
    )
    args = parser.parse_args(argv)

    verify_fn = None if args.skip_parquet else _load_verify_parquet()
    result = audit_failure_mix(
        args.output_dir,
        min_xy_m=float(args.min_xy_m),
        min_pairwise_xy_m=float(args.min_pairwise_xy_m),
        min_unique_triggered_at=args.min_unique_triggered_at,
        expect_delay_grid=parse_expect_delay_grid(args.expect_delay_grid),
        verify_parquet_fn=verify_fn,
    )

    print(format_drop_table(result.rows))
    print()
    for issue in result.issues:
        tag = issue.level.upper()
        print(f"[{tag}] {issue.message}")
    print()
    print(json.dumps(result.stats, indent=2))

    report_path = Path(args.output_dir) / "audit_report.json"
    report_path.write_text(json.dumps(audit_report_dict(result), indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")

    code = result.exit_code()
    if code == 0:
        print("AUDIT PASS")
    elif code == 2:
        print("AUDIT WARN")
    else:
        print("AUDIT FAIL")
    return code


if __name__ == "__main__":
    sys.exit(main())
