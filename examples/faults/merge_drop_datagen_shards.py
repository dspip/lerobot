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

"""Merge sharded drop datagen outputs into one training dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fault_system.datagen.merge_shards import MergeShardsError, merge_drop_datagen_shards


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards",
        type=Path,
        nargs="+",
        required=True,
        help="Shard output directories (each with run_manifest.json and dataset/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Empty directory for merged run artifacts.",
    )
    parser.add_argument(
        "--test-percent",
        type=int,
        default=15,
        help="Fraction of logical episodes (mod 100) assigned to test_in_distribution.",
    )
    parser.add_argument(
        "--control-hz",
        type=int,
        default=20,
        help="Control rate for dataset card (from recipe control_hz).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        out = merge_drop_datagen_shards(
            list(args.shards),
            args.output,
            test_percent=int(args.test_percent),
            control_hz=int(args.control_hz),
        )
    except (MergeShardsError, ValueError) as exc:
        print(f"Merge error: {exc}", file=sys.stderr)
        return 2
    print(f"Merged dataset written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
