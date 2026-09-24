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

"""Copy successful eval-recording episodes into a new dataset root.

Example::

    uv run python examples/faults/filter_eval_successes.py \\
        --root outputs/nominal_soup_baseline/recordings/libero_object_0 \\
        --output outputs/nominal_soup_kept \\
        --max-keep 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.success_filter import filter_successful_episodes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Eval recording dataset root (contains data/ and meta/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New dataset root for kept successes. Must not already exist.",
    )
    parser.add_argument(
        "--max-keep",
        type=int,
        default=100,
        help="Cap on successful episodes to keep (default 100). Use 0 for all successes.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="eval_recording_success",
        help="Repo id written into the filtered dataset metadata.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    max_keep = None if int(args.max_keep) == 0 else int(args.max_keep)
    result = filter_successful_episodes(
        args.root,
        args.output,
        max_keep=max_keep,
        repo_id=args.repo_id,
    )
    print(json.dumps(
        {
            "output_root": str(result.output_root),
            "source_episodes": result.source_episodes,
            "successful_episodes": result.successful_episodes,
            "kept_episodes": result.kept_episodes,
            "kept_indices": list(result.kept_indices),
            "discarded_failure_indices": list(result.discarded_failure_indices),
            "discarded_overflow_indices": list(result.discarded_overflow_indices),
        },
        indent=2,
    ))
    if result.kept_episodes < 100 and max_keep == 100:
        print(
            f"Warning: only {result.kept_episodes} successes kept "
            f"(need more eval attempts or lower --max-keep).",
            flush=True,
        )


if __name__ == "__main__":
    main()
