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
"""Open a recorded LeRobot dataset on disk and assert failure-annotation columns.

This is a physical check: it reads Parquet files, not mocks.

Example::

    uv run python examples/faults/verify_failure_parquet.py \\
        --root outputs/full_pipeline_demo/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_COLUMNS = (
    "is_failure",
    "failure_onset",
    "failure_type",
    "injection_active",
    "phase",
)


def _find_parquet_files(root: Path) -> list[Path]:
    data_dir = root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No data/ directory under {root}")
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return files


def verify_failure_parquet(root: Path, *, require_failure: bool = True) -> dict:
    import pandas as pd

    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    missing_schema = [c for c in REQUIRED_COLUMNS if c not in features]
    if missing_schema:
        raise AssertionError(f"meta/info.json missing features: {missing_schema}")

    frames = []
    for path in _find_parquet_files(root):
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise AssertionError(f"Parquet missing columns: {missing_cols}. Have {list(df.columns)}")

    def _as_bool_series(col: str):
        s = df[col]
        if getattr(s.dtype, "kind", None) in ("b", "i", "u", "f"):
            return s.astype(bool)
        # list/ndarray cells from parquet nested columns
        return s.map(lambda v: bool(np_item(v)))

    def np_item(v):
        if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
            try:
                return v[0]
            except Exception:
                return v
        return v

    is_failure = _as_bool_series("is_failure")
    onset = _as_bool_series("failure_onset")
    injection = _as_bool_series("injection_active")
    ftype = df["failure_type"].map(lambda v: int(np_item(v)))
    phase = df["phase"].map(lambda v: int(np_item(v)))

    summary = {
        "root": str(root),
        "n_frames": int(len(df)),
        "n_is_failure": int(is_failure.sum()),
        "n_failure_onset": int(onset.sum()),
        "n_injection_active": int(injection.sum()),
        "failure_type_max": int(ftype.max()) if len(ftype) else 0,
        "phase_max": int(phase.max()) if len(phase) else 0,
        "columns": list(df.columns),
    }

    if require_failure:
        if summary["n_is_failure"] < 1:
            raise AssertionError(f"Expected some is_failure=True rows, got none. {summary}")
        if summary["n_failure_onset"] < 1:
            raise AssertionError(f"Expected a failure_onset=True row, got none. {summary}")
        if summary["n_injection_active"] < 1:
            raise AssertionError(f"Expected injection_active=True on the glitch frame. {summary}")
        if summary["failure_type_max"] < 1:
            raise AssertionError(f"Expected failure_type > 0 during a faulted episode. {summary}")
        if onset.sum() > is_failure.sum():
            raise AssertionError("More onset frames than failure frames.")

    # Successful-episode sanity when require_failure is False.
    if not require_failure:
        if summary["n_is_failure"] != 0 or summary["n_injection_active"] != 0:
            raise AssertionError(f"Expected an all-nominal episode, got {summary}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument(
        "--allow-nominal",
        action="store_true",
        help="Assert the episode has zero failure flags (success-only recording).",
    )
    args = parser.parse_args(argv)
    summary = verify_failure_parquet(args.root, require_failure=not args.allow_nominal)
    print("SUCCESS: failure annotation parquet check")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
