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

"""Keep only successful eval-recording episodes for a nominal baseline."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import delete_episodes, split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import SUCCESS

_EPISODE_INDEX = "episode_index"


def _as_bool(value: Any) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).reshape(-1)
        return bool(arr[0]) if arr.size else False
    if value is None:
        return False
    return bool(value)


def data_parquet_paths(root: Path) -> list[Path]:
    """Return sorted ``data/**/*.parquet`` files under a dataset root."""
    data_dir = Path(root) / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"No data/ directory under {root}")
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    return files


def list_successful_episode_indices(root: str | Path) -> list[int]:
    """Episode indices where any frame has ``next.success`` true (eval convention)."""
    root = Path(root)
    frames = [pd.read_parquet(path, columns=[_EPISODE_INDEX, SUCCESS]) for path in data_parquet_paths(root)]
    df = pd.concat(frames, ignore_index=True)
    if _EPISODE_INDEX not in df.columns or SUCCESS not in df.columns:
        raise KeyError(f"Parquet must contain {_EPISODE_INDEX!r} and {SUCCESS!r}")
    succeeded: set[int] = set()
    for episode_index, group in df.groupby(_EPISODE_INDEX, sort=True):
        if any(_as_bool(value) for value in group[SUCCESS]):
            succeeded.add(int(episode_index))
    return sorted(succeeded)


def finish_eval_recorded_episode(
    dataset: LeRobotDataset,
    *,
    succeeded: bool,
    success_only: bool,
) -> None:
    """Commit a recorded episode, or drop the buffer when ``success_only`` and it failed."""
    if (not success_only) or succeeded:
        dataset.save_episode()
        return
    dataset.clear_episode_buffer(delete_images=True)


@dataclass(frozen=True)
class SuccessFilterResult:
    """Summary of a success-only copy of an eval recording."""

    output_root: Path
    source_episodes: int
    successful_episodes: int
    kept_episodes: int
    kept_indices: tuple[int, ...]
    discarded_failure_indices: tuple[int, ...]
    discarded_overflow_indices: tuple[int, ...]


def filter_successful_episodes(
    source_root: str | Path,
    output_root: str | Path,
    *,
    max_keep: int | None = None,
    repo_id: str | None = None,
) -> SuccessFilterResult:
    """Write a new dataset with successful episodes only, optionally capped at ``max_keep``."""
    source_root = Path(source_root)
    output_root = Path(output_root)
    if max_keep is not None and int(max_keep) <= 0:
        raise ValueError(f"max_keep must be a positive int (got {max_keep})")

    successful = list_successful_episode_indices(source_root)
    if not successful:
        raise ValueError(f"No successful episodes in {source_root} (no frame with {SUCCESS}=true)")

    dataset = LeRobotDataset(repo_id=repo_id or "eval_recording", root=source_root)
    source_n = int(dataset.meta.total_episodes)
    all_indices = list(range(source_n))
    failures = [idx for idx in all_indices if idx not in set(successful)]
    keep = successful if max_keep is None else successful[: int(max_keep)]
    overflow = successful[len(keep) :]
    drop = failures + overflow

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset root {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if drop:
        kept_ds = delete_episodes(
            dataset, drop, output_dir=output_root, repo_id=repo_id or "eval_recording_success"
        )
        output = Path(kept_ds.root)
    else:
        staging = output_root.parent / f".{output_root.name}.split_tmp"
        if staging.exists():
            shutil.rmtree(staging)
        splits = split_dataset(dataset, {"kept": keep}, output_dir=staging)
        shutil.move(str(splits["kept"].root), str(output_root))
        shutil.rmtree(staging, ignore_errors=True)
        output = output_root

    return SuccessFilterResult(
        output_root=output,
        source_episodes=source_n,
        successful_episodes=len(successful),
        kept_episodes=len(keep),
        kept_indices=tuple(keep),
        discarded_failure_indices=tuple(failures),
        discarded_overflow_indices=tuple(overflow),
    )
