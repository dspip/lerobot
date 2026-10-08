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

"""Keep LeRobot ``task_index`` equal to the official LIBERO-Object scene id.

LeRobot assigns ``task_index`` in the order language strings first appear.
That number is not the simulator scene id: butter is scene 6, but it is the
6th recorded sentence (index 5) when tomato sauce is held out. This module
writes the full official task table before any episode, and rewrites an
existing dataset so every stored ``task_index`` is the simulator scene id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fault_system.datagen.libero_object_tasks import LIBERO_OBJECT_TASKS

__all__ = [
    "align_official_task_indices",
    "ensure_official_libero_task_indices",
    "official_tasks_dataframe",
    "task_table_is_official",
]

_INDEX_STAT_KEYS = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")
_LANGUAGE_TO_TASK_ID = {row.language: row.task_id for row in LIBERO_OBJECT_TASKS}


def official_tasks_dataframe() -> pd.DataFrame:
    """All ten LIBERO-Object instructions, indexed by language, id in ``task_index``."""
    languages = [row.language for row in LIBERO_OBJECT_TASKS]
    task_ids = [row.task_id for row in LIBERO_OBJECT_TASKS]
    return pd.DataFrame({"task_index": task_ids}, index=pd.Index(languages, name="task"))


def task_table_is_official(tasks: pd.DataFrame | None) -> bool:
    """Return whether ``tasks`` is exactly the ten official scene ids."""
    if tasks is None or len(tasks) != len(LIBERO_OBJECT_TASKS):
        return False
    if "task_index" not in tasks.columns:
        return False
    expected = official_tasks_dataframe()
    if list(tasks.index) != list(expected.index):
        return False
    actual = [int(value) for value in tasks["task_index"].tolist()]
    return actual == [int(value) for value in expected["task_index"].tolist()]


def _tasks_are_official_languages(tasks: pd.DataFrame) -> bool:
    return all(str(language) in _LANGUAGE_TO_TASK_ID for language in tasks.index)


def _old_index_to_official(tasks: pd.DataFrame) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for language, row in tasks.iterrows():
        old = int(row["task_index"])
        new = _LANGUAGE_TO_TASK_ID[str(language)]
        if old in mapping and mapping[old] != new:
            raise ValueError(f"task_index {old} maps to both {mapping[old]} and {new}")
        mapping[old] = new
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"official task ids collide under the current table: {mapping}")
    return mapping


def _write_official_table(meta: Any) -> None:
    from lerobot.datasets.io_utils import write_info, write_tasks

    frame = official_tasks_dataframe()
    write_tasks(frame, meta.root)
    meta.tasks = frame
    meta.info.total_tasks = len(frame)
    write_info(meta.info, meta.root)


def _reload_metadata(meta: Any) -> None:
    from lerobot.datasets.io_utils import load_info, load_stats, load_tasks

    root = Path(meta.root)
    meta.info = load_info(root)
    meta.tasks = load_tasks(root)
    meta.stats = load_stats(root)


def ensure_official_libero_task_indices(dataset: Any) -> None:
    """Seed a new dataset, or rewrite a resumed one, so ``task_index`` is the scene id.

    Datasets whose task strings are not LIBERO-Object instructions are left unchanged.
    """
    meta = dataset.meta
    root = Path(meta.root)
    tasks_path = root / "meta" / "tasks.parquet"
    if meta.tasks is None or not tasks_path.is_file():
        _write_official_table(meta)
        return
    if task_table_is_official(meta.tasks):
        return
    if not _tasks_are_official_languages(meta.tasks):
        return
    align_official_task_indices(root)
    _reload_metadata(meta)


def align_official_task_indices(dataset_root: Path | str) -> bool:
    """Rewrite ``task_index`` in one dataset to the official LIBERO-Object scene id.

    Returns ``False`` when the table is already official, or when the dataset
    uses task strings that are not LIBERO-Object instructions.
    """
    from lerobot.datasets.compute_stats import get_feature_stats
    from lerobot.datasets.io_utils import (
        load_info,
        load_stats,
        load_tasks,
        write_info,
        write_stats,
        write_tasks,
    )

    root = Path(dataset_root)
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return False
    tasks = load_tasks(root)
    if task_table_is_official(tasks):
        return False
    if not _tasks_are_official_languages(tasks):
        return False

    mapping = _old_index_to_official(tasks)
    _rewrite_data_task_indices(root, mapping)
    _rewrite_episode_task_index_stats(root, mapping)
    write_tasks(official_tasks_dataframe(), root)

    info = load_info(root)
    info.total_tasks = len(LIBERO_OBJECT_TASKS)
    write_info(info, root)

    stats = load_stats(root)
    if stats is not None and "task_index" in stats:
        values = _read_task_index_values(root)
        if values.size:
            stats["task_index"] = get_feature_stats(values.reshape(-1, 1), axis=0, keepdims=False)
            write_stats(stats, root)
    return True


def _rewrite_data_task_indices(root: Path, mapping: dict[int, int]) -> None:
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        if "task_index" not in table.column_names:
            continue
        mapped = _map_arrow_column(table.column("task_index"), mapping, integer=True)
        table = _replace_column(table, "task_index", mapped)
        pq.write_table(table, path, compression="snappy")


def _rewrite_episode_task_index_stats(root: Path, mapping: dict[int, int]) -> None:
    episodes = root / "meta" / "episodes"
    if not episodes.is_dir():
        return
    for path in sorted(episodes.rglob("*.parquet")):
        table = pq.read_table(path)
        changed = False
        for key in _INDEX_STAT_KEYS:
            name = f"stats/task_index/{key}"
            if name not in table.column_names:
                continue
            mapped = _map_arrow_column(table.column(name), mapping, integer=False)
            table = _replace_column(table, name, mapped)
            changed = True
        if changed:
            pq.write_table(table, path, compression="snappy")


def _replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    return table.set_column(index, name, values)


def _map_arrow_column(column: pa.ChunkedArray, mapping: dict[int, int], *, integer: bool) -> pa.Array:
    values = column.to_pylist()
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        mapped_lists = [_map_sequence(row, mapping, integer=integer) for row in values]
        return pa.array(mapped_lists, type=column.type)
    mapped_scalars = [_map_one(value, mapping, integer=integer) for value in values]
    return pa.array(mapped_scalars, type=column.type)


def _map_sequence(row: list[Any] | None, mapping: dict[int, int], *, integer: bool) -> list[Any] | None:
    if row is None:
        return None
    return [_map_one(value, mapping, integer=integer) for value in row]


def _map_one(value: Any, mapping: dict[int, int], *, integer: bool) -> Any:
    if value is None:
        return None
    old = int(round(float(value)))
    if old not in mapping:
        raise ValueError(f"task_index {old} is not in the dataset task table")
    new = mapping[old]
    if integer:
        return int(new)
    return float(new)


def _read_task_index_values(root: Path) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["task_index"])
        values = table.column("task_index").to_numpy(zero_copy_only=False)
        chunks.append(np.asarray(values, dtype=np.float64).reshape(-1))
    if not chunks:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(chunks)
