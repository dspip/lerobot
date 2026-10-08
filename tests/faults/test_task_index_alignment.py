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

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch

pytest.importorskip("datasets")

from fault_system.datagen.libero_object_tasks import LIBERO_OBJECT_TASKS
from fault_system.datagen.task_index import (
    align_official_task_indices,
    ensure_official_libero_task_indices,
    task_table_is_official,
)
from lerobot.datasets.io_utils import load_info, load_tasks
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
}
_BUTTER = next(row for row in LIBERO_OBJECT_TASKS if row.instance_name == "butter_1")
_TOMATO = next(row for row in LIBERO_OBJECT_TASKS if row.instance_name == "tomato_sauce_1")


def _create(root: Path) -> LeRobotDataset:
    return LeRobotDataset.create(
        "local/task_index_alignment",
        fps=10,
        features=_FEATURES,
        root=root,
        use_videos=False,
    )


def _add_episode(dataset: LeRobotDataset, task: str) -> None:
    dataset.add_frame(
        {
            "observation.state": torch.zeros(8),
            "action": torch.zeros(7),
            "task": task,
        }
    )
    dataset.save_episode()


def _frame_task_indices(root: Path) -> list[int]:
    values: list[int] = []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["task_index"])
        values.extend(int(value) for value in table.column("task_index").to_pylist())
    return values


def test_recording_assigns_official_scene_id_before_the_first_episode(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    dataset = _create(root)
    ensure_official_libero_task_indices(dataset)
    _add_episode(dataset, _BUTTER.language)
    dataset.finalize()

    assert _frame_task_indices(root) == [_BUTTER.task_id]
    tasks = load_tasks(root)
    assert task_table_is_official(tasks)
    assert int(tasks.loc[_TOMATO.language, "task_index"]) == _TOMATO.task_id
    assert load_info(root).total_tasks == len(LIBERO_OBJECT_TASKS)


def test_align_rewrites_appearance_order_to_official_scene_id(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    dataset = _create(root)
    _add_episode(dataset, _BUTTER.language)
    dataset.finalize()
    assert _frame_task_indices(root) == [0]

    assert align_official_task_indices(root) is True
    assert align_official_task_indices(root) is False

    assert _frame_task_indices(root) == [_BUTTER.task_id]
    tasks = load_tasks(root)
    assert task_table_is_official(tasks)
    assert int(tasks.loc[_BUTTER.language, "task_index"]) == 6
    assert int(tasks.loc[_TOMATO.language, "task_index"]) == 5
    episodes = pd.read_parquet(next((root / "meta" / "episodes").rglob("*.parquet")))
    assert episodes["stats/task_index/min"].iloc[0] == [6.0]
    assert episodes["stats/task_index/max"].iloc[0] == [6.0]
    assert load_info(root).total_tasks == 10


def test_align_leaves_non_libero_task_strings_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    dataset = _create(root)
    _add_episode(dataset, "pick")
    dataset.finalize()

    assert align_official_task_indices(root) is False
    assert _frame_task_indices(root) == [0]
    tasks = load_tasks(root)
    assert list(tasks.index) == ["pick"]
