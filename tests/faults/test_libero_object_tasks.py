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

import re
from pathlib import Path

import pytest

from lerobot_faults.datagen.libero_object_tasks import (
    LIBERO_OBJECT_TASKS,
    official_task_id,
    supported_object_names,
)


def test_libero_object_task_table_matches_benchmark_bddl() -> None:
    from libero.libero.benchmark import get_benchmark

    bench = get_benchmark("libero_object")()
    assert bench.get_num_tasks() == 10
    assert len(LIBERO_OBJECT_TASKS) == 10
    assert [row.task_id for row in LIBERO_OBJECT_TASKS] == list(range(10))
    assert supported_object_names() == tuple(row.instance_name for row in LIBERO_OBJECT_TASKS)

    for row in LIBERO_OBJECT_TASKS:
        assert official_task_id(row.instance_name) == row.task_id
        task = bench.get_task(row.task_id)
        assert task.language == row.language
        bddl_text = Path(bench.get_task_bddl_file_path(row.task_id)).read_text()
        objects_m = re.search(r"\(:objects\s+([\s\S]*?)\n\s*\)", bddl_text)
        assert objects_m is not None
        instances = set(re.findall(r"(\w+_\d+)", objects_m.group(1)))
        assert row.instance_name in instances


def test_official_task_id_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown LIBERO-Object pick target"):
        official_task_id("not_a_real_object_1")
