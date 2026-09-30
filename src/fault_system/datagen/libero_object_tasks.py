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

"""Official LIBERO-Object task ids and pick-target instance names for datagen."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "LIBERO_OBJECT_TASKS",
    "LiberoObjectTask",
    "official_task_id",
    "supported_object_names",
]


@dataclass(frozen=True, slots=True)
class LiberoObjectTask:
    """One LIBERO-Object benchmark task (scene + pick target)."""

    task_id: int
    instance_name: str
    language: str


# Task order and instance ids verified from libero_object BDDL via get_task_bddl_file_path(i).
LIBERO_OBJECT_TASKS: tuple[LiberoObjectTask, ...] = (
    LiberoObjectTask(
        0,
        "alphabet_soup_1",
        "pick up the alphabet soup and place it in the basket",
    ),
    LiberoObjectTask(
        1,
        "cream_cheese_1",
        "pick up the cream cheese and place it in the basket",
    ),
    LiberoObjectTask(
        2,
        "salad_dressing_1",
        "pick up the salad dressing and place it in the basket",
    ),
    LiberoObjectTask(
        3,
        "bbq_sauce_1",
        "pick up the bbq sauce and place it in the basket",
    ),
    LiberoObjectTask(
        4,
        "ketchup_1",
        "pick up the ketchup and place it in the basket",
    ),
    LiberoObjectTask(
        5,
        "tomato_sauce_1",
        "pick up the tomato sauce and place it in the basket",
    ),
    LiberoObjectTask(
        6,
        "butter_1",
        "pick up the butter and place it in the basket",
    ),
    LiberoObjectTask(
        7,
        "milk_1",
        "pick up the milk and place it in the basket",
    ),
    LiberoObjectTask(
        8,
        "chocolate_pudding_1",
        "pick up the chocolate pudding and place it in the basket",
    ),
    LiberoObjectTask(
        9,
        "orange_juice_1",
        "pick up the orange juice and place it in the basket",
    ),
)

_INSTANCE_TO_TASK_ID: dict[str, int] = {row.instance_name: row.task_id for row in LIBERO_OBJECT_TASKS}


def supported_object_names() -> tuple[str, ...]:
    """Official pick-target MuJoCo instance names for LIBERO-Object."""
    return tuple(row.instance_name for row in LIBERO_OBJECT_TASKS)


def official_task_id(object_name: str) -> int:
    """Map a pick-target instance name to its LIBERO-Object task id."""
    try:
        return _INSTANCE_TO_TASK_ID[object_name]
    except KeyError as exc:
        supported = ", ".join(supported_object_names())
        raise ValueError(
            f"unknown LIBERO-Object pick target {object_name!r}; supported: {supported}"
        ) from exc
