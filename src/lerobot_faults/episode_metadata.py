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

"""Attach extra per-episode fields without patching ``LeRobotDataset.save_episode``."""

from __future__ import annotations

from typing import Any


def save_episode_adding_metadata(
    dataset: Any,
    episode_data: dict | None = None,
    *,
    parallel_encoding: bool = True,
    episode_metadata: dict | None = None,
) -> None:
    """Save an episode, merging ``episode_metadata`` into the metadata row.

    Upstream flushes the episode-metadata buffer inside ``save_episode``. The
    extra fields have to be on that row before the flush, so this wraps
    ``dataset.meta.save_episode`` for the duration of the call. Once a dataset
    has used an extra key, every later episode must pass that same key or the
    parquet writer rejects the row.
    """
    if not episode_metadata:
        dataset.save_episode(episode_data, parallel_encoding=parallel_encoding)
        return

    meta = dataset.meta
    original = meta.save_episode

    def _save_with_extra(
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        row: dict,
    ) -> None:
        merged = dict(row)
        merged.update(episode_metadata)
        original(episode_index, episode_length, episode_tasks, episode_stats, merged)

    meta.save_episode = _save_with_extra
    try:
        dataset.save_episode(episode_data, parallel_encoding=parallel_encoding)
    finally:
        meta.save_episode = original
