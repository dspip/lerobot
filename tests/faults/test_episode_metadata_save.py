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

from fault_system.episode_metadata import save_episode_adding_metadata


class _Meta:
    def __init__(self, *, flush_every: int) -> None:
        self.buffer: list[dict] = []
        self.flushed: list[dict] = []
        self.flush_every = flush_every

    def save_episode(self, episode_index, episode_length, episode_tasks, episode_stats, row) -> None:
        del episode_length, episode_tasks, episode_stats
        stored = dict(row)
        stored["episode_index"] = episode_index
        self.buffer.append(stored)
        if len(self.buffer) >= self.flush_every:
            self.flushed.extend(self.buffer)
            self.buffer.clear()


class _Dataset:
    def __init__(self, meta: _Meta) -> None:
        self.meta = meta

    def save_episode(self, episode_data=None, parallel_encoding: bool = True) -> None:
        del episode_data, parallel_encoding
        self.meta.save_episode(0, 1, ["task"], {}, {"from_index": 0})


def test_extra_fields_are_present_when_the_buffer_flushes_inside_save() -> None:
    meta = _Meta(flush_every=1)
    dataset = _Dataset(meta)
    save_episode_adding_metadata(dataset, episode_metadata={"success": True})
    assert meta.flushed == [{"from_index": 0, "success": True, "episode_index": 0}]
    assert meta.buffer == []
    assert meta.save_episode.__func__ is _Meta.save_episode  # type: ignore[attr-defined]


def test_save_without_extra_metadata_does_not_add_fields() -> None:
    meta = _Meta(flush_every=10)
    dataset = _Dataset(meta)
    save_episode_adding_metadata(dataset, episode_metadata=None)
    assert meta.buffer == [{"from_index": 0, "episode_index": 0}]
