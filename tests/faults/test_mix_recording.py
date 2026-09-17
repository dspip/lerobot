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

"""Mix recording helpers (no GPU/sim)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lerobot.faults.recovery.mix_recording import (
    MixCounters,
    MixWorkItem,
    commit_or_discard,
    compute_mix_max_attempts,
    default_seed_sequence,
    delay_grid_drop_target,
    episode_dir_for_seed,
    mix_output_layout,
    next_open_delay_grid_bin,
    parse_drops_per_delay,
    parse_int_list,
    parse_seed_list,
    plan_delay_grid_work_items,
    resolve_mix_drop_target,
    should_keep_episode,
)


def test_parse_seed_list() -> None:
    assert parse_seed_list("1000,1001,1010") == [1000, 1001, 1010]
    assert parse_seed_list(" 42 , 43 ") == [42, 43]


def test_parse_seed_list_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_seed_list(",")


def test_default_seed_sequence() -> None:
    assert default_seed_sequence(2000, 4) == [2000, 2001, 2002, 2003]


def test_mix_output_layout() -> None:
    root = Path("/tmp/mix_out")
    layout = mix_output_layout(root)
    assert layout["dataset"] == root / "dataset"
    assert layout["episodes"] == root / "episodes"
    assert layout["mix_log"] == root / "mix_log.json"


def test_episode_dir_for_seed() -> None:
    assert episode_dir_for_seed(Path("/out"), 2005) == Path("/out/episodes/seed_2005")


def test_should_keep_drop_requires_success() -> None:
    assert should_keep_episode({"success": True}, "drop")
    assert not should_keep_episode({"success": False}, "drop")


def test_should_keep_nominal_requires_success() -> None:
    assert should_keep_episode({"success": True}, "nominal")
    assert not should_keep_episode({"success": False}, "nominal")


def test_should_keep_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown"):
        should_keep_episode({"success": True}, "weird")


def test_commit_or_discard_end_episode() -> None:
    logger = MagicMock()
    commit_or_discard(logger, keep=True)
    logger.end_episode.assert_called_once()
    logger.clear_open_episode.assert_not_called()


def test_commit_or_discard_clear() -> None:
    logger = MagicMock()
    commit_or_discard(logger, keep=False)
    logger.clear_open_episode.assert_called_once()
    logger.end_episode.assert_not_called()


def test_mix_counters_interleave() -> None:
    c = MixCounters(n_drop_target=2, n_nominal_target=2)
    assert c.next_kind_interleave() == "drop"
    c.record_attempt("drop", kept=True)
    assert c.next_kind_interleave() == "nominal"
    c.record_attempt("nominal", kept=False)
    assert c.next_kind_interleave() == "drop"
    c.record_attempt("drop", kept=True)
    assert c.next_kind_interleave() == "nominal"
    c.record_attempt("nominal", kept=True)
    assert c.next_kind_interleave() == "nominal"
    c.record_attempt("nominal", kept=True)
    assert c.next_kind_interleave() is None
    assert c.quotas_met


def test_mix_counters_zero_nominal_quota_is_drop_only() -> None:
    c = MixCounters(n_drop_target=1, n_nominal_target=0)
    assert c.next_kind_interleave() == "drop"
    c.record_attempt("drop", kept=True)
    assert c.next_kind_interleave() is None
    assert c.quotas_met


def test_parse_int_list() -> None:
    assert parse_int_list("20,30,40") == [20, 30, 40]


def test_delay_grid_drop_target() -> None:
    assert delay_grid_drop_target([20, 30, 40], 3) == 9
    assert delay_grid_drop_target([0, 20, 40, 60], [4, 4, 4, 3]) == 15


def test_parse_drops_per_delay() -> None:
    grid = [0, 20, 40, 60]
    assert parse_drops_per_delay(grid, "3") == [3, 3, 3, 3]
    assert parse_drops_per_delay(grid, "4,4,4,3") == [4, 4, 4, 3]
    assert parse_drops_per_delay(grid, 2) == [2, 2, 2, 2]
    with pytest.raises(ValueError, match="length"):
        parse_drops_per_delay(grid, "4,4")


def test_next_open_delay_grid_bin_per_bin_quota() -> None:
    grid = [0, 20, 40, 60]
    quotas = [4, 4, 4, 3]
    kept = {0: 4, 20: 2, 40: 0, 60: 0}
    assert next_open_delay_grid_bin(grid, drops_per_bin=quotas, kept_per_delay=kept) == 20
    kept[60] = 3
    kept[20] = 4
    kept[40] = 4
    assert next_open_delay_grid_bin(grid, drops_per_bin=quotas, kept_per_delay=kept) is None


def test_plan_delay_grid_work_items() -> None:
    items = plan_delay_grid_work_items([20, 30], drops_per_delay=2, n_nominal=1)
    assert items == [
        MixWorkItem(kind="drop", post_grasp_delay_steps=20),
        MixWorkItem(kind="drop", post_grasp_delay_steps=20),
        MixWorkItem(kind="drop", post_grasp_delay_steps=30),
        MixWorkItem(kind="drop", post_grasp_delay_steps=30),
        MixWorkItem(kind="nominal", post_grasp_delay_steps=None),
    ]


def test_resolve_mix_drop_target_from_grid() -> None:
    assert (
        resolve_mix_drop_target(
            delay_grid=[20, 30],
            drops_per_delay=3,
            n_drop=None,
            n_drop_explicit=False,
        )
        == 6
    )
    assert (
        resolve_mix_drop_target(
            delay_grid=[0, 20, 40, 60],
            drops_per_delay="4,4,4,3",
            n_drop=None,
            n_drop_explicit=False,
        )
        == 15
    )


def test_resolve_mix_drop_target_grid_conflicts_with_n_drop() -> None:
    with pytest.raises(ValueError, match="n-drop"):
        resolve_mix_drop_target(
            delay_grid=[20],
            drops_per_delay=1,
            n_drop=5,
            n_drop_explicit=True,
        )


def test_resolve_mix_drop_target_zero_n_drop_without_grid() -> None:
    assert (
        resolve_mix_drop_target(
            delay_grid=None,
            drops_per_delay=3,
            n_drop=0,
            n_drop_explicit=True,
        )
        == 0
    )


def test_compute_mix_max_attempts_bumps_for_grid() -> None:
    assert compute_mix_max_attempts(
        n_drop=15,
        n_nominal=8,
        delay_grid=[20, 30],
        max_attempts_arg=40,
    ) == max(40, 23 * 4)