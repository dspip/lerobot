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

"""Helpers for multi-episode failure-mix dataset recording (no GPU/sim)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


def parse_int_list(values_str: str) -> list[int]:
    """Parse a comma-separated integer list, e.g. ``\"20,30,40\"``."""
    parts = [p.strip() for p in values_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("integer list is empty after parsing")
    return [int(p) for p in parts]


def parse_seed_list(seeds_str: str) -> list[int]:
    """Parse a comma-separated seed list, e.g. ``\"1000,1001,1010\"``."""
    parts = [p.strip() for p in seeds_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("seeds string is empty after parsing")
    return parse_int_list(seeds_str)


def default_seed_sequence(seed_start: int, count: int) -> list[int]:
    """Return ``count`` consecutive seeds starting at ``seed_start``."""
    if count < 0:
        raise ValueError(f"count must be >= 0 (got {count})")
    start = int(seed_start)
    n = int(count)
    return [start + i for i in range(n)]


def parse_drops_per_delay(delay_grid: list[int], drops_arg: str | int | list[int]) -> list[int]:
    """Per-bin drop quotas aligned with ``delay_grid``."""
    if not delay_grid:
        raise ValueError("delay_grid must not be empty")
    if isinstance(drops_arg, list):
        values = [int(v) for v in drops_arg]
        if len(values) == 1:
            return values * len(delay_grid)
        if len(values) != len(delay_grid):
            raise ValueError(
                f"drops_per_delay length {len(values)} must match delay_grid length "
                f"{len(delay_grid)} (or be a single value to broadcast)"
            )
        if any(v < 0 for v in values):
            raise ValueError(f"drops_per_delay values must be >= 0 (got {values})")
        return values
    if isinstance(drops_arg, int):
        if drops_arg < 0:
            raise ValueError(f"drops_per_delay must be >= 0 (got {drops_arg})")
        return [int(drops_arg)] * len(delay_grid)
    parts = [p.strip() for p in str(drops_arg).split(",") if p.strip()]
    if not parts:
        raise ValueError("drops_per_delay is empty after parsing")
    if len(parts) == 1:
        n = int(parts[0])
        if n < 0:
            raise ValueError(f"drops_per_delay must be >= 0 (got {n})")
        return [n] * len(delay_grid)
    values = [int(p) for p in parts]
    if len(values) != len(delay_grid):
        raise ValueError(
            f"drops_per_delay length {len(values)} must match delay_grid length "
            f"{len(delay_grid)} (or pass a single integer to broadcast)"
        )
    if any(v < 0 for v in values):
        raise ValueError(f"drops_per_delay values must be >= 0 (got {values})")
    return values


def delay_grid_drop_target(delay_grid: list[int], drops_per_delay: int | list[int] | str) -> int:
    """Total drop episodes when recording a fixed delay grid."""
    if not delay_grid:
        raise ValueError("delay_grid must not be empty")
    if isinstance(drops_per_delay, int):
        if drops_per_delay < 0:
            raise ValueError(f"drops_per_delay must be >= 0 (got {drops_per_delay})")
        return len(delay_grid) * int(drops_per_delay)
    per_bin = parse_drops_per_delay(delay_grid, drops_per_delay)
    return sum(per_bin)


def resolve_mix_drop_target(
    *,
    delay_grid: list[int] | None,
    drops_per_delay: int | list[int] | str,
    n_drop: int | None,
    n_drop_explicit: bool,
) -> int:
    """Compute drop quota for a mix run."""
    if delay_grid is not None:
        if n_drop_explicit:
            raise ValueError(
                "--n-drop cannot be used with --delay-grid; "
                "drop target is the sum of per-bin drops_per_delay counts"
            )
        return delay_grid_drop_target(delay_grid, drops_per_delay)
    if n_drop is None:
        return 8
    if n_drop < 0:
        raise ValueError(f"n_drop must be >= 0 (got {n_drop})")
    return int(n_drop)


def compute_mix_max_attempts(
    *,
    n_drop: int,
    n_nominal: int,
    delay_grid: list[int] | None,
    max_attempts_arg: int,
) -> int:
    """Default attempt budget; larger when a delay grid needs many drop retries."""
    base = max(1, int(max_attempts_arg))
    if delay_grid is None:
        return base
    return max(base, (int(n_drop) + int(n_nominal)) * 4)


@dataclass(frozen=True)
class MixWorkItem:
    """One pipeline attempt in a mix recording plan."""

    kind: Literal["drop", "nominal"]
    post_grasp_delay_steps: int | None = None


def plan_delay_grid_work_items(
    delay_grid: list[int],
    *,
    drops_per_delay: int | list[int] | str,
    n_nominal: int,
) -> list[MixWorkItem]:
    """Expand drop bins (fixed delay each) then nominal episodes."""
    per_bin = parse_drops_per_delay(delay_grid, drops_per_delay)
    items: list[MixWorkItem] = []
    for delay, quota in zip(delay_grid, per_bin, strict=True):
        for _ in range(int(quota)):
            items.append(MixWorkItem(kind="drop", post_grasp_delay_steps=int(delay)))
    for _ in range(int(n_nominal)):
        items.append(MixWorkItem(kind="nominal", post_grasp_delay_steps=None))
    return items


def next_open_delay_grid_bin(
    delay_grid: list[int],
    *,
    drops_per_bin: list[int],
    kept_per_delay: dict[int, int],
) -> int | None:
    """Return the next delay bin that still needs successful drop episodes."""
    if len(drops_per_bin) != len(delay_grid):
        raise ValueError(
            f"drops_per_bin length {len(drops_per_bin)} must match delay_grid length {len(delay_grid)}"
        )
    for delay, quota in zip(delay_grid, drops_per_bin, strict=True):
        if kept_per_delay.get(int(delay), 0) < int(quota):
            return int(delay)
    return None


def mix_output_layout(output_dir: Path) -> dict[str, Path]:
    """Standard directory layout for a mixed failure dataset run."""
    root = Path(output_dir)
    return {
        "dataset": root / "dataset",
        "episodes": root / "episodes",
        "mix_log": root / "mix_log.json",
    }


def episode_dir_for_seed(output_dir: Path, seed: int) -> Path:
    """Per-attempt logs/videos under ``output_dir/episodes/seed_{seed}/``."""
    return Path(output_dir) / "episodes" / f"seed_{int(seed)}"


def should_keep_episode(summary: dict[str, Any], kind: str) -> bool:
    """Return whether a pipeline summary should be committed to the shared dataset."""
    if kind not in ("drop", "nominal"):
        raise ValueError(f"unknown episode kind {kind!r}; expected 'drop' or 'nominal'")
    return bool(summary.get("success"))


class _EpisodeLogger(Protocol):
    def end_episode(self) -> None: ...

    def clear_open_episode(self) -> None: ...


def commit_or_discard(logger: _EpisodeLogger, *, keep: bool) -> None:
    """Persist the open episode or drop buffered frames."""
    if keep:
        logger.end_episode()
    else:
        logger.clear_open_episode()


@dataclass
class MixCounters:
    """Running quotas for interleaved mix recording."""

    n_drop_target: int
    n_nominal_target: int
    drop_kept: int = 0
    nominal_kept: int = 0
    attempts: int = 0
    drop_discarded: int = 0
    nominal_discarded: int = 0

    @property
    def drop_full(self) -> bool:
        return self.drop_kept >= self.n_drop_target

    @property
    def nominal_full(self) -> bool:
        return self.nominal_kept >= self.n_nominal_target

    @property
    def quotas_met(self) -> bool:
        return self.drop_full and self.nominal_full

    def next_kind_interleave(self) -> str | None:
        """Alternate drop / nominal until both quotas are filled."""
        if self.quotas_met:
            return None
        if self.drop_full:
            return "nominal"
        if self.nominal_full:
            return "drop"
        return "drop" if self.attempts % 2 == 0 else "nominal"

    def record_attempt(self, kind: str, kept: bool) -> None:
        self.attempts += 1
        if kept:
            if kind == "drop":
                self.drop_kept += 1
            elif kind == "nominal":
                self.nominal_kept += 1
            else:
                raise ValueError(f"unknown kind {kind!r}")
        elif kind == "drop":
            self.drop_discarded += 1
        elif kind == "nominal":
            self.nominal_discarded += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_drop_target": self.n_drop_target,
            "n_nominal_target": self.n_nominal_target,
            "drop_kept": self.drop_kept,
            "nominal_kept": self.nominal_kept,
            "attempts": self.attempts,
            "drop_discarded": self.drop_discarded,
            "nominal_discarded": self.nominal_discarded,
        }
