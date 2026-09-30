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

"""Logical episode shard range validation for datagen runs."""

from __future__ import annotations

from fault_system.datagen.recipe import DropDatagenRecipe, RecipeError

__all__ = [
    "logical_range_from_manifest_raw",
    "max_logical_episodes",
    "ranges_overlap",
    "validate_logical_shard_range",
]


def max_logical_episodes(recipe: DropDatagenRecipe) -> int:
    """Upper bound on logical episode indices for this recipe (exclusive end for full run)."""
    if not recipe.experiment_matrix:
        raise RecipeError("experiment_matrix must be non-empty")
    return max(int(variant.episodes) for variant in recipe.experiment_matrix)


def validate_logical_shard_range(
    recipe: DropDatagenRecipe,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Validate half-open ``[start, end)`` against the recipe's logical episode budget."""
    max_ep = max_logical_episodes(recipe)
    start_i, end_i = int(start), int(end)
    if start_i < 0:
        raise RecipeError(f"logical-start must be >= 0 (got {start_i})")
    if end_i <= start_i:
        raise RecipeError(f"logical-end must be > logical-start (got [{start_i}, {end_i}))")
    if end_i > max_ep:
        raise RecipeError(
            f"logical-end must be <= {max_ep} (max logical episodes for recipe; got {end_i})"
        )
    return start_i, end_i


def logical_range_from_manifest_raw(raw: dict) -> tuple[int, int] | None:
    """Read ``logical_range: [start, end]`` from a manifest dict, if present."""
    value = raw.get("logical_range")
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("logical_range must be a two-element list [start, end]")
    return int(value[0]), int(value[1])


def ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when half-open ranges ``[a0,a1)`` and ``[b0,b1)`` intersect."""
    return a[0] < b[1] and b[0] < a[1]
