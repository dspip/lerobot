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

"""Unified drop datagen entry point (recipe-driven experiment matrix)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path

from lerobot.faults.datagen.dataset_writer import RunDatasetWriter
from lerobot.faults.datagen.recipe import (
    DropDatagenRecipe,
    RecipeError,
    apply_recording_episode_total,
    load_drop_datagen_recipe,
)
from lerobot.faults.datagen.recipe_identity import recipe_content_hash_from_json_path
from lerobot.faults.datagen.runner import run_drop_datagen_matrix
from lerobot.faults.datagen.shard_range import validate_logical_shard_range


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        type=Path,
        required=True,
        help="Path to can_drop_datagen.json (typed unified recipe).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Override recording.base_seed from the recipe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override recording.output_dir from the recipe.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override recording.episodes (dataset total across matrix rows; must be >= 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for SmolVLA runs (e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--logical-start",
        type=int,
        default=None,
        help="Half-open logical episode range start (requires --logical-end).",
    )
    parser.add_argument(
        "--logical-end",
        type=int,
        default=None,
        help="Half-open logical episode range end (exclusive).",
    )
    return parser


def _apply_overrides(
    recipe: DropDatagenRecipe,
    *,
    base_seed: int | None,
    output: Path | None,
    episodes: int | None,
) -> DropDatagenRecipe:
    recording = recipe.recording
    if base_seed is not None:
        recording = replace(recording, base_seed=int(base_seed))
    if output is not None:
        recording = replace(recording, output_dir=str(output))
    updated = replace(recipe, recording=recording)
    if episodes is not None:
        if int(episodes) <= 0:
            raise RecipeError("episodes override must be >= 1")
        updated = apply_recording_episode_total(updated, int(episodes))
    return updated


def _serialize_result(result: object) -> dict:
    if is_dataclass(result):
        payload = asdict(result)
        payload["output_dir"] = str(payload["output_dir"])
        payload["controller"] = getattr(result.controller, "value", result.controller)
        payload["post_drop_mode"] = getattr(result.post_drop_mode, "value", result.post_drop_mode)
        return payload
    raise TypeError(f"Cannot serialize {type(result)!r}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        recipe = load_drop_datagen_recipe(args.recipe)
        recipe = _apply_overrides(
            recipe,
            base_seed=args.base_seed,
            output=args.output,
            episodes=args.episodes,
        )
    except RecipeError as exc:
        print(f"Recipe error: {exc}", file=sys.stderr)
        return 2

    logical_indices: range | None = None
    logical_range: tuple[int, int] | None = None
    if args.logical_start is not None or args.logical_end is not None:
        if args.logical_start is None or args.logical_end is None:
            print("Both --logical-start and --logical-end are required for sharded runs.", file=sys.stderr)
            return 2
        try:
            logical_range = validate_logical_shard_range(recipe, args.logical_start, args.logical_end)
        except RecipeError as exc:
            print(f"Recipe error: {exc}", file=sys.stderr)
            return 2
        logical_indices = range(logical_range[0], logical_range[1])

    recipe_hash = recipe_content_hash_from_json_path(args.recipe)
    writer = RunDatasetWriter(
        recipe,
        logical_range=logical_range,
        recipe_content_hash=recipe_hash,
    )
    results = run_drop_datagen_matrix(
        recipe,
        logical_episode_indices=logical_indices,
        device=str(args.device),
        dataset_writer=writer,
    )
    summary_path = Path(recipe.recording.output_dir) / "matrix_results.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps([_serialize_result(r) for r in results], indent=2),
        encoding="utf-8",
    )
    ok = sum(1 for r in results if r.success)
    print(f"Completed {ok}/{len(results)} matrix episode(s)", flush=True)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
