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
from dataclasses import asdict, is_dataclass
from pathlib import Path

from lerobot.faults.datagen.recipe import DropDatagenRecipe, RecipeError, load_drop_datagen_recipe
from lerobot.faults.datagen.runner import run_drop_datagen_matrix


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
        help="Override episodes count for every matrix variant (must be >= 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for SmolVLA runs (e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable interactive viewers when supported.",
    )
    return parser


def _apply_overrides(
    recipe: DropDatagenRecipe,
    *,
    base_seed: int | None,
    output: Path | None,
    episodes: int | None,
) -> DropDatagenRecipe:
    if episodes is not None and int(episodes) <= 0:
        raise RecipeError("episodes override must be >= 1")
    recording = recipe.recording
    if base_seed is not None:
        recording = recording.__class__(
            base_seed=int(base_seed),
            output_dir=recording.output_dir,
            dataset_fps=recording.dataset_fps,
        )
    if output is not None:
        recording = recording.__class__(
            base_seed=recording.base_seed,
            output_dir=str(output),
            dataset_fps=recording.dataset_fps,
        )
    matrix = recipe.experiment_matrix
    if episodes is not None:
        matrix = tuple(
            variant.__class__(
                controller=variant.controller,
                post_drop_mode=variant.post_drop_mode,
                episodes=int(episodes),
            )
            for variant in matrix
        )
    return recipe.__class__(
        name=recipe.name,
        task=recipe.task,
        task_id=recipe.task_id,
        control_hz=recipe.control_hz,
        q=recipe.q,
        object_names=recipe.object_names,
        basket_name=recipe.basket_name,
        placement=recipe.placement,
        simple_ik=recipe.simple_ik,
        smolvla=recipe.smolvla,
        post_drop=recipe.post_drop,
        recording=recording,
        experiment_matrix=matrix,
    )


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
    results = run_drop_datagen_matrix(
        recipe,
        headless=args.headless,
        device=str(args.device),
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
