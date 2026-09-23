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

"""SmolVLA controller adapter delegating to the existing drop-recovery pipeline."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.faults.datagen.drop_trigger import sample_smolvla_band_target
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.recipe import DropDatagenRecipe, effective_post_drop_dwell_steps

PipelineRunner = Callable[..., dict[str, Any]]


def _load_run_pipeline() -> PipelineRunner:
    repo_root = Path(__file__).resolve().parents[5]
    path = repo_root / "examples" / "faults" / "run_full_drop_recovery_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_full_drop_recovery_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pipeline module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = getattr(module, "run_pipeline", None)
    if runner is None:
        raise ImportError("run_pipeline not found in run_full_drop_recovery_pipeline.py")
    return runner


class SmolVLADatagenAdapter:
    """Adapts ``run_pipeline`` for unified matrix episodes."""

    def __init__(
        self,
        recipe: DropDatagenRecipe,
        *,
        pipeline_runner: PipelineRunner | None = None,
        device: str = "cuda",
    ) -> None:
        self._recipe = recipe
        self._pipeline_runner = pipeline_runner or _load_run_pipeline()
        self._device = device

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        recipe = request.recipe
        manifest = request.manifest
        dwell_steps = effective_post_drop_dwell_steps(recipe, manifest.post_drop_mode)
        band_rng = np.random.default_rng(manifest.controller_seed)
        band_target = sample_smolvla_band_target(band_rng, recipe.smolvla.drop_xy_bands)
        margin = 0.002
        band_min = max(band_target.band.min_m, band_target.target_m - margin)
        band_max = min(band_target.band.max_m, band_target.target_m + margin)

        summary = self._pipeline_runner(
            request.output_dir,
            policy_path=recipe.smolvla.policy_path,
            device=self._device,
            seed=manifest.controller_seed,
            post_grasp_delay_steps=recipe.smolvla.post_grasp_delay_steps,
            post_drop_dwell_steps=dwell_steps,
            post_drop_mode=manifest.post_drop_mode.value,
            drop_xy_band_min=band_min,
            drop_xy_band_max=band_max,
            min_drop_distance_from_basket_m=(
                recipe.simple_ik.path_drop.min_drop_distance_from_basket_m
                if recipe.simple_ik.path_drop is not None
                else 0.30
            ),
            wipe_output_dir=True,
            raise_on_failure=False,
            copy_demo_gif=False,
            fault_overrides={
                "object_name": request.object_name,
                "basket_name": recipe.basket_name,
                "probability": 1.0,
            },
        )
        success = bool(summary.get("task_success"))
        return EpisodeResult.ok(
            request,
            outcome="completed" if success else "pipeline_finished",
            drop_trigger={
                "kind": "smolvla_xy_band",
                "band_name": band_target.band.name,
                "band_min_m": band_target.band.min_m,
                "band_max_m": band_target.band.max_m,
                "target_m": band_target.target_m,
                "configured_band_min_m": band_min,
                "configured_band_max_m": band_max,
            },
            pipeline_summary=summary,
        )
