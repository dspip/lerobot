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

"""SmolVLA controller adapter using the in-package drop-recovery pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from lerobot.faults.datagen.drop_trigger import smolvla_fault_drop_fields
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.recipe import DropDatagenRecipe, effective_post_drop_dwell_steps
from lerobot.faults.datagen.smolvla_pipeline import run_pipeline

SMOLVLA_DEFAULT_MIN_DROP_DISTANCE_M = 0.30

PipelineRunner = Callable[..., dict[str, Any]]


class SmolVLADatagenAdapter:
    def __init__(
        self,
        recipe: DropDatagenRecipe,
        *,
        pipeline_runner: PipelineRunner | None = None,
        min_drop_distance_from_basket_m: float = SMOLVLA_DEFAULT_MIN_DROP_DISTANCE_M,
    ) -> None:
        self._recipe = recipe
        self._pipeline_runner = pipeline_runner or run_pipeline
        self._min_drop_distance_m = float(min_drop_distance_from_basket_m)

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        recipe = request.recipe
        manifest = request.manifest
        plan = request.paired_plan
        dwell_steps = effective_post_drop_dwell_steps(recipe, manifest.post_drop_mode)
        drop_fields = smolvla_fault_drop_fields(plan.smolvla_target)

        summary = self._pipeline_runner(
            request.output_dir,
            policy_path=recipe.smolvla.policy_path,
            device=request.device,
            seed=manifest.controller_seed,
            post_grasp_delay_steps=recipe.smolvla.post_grasp_delay_steps,
            post_drop_dwell_steps=dwell_steps,
            post_drop_mode=manifest.post_drop_mode.value,
            min_drop_distance_from_basket_m=self._min_drop_distance_m,
            drop_xy_band_min=drop_fields["drop_xy_band_min"],
            drop_xy_band_max=drop_fields["drop_xy_band_max"],
            drop_xy_target_m=drop_fields["drop_xy_target_m"],
            object_name=request.object_name,
            init_state_id=plan.init_state_id,
            shared_layout=request.shared_layout,
            layout_seed=plan.layout_seed if request.shared_layout is None else None,
            placement_config=recipe if request.shared_layout is None else None,
            wipe_output_dir=True,
            raise_on_failure=True,
            copy_demo_gif=False,
            fault_overrides={
                "object_name": request.object_name,
                "basket_name": recipe.basket_name,
                "probability": 1.0,
            },
        )
        success = bool(summary.get("success", summary.get("behavioral_success", False)))
        return EpisodeResult.from_run(
            request,
            success=success,
            outcome="completed" if success else "pipeline_failed",
            drop_trigger={
                "kind": "smolvla_xy_target",
                **drop_fields,
                "band_name": plan.smolvla_target.band.name,
            },
            actual_dwell_steps=int(
                summary.get("fault_config", {}).get("post_drop_dwell_steps", dwell_steps)
            ),
            pipeline_summary=summary,
        )
