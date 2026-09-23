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
from dataclasses import asdict
from typing import Any

from lerobot.faults.datagen.drop_trigger import smolvla_fault_drop_fields
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.recipe import DropDatagenRecipe, effective_post_drop_dwell_steps
from lerobot.faults.datagen.smolvla_pipeline import run_pipeline
from lerobot.faults.datagen.smolvla_resources import SmolVLAPolicyResources, load_smolvla_policy_resources

PipelineRunner = Callable[..., dict[str, Any]]

_BEHAVIORAL_OUTCOME_ORDER: tuple[tuple[str, str], ...] = (
    ("fault_triggered", "drop_not_triggered"),
    ("grasped_before_drop", "no_grasp_before_drop"),
    ("was_midair_at_drop", "not_midair_at_drop"),
    ("drop_moved_object", "drop_did_not_move_object"),
    ("object_in_view_after_drop", "object_not_in_view_after_drop"),
    ("regrasped_after_drop", "no_regrasp_after_drop"),
    ("object_in_basket", "not_in_basket"),
    ("dropped_after_carry", "dropped_before_carry_complete"),
    ("seat_assist_ok", "seat_assist_violation"),
)


def smolvla_behavioral_outcome(summary: dict[str, Any]) -> str:
    """Map SmolVLA pipeline behavioral checks to a stable reject outcome string."""
    if bool(summary.get("behavioral_success", summary.get("success", False))):
        return "completed"
    checks = summary.get("checks")
    if isinstance(checks, dict):
        for check_key, outcome in _BEHAVIORAL_OUTCOME_ORDER:
            if check_key in checks and not bool(checks[check_key]):
                return outcome
    return "behavioral_failed"


def _drop_trigger_from_summary(
    summary: dict[str, Any],
    *,
    drop_fields: dict[str, Any],
    band_name: str,
) -> dict[str, Any]:
    return {
        "kind": "smolvla_xy_target",
        **drop_fields,
        "band_name": band_name,
        "triggered_at": summary.get("triggered_at"),
        "drop_basket_xy_dist": summary.get("drop_basket_xy_dist"),
        "drop_trigger_reason": summary.get("drop_trigger_reason"),
        "pre_drop_pose": summary.get("pre_drop_pose"),
        "post_drop_pose": summary.get("post_drop_pose"),
    }


class SmolVLADatagenAdapter:
    """Datagen controller adapter that delegates episodes to the SmolVLA pipeline."""

    def __init__(
        self,
        recipe: DropDatagenRecipe,
        *,
        pipeline_runner: PipelineRunner | None = None,
        policy_resources: SmolVLAPolicyResources | None = None,
    ) -> None:
        """Store recipe settings and an optional injectable pipeline runner."""
        self._recipe = recipe
        self._pipeline_runner = pipeline_runner or run_pipeline
        self._policy_resources = policy_resources

    def _policy_bundle(self, device: str) -> SmolVLAPolicyResources:
        if self._policy_resources is not None and (
            self._policy_resources.policy_path != self._recipe.smolvla.policy_path
            or self._policy_resources.device != device
        ):
            self._policy_resources = None
        if self._policy_resources is None:
            self._policy_resources = load_smolvla_policy_resources(
                policy_path=self._recipe.smolvla.policy_path,
                device=device,
                task=self._recipe.task,
                task_id=self._recipe.task_id,
            )
        return self._policy_resources

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        """Run the SmolVLA drop-recovery pipeline and map its summary to ``EpisodeResult``."""
        recipe = request.recipe
        manifest = request.manifest
        plan = request.paired_plan
        dwell_steps = effective_post_drop_dwell_steps(recipe, manifest.post_drop_mode)
        session = request.episode_session
        motion = plan.motion_profile
        profile_dict = asdict(motion)

        base_pipeline_kwargs: dict[str, Any] = {
            "policy_path": recipe.smolvla.policy_path,
            "device": request.device,
            "episode_seed": manifest.episode_seed,
            "drop_seed": manifest.drop_seed,
            "variant_seed": manifest.controller_seed,
            "seed": manifest.episode_seed,
            "task": recipe.task,
            "task_id": recipe.task_id,
            "control_hz": recipe.control_hz,
            "post_grasp_delay_steps": recipe.smolvla.post_grasp_delay_steps,
            "post_drop_dwell_steps": dwell_steps,
            "post_drop_mode": manifest.post_drop_mode.value,
            "min_drop_distance_from_basket_m": recipe.smolvla.min_drop_distance_from_basket_m,
            "object_name": request.object_name,
            "basket_name": recipe.basket_name,
            "init_state_id": plan.init_state_id,
            "shared_layout": request.shared_layout,
            "motion_profile": motion,
            "policy_resources": self._policy_bundle(request.device),
            "wipe_output_dir": session is None,
            "raise_on_failure": False,
            "copy_demo_gif": False,
            "record_dataset": session is not None,
            "dataset_fps": recipe.recording.dataset_fps,
        }
        if session is not None:
            base_pipeline_kwargs["episode_session"] = session
            base_pipeline_kwargs["defer_dataset_commit"] = True

        if not plan.drop_decision.drop:
            summary = self._pipeline_runner(
                request.output_dir,
                **base_pipeline_kwargs,
                episode_kind="nominal",
            )
            success = bool(summary.get("behavioral_success", summary.get("success", False)))
            return EpisodeResult.from_run(
                request,
                success=success,
                outcome="nominal_no_drop",
                drop_trigger={"kind": "paired_skipped", "reason": plan.drop_decision.reason},
                actual_dwell_steps=0,
                pipeline_summary=summary,
                motion_profile=profile_dict,
            )

        drop_fields = smolvla_fault_drop_fields(plan.smolvla_target)
        summary = self._pipeline_runner(
            request.output_dir,
            **base_pipeline_kwargs,
            episode_kind="drop",
            drop_xy_band_min=drop_fields["drop_xy_band_min"],
            drop_xy_band_max=drop_fields["drop_xy_band_max"],
            drop_xy_target_m=drop_fields["drop_xy_target_m"],
            fault_overrides={
                "object_name": request.object_name,
                "basket_name": recipe.basket_name,
                "probability": 1.0,
            },
        )
        success = bool(summary.get("behavioral_success", summary.get("success", False)))
        outcome = smolvla_behavioral_outcome(summary) if not success else "completed"
        actual_dwell = summary.get("actual_dwell_steps")
        trigger_pose = summary.get("trigger_pose") or summary.get("pre_drop_pose")
        trigger_pose_list = [float(x) for x in trigger_pose] if isinstance(trigger_pose, list) else None
        return EpisodeResult.from_run(
            request,
            success=success,
            outcome=outcome,
            drop_trigger=_drop_trigger_from_summary(
                summary,
                drop_fields=drop_fields,
                band_name=plan.smolvla_target.band.name,
            ),
            trigger_pose=trigger_pose_list,
            actual_dwell_steps=int(actual_dwell) if actual_dwell is not None else None,
            pipeline_summary=summary,
            motion_profile=profile_dict,
        )
