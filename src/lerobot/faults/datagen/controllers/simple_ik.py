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

"""SimpleIK controller adapter and episode loop."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lerobot.envs.factory import make_env
from lerobot.faults.datagen.drop_timing import DropDecision, keepout_m
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.frame_logging import log_post_step_to_session, should_log_sim_step
from lerobot.faults.datagen.paired_context import PairedEpisodePlan, resolve_path_drop_trigger
from lerobot.faults.datagen.path_drop import (
    EligiblePath,
    PathTrigger,
    eligible_path,
    sample_path_drop,  # noqa: F401 — unit tests patch ``simple_ik.sample_path_drop``
)
from lerobot.faults.datagen.recipe import (
    DropDatagenRecipe,
    DropRecipe,
    effective_post_drop_dwell_steps,
    legacy_drop_recipe,
)
from lerobot.faults.datagen.runtime import stabilize_carry_action
from lerobot.faults.datagen.scene import apply_serializable_layout
from lerobot.faults.datagen.task_label import read_libero_task_description
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from lerobot.faults.recovery.planner import CARRY_PHASES, SimpleIKRecoveryPlanner
from lerobot.faults.sim.libero import (
    force_close_gripper,
    force_open_gripper,
    get_eef_pose,
    get_gripper_closing_axis,
    get_object_pose,
    get_place_destination,
    get_robosuite_env,
    hold_gripper_closed,
    is_object_grasped,  # noqa: F401 — unit tests patch ``simple_ik.is_object_grasped``
    is_object_held_midair,
    is_object_in_basket,
    object_pose_orientation,
    read_control_freq,
    unwrap_libero_env,
)
from lerobot.faults.wrappers import DropRecoveryEnvWrapper

MAX_PLAN_STEPS = 800
POST_RECOVERY_SETTLE_STEPS = 40


@dataclass
class SimpleIKEpisodeFacts:
    """Outcome fields collected after a SimpleIK datagen episode loop."""

    success: bool
    outcome: str
    drop_trigger: dict[str, Any] | None
    trigger_pose: list[float] | None
    actual_dwell_steps: int
    layout: dict[str, Any] | None = None


def _nominal_action(
    planner: SimpleIKRecoveryPlanner,
    rs_env: Any,
    object_name: str,
    *,
    gripper_settle_steps: int,
) -> np.ndarray | None:
    eef_pos, _ = get_eef_pose(rs_env)
    object_pos = get_object_pose(rs_env, object_name)["pos"]
    action = planner.next_action(
        eef_pos=eef_pos,
        object_pos=object_pos,
        closing_axis=get_gripper_closing_axis(rs_env),
        object_axis=object_pose_orientation(rs_env, object_name)["axis"],
    )
    if planner.just_entered_close:
        force_close_gripper(rs_env, gripper_settle_steps=gripper_settle_steps)
    if planner.just_entered_open:
        force_open_gripper(rs_env, gripper_settle_steps=gripper_settle_steps)
    if action is not None and planner.phase_name in CARRY_PHASES:
        hold_gripper_closed(rs_env)
        action = stabilize_carry_action(action)
    return action


def run_simple_ik_episode_loop(
    env: Any,
    rs_env: Any,
    *,
    fault: MidAirDropFault,
    planner: SimpleIKRecoveryPlanner,
    recipe_drop: DropRecipe,
    object_name: str,
    basket_name: str,
    q: float,
    drop_rng: np.random.Generator,
    paired_plan: PairedEpisodePlan | None = None,
    max_steps: int = MAX_PLAN_STEPS,
    gripper_settle_steps: int = 0,
    episode_session: Any | None = None,
    recording_stride: int = 1,
    task: str = "pick up the alphabet soup and place it in the basket",
    eligible_phases: tuple[str, ...] | None = None,
) -> SimpleIKEpisodeFacts:
    """Drive one SimpleIK episode with path-based drop timing and optional recording."""
    keepout = keepout_m(
        recipe_drop.min_drop_distance_from_basket_m,
        recipe_drop.hard_keepout_floor_m,
    )
    path: EligiblePath | None = None
    decision: DropDecision | None = None
    path_trigger: PathTrigger | None = None
    dropped = False
    trigger_pose: list[float] | None = None
    settle_left = POST_RECOVERY_SETTLE_STEPS
    dwell_before_recovery = 0

    for step in range(max_steps):
        state = fault._states[0]

        if state.recovery_active:
            action = np.zeros((1, 7), dtype=np.float32)
        elif dropped and not state.recovery_active:
            skipped, skip_reason = fault.post_drop_recovery_skipped(0)
            if skipped:
                return SimpleIKEpisodeFacts(
                    False,
                    skip_reason or "recovery_skipped",
                    _drop_trigger_payload(decision, path_trigger),
                    trigger_pose,
                    int(state.dwell_steps_completed),
                )
            nominal = _nominal_action(planner, rs_env, object_name, gripper_settle_steps=gripper_settle_steps)
            if nominal is None or planner.done:
                return SimpleIKEpisodeFacts(
                    False,
                    "nominal_completed_after_drop",
                    _drop_trigger_payload(decision, path_trigger),
                    trigger_pose,
                    int(state.dwell_steps_completed),
                )
            action = nominal.reshape(1, 7)
        else:
            phase = planner.phase_name
            obj = get_object_pose(rs_env, object_name)["pos"].copy()
            basket = get_place_destination(rs_env, object_name, basket_name=basket_name)
            distance = float(np.linalg.norm(obj[:2] - basket[:2]))

            if decision is None and phase == "lift" and planner.carry_path is not None:
                path = eligible_path(
                    planner.carry_path,
                    basket_xy=basket[:2],
                    keepout_m=keepout,
                    eligible_phases=eligible_phases,
                )
                if paired_plan is None:
                    raise ValueError("paired_plan is required for unified drop datagen")
                decision = paired_plan.drop_decision
                if paired_plan.drop_decision.drop and paired_plan.drop_u is not None:
                    path_trigger = resolve_path_drop_trigger(
                        paired_plan,
                        planner.carry_path,
                        basket_xy=basket[:2],
                        keepout_m=keepout,
                        eligible_phases=eligible_phases,
                    )
                else:
                    path_trigger = None

            held_midair = bool(is_object_held_midair(rs_env, object_name))
            fire = (
                path_trigger is not None
                and not dropped
                and decision is not None
                and decision.drop
                and path_trigger.fires(
                    phase=phase,
                    object_xyz=obj,
                    carry_path=planner.carry_path,
                    held_midair=held_midair,
                )
            )
            if fire and distance < keepout:
                decision = DropDecision(False, None, "runtime_keepout")
                path_trigger = None
                fire = False

            if fire:
                nominal = _nominal_action(
                    planner, rs_env, object_name, gripper_settle_steps=gripper_settle_steps
                )
                if nominal is None:
                    return SimpleIKEpisodeFacts(
                        False,
                        "nominal_missing_at_drop",
                        _drop_trigger_payload(decision, path_trigger),
                        trigger_pose,
                        0,
                    )
                trigger_pose = obj.astype(float).tolist()
                executed = fault.trigger_scheduled_drop(env, 0, nominal, reason="path_uniform")
                dropped = True
                action = np.asarray(executed, dtype=np.float32).reshape(1, 7)
            else:
                nominal = _nominal_action(
                    planner, rs_env, object_name, gripper_settle_steps=gripper_settle_steps
                )
                if nominal is None or planner.done:
                    if dropped:
                        return SimpleIKEpisodeFacts(
                            False,
                            "nominal_completed_after_drop",
                            _drop_trigger_payload(decision, path_trigger),
                            trigger_pose,
                            int(state.dwell_steps_completed),
                        )
                    if not paired_plan.drop_decision.drop and not dropped:
                        return _paired_nominal_no_drop_facts(
                            rs_env,
                            object_name=object_name,
                            basket_name=basket_name,
                            reason=paired_plan.drop_decision.reason,
                            trigger_pose=trigger_pose,
                        )
                    if decision is not None and decision.reason == "runtime_keepout":
                        return SimpleIKEpisodeFacts(
                            False,
                            "runtime_keepout",
                            _drop_trigger_payload(decision, path_trigger),
                            trigger_pose,
                            0,
                        )
                    return SimpleIKEpisodeFacts(
                        False,
                        "nominal_completed_without_drop",
                        _drop_trigger_payload(decision, path_trigger),
                        trigger_pose,
                        0,
                    )
                action = nominal.reshape(1, 7)

        step_out = env.step(action)
        observation = step_out[0] if isinstance(step_out, tuple) and step_out else None
        state = fault._states[0]
        is_drop_episode = paired_plan is None or paired_plan.drop_decision.drop
        drop_injection = bool(is_drop_episode and state.drop_injection_step)
        if (
            episode_session is not None
            and observation is not None
            and should_log_sim_step(
                step,
                recording_stride=recording_stride,
                force_drop_injection=drop_injection,
            )
        ):
            executed = env.last_executed_action
            if executed is None:
                executed = action
            log_post_step_to_session(
                episode_session,
                env=env,
                post_step_observation=observation,
                executed_action=np.asarray(executed),
                task=task,
                phase=planner.phase_name,
                is_drop_episode=is_drop_episode,
                sim_step=step,
            )

        state = fault._states[0]
        if dropped:
            dwell_before_recovery = int(state.dwell_steps_completed)

        if state.recovery_active and state.planner is not None and state.planner.done:
            settle_left -= 1
            if settle_left <= 0:
                in_basket = is_object_in_basket(rs_env, object_name, basket_name=basket_name, z_max=0.14)
                return SimpleIKEpisodeFacts(
                    bool(in_basket),
                    "recovery_completed_in_basket" if in_basket else "recovery_finished_outside_basket",
                    _drop_trigger_payload(decision, path_trigger),
                    trigger_pose,
                    dwell_before_recovery,
                )

    if paired_plan is not None and not paired_plan.drop_decision.drop and not dropped:
        return _paired_nominal_no_drop_facts(
            rs_env,
            object_name=object_name,
            basket_name=basket_name,
            reason=paired_plan.drop_decision.reason,
            trigger_pose=trigger_pose,
        )

    return SimpleIKEpisodeFacts(
        False,
        "max_steps_exceeded",
        _drop_trigger_payload(decision, path_trigger),
        trigger_pose,
        dwell_before_recovery,
    )


def _paired_skipped_drop_trigger(reason: str) -> dict[str, Any]:
    return {"kind": "paired_skipped", "reason": reason}


def _paired_nominal_no_drop_facts(
    rs_env: Any,
    *,
    object_name: str,
    basket_name: str,
    reason: str,
    trigger_pose: list[float] | None,
) -> SimpleIKEpisodeFacts:
    in_basket = is_object_in_basket(rs_env, object_name, basket_name=basket_name, z_max=0.14)
    return SimpleIKEpisodeFacts(
        bool(in_basket),
        "nominal_no_drop",
        _paired_skipped_drop_trigger(reason),
        trigger_pose,
        0,
    )


def _drop_trigger_payload(
    decision: DropDecision | None,
    trigger: PathTrigger | None,
) -> dict[str, Any] | None:
    if decision is None or trigger is None:
        return None
    return {
        "kind": "simple_ik_path",
        "drop": decision.drop,
        "reason": decision.reason,
        "segment_name": trigger.segment_name,
        "target_t": trigger.target_t,
    }


class SimpleIKDatagenAdapter:
    """Datagen controller adapter that runs episodes via the SimpleIK recovery loop."""

    def __init__(self, recipe: DropDatagenRecipe) -> None:
        """Store the unified drop datagen recipe for episode construction."""
        self._recipe = recipe

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        """Build a LIBERO env, run the IK loop, and return an ``EpisodeResult``."""
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lerobot.envs.configs import LiberoEnv
        from lerobot.faults.config import FaultInjectionConfig

        recipe = request.recipe
        manifest = request.manifest
        plan = request.paired_plan
        drop_recipe = legacy_drop_recipe(recipe)
        dwell_steps = effective_post_drop_dwell_steps(recipe, manifest.post_drop_mode)

        env_cfg = LiberoEnv(
            task=recipe.task,
            task_ids=[recipe.task_id],
            observation_height=256,
            observation_width=256,
            episode_length=4000,
        )
        envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
        suite = next(iter(envs.values()))
        vec = next(iter(suite.values()))
        config = FaultInjectionConfig(
            enabled=True,
            type="midair_drop",
            probability=0.0,
            t_min=0,
            t_max=10_000,
            object_name=request.object_name,
            basket_name=recipe.basket_name,
            seat_assist_enabled=False,
            waypoint_noise_m=0.0,
            recovery_action_noise_std=0.0,
            arm_posture_noise_deg=0.0,
            speed_multiplier_min=1.0,
            speed_multiplier_max=1.0,
            waypoint_blend_radius_m=recipe.simple_ik.waypoint_blend_radius_m,
            post_grasp_delay_steps=0,
            post_drop_dwell_steps=dwell_steps,
            post_drop_mode=manifest.post_drop_mode.value,
            drop_xy_band_min=None,
            drop_xy_band_max=None,
            seed=plan.drop_seed,
            log_path=request.output_dir / "fault_events.jsonl",
        )
        env: Any | None = None
        try:
            env = DropRecoveryEnvWrapper(vec, config)
            libero_env = unwrap_libero_env(vec)
            libero_env.init_state_id = int(plan.init_state_id)
            env.reset(seed=plan.episode_seed)
            rs_env = get_robosuite_env(env, 0)
            apply_serializable_layout(rs_env, request.shared_layout)
            drop_rng = np.random.default_rng(plan.drop_seed)
            fault = env.fault
            fault.set_recovery_motion_profile(
                0,
                speed_multiplier=plan.motion_profile.speed_multiplier,
                pickup_offset_xy_m=plan.motion_profile.recovery.pickup_offset_xy_m,
                transport_offset_m=plan.motion_profile.recovery.transport_offset_m,
                posture_bias_rad=plan.motion_profile.recovery.posture_bias_rad,
            )
            planner = _new_planner(
                rs_env,
                recipe,
                drop_recipe,
                plan.drop_seed,
                plan.motion_profile,
                request.object_name,
            )
            from lerobot.faults.recovery.fps import recording_stride

            control_freq = read_control_freq(rs_env)
            if int(round(control_freq)) != int(recipe.control_hz):
                raise ValueError(
                    f"sim control_hz={control_freq} does not match recipe.control_hz={recipe.control_hz}"
                )
            stride = recording_stride(control_freq, recipe.recording.dataset_fps)
            path_drop = recipe.simple_ik.path_drop
            assert path_drop is not None
            task = read_libero_task_description(vec)
            facts = run_simple_ik_episode_loop(
                env,
                rs_env,
                fault=fault,
                planner=planner,
                recipe_drop=drop_recipe,
                object_name=request.object_name,
                basket_name=recipe.basket_name,
                q=recipe.q,
                drop_rng=drop_rng,
                paired_plan=plan,
                gripper_settle_steps=config.gripper_settle_steps,
                episode_session=request.episode_session,
                recording_stride=stride,
                task=task,
                eligible_phases=path_drop.eligible_phases,
            )
            return EpisodeResult.from_run(
                request,
                success=facts.success,
                outcome=facts.outcome,
                drop_trigger=facts.drop_trigger,
                trigger_pose=facts.trigger_pose,
                actual_dwell_steps=facts.actual_dwell_steps,
                layout=request.shared_layout,
                motion_profile=asdict(plan.motion_profile),
            )
        finally:
            if env is not None:
                env.close()


def build_simple_ik_planner(
    rs_env: Any,
    *,
    object_name: str,
    basket_name: str,
    drop_recipe: DropRecipe,
    waypoint_blend_radius_m: float,
    seed: int,
    motion_profile: Any,
) -> SimpleIKRecoveryPlanner:
    """Construct a ``SimpleIKRecoveryPlanner`` from datagen motion profile settings."""
    leg = motion_profile.nominal
    planner = SimpleIKRecoveryPlanner(
        speed_multiplier=motion_profile.speed_multiplier,
        waypoint_noise_m=0.0,
        arm_posture_noise_rad=np.asarray(leg.posture_bias_rad),
        pickup_via_offset_xy_m=leg.pickup_offset_xy_m,
        transport_via_offset_m=leg.transport_offset_m,
        waypoint_blend_radius_m=waypoint_blend_radius_m,
        basket_keepout_m=keepout_m(
            drop_recipe.min_drop_distance_from_basket_m,
            drop_recipe.hard_keepout_floor_m,
        ),
        seed=seed,
    )
    eef_pos, eef_quat = get_eef_pose(rs_env)
    object_pose = get_object_pose(rs_env, object_name)
    destination = get_place_destination(
        rs_env,
        object_name,
        basket_name=basket_name,
    )
    planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=object_pose["pos"],
        object_axis=object_pose_orientation(rs_env, object_name)["axis"],
        destination_pos=destination,
        gripper_open=True,
    )
    return planner


def _new_planner(
    rs_env: Any,
    recipe: DropDatagenRecipe,
    drop_recipe: DropRecipe,
    seed: int,
    motion_profile: Any,
    object_name: str,
) -> SimpleIKRecoveryPlanner:
    return build_simple_ik_planner(
        rs_env,
        object_name=object_name,
        basket_name=recipe.basket_name,
        drop_recipe=drop_recipe,
        waypoint_blend_radius_m=recipe.simple_ik.waypoint_blend_radius_m,
        seed=seed,
        motion_profile=motion_profile,
    )
