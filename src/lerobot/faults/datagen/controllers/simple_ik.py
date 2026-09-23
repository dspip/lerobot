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

"""SimpleIK controller adapter and post-drop transition helpers."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np

from lerobot.faults.datagen.drop_timing import DropDecision, keepout_m
from lerobot.faults.datagen.episode import EpisodeRequest, EpisodeResult
from lerobot.faults.datagen.events import DatagenEventLog
from lerobot.faults.datagen.layout import ObjectPose2d, sample_layout
from lerobot.faults.datagen.motion_profile import sample_episode_motion_profile
from lerobot.faults.datagen.path_drop import (
    EligiblePath,
    PathTrigger,
    eligible_path,
    sample_path_drop,
)
from lerobot.faults.datagen.recipe import (
    DropDatagenRecipe,
    DropRecipe,
    PostDropMode,
    effective_post_drop_dwell_steps,
    legacy_drop_recipe,
)
from lerobot.faults.datagen.runtime import (
    movable_object_names,
    rotate_quat_about_world_z,
    stabilize_carry_action,
)
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
    is_object_grasped,
    is_object_held_midair,
    is_object_in_basket,
    object_pose_orientation,
    set_object_pose,
    unwrap_libero_env,
)

MAX_PLAN_STEPS = 800
POST_RECOVERY_SETTLE_STEPS = 40
LAYOUT_SETTLE_STEPS = 40
TABLE_XY_LIMIT_M = 0.7


class SimpleIKPostDropAction(str, Enum):
    CONTINUE_NOMINAL = "continue_nominal"
    REQUEST_RECOVERY_NOW = "request_recovery_now"
    SKIP_RECOVERY = "skip_recovery"
    IN_RECOVERY = "in_recovery"


def decide_simple_ik_post_drop(
    *,
    mode: PostDropMode,
    dwell_steps: int,
    just_dropped: bool,
    dwell_elapsed: int,
    recovery_started: bool,
    regrasped: bool,
    in_basket: bool,
) -> SimpleIKPostDropAction:
    if recovery_started:
        return SimpleIKPostDropAction.IN_RECOVERY
    if regrasped or in_basket:
        return SimpleIKPostDropAction.SKIP_RECOVERY
    if just_dropped:
        if mode is PostDropMode.IMMEDIATE_IK:
            return SimpleIKPostDropAction.REQUEST_RECOVERY_NOW
        return SimpleIKPostDropAction.CONTINUE_NOMINAL
    if mode is PostDropMode.CONTINUE_THEN_IK and dwell_elapsed < dwell_steps:
        return SimpleIKPostDropAction.CONTINUE_NOMINAL
    if mode is PostDropMode.CONTINUE_THEN_IK:
        return SimpleIKPostDropAction.REQUEST_RECOVERY_NOW
    return SimpleIKPostDropAction.IN_RECOVERY


@dataclass(frozen=True)
class ObjectPose:
    pos: np.ndarray
    quat_wxyz: np.ndarray


def _pause_and_pump(viewer: Any | None) -> bool:
    if viewer is None:
        return False
    viewer.pump()
    while viewer.pause and not viewer.quit_requested:
        viewer.pump()
        time.sleep(0.02)
    return viewer.quit_requested


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
    fault: Any,
    planner: SimpleIKRecoveryPlanner,
    recipe_drop: DropRecipe,
    object_name: str,
    basket_name: str,
    q: float,
    drop_rng: np.random.Generator,
    post_drop_mode: PostDropMode,
    dwell_steps: int,
    path_trigger: PathTrigger | None = None,
    max_steps: int = MAX_PLAN_STEPS,
    gripper_settle_steps: int = 0,
    viewer: Any | None = None,
    on_step_end: Callable[..., None] | None = None,
) -> bool:
    """Execute one SimpleIK episode with optional post-drop dwell before recovery."""
    keepout = keepout_m(
        recipe_drop.min_drop_distance_from_basket_m,
        recipe_drop.hard_keepout_floor_m,
    )
    path: EligiblePath | None = None
    if path_trigger is not None:
        decision = DropDecision(drop=True, step=None, reason="injected")
        trigger = path_trigger
    else:
        decision = None
        trigger = None
    dropped = False
    recovery_started = False
    dwell_elapsed = 0
    just_dropped = False
    settle_left = POST_RECOVERY_SETTLE_STEPS
    regrasped = False
    landed_in_basket_logged = False

    for step in range(max_steps):
        if _pause_and_pump(viewer):
            return False
        just_dropped = False
        state = fault._states[0]
        if recovery_started:
            action = np.zeros((1, 7), dtype=np.float32)
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
                )
                decision, trigger = sample_path_drop(q, path, drop_rng)

            held_midair = bool(is_object_held_midair(rs_env, object_name))
            fire = (
                trigger is not None
                and not dropped
                and decision is not None
                and decision.drop
                and trigger.fires(
                    phase=phase,
                    object_xyz=obj,
                    carry_path=planner.carry_path,
                    held_midair=held_midair,
                )
            )
            if fire and distance < keepout:
                decision = DropDecision(False, None, "runtime_keepout")
                trigger = None
                fire = False

            if fire:
                if not fault.trigger_manual_drop(env, 0, reason="path_uniform"):
                    return False
                dropped = True
                just_dropped = True
                trigger = None
                post_action = decide_simple_ik_post_drop(
                    mode=post_drop_mode,
                    dwell_steps=dwell_steps,
                    just_dropped=True,
                    dwell_elapsed=dwell_elapsed,
                    recovery_started=recovery_started,
                    regrasped=regrasped,
                    in_basket=bool(
                        is_object_in_basket(
                            rs_env,
                            object_name,
                            basket_name=basket_name,
                            z_max=0.14,
                        )
                    ),
                )
                if post_action is SimpleIKPostDropAction.REQUEST_RECOVERY_NOW:
                    fault.request_recovery(env, 0, reason="path_uniform", consume_first_action=False)
                    recovery_started = True
                action = np.zeros((1, 7), dtype=np.float32)
            elif dropped and not recovery_started:
                regrasped = regrasped or bool(is_object_grasped(rs_env, object_name))
                in_basket = bool(
                    is_object_in_basket(
                        rs_env,
                        object_name,
                        basket_name=basket_name,
                        z_max=0.14,
                    )
                )
                post_action = decide_simple_ik_post_drop(
                    mode=post_drop_mode,
                    dwell_steps=dwell_steps,
                    just_dropped=False,
                    dwell_elapsed=dwell_elapsed,
                    recovery_started=recovery_started,
                    regrasped=regrasped,
                    in_basket=in_basket,
                )
                if post_action is SimpleIKPostDropAction.SKIP_RECOVERY:
                    return True
                if post_action is SimpleIKPostDropAction.REQUEST_RECOVERY_NOW:
                    fault.request_recovery(env, 0, reason="path_uniform", consume_first_action=False)
                    recovery_started = True
                    action = np.zeros((1, 7), dtype=np.float32)
                else:
                    nominal = _nominal_action(
                        planner,
                        rs_env,
                        object_name,
                        gripper_settle_steps=gripper_settle_steps,
                    )
                    if nominal is None or planner.done:
                        return True
                    action = nominal.reshape(1, 7)
                    dwell_elapsed += 1
            else:
                nominal = _nominal_action(
                    planner,
                    rs_env,
                    object_name,
                    gripper_settle_steps=gripper_settle_steps,
                )
                if nominal is None or planner.done:
                    return True
                action = nominal.reshape(1, 7)

        env.step(action)
        if on_step_end is not None:
            on_step_end(step=step)

        if recovery_started:
            regrasped = regrasped or bool(is_object_grasped(rs_env, object_name))
            if (
                not regrasped
                and not landed_in_basket_logged
                and is_object_in_basket(
                    rs_env,
                    object_name,
                    basket_name=basket_name,
                    z_max=0.14,
                )
            ):
                landed_in_basket_logged = True

            if state.planner is not None and state.planner.done:
                settle_left -= 1
                if settle_left <= 0:
                    return bool(
                        is_object_in_basket(
                            rs_env,
                            object_name,
                            basket_name=basket_name,
                            z_max=0.14,
                        )
                    )

    return False


class SimpleIKDatagenAdapter:
    """Runs live SimpleIK episodes for unified drop datagen."""

    def __init__(self, recipe: DropDatagenRecipe) -> None:
        self._recipe = recipe

    def run_episode(self, request: EpisodeRequest) -> EpisodeResult:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env
        from lerobot.faults.config import FaultInjectionConfig
        from lerobot.faults.wrappers import DropRecoveryEnvWrapper

        recipe = request.recipe
        manifest = request.manifest
        drop_recipe = legacy_drop_recipe(recipe)
        dwell_steps = effective_post_drop_dwell_steps(recipe, manifest.post_drop_mode)
        motion_profile = sample_episode_motion_profile(recipe.simple_ik, manifest.controller_seed)
        event_log = DatagenEventLog(request.output_dir / "events.jsonl", stdout=False)

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
            drop_xy_band_min=None,
            drop_xy_band_max=None,
            seed=manifest.controller_seed,
            log_path=request.output_dir / "fault_events.jsonl",
        )
        env = DropRecoveryEnvWrapper(vec, config)
        libero_env = unwrap_libero_env(vec)
        baseline_init_state_id = int(libero_env.init_state_id)

        try:
            layout_rng = np.random.default_rng(manifest.layout_seed)
            drop_rng = np.random.default_rng(manifest.drop_seed)
            libero_env.init_state_id = baseline_init_state_id
            env.reset(seed=manifest.episode_seed)
            rs_env = get_robosuite_env(env, 0)
            layout = _sample_pose_layout(
                rs_env,
                recipe,
                request.object_name,
                layout_rng,
            )
            if layout is None:
                return EpisodeResult.failed(
                    request,
                    outcome="layout_failed",
                    error="no legal layout found",
                )
            _apply_layout(rs_env, layout)
            fault = env.fault
            fault.set_recovery_motion_profile(
                0,
                speed_multiplier=motion_profile.speed_multiplier,
                pickup_offset_xy_m=motion_profile.recovery.pickup_offset_xy_m,
                transport_offset_m=motion_profile.recovery.transport_offset_m,
                posture_bias_rad=motion_profile.recovery.posture_bias_rad,
            )
            planner = _new_planner(
                rs_env,
                recipe,
                drop_recipe,
                manifest.controller_seed,
                motion_profile,
                request.object_name,
            )
            success = run_simple_ik_episode_loop(
                env,
                rs_env,
                fault=fault,
                planner=planner,
                recipe_drop=drop_recipe,
                object_name=request.object_name,
                basket_name=recipe.basket_name,
                q=recipe.q,
                drop_rng=drop_rng,
                post_drop_mode=manifest.post_drop_mode,
                dwell_steps=dwell_steps,
                gripper_settle_steps=config.gripper_settle_steps,
            )
            return EpisodeResult.ok(
                request,
                outcome="completed" if success else "recovery_failed",
                layout={name: pose.pos.tolist() for name, pose in layout.items()},
                motion_profile=asdict(motion_profile),
            )
        except Exception as exc:  # noqa: BLE001
            return EpisodeResult.failed(request, outcome="error", error=str(exc))
        finally:
            env.close()
            event_log.close()


def _new_planner(
    rs_env: Any,
    recipe: DropDatagenRecipe,
    drop_recipe: DropRecipe,
    seed: int,
    motion_profile: Any,
    object_name: str,
) -> SimpleIKRecoveryPlanner:
    leg = motion_profile.nominal
    planner = SimpleIKRecoveryPlanner(
        speed_multiplier=motion_profile.speed_multiplier,
        waypoint_noise_m=0.0,
        arm_posture_noise_rad=np.asarray(leg.posture_bias_rad),
        pickup_via_offset_xy_m=leg.pickup_offset_xy_m,
        transport_via_offset_m=leg.transport_offset_m,
        waypoint_blend_radius_m=recipe.simple_ik.waypoint_blend_radius_m,
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
        basket_name=recipe.basket_name,
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


def _sample_pose_layout(
    rs_env: Any,
    recipe: DropDatagenRecipe,
    object_name: str,
    rng: np.random.Generator,
) -> dict[str, ObjectPose] | None:
    names = movable_object_names(
        rs_env,
        basket_name=recipe.basket_name,
        required_object_name=object_name,
    )
    reset = {name: get_object_pose(rs_env, name) for name in names}
    basket_xy = get_place_destination(
        rs_env,
        object_name,
        basket_name=recipe.basket_name,
    )[:2]
    sampled = sample_layout(
        objects=[
            ObjectPose2d(name=name, xy=pose["pos"][:2].copy(), yaw_rad=0.0)
            for name, pose in reset.items()
        ],
        basket_xy=basket_xy,
        table_xy_lim=TABLE_XY_LIMIT_M,
        rng=rng,
        placement=recipe.placement,
        target_name=object_name,
    )
    if sampled is None:
        return None
    layout: dict[str, ObjectPose] = {}
    for pose2d in sampled:
        source = reset[pose2d.name]
        pos = source["pos"].copy()
        pos[:2] = pose2d.xy
        layout[pose2d.name] = ObjectPose(
            pos=pos,
            quat_wxyz=rotate_quat_about_world_z(
                source["quat_wxyz"],
                pose2d.yaw_rad,
            ),
        )
    return layout


def _apply_layout(rs_env: Any, layout: dict[str, ObjectPose]) -> None:
    for name, pose in layout.items():
        set_object_pose(
            rs_env,
            name,
            pos=pose.pos,
            quat_wxyz=pose.quat_wxyz,
            settle_steps=0,
        )
    for _ in range(LAYOUT_SETTLE_STEPS):
        rs_env.sim.step()
