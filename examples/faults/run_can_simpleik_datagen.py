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

"""Live can-only SimpleIK generation with exact run-level drop sampling."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from can_simpleik_viewer import DEFAULT_CAMERA_PX, DatagenViewer  # noqa: E402
from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.factory import make_env  # noqa: E402
from lerobot.faults.config import FaultInjectionConfig  # noqa: E402
from lerobot.faults.datagen.drop_timing import DropDecision, keepout_m  # noqa: E402
from lerobot.faults.datagen.events import DatagenEventLog  # noqa: E402
from lerobot.faults.datagen.layout import ObjectPose2d, sample_layout  # noqa: E402
from lerobot.faults.datagen.motion_profile import (  # noqa: E402
    EpisodeMotionProfile,
    MotionLegProfile,
    sample_episode_motion_profile,
)
from lerobot.faults.datagen.path_drop import (  # noqa: E402
    LIFT_PHASE,
    EligiblePath,
    PathTrigger,
    eligible_path,
    sample_path_drop,
)
from lerobot.faults.datagen.controllers.simple_ik import run_simple_ik_episode_loop  # noqa: E402
from lerobot.faults.datagen.recipe import DatagenRecipe, PostDropMode, load_recipe  # noqa: E402
from lerobot.faults.datagen.runtime import (  # noqa: E402
    movable_object_names,
    rotate_quat_about_world_z,
    stabilize_carry_action,
)
from lerobot.faults.recovery.planner import (  # noqa: E402
    CARRY_PHASES,
    SimpleIKRecoveryPlanner,
)
from lerobot.faults.sim.libero import (  # noqa: E402
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
from lerobot.faults.wrappers import DropRecoveryEnvWrapper  # noqa: E402

MAX_PLAN_STEPS = 800
POST_RECOVERY_SETTLE_STEPS = 40
LAYOUT_SETTLE_STEPS = 40
TABLE_XY_LIMIT_M = 0.7


@dataclass(frozen=True)
class ObjectPose:
    pos: np.ndarray
    quat_wxyz: np.ndarray


def _vec_env(envs: dict[str, Any]) -> Any:
    suite = next(iter(envs.values()))
    return next(iter(suite.values()))


def _terminated(result: tuple[Any, ...]) -> bool:
    if len(result) != 5:
        return False
    return bool(
        np.asarray(result[2]).reshape(-1).any()
        or np.asarray(result[3]).reshape(-1).any()
    )


def _render(env: Any) -> np.ndarray | None:
    try:
        raw = env.call("render") if hasattr(env, "call") else [env.envs[0].render()]
        return np.asarray(raw[0], dtype=np.uint8)
    except Exception:
        return None


def _pause_and_pump(viewer: DatagenViewer | None) -> bool:
    if viewer is None:
        return False
    viewer.pump()
    while viewer.pause and not viewer.quit_requested:
        viewer.pump()
        time.sleep(0.02)
    return viewer.quit_requested


def _hud(
    *,
    episode: int,
    step: int,
    phase: str,
    q: float,
    path: EligiblePath | None,
    decision: DropDecision | None,
    trigger: PathTrigger | None,
    speed_multiplier: float,
    basket_distance: float,
    grasped: bool,
) -> str:
    if decision is None:
        decision_text = "pending"
    elif decision.drop and trigger is not None:
        decision_text = f"drop@{trigger.segment_name}:{trigger.target_t:.2f}"
    else:
        decision_text = decision.reason
    eligible_text = "-" if path is None else f"{path.total:.2f}m"
    return (
        f"episode={episode} step={step} phase={phase} speed={speed_multiplier:.2f}\n"
        f"q={q:.3f} eligible={eligible_text} decision={decision_text}\n"
        f"can-basket={basket_distance:.3f}m grasped={grasped}"
    )


def _show(
    viewer: DatagenViewer | None,
    env: Any,
    *,
    episode: int,
    step: int,
    phase: str,
    q: float,
    path: EligiblePath | None,
    decision: DropDecision | None,
    trigger: PathTrigger | None,
    speed_multiplier: float,
    rs_env: Any,
    recipe: DatagenRecipe,
) -> None:
    if viewer is None:
        return
    frame = _render(env)
    if frame is None:
        return
    obj = get_object_pose(rs_env, recipe.object_name)["pos"]
    dest = get_place_destination(
        rs_env,
        recipe.object_name,
        basket_name=recipe.basket_name,
    )
    viewer.show(
        frame,
        _hud(
            episode=episode,
            step=step,
            phase=phase,
            q=q,
            path=path,
            decision=decision,
            trigger=trigger,
            speed_multiplier=speed_multiplier,
            basket_distance=float(np.linalg.norm(obj[:2] - dest[:2])),
            grasped=bool(is_object_grasped(rs_env, recipe.object_name)),
        ),
    )


def _new_planner(
    rs_env: Any,
    recipe: DatagenRecipe,
    seed: int,
    leg: MotionLegProfile,
    speed_multiplier: float,
) -> SimpleIKRecoveryPlanner:
    planner = SimpleIKRecoveryPlanner(
        speed_multiplier=speed_multiplier,
        waypoint_noise_m=0.0,
        arm_posture_noise_rad=np.asarray(leg.posture_bias_rad),
        pickup_via_offset_xy_m=leg.pickup_offset_xy_m,
        transport_via_offset_m=leg.transport_offset_m,
        waypoint_blend_radius_m=recipe.simple_ik.waypoint_blend_radius_m,
        basket_keepout_m=keepout_m(
            recipe.drop.min_drop_distance_from_basket_m,
            recipe.drop.hard_keepout_floor_m,
        ),
        seed=seed,
    )
    eef_pos, eef_quat = get_eef_pose(rs_env)
    object_pose = get_object_pose(rs_env, recipe.object_name)
    destination = get_place_destination(
        rs_env,
        recipe.object_name,
        basket_name=recipe.basket_name,
    )
    planner.plan(
        eef_pos=eef_pos,
        eef_quat=eef_quat,
        object_pos=object_pose["pos"],
        object_axis=object_pose_orientation(rs_env, recipe.object_name)["axis"],
        destination_pos=destination,
        gripper_open=True,
    )
    return planner


def _nominal_action(
    planner: SimpleIKRecoveryPlanner,
    rs_env: Any,
    recipe: DatagenRecipe,
    *,
    gripper_settle_steps: int,
) -> np.ndarray | None:
    eef_pos, _ = get_eef_pose(rs_env)
    object_pos = get_object_pose(rs_env, recipe.object_name)["pos"]
    action = planner.next_action(
        eef_pos=eef_pos,
        object_pos=object_pos,
        closing_axis=get_gripper_closing_axis(rs_env),
        object_axis=object_pose_orientation(rs_env, recipe.object_name)["axis"],
    )
    if planner.just_entered_close:
        force_close_gripper(rs_env, gripper_settle_steps=gripper_settle_steps)
    if planner.just_entered_open:
        force_open_gripper(rs_env, gripper_settle_steps=gripper_settle_steps)
    if action is not None and planner.phase_name in CARRY_PHASES:
        hold_gripper_closed(rs_env)
        action = stabilize_carry_action(action)
    return action


def _sample_pose_layout(
    rs_env: Any,
    recipe: DatagenRecipe,
    rng: np.random.Generator,
) -> dict[str, ObjectPose] | None:
    names = movable_object_names(
        rs_env,
        basket_name=recipe.basket_name,
        required_object_name=recipe.object_name,
    )
    reset = {name: get_object_pose(rs_env, name) for name in names}
    basket_xy = get_place_destination(
        rs_env,
        recipe.object_name,
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
        target_name=recipe.object_name,
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


def _run_episode(
    env: Any,
    rs_env: Any,
    recipe: DatagenRecipe,
    event_log: DatagenEventLog,
    viewer: DatagenViewer | None,
    *,
    episode: int,
    seed: int,
    q: float,
    drop_rng: np.random.Generator,
    motion_profile: EpisodeMotionProfile,
    gripper_settle_steps: int,
) -> bool:
    """Execute one legacy SimpleIK episode (immediate recovery after drop)."""
    fault = env.fault
    fault.set_recovery_motion_profile(
        0,
        speed_multiplier=motion_profile.speed_multiplier,
        pickup_offset_xy_m=motion_profile.recovery.pickup_offset_xy_m,
        transport_offset_m=motion_profile.recovery.transport_offset_m,
        posture_bias_rad=motion_profile.recovery.posture_bias_rad,
    )
    try:
        planner = _new_planner(
            rs_env,
            recipe,
            seed,
            motion_profile.nominal,
            motion_profile.speed_multiplier,
        )
    except Exception as exc:
        event_log.emit("plan_failed", f"planner failed: {exc}", episode=episode)
        return False
    event_log.emit(
        "plan_ok",
        "SimpleIK plan initialized",
        episode=episode,
        pickup_via=asdict(planner.pickup_via) if planner.pickup_via is not None else None,
    )
    return run_simple_ik_episode_loop(
        env,
        rs_env,
        fault=fault,
        planner=planner,
        recipe_drop=recipe.drop,
        object_name=recipe.object_name,
        basket_name=recipe.basket_name,
        q=q,
        drop_rng=drop_rng,
        post_drop_mode=PostDropMode.IMMEDIATE_IK,
        dwell_steps=0,
        viewer=viewer,
        gripper_settle_steps=gripper_settle_steps,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("outputs/can_simpleik_datagen"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--viewer-camera-px",
        type=int,
        default=DEFAULT_CAMERA_PX,
        help="Longest edge of the viewer camera panel. Display only; does not "
        "change the rendered observation size.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    recipe = load_recipe(args.recipe)
    args.output.mkdir(parents=True, exist_ok=True)
    event_log = DatagenEventLog(args.output / "events.jsonl", stdout=True)
    viewer = None if args.headless else DatagenViewer.try_create(args.viewer_camera_px)
    if viewer is not None:
        event_log.add_listener(viewer.append_log)
    event_log.emit("recipe", f"loaded {args.recipe}", q=recipe.q)

    env_cfg = LiberoEnv(
        task="libero_object",
        task_ids=[0],
        observation_height=256,
        observation_width=256,
        episode_length=4000,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = _vec_env(envs)
    config = FaultInjectionConfig(
        enabled=True,
        type="midair_drop",
        probability=0.0,
        t_min=0,
        t_max=10_000,
        object_name=recipe.object_name,
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
        seed=args.seed,
        log_path=args.output / "fault_events.jsonl",
    )
    env = DropRecoveryEnvWrapper(vec, config)
    libero_env = unwrap_libero_env(vec)
    baseline_init_state_id = int(libero_env.init_state_id)
    completed = 0

    try:
        for episode in range(args.episodes):
            if viewer is not None and viewer.quit_requested:
                break
            episode_seed = args.seed + episode
            layout_rng = np.random.default_rng(
                np.random.SeedSequence([episode_seed, 0x4C41594F])
            )
            drop_rng = np.random.default_rng(
                np.random.SeedSequence([episode_seed, 0x44524F50])
            )
            motion_profile = sample_episode_motion_profile(recipe.simple_ik, episode_seed)
            libero_env.init_state_id = baseline_init_state_id
            env.reset(seed=episode_seed)
            rs_env = get_robosuite_env(env, 0)
            layout = _sample_pose_layout(rs_env, recipe, layout_rng)
            if layout is None:
                event_log.emit("layout_failed", "no legal layout found", episode=episode)
                continue
            _apply_layout(rs_env, layout)
            event_log.emit(
                "layout_ok",
                "randomized movable-object layout applied",
                episode=episode,
                objects={name: pose.pos.tolist() for name, pose in layout.items()},
            )
            event_log.emit(
                "motion_profile",
                "sampled coherent trajectory profile",
                episode=episode,
                **asdict(motion_profile),
            )

            q_effective = (
                viewer.force_q
                if viewer is not None and viewer.force_q is not None
                else recipe.q
            )
            if viewer is not None:
                viewer.force_q = None

            if _run_episode(
                env,
                rs_env,
                recipe,
                event_log,
                viewer,
                episode=episode,
                seed=episode_seed,
                q=q_effective,
                drop_rng=drop_rng,
                motion_profile=motion_profile,
                gripper_settle_steps=config.gripper_settle_steps,
            ):
                completed += 1
    finally:
        if viewer is not None:
            viewer.destroy()
        env.close()
        event_log.close()
    print(f"Completed {completed}/{args.episodes} episode(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
