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

#!/usr/bin/env python3
"""Record SmolVLA motion after a mid-air drop without IK recovery (diagnostic only)."""

from __future__ import annotations

import argparse
import json
import os
import types
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

OBJECT_NAME = "alphabet_soup_1"
T_MIN = 40
T_MAX = 400
POST_GRASP_DELAY_MIN = 20
POST_GRASP_DELAY_MAX = 60


def _parse_seeds(text: str) -> list[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def _info_flag(info: Any, key: str) -> bool | None:
    if not isinstance(info, dict) or key not in info:
        return None
    arr = np.asarray(info[key]).reshape(-1)
    if arr.size == 0:
        return None
    return bool(arr[0])


def _pre_drop_block(rs: Any, state: Any, config: Any) -> str:
    """Why ``_should_trigger`` would refuse on this step. Mirrors that function."""
    from lerobot.faults.sim.libero import (
        get_eef_pose,
        get_object_pose,
        is_object_grasped,
        object_basket_xy_distance,
    )

    if not (int(config.t_min) <= int(state.episode_step) <= int(config.t_max)):
        return "outside_time_window"
    if config.require_grasp and not is_object_grasped(rs, config.object_name):
        return "not_grasped"
    obj = np.asarray(get_object_pose(rs, config.object_name)["pos"], dtype=np.float64)
    if float(config.min_object_z) > 0.0 and float(obj[2]) < float(config.min_object_z):
        return "below_min_z"
    if config.require_grasp:
        eef_pos, _ = get_eef_pose(rs)
        if float(np.linalg.norm(np.asarray(eef_pos) - obj)) > 0.08:
            return "eef_far"
    basket_dist = object_basket_xy_distance(rs, config.object_name, basket_name=config.basket_name)
    min_dist = float(config.min_drop_distance_from_basket_m)
    if min_dist > 0.0 and basket_dist is not None:
        from lerobot.faults.recovery.midair_drop import HARD_BASKET_KEEPOUT_M

        if basket_dist < min(min_dist, float(HARD_BASKET_KEEPOUT_M)):
            return "hard_keepout"
        if basket_dist < min_dist:
            return "inside_min_drop"
    if state.eligible_since is None:
        return "not_eligible"
    band_lo = config.drop_xy_band_min
    band_hi = config.drop_xy_band_max
    if band_lo is not None and band_hi is not None:
        if basket_dist is None or not (float(band_lo) <= float(basket_dist) <= float(band_hi)):
            return "outside_xy_band"
        return "would_trigger"
    delay = int(config.post_grasp_delay_steps)
    if int(state.episode_step) < int(state.eligible_since) + delay:
        return "delay_pending"
    return "would_trigger"


def _disable_libero_autoreset(env: Any) -> None:
    """Keep the LIBERO scene after success/horizon when recovery is inactive."""
    from lerobot.faults.sim.libero import unwrap_libero_env

    target = env
    if hasattr(target, "envs"):
        target = target.envs[0]
    libero = unwrap_libero_env(target)

    def step_without_autoreset(action: Any):
        libero._ensure_env()
        assert libero._env is not None
        raw_obs, reward, done, info = libero._env.step(action)
        is_success = libero._env.check_success()
        terminated = bool(done or is_success)
        info = dict(info) if info is not None else {}
        info.update(
            {
                "task": getattr(libero, "task", None),
                "task_id": getattr(libero, "task_id", None),
                "done": done,
                "is_success": is_success,
            }
        )
        observation = libero._format_raw_obs(raw_obs)
        truncated = False
        return observation, reward, terminated, truncated, info

    libero.step = step_without_autoreset  # type: ignore[method-assign]


def _install_diagnostic_trigger_drop(fault: Any) -> None:
    from lerobot.faults.sim.libero import get_arm_qpos

    def diagnostic_trigger_drop(
        self: Any,
        env: Any,
        env_idx: int,
        state: Any,
        *,
        proposed_action: np.ndarray,
    ) -> np.ndarray:
        telemetry, rs_env = self._drop_object(env, env_idx, state)
        state.triggered = True
        executed = np.asarray(proposed_action, dtype=np.float32).copy()
        self._log_event(
            env_idx=env_idx,
            status="diagnostic_no_recovery",
            telemetry=telemetry,
            arm_q=get_arm_qpos(rs_env),
            proposed_action=proposed_action,
            executed_recovery_action=executed,
        )
        return executed

    fault._trigger_drop = types.MethodType(diagnostic_trigger_drop, fault)


def _gripper_open_from_action(action: np.ndarray) -> bool:
    # LIBERO / robosuite Panda: negative last dim opens (see pipeline hold at -1.0).
    return float(np.asarray(action).reshape(-1)[-1]) < 0.0


def _wrist_object_visible(rs: Any, object_name: str, *, width: int = 256, height: int = 256) -> bool | None:
    try:
        import mujoco

        from lerobot.faults.sim.libero import get_object_pose

        sim = rs.sim
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_eye_in_hand")
        if cam_id < 0:
            return None
        obj_pos = np.asarray(get_object_pose(rs, object_name)["pos"], dtype=np.float64)
        cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
        cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
        rel = obj_pos - cam_pos
        p_cam = cam_mat.T @ rel
        depth = -float(p_cam[2])
        if depth <= 1e-5:
            return False
        fovy_rad = float(model.cam_fovy[cam_id]) * np.pi / 180.0
        tan_half = np.tan(fovy_rad / 2.0)
        x_ndc = float(p_cam[0]) / depth
        y_ndc = float(p_cam[1]) / depth
        px = width * 0.5 * (1.0 + x_ndc / tan_half)
        py = height * 0.5 * (1.0 - y_ndc / tan_half)
        return bool(0.0 <= px < width and 0.0 <= py < height)
    except Exception:
        return None


def _build_frame(
    rs: Any,
    *,
    t_since_drop: int,
    executed_action: np.ndarray,
) -> dict[str, Any]:
    from lerobot.faults.sim.libero import (
        get_eef_pose,
        get_object_pose,
        is_object_grasped,
        is_object_in_basket,
    )

    obj = get_object_pose(rs, OBJECT_NAME)
    obj_pos = np.asarray(obj["pos"], dtype=np.float64)
    eef_pos, _ = get_eef_pose(rs)
    eef_pos = np.asarray(eef_pos, dtype=np.float64)
    xy = float(np.linalg.norm(eef_pos[:2] - obj_pos[:2]))
    above = float(eef_pos[2] - obj_pos[2])
    return {
        "t_since_drop": int(t_since_drop),
        "grasped": bool(is_object_grasped(rs, OBJECT_NAME)),
        "eef_obj_xy_m": xy,
        "eef_above_obj_m": above,
        "gripper_open": _gripper_open_from_action(executed_action),
        "in_basket_z014": bool(is_object_in_basket(rs, OBJECT_NAME, z_max=0.14)),
        "in_basket_z020": bool(is_object_in_basket(rs, OBJECT_NAME, z_max=0.20)),
        "obj_z": float(obj_pos[2]),
        "wrist_visible": _wrist_object_visible(rs, OBJECT_NAME),
    }


def _policy_action(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    env: Any,
    observation: Any,
    task: str,
) -> np.ndarray:
    import torch

    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    obs_dict = preprocess_observation(observation)
    try:
        obs_dict["task"] = list(env.call("task_description"))
    except Exception:
        try:
            obs_dict["task"] = list(env.call("task"))
        except Exception:
            obs_dict["task"] = [task]
    obs_dict = env_preprocessor(obs_dict)
    obs_dict = preprocessor(obs_dict)
    with torch.inference_mode():
        action = policy.select_action(obs_dict)
    action = postprocessor(action)
    action_transition = env_postprocessor({ACTION: action})
    action = action_transition[ACTION]
    action_numpy = np.asarray(action.to("cpu").numpy(), dtype=np.float32)
    if action_numpy.ndim == 1:
        action_numpy = action_numpy[None, ...]
    return action_numpy


def run_seed(
    *,
    seed: int,
    env: Any,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    max_steps_before_drop: int,
    post_drop_steps: int,
    task: str,
    reset_policy_on_drop: bool = False,
) -> dict[str, Any]:
    from lerobot.faults.sim.libero import get_object_pose, get_robosuite_env
    from lerobot.utils.random_utils import set_seed

    set_seed(seed)
    observation, _info = env.reset(seed=seed)
    _disable_libero_autoreset(env)
    _install_diagnostic_trigger_drop(env.fault)
    rs = get_robosuite_env(env)

    pre_drop_steps = 0
    triggered_at: int | None = None
    frames: list[dict[str, Any]] = []
    env_reset = False
    stop_reason: str | None = None
    is_success: bool | None = None
    last_block = "not_started"
    block_counts: dict[str, int] = {}

    st = env.fault._states[0]

    while pre_drop_steps < max_steps_before_drop and not st.triggered:
        action_numpy = _policy_action(
            policy,
            preprocessor,
            postprocessor,
            env_preprocessor,
            env_postprocessor,
            env,
            observation,
            task,
        )
        observation, _reward, terminated, truncated, info = env.step(action_numpy)
        pre_drop_steps += 1
        is_success = _info_flag(info, "is_success")
        if not st.triggered:
            last_block = _pre_drop_block(rs, st, env.fault.config)
            block_counts[last_block] = block_counts.get(last_block, 0) + 1
        if bool(np.asarray(terminated).any() or np.asarray(truncated).any()):
            if not st.triggered:
                stop_reason = "success_before_drop" if is_success else "terminated_before_drop"
                break

    if not st.triggered:
        return {
            "seed": seed,
            "dropped": False,
            "triggered_at": None,
            "pre_drop_steps": pre_drop_steps,
            "frames": [],
            "env_reset": False,
            "stop_reason": stop_reason or "max_steps_before_drop",
            "is_success": is_success,
            "last_block": last_block,
            "block_counts": block_counts,
        }

    triggered_at = pre_drop_steps - 1
    if reset_policy_on_drop:
        # Drop the queued carry chunk so the next action is planned over the can.
        policy.reset()
    executed = np.asarray(getattr(env, "last_executed_action", action_numpy), dtype=np.float32)
    if executed.ndim == 2:
        executed = executed[0]

    n_logged = 0
    seen_free = False
    prev_obj = np.asarray(get_object_pose(rs, OBJECT_NAME)["pos"], dtype=np.float64)
    while n_logged < post_drop_steps:
        if n_logged > 0:
            action_numpy = _policy_action(
                policy,
                preprocessor,
                postprocessor,
                env_preprocessor,
                env_postprocessor,
                env,
                observation,
                task,
            )
            observation, _reward, _terminated, _truncated, _info = env.step(action_numpy)
            executed = np.asarray(env.last_executed_action, dtype=np.float32)
            if executed.ndim == 2:
                executed = executed[0]
            obj_now = np.asarray(get_object_pose(rs, OBJECT_NAME)["pos"], dtype=np.float64)
            # Horizon termination clears fault state without moving the can.
            # A real reset teleports it. Only that ends the trace.
            if float(np.linalg.norm(obj_now - prev_obj)) > 0.30:
                env_reset = True
                stop_reason = "env_reset"
                break
            prev_obj = obj_now

        frame = _build_frame(rs, t_since_drop=n_logged, executed_action=executed)
        frames.append(frame)
        if not frame["grasped"]:
            seen_free = True
        elif seen_free:
            stop_reason = "regrasp"
            break
        n_logged += 1

    return {
        "seed": seed,
        "dropped": True,
        "triggered_at": triggered_at,
        "pre_drop_steps": pre_drop_steps,
        "frames": frames,
        "env_reset": env_reset,
        "stop_reason": stop_reason,
        "is_success": is_success,
        "last_block": last_block,
        "block_counts": block_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "post_drop_vla_diagnostic")
    parser.add_argument("--seeds", default="9000,9001,9002,9003,9004,9005")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--post-drop-steps", type=int, default=400)
    parser.add_argument("--max-steps-before-drop", type=int, default=500)
    parser.add_argument(
        "--post-grasp-delay-steps",
        type=int,
        default=None,
        help="Fixed carry delay. Default samples 20–60 like the training recipe.",
    )
    parser.add_argument("--t-min", type=int, default=T_MIN)
    parser.add_argument("--drop-xy-band-min", type=float, default=None)
    parser.add_argument("--drop-xy-band-max", type=float, default=None)
    parser.add_argument(
        "--reset-policy-on-drop",
        action="store_true",
        help="Clear the SmolVLA action queue on the drop step so the next chunk is replanned.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

    seeds = _parse_seeds(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.faults.config import FaultInjectionConfig
    from lerobot.faults.recovery.fps import (
        DEFAULT_LIBERO_CONTROL_FREQ,
        SMOLVLA_LIBERO_TARGET_FPS,
        configure_libero_control_freq,
    )
    from lerobot.faults.recovery.libero_hook import install_libero_control_freq_hook
    from lerobot.faults.recovery.post_drop_trace import aggregate_seed_summaries, summarize_post_drop_trace
    from lerobot.faults.recovery.recording_recipe import (
        sample_post_grasp_delay_steps,
        training_midair_drop_kwargs,
    )
    from lerobot.faults.sim.libero import unwrap_libero_env
    from lerobot.faults.wrappers import DropRecoveryEnvWrapper
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_path = "lerobot/smolvla_libero"
    control_freq_target = DEFAULT_LIBERO_CONTROL_FREQ
    policy_fps = SMOLVLA_LIBERO_TARGET_FPS
    configure_libero_control_freq(control_freq_target)
    install_libero_control_freq_hook(control_freq_target)

    import torch

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU.", flush=True)
        device = "cpu"

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    if hasattr(policy_cfg, "empty_cameras"):
        policy_cfg.empty_cameras = 1

    env_cfg = LiberoEnv(
        task="libero_object",
        task_ids=[0],
        control_mode="relative",
        camera_name_mapping={
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        },
    )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = envs["libero_object"][0]

    task = "pick up the alphabet soup and place it in the basket"
    per_seed_summary: dict[str, dict[str, Any]] = {}

    for seed in seeds:
        if args.post_grasp_delay_steps is None:
            delay_rng = np.random.default_rng(seed)
            delay_steps = sample_post_grasp_delay_steps(
                delay_rng, POST_GRASP_DELAY_MIN, POST_GRASP_DELAY_MAX
            )
        else:
            delay_steps = int(args.post_grasp_delay_steps)
        fault_kwargs = training_midair_drop_kwargs(
            t_min=int(args.t_min),
            t_max=T_MAX,
            seed=seed,
            post_grasp_delay_steps=delay_steps,
            log_path=args.output_dir / f"fault_events_seed_{seed}.jsonl",
            policy_fps=policy_fps,
            seat_assist_enabled=False,
            drop_xy_band_min=args.drop_xy_band_min,
            drop_xy_band_max=args.drop_xy_band_max,
        )
        fault_cfg = FaultInjectionConfig(**fault_kwargs)
        env = DropRecoveryEnvWrapper(vec, fault_cfg)

        libero_env = unwrap_libero_env(vec)
        init_states = getattr(libero_env, "_init_states", None)
        if getattr(libero_env, "init_states", False) and init_states is not None:
            n_init = len(init_states)
            if n_init > 0:
                libero_env.init_state_id = int(seed) % n_init

        result = run_seed(
            seed=seed,
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            max_steps_before_drop=args.max_steps_before_drop,
            post_drop_steps=args.post_drop_steps,
            task=task,
            reset_policy_on_drop=bool(args.reset_policy_on_drop),
        )

        summary = summarize_post_drop_trace(result["frames"])
        summary["seed"] = seed
        summary["dropped"] = result["dropped"]
        summary["triggered_at"] = result["triggered_at"]
        summary["pre_drop_steps"] = result["pre_drop_steps"]
        summary["env_reset"] = result["env_reset"]
        summary["stop_reason"] = result.get("stop_reason")
        summary["is_success"] = result.get("is_success")
        summary["last_block"] = result.get("last_block")
        summary["block_counts"] = result.get("block_counts")
        summary["post_grasp_delay_steps"] = delay_steps
        per_seed_summary[str(seed)] = summary

        seed_path = args.output_dir / f"seed_{seed}.jsonl"
        with seed_path.open("w", encoding="utf-8") as f:
            meta = {
                "type": "meta",
                "seed": seed,
                "pre_drop_steps": result["pre_drop_steps"],
                "dropped": result["dropped"],
                "triggered_at": result["triggered_at"],
                "env_reset": result["env_reset"],
                "stop_reason": result.get("stop_reason"),
            }
            f.write(json.dumps(meta) + "\n")
            for frame in result["frames"]:
                f.write(json.dumps(frame) + "\n")

        med_xy = summary.get("median_eef_obj_xy_m")
        med_above = summary.get("median_eef_above_obj_m")
        print(
            f"seed={seed} delay={delay_steps} triggered_at={result['triggered_at']} "
            f"stop={result.get('stop_reason')} success={result.get('is_success')} "
            f"last_block={result.get('last_block')} "
            f"regrasped={summary['regrasped']} "
            f"median_xy={med_xy} median_above={med_above}",
            flush=True,
        )

        policy.reset()

    aggregate = aggregate_seed_summaries(per_seed_summary)
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"seeds": per_seed_summary, "aggregate": aggregate}, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
