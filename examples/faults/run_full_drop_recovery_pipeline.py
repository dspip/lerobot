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
"""Full pipeline: SmolVLA → wait for grasp → mid-air drop → IK recovery → video.

Important:
  - LIBERO control stays at the default **20 Hz** (same as successful baseline eval).
  - Dataset / recovery planner target remains **10 FPS** (stride=2 recording).
  - Drop only fires when the object is actually grasped (no teleport hacks).
  - ``success`` means behavioral checks passed, not merely \"fault triggered\".
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _overlay_banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Draw a simple top banner (no OpenCV dependency)."""
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = max(28, h // 12)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.rectangle([0, 0, w, banner_h], fill=color)
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        return np.asarray(img)
    except Exception:
        out[:banner_h] = (out[:banner_h].astype(np.float32) * 0.35).astype(np.uint8)
        return out


def _write_gif(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / max(fps, 1))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from lerobot.utils.io_utils import write_video

        write_video(str(path), np.stack(frames), fps)
        return
    except Exception:
        pass
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
    except Exception as exc:
        raise RuntimeError(f"Could not write mp4: {exc}") from exc


def _in_view(pos: np.ndarray, *, z_min: float = 0.0, z_max: float = 0.8, xy_lim: float = 0.7) -> bool:
    """Rough workspace visibility check (object not under table / blasted away)."""
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
    return (z_min <= z <= z_max) and (abs(x) <= xy_lim) and (abs(y) <= xy_lim)


def _render_looking_into_basket(rs: Any, size: int = 384) -> np.ndarray | None:
    """Render steeply down onto the basket using a free camera.

    Every camera baked into the scene views the basket from the side, and the
    basket rim (~0.15 m) is taller than a can standing on its floor (top ~0.10 m),
    so a correctly placed can is fully occluded and the basket reads as empty.
    Only a view from above can show the outcome.
    """
    try:
        import mujoco

        from lerobot.faults.sim.libero import DEFAULT_BASKET_NAME, _body_xpos

        model = rs.sim.model._model
        data = rs.sim.data._data
        basket = _body_xpos(rs, DEFAULT_BASKET_NAME)
        if basket is None:
            return None

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [float(basket[0]), float(basket[1]), float(basket[2])]
        cam.distance = 0.55
        cam.azimuth = 90.0
        cam.elevation = -75.0  # steep, but not exactly top-down, to keep depth cues

        with mujoco.Renderer(model, height=size, width=size) as renderer:
            renderer.update_scene(data, camera=cam)
            return np.asarray(renderer.render(), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] basket top-view render failed: {exc}", flush=True)
        return None


def _final_proof_shot(rs: Any, out_path: Path) -> str | None:
    """Save a multi-camera still of the end state.

    The recording camera looks at the basket side-on, so a can resting on the
    basket floor is hidden behind the near wall — indistinguishable from an empty
    basket. Extra viewpoints make the outcome checkable instead of inferred from
    coordinates.
    """
    shots: list[np.ndarray] = []
    top = _render_looking_into_basket(rs)
    if top is not None:
        shots.append(top)
    for camera in ("agentview", "frontview", "birdview", "sideview"):
        try:
            img = rs.sim.render(height=384, width=384, camera_name=camera)
        except Exception:
            continue
        shots.append(np.asarray(img, dtype=np.uint8)[::-1])
    if not shots:
        return None
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, np.concatenate(shots, axis=1))
    return str(out_path)


def _settle_after_release(
    env: Any,
    rs: Any,
    action_template: np.ndarray,
    last_frame: np.ndarray | None,
    *,
    max_steps: int = 40,
) -> tuple[list[np.ndarray], list[str]]:
    """Hold the arm still with the gripper open until the released can settles.

    Returns the rendered frames so the video shows the landing rather than cutting
    on a can frozen in mid-air. Stops early once the can is resting (z stops
    changing), and bails out if the vec env auto-resets, since a reset would
    teleport the can and fake either outcome.
    """
    from lerobot.faults.sim.libero import get_object_pose

    frames: list[np.ndarray] = []
    phases: list[str] = []
    hold = np.zeros_like(np.asarray(action_template, dtype=np.float32))
    if hold.ndim == 1:
        hold = hold[None, ...]
    hold[..., -1] = -1.0  # keep the gripper commanded open

    prev_z = float(get_object_pose(rs, "alphabet_soup_1")["pos"][2])
    still = 0
    for _ in range(max_steps):
        try:
            _obs, _r, terminated, truncated, _info = env.step(hold)
        except Exception as exc:  # noqa: BLE001
            print(f"[pipeline] settle step failed: {exc}", flush=True)
            break
        z = float(get_object_pose(rs, "alphabet_soup_1")["pos"][2])
        try:
            raw = env.call("render") if hasattr(env, "call") else [env.envs[0].render()]
            frames.append(_overlay_banner(np.asarray(raw[0]), "PHASE: SETTLE", (40, 90, 160)))
            phases.append("recovery")
        except Exception:
            if last_frame is not None:
                frames.append(last_frame.copy())
                phases.append("recovery")
        still = still + 1 if abs(z - prev_z) < 1e-4 else 0
        prev_z = z
        if still >= 5:
            break
        if bool(np.asarray(terminated).any() or np.asarray(truncated).any()):
            print("[pipeline] settle stopped: env reported done", flush=True)
            break
    print(f"[pipeline] settled after release: object_z={prev_z:.3f}", flush=True)
    return frames, phases


def _carry_evidence(
    object_traj: list[list[float]],
    grasp_flags: list[bool],
    triggered_at: int | None,
) -> dict[str, Any]:
    """Summarise post-drop grasp spans so "carried" can be told from "shoved".

    A can that ends up in the basket proves nothing on its own: the gripper can
    bulldoze it across the table, or knock it over the rim. A genuine recovery has
    at least one span where the object is held *and* is lifted clear of the table
    while it travels. Reports the best such span.
    """
    if triggered_at is None or not object_traj:
        return {"spans": 0, "best": None}
    traj = np.asarray(object_traj, dtype=float)
    start = min(int(triggered_at) + 1, len(traj))
    flags = [bool(g) for g in grasp_flags[:len(traj)]]

    spans: list[dict[str, Any]] = []
    i = start
    while i < len(flags):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(flags) and flags[j + 1]:
            j += 1
        seg = traj[i : j + 1]
        spans.append(
            {
                "start_step": i,
                "end_step": j,
                "n_steps": int(j - i + 1),
                "displacement_m": float(np.linalg.norm(seg[-1] - seg[0])),
                "max_z": float(seg[:, 2].max()),
                "lift_m": float(seg[:, 2].max() - seg[0, 2]),
            }
        )
        i = j + 1

    if not spans:
        return {"spans": 0, "best": None}
    best = max(spans, key=lambda s: (s["lift_m"], s["displacement_m"]))
    return {"spans": len(spans), "best": best, "all": spans}


def run_pipeline(
    output_dir: Path,
    *,
    policy_path: str = "lerobot/smolvla_libero",
    device: str = "cuda",
    t_min: int = 40,
    t_max: int = 400,
    max_steps: int = 500,
    seed: int = 1000,
    recovery_horizon: int = 450,
    fault_overrides: dict | None = None,
    post_drop_hook: Any | None = None,
) -> dict:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION
    from lerobot.utils.random_utils import set_seed

    from lerobot.faults.config import FaultInjectionConfig
    from lerobot.faults.recovery.fps import (
        DEFAULT_LIBERO_CONTROL_FREQ,
        SMOLVLA_LIBERO_TARGET_FPS,
        assert_control_rate_aligned,
        assert_dataset_fps,
        configure_libero_control_freq,
        recording_stride,
    )
    from lerobot.faults.recovery.libero_hook import install_libero_control_freq_hook
    from lerobot.faults.sim.libero import (
        get_eef_pose,
        get_gripper_closing_axis,
        get_object_pose,
        get_place_destination,
        get_robosuite_env,
        is_object_grasped,
        is_object_held_midair,
        is_object_in_basket,
        object_basket_xy_distance,
        object_pose_orientation,
        read_control_freq,
        read_model_timestep,
        seat_object_in_basket_if_above,
    )
    from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger
    from lerobot.faults.wrappers import DropRecoveryEnvWrapper

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir()

    set_seed(seed)

    # Match successful baseline eval: LIBERO default 20 Hz control.
    # Record dataset / planner at SmolVLA's 10 FPS via stride=2.
    control_freq_target = DEFAULT_LIBERO_CONTROL_FREQ
    policy_fps = SMOLVLA_LIBERO_TARGET_FPS
    configure_libero_control_freq(control_freq_target)
    hook_ok = install_libero_control_freq_hook(control_freq_target)

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU (slow).", flush=True)
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
    policy.reset()  # required by SmolVLA action-chunk queue (matches lerobot_eval.rollout)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = envs["libero_object"][0]
    fault_kwargs: dict = {
        "enabled": True,
        "type": "midair_drop",
        "t_min": t_min,
        "t_max": t_max,
        "require_grasp": True,  # ONLY after real grasp — no teleport
        "min_object_z": 0.12,  # must be lifted mid-air, not table-contact
        # Downward-biased mild impulse: visible fall, stay in camera.
        "impulse_lin_std": 0.05,
        "impulse_ang_std": 0.05,
        "impulse_lin_bias": (0.0, 0.0, -0.55),
        "post_grasp_delay_steps": 0,
        "settle_steps": 100,  # MuJoCo substeps (~0.2s) so the fall is visible
        "gripper_settle_steps": 25,
        "recovery_fps": policy_fps,
        "seed": seed,
        "log_path": output_dir / "fault_events.jsonl",
    }
    if fault_overrides:
        fault_kwargs.update(fault_overrides)
    fault_cfg = FaultInjectionConfig(**fault_kwargs)
    env = DropRecoveryEnvWrapper(vec, fault_cfg)

    ds_logger = FaultRecoveryDatasetLogger(
        root=output_dir / "dataset",
        repo_id="local/full_pipeline_drop_recovery",
        policy_fps=policy_fps,
    )

    observation, info = env.reset(seed=seed)
    rs = get_robosuite_env(env)
    control_freq = read_control_freq(rs)
    model_timestep = read_model_timestep(rs)
    assert_control_rate_aligned(control_freq, model_timestep, policy_fps)
    stride = recording_stride(control_freq, policy_fps)

    frames: list[np.ndarray] = []
    phases: list[str] = []
    object_traj: list[list[float]] = []
    grasp_flags: list[bool] = []
    triggered_at: int | None = None
    first_grasp_step: int | None = None
    regrasped_after_drop = False
    pre_drop_pose: list[float] | None = None
    post_drop_pose: list[float] | None = None
    closing_axis_at_recovery: list[float] | None = None
    yaw_error_trace: list[dict[str, float | int | str | None]] = []
    task = "pick up the alphabet soup and place it in the basket"

    try:
        raw = env.call("render") if hasattr(env, "call") else [env.envs[0].render()]
        frame0 = np.asarray(raw[0])
    except Exception:
        frame0 = np.zeros((256, 256, 3), dtype=np.uint8)
    frames.append(_overlay_banner(frame0, "PHASE: VLA (SmolVLA)", (30, 90, 200)))
    phases.append("vla")

    import torch

    print(
        f"[pipeline] control_freq={control_freq} (target={control_freq_target}), "
        f"dataset_fps={policy_fps}, stride={stride}, require_grasp=True, "
        f"t_window=[{t_min},{t_max}]",
        flush=True,
    )

    for step in range(max_steps):
        st = env.fault._states[0]
        obj_z = float(get_object_pose(rs, "alphabet_soup_1")["pos"][2])
        grasped_now = bool(is_object_grasped(rs, "alphabet_soup_1"))
        midair_grasp = bool(
            is_object_held_midair(rs, "alphabet_soup_1", min_object_z=0.12, max_eef_distance=0.08)
        )
        if midair_grasp and first_grasp_step is None:
            first_grasp_step = step
            print(f"[pipeline] MID-AIR GRASP (soup near EEF) at step={step} z={obj_z:.3f}", flush=True)
        if (
            triggered_at is not None
            and midair_grasp
            and st.recovery_active
            and not regrasped_after_drop
        ):
            regrasped_after_drop = True
            print(
                f"[pipeline] REGRASP after drop at step={step} z={obj_z:.3f}",
                flush=True,
            )
        elif grasped_now and first_grasp_step is None and step % 20 == 0:
            print(
                f"[pipeline] contact/false grasp? step={step} z={obj_z:.3f} "
                f"(waiting for soup held mid-air near EEF)",
                flush=True,
            )

        if st.recovery_active:
            phase = "recovery"
            wp = getattr(st.planner, "phase_name", "?") if st.planner is not None else "?"
            retries = getattr(st, "grasp_retries", 0)
            yaw_err = getattr(st.planner, "yaw_error", None) if st.planner is not None else None
            lying = getattr(st.planner, "object_lying", None) if st.planner is not None else None
            banner = f"PHASE: RECOVERY [{wp}] z={obj_z:.2f} r={retries}"
            color = (40, 160, 70)
            if yaw_err is not None:
                yaw_error_trace.append(
                    {
                        "step": step,
                        "phase": str(wp),
                        "yaw_error": float(yaw_err),
                        "object_lying": bool(lying) if lying is not None else None,
                    }
                )
                print(
                    f"[pipeline] yaw_error step={step} phase={wp} "
                    f"yaw_error={float(yaw_err):+.4f} lying={lying}",
                    flush=True,
                )
            if step % 25 == 0:
                print(
                    f"[pipeline] recovery step={step} phase={wp} obj_z={obj_z:.3f} "
                    f"grasped={grasped_now} retries={retries}",
                    flush=True,
                )
        elif st.triggered or (triggered_at is not None and step >= triggered_at):
            phase = "drop"
            banner = "PHASE: DROP (midair_drop)"
            color = (200, 50, 50)
        else:
            phase = "vla"
            g = "GRASPED" if grasped_now else "seeking"
            banner = f"PHASE: VLA (SmolVLA) [{g}]"
            color = (30, 90, 200)

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

        pose_before = get_object_pose(rs, "alphabet_soup_1")["pos"].astype(float).copy()
        observation, reward, terminated, truncated, info = env.step(action_numpy)

        # Post-step regrasp check (catches lift on the same step the planner finishes).
        if triggered_at is not None and step > triggered_at and env.fault._states[0].recovery_active:
            try:
                if is_object_held_midair(
                    rs, "alphabet_soup_1", min_object_z=0.12, max_eef_distance=0.08
                ):
                    if not regrasped_after_drop:
                        z_now = float(get_object_pose(rs, "alphabet_soup_1")["pos"][2])
                        print(
                            f"[pipeline] REGRASP after drop at step={step} z={z_now:.3f}",
                            flush=True,
                        )
                    regrasped_after_drop = True
            except Exception:
                pass

        if env.fault._states[0].triggered and triggered_at is None:
            triggered_at = step
            pre_drop_pose = pose_before.tolist()
            # Test hook: lets a harness force a specific landing pose (e.g. the can
            # tipped onto its side) so recovery can be probed deterministically.
            if post_drop_hook is not None:
                post_drop_hook(rs)
                # Drop fault already planned from the pre-hook pose; rebuild waypoints
                # so approach / grasp heights match the staged tipped can.
                st_hook = env.fault._states[0]
                if st_hook.planner is not None:
                    eef_pos, eef_quat = get_eef_pose(rs)
                    object_pose = get_object_pose(rs, "alphabet_soup_1")
                    dest = (
                        st_hook.destination_pos
                        if st_hook.destination_pos is not None
                        else get_place_destination(rs, "alphabet_soup_1")
                    )
                    st_hook.planner.plan(
                        eef_pos=eef_pos,
                        eef_quat=eef_quat,
                        object_pos=object_pose["pos"],
                        object_axis=object_pose_orientation(rs, "alphabet_soup_1")["axis"],
                        destination_pos=dest,
                        gripper_open=True,
                    )
                    print(
                        f"[pipeline] replan after post_drop_hook: "
                        f"lying={getattr(st_hook.planner, 'object_lying', None)} "
                        f"target_yaw={getattr(st_hook.planner, 'target_closing_yaw', None)}",
                        flush=True,
                    )
            axis = get_gripper_closing_axis(rs)
            if axis is not None:
                closing_axis_at_recovery = [float(x) for x in axis]
                axis_n = float(np.linalg.norm(axis))
                print(
                    f"[pipeline] gripper_closing_axis={closing_axis_at_recovery} "
                    f"norm={axis_n:.4f} z={float(axis[2]):+.4f}",
                    flush=True,
                )
            else:
                print("[pipeline] gripper_closing_axis=None (finger geoms not found)", flush=True)
            post_drop_pose = get_object_pose(rs, "alphabet_soup_1")["pos"].astype(float).tolist()
            print(
                f"[pipeline] midair_drop TRIGGERED at step={step} "
                f"(first_grasp={first_grasp_step}, pre_z={pre_drop_pose[2]:.3f}, "
                f"post_z={post_drop_pose[2]:.3f})",
                flush=True,
            )

        try:
            raw = env.call("render") if hasattr(env, "call") else [env.envs[0].render()]
            fr = np.asarray(raw[0])
        except Exception:
            fr = frames[-1].copy()
        frames.append(_overlay_banner(fr, f"{banner}  step={step}", color))
        phases.append(phase)

        pose = get_object_pose(rs, "alphabet_soup_1")
        object_traj.append(pose["pos"].astype(float).tolist())
        grasp_flags.append(grasped_now)

        # Always log the drop injection frame (loss_mask=0); otherwise stride may skip it.
        is_drop_frame = bool(env.fault._states[0].drop_injection_step)
        if step % stride == 0 or is_drop_frame:
            executed = env.last_executed_action
            if executed is None:
                executed = action_numpy
            if np.asarray(executed).ndim == 2:
                executed = np.asarray(executed)[0]
            mask = float(env.loss_mask())
            try:
                from lerobot.faults.recovery.dataset_logger import libero_obs_to_frame

                frame = libero_obs_to_frame(preprocess_observation(observation))
                ds_logger.log_step(frame, executed, task, mask, phase=phase)
            except Exception as exc:
                print(f"[pipeline] dataset log warning at step={step}: {exc}", flush=True)

        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        planner_done = bool(
            env.fault._states[0].planner is not None and env.fault._states[0].planner.done
        )
        if triggered_at is not None and step >= triggered_at + recovery_horizon:
            break
        if triggered_at is not None and planner_done and phases.count("recovery") > 5:
            print(f"[pipeline] recovery planner finished at step={step}", flush=True)
            # The can is still in the air the instant the planner reports done —
            # it has to fall the ~13 cm from the release hover into the basket.
            # Measuring here scores a successful place as a miss, so hold the arm
            # still (zero delta, gripper open) and let physics finish.
            settle_frames, settle_phases = _settle_after_release(
                env, rs, action_numpy, frames[-1] if frames else None
            )
            frames.extend(settle_frames)
            phases.extend(settle_phases)
            break
        if done and not env.fault._states[0].triggered:
            print(f"[pipeline] episode ended at step={step} before drop (no grasp in window?)", flush=True)
            break

    try:
        ds_logger.end_episode()
        ds_logger.finalize()
        assert_dataset_fps(ds_logger.dataset.fps, policy_fps)
        loss_counts = dict(ds_logger.loss_mask_counts)
    except Exception as exc:
        print(f"[pipeline] dataset finalize warning: {exc}", flush=True)
        loss_counts = {}

    basket_place_ok = False
    seat_assisted = False
    basket_dest = None
    final_object_pos = None
    planned_dest = None
    try:
        st0 = env.fault._states[0]
        # Prefer proof captured at seat time via after_physics_step.
        seat_assisted = bool(getattr(st0, "seat_assisted", False))
        # Strict in-basket (z_max=0.14). Prefer live sim; fall back to fault proof.
        basket_place_ok = bool(is_object_in_basket(rs, "alphabet_soup_1", z_max=0.14))
        final_object_pos = get_object_pose(rs, "alphabet_soup_1")["pos"].astype(float).tolist()
        basket_dest = get_place_destination(rs, "alphabet_soup_1").astype(float).tolist()
        if (
            not basket_place_ok
            and st0.place_succeeded
            and st0.proof_object_pos is not None
            and st0.proof_basket_pos is not None
        ):
            # Apply the same geometry as is_object_in_basket. Comparing an
            # absolute world height against 0.14 accepts a can resting anywhere
            # on the table (z ~ 0.05), which would score a miss as a placement.
            proof_obj = np.asarray(st0.proof_object_pos, dtype=float)
            proof_basket = np.asarray(st0.proof_basket_pos, dtype=float)
            proof_xy = float(np.linalg.norm(proof_obj[:2] - proof_basket[:2]))
            proof_z = float(proof_obj[2] - proof_basket[2])
            basket_place_ok = proof_xy <= 0.07 and 0.0 <= proof_z <= 0.14
            final_object_pos = proof_obj.tolist()
            if st0.proof_basket_pos is not None:
                basket_dest = np.asarray(st0.proof_basket_pos, dtype=float).tolist()
        print(
            f"[pipeline] place check: in_basket={basket_place_ok} "
            f"seat_assisted={seat_assisted} obj={final_object_pos} basket~={basket_dest}",
            flush=True,
        )
        if not basket_place_ok:
            # Diagnostic-only post-hoc seat — never counts as success.
            seated = seat_object_in_basket_if_above(rs, "alphabet_soup_1")
            if seated:
                print(
                    "[pipeline] post-hoc seat: diagnostic only (not counted as success)",
                    flush=True,
                )
                for _ in range(40):
                    rs.sim.step()
                final_object_pos = get_object_pose(rs, "alphabet_soup_1")["pos"].astype(float).tolist()
        if st0.destination_pos is not None:
            planned_dest = np.asarray(st0.destination_pos, dtype=float).tolist()
        # Render proof frames when place succeeded behaviorally.
        if basket_place_ok:
            for _ in range(6):
                try:
                    raw = env.call("render") if hasattr(env, "call") else [env.envs[0].render()]
                    frames.append(
                        _overlay_banner(np.asarray(raw[0]), "PHASE: IN BASKET", (20, 140, 60))
                    )
                    phases.append("recovery")
                except Exception:
                    break
    except Exception as exc:
        print(f"[pipeline] place check warning: {exc}", flush=True)

    proof_shot = None
    try:
        proof_shot = _final_proof_shot(rs, output_dir / "final_state_multicam.png")
        print(f"[pipeline] final multi-camera proof: {proof_shot}", flush=True)
    except Exception as exc:
        print(f"[pipeline] proof shot warning: {exc}", flush=True)

    env.close()

    gif_path = videos_dir / "full_pipeline.gif"
    mp4_path = videos_dir / "full_pipeline.mp4"
    # Keep GIF manageable; prefer every other frame when long.
    gif_stride = max(1, len(frames) // 100)
    _write_gif(gif_path, frames[::gif_stride], fps=8)
    _write_mp4(mp4_path, frames, fps=int(round(control_freq)))

    demo_dir = REPO_ROOT / "docs" / "assets" / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(gif_path, demo_dir / "full_pipeline_drop_recovery.gif")

    # Behavioral success criteria (honest).
    grasped_before_drop = first_grasp_step is not None and (
        triggered_at is not None and first_grasp_step <= triggered_at
    )
    # Prefer "still on the table after drop" over end-of-episode (recovery may shove objects).
    object_visible_after_drop = True
    drop_moved = False
    was_midair_at_drop = bool(pre_drop_pose is not None and pre_drop_pose[2] >= 0.12)
    if pre_drop_pose is not None and triggered_at is not None and object_traj:
        later_idx = min(triggered_at + 10, len(object_traj) - 1)
        later = np.asarray(object_traj[later_idx], dtype=float)
        pre = np.asarray(pre_drop_pose, dtype=float)
        delta = float(np.linalg.norm(later - pre))
        z_drop = float(pre[2] - later[2])
        # Must actually fall (z decreases). Lateral-only motion is NOT a drop.
        drop_moved = z_drop > 0.03 and delta < 0.45
        object_visible_after_drop = _in_view(later)
        summary_drop_metrics = {
            "delta": delta,
            "z_drop": z_drop,
            "later_idx": later_idx,
            "later_pos": later.tolist(),
        }
    else:
        summary_drop_metrics = {}

    behavioral_success = bool(
        grasped_before_drop
        and was_midair_at_drop
        and triggered_at is not None
        and drop_moved
        and object_visible_after_drop
        and phases.count("recovery") > 10
        and basket_place_ok
    )

    # Yaw-convention diagnostic: |yaw_error| should shrink when side-grasp is on.
    yaw_abs = [abs(float(t["yaw_error"])) for t in yaw_error_trace if t.get("yaw_error") is not None]
    yaw_diag: dict[str, Any] = {
        "n_samples": len(yaw_abs),
        "first_abs": float(yaw_abs[0]) if yaw_abs else None,
        "last_abs": float(yaw_abs[-1]) if yaw_abs else None,
        "min_abs": float(min(yaw_abs)) if yaw_abs else None,
        "max_abs": float(max(yaw_abs)) if yaw_abs else None,
        "converged": bool(yaw_abs and yaw_abs[-1] < yaw_abs[0] and yaw_abs[-1] <= 0.15)
        if yaw_abs
        else None,
    }

    summary = {
        "success": behavioral_success,
        "checks": {
            "grasped_before_drop": grasped_before_drop,
            "was_midair_at_drop": was_midair_at_drop,
            "fault_triggered": triggered_at is not None,
            "drop_moved_object": drop_moved,
            "object_in_view_after_drop": object_visible_after_drop,
            "recovery_steps": phases.count("recovery"),
            "regrasped_after_drop": regrasped_after_drop,
            "object_in_basket": basket_place_ok,
            "seat_assisted": seat_assisted,
        },
        "regrasped_after_drop": regrasped_after_drop,
        "closing_axis_at_recovery": closing_axis_at_recovery,
        "yaw_error_trace": yaw_error_trace,
        "yaw_diag": yaw_diag,
        "final_object_pos": final_object_pos,
        "basket_destination": basket_dest,
        "policy_path": policy_path,
        "device": device,
        "control_freq": control_freq,
        "control_freq_hook": hook_ok,
        "policy_fps": policy_fps,
        "recording_stride": stride,
        "t_min": t_min,
        "t_max": t_max,
        "seed": seed,
        "fault_config": {
            "post_grasp_delay_steps": int(fault_cfg.post_grasp_delay_steps),
            "impulse_lin_std": float(fault_cfg.impulse_lin_std),
            "impulse_ang_std": float(fault_cfg.impulse_ang_std),
            "impulse_lin_bias": list(fault_cfg.impulse_lin_bias),
            "waypoint_noise_m": float(fault_cfg.waypoint_noise_m),
            "recovery_action_noise_std": float(fault_cfg.recovery_action_noise_std),
            "speed_multiplier_min": float(fault_cfg.speed_multiplier_min),
            "speed_multiplier_max": float(fault_cfg.speed_multiplier_max),
            "arm_posture_noise_deg": float(fault_cfg.arm_posture_noise_deg),
        },
        "first_grasp_step": first_grasp_step,
        "triggered_at": triggered_at,
        "object_traj": object_traj,
        "grasp_flags": [bool(g) for g in grasp_flags],
        "carry_evidence": _carry_evidence(object_traj, grasp_flags, triggered_at),
        "yaw_error_abs_final": yaw_diag.get("last_abs"),
        "yaw_error_abs_min": yaw_diag.get("min_abs"),
        "pre_drop_pose": pre_drop_pose,
        "post_drop_pose": post_drop_pose,
        "drop_metrics": summary_drop_metrics,
        "num_frames_video": len(frames),
        "phase_counts": {p: phases.count(p) for p in sorted(set(phases))},
        "loss_mask_counts": loss_counts,
        "object_pos_delta": float(
            np.linalg.norm(np.array(object_traj[-1]) - np.array(object_traj[triggered_at]))
        )
        if triggered_at is not None and object_traj
        else None,
        "fault_events": str(output_dir / "fault_events.jsonl"),
        "video_mp4": str(mp4_path),
        "video_gif": str(gif_path),
        "final_state_multicam": proof_shot,
        "dataset_dir": str(output_dir / "dataset"),
        "planned_destination": planned_dest,
        "seat_assisted": seat_assisted,
        "note": (
            "control_freq=20 (LIBERO default, same as baseline success). "
            "Dataset recorded at 10 FPS via stride=2. "
            "Basket success requires live in-basket or fault place_succeeded proof; "
            "post-hoc seat_object_in_basket_if_above is diagnostic only."
        ),
    }
    (output_dir / "pipeline_log.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    if not behavioral_success:
        failed = [k for k, v in summary["checks"].items() if not v]
        raise SystemExit(f"FAIL: behavioral checks failed: {failed}")
    print("SUCCESS: grasp → midair drop → recovery place into basket", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "full_pipeline_demo")
    parser.add_argument("--policy-path", default="lerobot/smolvla_libero")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--t-min", type=int, default=40, help="Earliest step to allow drop (after grasp)")
    parser.add_argument("--t-max", type=int, default=400, help="Latest step to wait for grasp+drop")
    parser.add_argument("--max-steps", type=int, default=750)
    parser.add_argument("--recovery-horizon", type=int, default=450)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args(argv)
    run_pipeline(
        args.output_dir,
        policy_path=args.policy_path,
        device=args.device,
        t_min=args.t_min,
        t_max=args.t_max,
        max_steps=args.max_steps,
        seed=args.seed,
        recovery_horizon=args.recovery_horizon,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
