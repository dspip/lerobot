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

"""One LIBERO rollout where only the failure head may start IK recovery.

The drop injector is constructed so ``request_recovery`` can run, but this
script never calls ``on_step`` until recovery is already active, so a grasp
does not throw the soup or start the planner by itself.

The two-pass head is a reconstruction from the checkpoint weight shapes and
Eran's written placement. His ``head_vlm_conditioning.py`` is not in this repo.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from robosuite.utils.transform_utils import quat2axisangle

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "third_party" / "Gangelia_Project" / "smolvla_r_package"
sys.path.insert(0, str(PACKAGE))

from head_vlm_conditioning import HeadToVlmPolicy  # noqa: E402
from train_phase1 import load_policy  # noqa: E402

from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.factory import make_env  # noqa: E402
from lerobot.faults.config import FaultInjectionConfig  # noqa: E402
from lerobot.faults.logging import FaultEventLogger  # noqa: E402
from lerobot.faults.recovery.midair_drop import MidAirDropFault  # noqa: E402
from lerobot.faults.sim.libero import get_robosuite_env, is_object_grasped, midair_drop  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

DEFAULT_CHECKPOINT = (
    REPO
    / "checkpoints"
    / "no_calibration_predicted_head_to_vlm-total3000-20260922T091225Z-1-001"
    / "no_calibration_predicted_head_to_vlm-total3000"
)


def _vec_env(envs):
    suite = next(iter(envs.values()))
    return next(iter(suite.values()))


def _policy_observation(obs):
    """Match the dataset state layout: xyz, axis-angle, two finger joints."""
    robot = obs["robot_state"]
    pos = np.asarray(robot["eef"]["pos"], dtype=np.float32).reshape(-1)
    quat = np.asarray(robot["eef"]["quat"], dtype=np.float64).reshape(-1)
    fingers = np.asarray(robot["gripper"]["qpos"], dtype=np.float32).reshape(-1)
    state = np.concatenate([pos, quat2axisangle(quat).astype(np.float32), fingers])
    images = {}
    for name, frame in obs["pixels"].items():
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        images[name] = np.ascontiguousarray(frame[::-1, ::-1])
    return state.astype(np.float32), images


def _query(policy, head, pre, post, state, images, device, seed):
    raw = {"observation.state": torch.from_numpy(state.copy())[None], "task": [""]}
    for key, frame in images.items():
        raw[f"observation.images.{key}"] = (
            torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255)[None].to(device)
        )
    raw["observation.state"] = raw["observation.state"].to(device)
    policy.reset()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    with torch.inference_mode():
        batch = pre(raw)
        actions, probabilities = head.predict(batch, noise=noise)
        actions = post(actions)[0].detach().float().cpu().numpy()
    return np.clip(actions, -1.0, 1.0), probabilities[0].detach().float().cpu().numpy()


def _overlay(frame, text):
    image = Image.fromarray(np.asarray(frame).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 36), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return np.asarray(image)


def _replay_then_release(vec, frames, log):
    """Play recorded nominal actions, then open the hand if the can is held.

    This is scene setup. It does not call ``request_recovery``.
    """
    import pyarrow.parquet as pq

    root = Path("/tmp/xy_band_mix_60ep_extract/dataset/data/chunk-000")
    tables = [pq.read_table(path, columns=["episode_index", "action"]) for path in sorted(root.glob("*.parquet"))]
    actions = []
    for table in tables:
        episodes = table.column("episode_index").to_pylist()
        stored = table.column("action").to_pylist()
        actions.extend(
            np.asarray(action, dtype=np.float32)
            for episode, action in zip(episodes, stored, strict=True)
            if episode == 12
        )
    if len(actions) < 36:
        raise RuntimeError(f"nominal episode 12 has {len(actions)} actions; need 36 to lift before a release")
    obs = None
    for index, command in enumerate(actions[:36]):
        for _hold in range(2):
            if obs is not None:
                _state, images = _policy_observation(obs)
                frames.append(_overlay(images["image"], f"setup replay {index}  no recovery"))
                log.append({"phase": "replay", "p_drop": None, "recovery": False})
            obs, _reward, _terminated, _truncated, _info = vec.step(command.reshape(1, 7))
    rs_env = get_robosuite_env(vec, 0)
    grasped = bool(is_object_grasped(rs_env, "alphabet_soup_1"))
    if grasped:
        midair_drop(rs_env, "alphabet_soup_1", lin_vel=[0.0, 0.08, -0.45], ang_vel=[0.0, -0.06, 0.01])
        obs, _reward, _terminated, _truncated, _info = vec.step(np.array([[0, 0, 0, 0, 0, 0, -1]], dtype=np.float32))
        _state, images = _policy_observation(obs)
        frames.append(_overlay(images["image"], "setup release  recovery still off"))
        log.append({"phase": "release", "p_drop": None, "recovery": False})
    return obs, grasped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=REPO / "reports" / "xy60_verify" / "head_recovery_rollout")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stock = load_policy(args.checkpoint, device).float().eval()
    head = HeadToVlmPolicy(stock).load_auxiliary(args.checkpoint / "auxiliary").to(device).eval()
    for parameter in stock.parameters():
        parameter.requires_grad_(False)
    pre, post = make_pre_post_processors(
        stock.config,
        pretrained_path=str(args.checkpoint / "processors"),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
            "tokenizer_processor": {"tokenizer_name": stock.config.vlm_model_name},
        },
    )

    env_cfg = LiberoEnv(
        task="libero_object",
        task_ids=[0],
        observation_height=256,
        observation_width=256,
        episode_length=args.steps,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = _vec_env(envs)
    args.output.mkdir(parents=True, exist_ok=True)
    logger = FaultEventLogger(args.output / "fault_events.jsonl")
    fault = MidAirDropFault(
        FaultInjectionConfig(enabled=True, type="midair_drop", seed=1000),
        num_envs=1,
        event_logger=logger,
    )
    fault.reset(episode_ids=[0])

    obs, _info = vec.reset(seed=1000)
    frames = []
    log = []
    obs, grasped_before_release = _replay_then_release(vec, frames, log)
    recovery_started_at = None
    chunk = None
    chunk_index = 0
    probabilities = np.array([1.0, 0.0], dtype=np.float32)
    try:
        for step in range(args.steps):
            state, images = _policy_observation(obs)
            need_query = chunk is None or chunk_index >= 5
            if recovery_started_at is None and need_query:
                chunk, probabilities = _query(stock, head, pre, post, state, images, device, 5100 + step)
                chunk_index = 0
                if float(probabilities[1]) >= args.threshold:
                    first = fault.request_recovery(vec, 0, reason="head")
                    recovery_started_at = step
                    action = np.asarray(first, dtype=np.float32).reshape(-1)[:7]
                    chunk = None
                else:
                    action = chunk[chunk_index]
                    chunk_index += 1
            elif recovery_started_at is not None:
                action = fault.on_step(vec, np.asarray(action, dtype=np.float32).reshape(1, -1))[0]
            else:
                action = chunk[chunk_index]
                chunk_index += 1
            action = np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[:7], -1.0, 1.0)
            label = (
                f"step {step}  P(drop)={probabilities[1]:.2f}  "
                + ("RECOVERY" if recovery_started_at is not None else "policy")
            )
            frames.append(_overlay(images["image"], label))
            log.append(
                {
                    "phase": "head",
                    "step": step,
                    "p_drop": float(probabilities[1]),
                    "recovery": recovery_started_at is not None,
                    "action": action.tolist(),
                }
            )
            obs, _reward, _terminated, _truncated, _info = vec.step(action.reshape(1, 7))
    finally:
        vec.close()

    (args.output / "steps.json").write_text(json.dumps(log, indent=2) + "\n")
    summary = {
        "checkpoint": str(args.checkpoint),
        "steps": args.steps,
        "threshold": args.threshold,
        "recovery_started_at": recovery_started_at,
        "grasped_before_release": grasped_before_release,
        "max_p_drop": max(row["p_drop"] for row in log if row["p_drop"] is not None),
        "note": (
            "Recovery starts only from HeadToVlmPolicy probability, via request_recovery. "
            "The automatic drop trigger is never called. The two-pass module is a "
            "reconstruction; Eran's head_vlm_conditioning.py is not in this repo."
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    video = args.output / "rollout.mp4"
    ffmpeg = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frames[0].shape[1]}x{frames[0].shape[0]}",
            "-r",
            "20",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        input=b"".join(frame.astype(np.uint8).tobytes() for frame in frames),
        check=False,
        capture_output=True,
    )
    if ffmpeg.returncode != 0:
        raise RuntimeError(ffmpeg.stderr.decode()[-2000:])
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
