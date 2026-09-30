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

"""Live LIBERO alphabet-soup scene: Aviya drops and recovers via keyboard or buttons.

Replays a nominal dataset episode (default 12, full pick-and-place) in a loop until
manual ``midair_drop``, then ``request_recovery``. No prediction head is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from robosuite.utils.transform_utils import quat2axisangle

try:
    import tkinter as tk

    from PIL import ImageTk
except ImportError:
    raise SystemExit(
        "Tkinter/PIL.ImageTk is required to run the interactive manual drop/recovery script.\n"
        "Please install it (e.g. `sudo apt-get install python3-tk` on Ubuntu/Debian) and try again."
    ) from None

REPO = Path(__file__).resolve().parents[2]

from lerobot.envs.configs import LiberoEnv  # noqa: E402
from lerobot.envs.factory import make_env  # noqa: E402
from lerobot.faults.config import FaultInjectionConfig  # noqa: E402
from lerobot.faults.recovery.midair_drop import MidAirDropFault  # noqa: E402
from lerobot.faults.sim.libero import (  # noqa: E402
    get_robosuite_env,
    is_object_grasped,
    unwrap_libero_env,
)
from lerobot.faults.wrappers import DropRecoveryEnvWrapper  # noqa: E402

_XY60_EXTRACT_ROOT = Path(tempfile.gettempdir()) / "xy_band_mix_60ep_extract"
AUDIT_REPORT = _XY60_EXTRACT_ROOT / "audit_report.json"
DATASET_ROOT = _XY60_EXTRACT_ROOT / "dataset" / "data" / "chunk-000"
DEFAULT_OUTPUT = REPO / "reports" / "xy60_verify" / "manual_drop_recovery"
DEFAULT_REPLAY_EPISODE = 12
OBJECT_NAME = "alphabet_soup_1"
ACTION_HOLD_STEPS = 2
MAX_TOTAL_STEPS = 4000
LIBERO_EPISODE_LENGTH = 4000
POST_RECOVERY_HOLD_FRAMES = 60


def _vec_env(envs):
    suite = next(iter(envs.values()))
    return next(iter(suite.values()))


def _policy_observation(obs):
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


def _overlay(frame, text):
    image = Image.fromarray(np.asarray(frame).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 52), fill=(0, 0, 0))
    for line_index, line in enumerate(text.split("\n")):
        draw.text((8, 8 + line_index * 14), line, fill=(255, 255, 255))
    return np.asarray(image)


def _load_nominal_episode_ids() -> list[int]:
    if not AUDIT_REPORT.is_file():
        raise SystemExit(f"Missing audit report at {AUDIT_REPORT}; extract xy_band_mix_60ep first.")
    payload = json.loads(AUDIT_REPORT.read_text())
    rows = payload.get("rows", [])
    nominal = sorted(
        int(row["episode_id"])
        for row in rows
        if row.get("type") == "nominal" and row.get("episode_id") is not None
    )
    if not nominal:
        raise SystemExit(f"No nominal episodes in {AUDIT_REPORT}")
    return nominal


def _resolve_replay_episode(episode_id: int, nominal_ids: list[int]) -> int:
    if episode_id not in nominal_ids:
        raise SystemExit(f"Episode {episode_id} is not nominal; choose one of {nominal_ids}")
    return episode_id


def _load_episode_actions(episode_id: int) -> list[np.ndarray]:
    import pyarrow.parquet as pq

    if not DATASET_ROOT.is_dir():
        raise SystemExit(
            f"Missing dataset at {DATASET_ROOT.parent.parent}; "
            "extract xy_band_mix_60ep before running this script."
        )
    pairs: list[tuple[int, np.ndarray]] = []
    for path in sorted(DATASET_ROOT.glob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action"])
        episodes = table.column("episode_index").to_pylist()
        frames = table.column("frame_index").to_pylist()
        stored = table.column("action").to_pylist()
        for episode, frame, action in zip(episodes, frames, stored, strict=True):
            if int(episode) == episode_id:
                pairs.append((int(frame), np.asarray(action, dtype=np.float32)))
    if not pairs:
        raise RuntimeError(f"nominal episode {episode_id} has no actions in {DATASET_ROOT}")
    pairs.sort(key=lambda item: item[0])
    return [action for _frame, action in pairs]


class ManualControlUI:
    """Tk window; callbacks only set flags (sim runs on the main thread)."""

    def __init__(self) -> None:
        self.pending_drop = False
        self.pending_recover = False
        self.quit_requested = False
        self._photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("Manual drop / recovery (LIBERO object)")
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self.root.bind("<d>", lambda _event: self._arm_drop())
        self.root.bind("<r>", lambda _event: self._arm_recover())
        self.root.bind("<q>", lambda _event: self._request_quit())
        self.root.bind("<Escape>", lambda _event: self._request_quit())

        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Button(controls, text="Drop (d)", command=self._arm_drop, width=12).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Recover (r)", command=self._arm_recover, width=12).pack(
            side=tk.LEFT, padx=4
        )
        tk.Label(controls, text="Focus this window — q or Esc to quit").pack(side=tk.LEFT, padx=8)

        self._label = tk.Label(self.root)
        self._label.pack(side=tk.TOP)

    def _arm_drop(self) -> None:
        self.pending_drop = True

    def _arm_recover(self) -> None:
        self.pending_recover = True

    def _request_quit(self) -> None:
        self.quit_requested = True

    def pump(self) -> None:
        if self.quit_requested:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.quit_requested = True

    def show(self, frame_rgb: np.ndarray) -> None:
        if self.quit_requested:
            return
        try:
            image = Image.fromarray(frame_rgb.astype(np.uint8))
            self._photo = ImageTk.PhotoImage(image=image)
            self._label.configure(image=self._photo)
        except tk.TclError:
            self.quit_requested = True

    def destroy(self) -> None:
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass


def _phase_label(*, replay: bool, dropped: bool, recovery_active: bool, recovery_done: bool) -> str:
    if recovery_done:
        return "recovery_done"
    if recovery_active:
        return "recovery"
    if replay:
        return "replay"
    if dropped:
        return "dropped"
    return "holding"


def _hold_action(*, dropped: bool) -> np.ndarray:
    gripper = -1.0 if dropped else 1.0
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper], dtype=np.float32)


def _recovery_planner_done(fault: MidAirDropFault) -> bool:
    state = fault._states[0]
    planner = state.planner
    return planner is not None and planner.phase_name == "done"


def _write_video(output: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to write recovery videos")
    ffmpeg = subprocess.run(
        [
            ffmpeg_exe,
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
            str(output),
        ],
        input=b"".join(frame.astype(np.uint8).tobytes() for frame in frames),
        check=False,
        capture_output=True,
    )
    if ffmpeg.returncode != 0:
        raise RuntimeError(ffmpeg.stderr.decode()[-2000:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual midair drop and IK recovery (no head).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--episode",
        type=int,
        default=DEFAULT_REPLAY_EPISODE,
        help=f"Nominal dataset episode id to replay (default: {DEFAULT_REPLAY_EPISODE}).",
    )
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    nominal_ids = _load_nominal_episode_ids()
    replay_episode_id = _resolve_replay_episode(args.episode, nominal_ids)
    replay_commands = _load_episode_actions(replay_episode_id)

    env_cfg = LiberoEnv(
        task="libero_object",
        task_ids=[0],
        observation_height=256,
        observation_width=256,
        episode_length=LIBERO_EPISODE_LENGTH,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = _vec_env(envs)
    libero_env = unwrap_libero_env(vec)
    baseline_init_state_id = libero_env.init_state_id

    args.output.mkdir(parents=True, exist_ok=True)
    fault_config = FaultInjectionConfig(
        enabled=True,
        type="midair_drop",
        seed=1000,
        probability=0.0,
        impulse_lin_bias=(0.0, 0.08, -0.45),
        impulse_lin_std=0.0,
        impulse_ang_std=0.0,
        log_path=args.output / "fault_events.jsonl",
    )
    env = DropRecoveryEnvWrapper(vec, fault_config)
    fault: MidAirDropFault = env.fault

    ui = ManualControlUI()
    frames: list[np.ndarray] = []
    global_step = 0
    dropped = False
    recovery_active = False
    recovery_finished = False
    post_recovery_frames = 0
    drop_pressed_at: int | None = None
    recovery_pressed_at: int | None = None
    grasped_before_drop: bool | None = None
    quit_reason = "step_cap"
    obs = None
    replay_index = 0
    replay_hold = 0
    reset_seed = 1000

    print(f"Replaying nominal episode {replay_episode_id} ({len(replay_commands)} actions)")

    def _restart_same_episode() -> None:
        nonlocal obs, replay_index, replay_hold, reset_seed
        libero_env.init_state_id = baseline_init_state_id
        reset_seed += 1
        obs, _info = env.reset(seed=reset_seed)
        replay_index = 0
        replay_hold = 0
        print(f"Restarting nominal episode {replay_episode_id} from action 0")

    try:
        obs, _info = env.reset(seed=reset_seed)
        while global_step < MAX_TOTAL_STEPS and not ui.quit_requested:
            ui.pump()
            if recovery_finished and post_recovery_frames >= POST_RECOVERY_HOLD_FRAMES:
                quit_reason = "recovery_done"
                break

            if ui.pending_drop:
                ui.pending_drop = False
                if not dropped:
                    rs_env = get_robosuite_env(env, 0)
                    grasped_before_drop = bool(is_object_grasped(rs_env, OBJECT_NAME))
                    print(f"Drop: is_object_grasped={grasped_before_drop} (step {global_step})")
                    if fault.trigger_manual_drop(env, 0):
                        dropped = True
                        drop_pressed_at = global_step
                        replay_index = len(replay_commands)

            if ui.pending_recover:
                ui.pending_recover = False
                if not recovery_active:
                    if not dropped:
                        print("Press Drop while the can is in the hand first.")
                    else:
                        fault.request_recovery(env, 0, reason="manual", consume_first_action=False)
                        recovery_active = True
                        recovery_pressed_at = global_step
                        print(f"Recovery started at step {global_step}")

            in_replay = replay_index < len(replay_commands) and not dropped and not recovery_active
            if recovery_finished or recovery_active:
                action = _hold_action(dropped=True)
            elif in_replay:
                if replay_hold == 0:
                    action = replay_commands[replay_index]
                    replay_hold = ACTION_HOLD_STEPS
                replay_hold -= 1
                if replay_hold == 0:
                    replay_index += 1
            else:
                action = _hold_action(dropped=dropped)

            _state, images = _policy_observation(obs)
            phase = _phase_label(
                replay=in_replay,
                dropped=dropped,
                recovery_active=recovery_active,
                recovery_done=recovery_finished,
            )
            overlay = (
                f"phase: {phase}  ep: {replay_episode_id}  step: {global_step}\n"
                f"d=Drop (hold)  r=Recover (IK now)  q/Esc=quit"
            )
            frames.append(_overlay(images["image"], overlay))
            ui.show(frames[-1])

            obs, _reward, terminated, truncated, _info = env.step(action.reshape(1, 7))
            global_step += 1

            if recovery_active and not recovery_finished:
                if _recovery_planner_done(fault):
                    recovery_finished = True
                    recovery_active = False
                    post_recovery_frames = 0
            elif recovery_finished:
                post_recovery_frames += 1

            if not dropped and not recovery_active and not recovery_finished:
                term = bool(np.asarray(terminated).reshape(-1)[0])
                trunc = bool(np.asarray(truncated).reshape(-1)[0])
                if term or trunc:
                    # The vector env resets on the next step. Point that reset at the
                    # original soup pose. A second reset here would move the can.
                    libero_env.init_state_id = baseline_init_state_id
                    replay_index = 0
                    replay_hold = 0
                    print(f"Restarting nominal episode {replay_episode_id} from action 0")
                elif replay_index >= len(replay_commands):
                    _restart_same_episode()

        if ui.quit_requested:
            quit_reason = "user_quit"
    finally:
        ui.destroy()
        env.close()
        if ui.quit_requested:
            quit_reason = "user_quit"
        summary = {
            "drop_pressed_at": drop_pressed_at,
            "grasped_before_drop": grasped_before_drop,
            "recovery_pressed_at": recovery_pressed_at,
            "replay_episode_id": replay_episode_id,
            "quit_reason": quit_reason,
            "total_steps": global_step,
            "max_total_steps": MAX_TOTAL_STEPS,
            "note": "Manual controls only; no prediction head loaded.",
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _write_video(args.output / "rollout.mp4", frames)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
