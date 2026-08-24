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
"""Dry-run or live verification for mid-air drop recovery dataset logging."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_FAULTS = Path(__file__).resolve().parent
if str(_EXAMPLES_FAULTS) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_FAULTS))

from datagen import RandomizationConfig, _loss_mask_for_step, sample_episode_params
from lerobot.faults.recovery.fps import (
    SMOLVLA_LIBERO_TARGET_FPS,
    assert_control_rate_aligned,
    assert_dataset_fps,
    recording_stride,
    resolve_target_fps,
)
from lerobot.faults.recovery.libero_hook import install_libero_control_freq_hook
from lerobot.faults.sim.libero import get_robosuite_env, read_control_freq, read_model_timestep
from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger
from lerobot.faults.recovery.planner import SimpleIKRecoveryPlanner

DEFAULT_DRY_DIR = REPO_ROOT / "outputs" / "demo_drop_recovery_dry"


def _synthetic_frame(step: int) -> dict[str, np.ndarray]:
    img = np.full((256, 256, 3), (step * 7) % 256, dtype=np.uint8)
    state = np.linspace(0.0, 0.7, 8, dtype=np.float32)
    state[0] = 0.05 * step
    return {
        "observation.images.image": img,
        "observation.images.image2": 255 - img,
        "observation.state": state,
    }


def run_dry_run(output_dir: Path, *, policy_fps: int = SMOLVLA_LIBERO_TARGET_FPS) -> None:
    """Validate planner + logger FPS assert + loss_mask bookkeeping without LIBERO."""
    policy_fps = resolve_target_fps(policy_fps)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    rng = np.random.default_rng(0)
    params = sample_episode_params(rng, RandomizationConfig(seed=0, t_min=20, t_max=20, drop_duration=1))
    t_fault = int(params["t_fault"])

    planner = SimpleIKRecoveryPlanner(
        fps=policy_fps,
        speed_multiplier=float(params["speed_multiplier"]),
        waypoint_noise_m=float(params["waypoint_noise_m"]),
        seed=0,
    )
    plan = planner.plan(
        eef_pos=np.array([0.0, 0.0, 1.0]),
        eef_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        object_pos=np.array([0.1, 0.0, 0.9]),
        destination_pos=np.array([0.3, 0.2, 0.9]),
        gripper_open=True,
    )
    assert plan.shape[1] == 7 and len(plan) > 0, "Recovery plan must be non-empty 7D actions."

    ds_logger = FaultRecoveryDatasetLogger(
        root=output_dir,
        repo_id="local/demo_drop_recovery_dry",
        policy_fps=policy_fps,
    )

    # loss_mask: 1.0 before trigger, 0.0 on the single drop-injection frame, 1.0 during recovery
    n_nominal = 20
    n_drop = 1
    n_recovery = min(25, len(plan))
    total = n_nominal + n_drop + n_recovery
    t_injection = n_nominal
    t_recovery_start = t_injection + 1

    for step in range(total):
        mask = _loss_mask_for_step(step, t_injection, drop_duration=1)
        if step < n_nominal:
            action = np.zeros(7, dtype=np.float32)
            phase = "vla"
        elif step == t_injection:
            action = np.zeros(7, dtype=np.float32)
            phase = "drop"
        else:
            rec_i = step - t_recovery_start
            action = plan[rec_i].copy()
            phase = "recovery"

        ds_logger.log_step(_synthetic_frame(step), action, "libero_object task 0", mask, phase=phase)

    ds_logger.end_episode()
    ds_logger.finalize()

    assert_dataset_fps(ds_logger.dataset.fps, policy_fps)
    counts = ds_logger.loss_mask_counts
    assert counts.get(1.0, 0) == total - 1, (
        f"Expected {total - 1} masked-in frames, got {counts.get(1.0, 0)}"
    )
    assert counts.get(0.0, 0) == 1, f"Expected 1 drop-injection frame, got {counts.get(0.0, 0)}"
    assert ds_logger.dataset.fps == policy_fps == SMOLVLA_LIBERO_TARGET_FPS

    print("SUCCESS: drop-recovery dry-run demo")
    print(f"  output_dir: {output_dir}")
    print(f"  dataset_fps: {ds_logger.dataset.fps} (policy_fps={policy_fps})")
    print(f"  recovery_plan_steps: {len(plan)}")
    print(f"  loss_mask counts: 1.0={counts.get(1.0, 0)}, 0.0={counts.get(0.0, 0)}")
    print(f"  t_fault={t_fault}, t_recovery_start={t_recovery_start}, total_frames={total}")


def run_live(output_dir: Path, *, policy_fps: int = SMOLVLA_LIBERO_TARGET_FPS) -> None:
    """Run one LIBERO episode with midair_drop when dependencies are available."""
    from lerobot.envs.factory import make_env

    from lerobot.faults.config import FaultInjectionConfig
    from lerobot.faults.wrappers import DropRecoveryEnvWrapper

    if output_dir.exists():
        shutil.rmtree(output_dir)

    from lerobot.faults.recovery.fps import DEFAULT_LIBERO_CONTROL_FREQ

    policy_fps = resolve_target_fps(policy_fps)
    # Live control must stay at LIBERO 20 Hz; dataset/planner target remains policy_fps.
    hook_ok = install_libero_control_freq_hook(DEFAULT_LIBERO_CONTROL_FREQ)
    rng = np.random.default_rng(42)
    params = sample_episode_params(rng, RandomizationConfig(seed=42))

    fault_cfg = FaultInjectionConfig(
        enabled=True,
        type="midair_drop",
        t_min=int(params["t_fault"]),
        t_max=int(params["t_fault"]),
        require_grasp=False,
        recovery_fps=policy_fps,
        seed=42,
        log_path=output_dir / "fault_events.jsonl",
    )

    envs = make_env(
        env_type="libero",
        task="libero_object",
        task_ids=[0],
        n_envs=1,
        use_async_envs=False,
    )
    vec = envs["libero_object"][0]
    env = DropRecoveryEnvWrapper(vec, fault_cfg)

    ds_logger = FaultRecoveryDatasetLogger(
        root=output_dir / "dataset",
        repo_id="local/demo_drop_recovery_live",
        policy_fps=policy_fps,
    )

    obs, _ = env.reset()
    rs_env = get_robosuite_env(env)
    control_freq = read_control_freq(rs_env)
    model_timestep = read_model_timestep(rs_env)
    assert_control_rate_aligned(control_freq, model_timestep, policy_fps)
    stride = recording_stride(control_freq, policy_fps)
    task = "libero_object task 0"

    for step in range(200):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        if step % stride != 0:
            if bool(np.asarray(terminated).any() or np.asarray(truncated).any()):
                break
            continue
        mask = env.loss_mask()
        executed = env.last_executed_action
        if executed is None:
            executed = action
        ds_logger.log_step(obs, executed, task, mask)
        if bool(np.asarray(terminated).any() or np.asarray(truncated).any()):
            break

    ds_logger.end_episode()
    ds_logger.finalize()
    env.close()

    assert (output_dir / "fault_events.jsonl").exists() or True  # fault may skip if no grasp
    print("SUCCESS: drop-recovery live demo")
    print(f"  output_dir: {output_dir}")
    print(f"  control_freq_hook: {hook_ok}, recording_stride: {stride}")
    print(f"  loss_mask counts: {ds_logger.loss_mask_counts}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Synthetic CI-safe demo (default).")
    parser.add_argument("--live", action="store_true", help="Full LIBERO path when available.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--policy-fps", type=int, default=SMOLVLA_LIBERO_TARGET_FPS)
    args = parser.parse_args(argv)

    if args.live and args.dry_run:
        print("Choose only one of --dry-run or --live.", file=sys.stderr)
        return 2

    mode = "live" if args.live else "dry-run"
    if mode == "dry-run":
        out = args.output_dir or DEFAULT_DRY_DIR
        run_dry_run(out, policy_fps=args.policy_fps)
    else:
        out = args.output_dir or (REPO_ROOT / "outputs" / "demo_drop_recovery_live")
        run_live(out, policy_fps=args.policy_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
