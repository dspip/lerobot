# Copyright 2026 Gangelia. All rights reserved.
"""Trajectory datagen for mid-air drop recovery datasets."""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.recovery.fps import assert_control_rate_aligned, recording_stride, resolve_target_fps
from lerobot.faults.recovery.libero_hook import install_libero_control_freq_hook
from lerobot.faults.sim.libero import get_robosuite_env, read_control_freq, read_model_timestep
from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger
from lerobot.faults.recovery.planner import SimpleIKRecoveryPlanner

logger = logging.getLogger(__name__)


@dataclass
class RandomizationConfig:
    """Randomization axes for recovery trajectory datagen."""

    t_min: int = 10
    t_max: int = 30
    impulse_lin_std: float = 0.5
    impulse_ang_std: float = 0.2
    impulse_lin_bias: tuple[float, float, float] = (0.0, 0.0, -0.5)
    recovery_action_noise_std: float = 0.02
    arm_posture_noise_deg: float = 3.0
    waypoint_noise_m: float = 0.015
    speed_multiplier_min: float = 0.8
    speed_multiplier_max: float = 1.2
    drop_duration: int = 3
    seed: int | None = 42

    def __post_init__(self) -> None:
        if self.t_min < 0:
            raise ValueError(f"t_min must be >= 0, got {self.t_min}.")
        if self.t_max < self.t_min:
            raise ValueError(f"t_max must be >= t_min (got {self.t_max} < {self.t_min}).")
        if self.impulse_lin_std < 0 or self.impulse_ang_std < 0:
            raise ValueError("impulse std values must be non-negative.")
        if self.recovery_action_noise_std < 0:
            raise ValueError("recovery_action_noise_std must be non-negative.")
        if self.waypoint_noise_m < 0:
            raise ValueError("waypoint_noise_m must be non-negative.")
        if self.arm_posture_noise_deg < 0:
            raise ValueError("arm_posture_noise_deg must be non-negative.")
        if self.speed_multiplier_min <= 0 or self.speed_multiplier_max <= 0:
            raise ValueError("speed_multiplier bounds must be positive.")
        if self.speed_multiplier_max < self.speed_multiplier_min:
            raise ValueError("speed_multiplier_max must be >= speed_multiplier_min.")
        if self.drop_duration < 1:
            raise ValueError("drop_duration must be >= 1.")


def sample_episode_params(rng: np.random.Generator, config: RandomizationConfig | None = None) -> dict[str, Any]:
    """Sample one episode's randomized parameters."""
    cfg = config or RandomizationConfig()
    t_fault = int(rng.integers(cfg.t_min, cfg.t_max + 1))
    lin = np.asarray(cfg.impulse_lin_bias, dtype=np.float64) + rng.normal(
        0.0, cfg.impulse_lin_std, size=3
    )
    ang = rng.normal(0.0, cfg.impulse_ang_std, size=3)
    speed_multiplier = float(rng.uniform(cfg.speed_multiplier_min, cfg.speed_multiplier_max))
    arm_noise_rad = float(np.deg2rad(cfg.arm_posture_noise_deg))
    arm_delta = rng.uniform(-arm_noise_rad, arm_noise_rad, size=3)

    return {
        "t_fault": t_fault,
        "t_recovery_start": t_fault + cfg.drop_duration,
        "drop_duration": cfg.drop_duration,
        "impulse_lin_vel": lin.astype(np.float64),
        "impulse_ang_vel": ang.astype(np.float64),
        "recovery_action_noise_std": cfg.recovery_action_noise_std,
        "arm_posture_noise_rad": arm_delta.astype(np.float64),
        "waypoint_noise_m": cfg.waypoint_noise_m,
        "speed_multiplier": speed_multiplier,
        "seed": cfg.seed,
    }


def _synthetic_obs(step: int) -> dict[str, np.ndarray]:
    """Build a minimal processed observation for scripted / dry datagen."""
    img = np.full((256, 256, 3), step % 256, dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    state[0] = 0.1 * step
    return {
        "observation.images.image": img,
        "observation.images.image2": img.copy(),
        "observation.state": state,
    }


def _loss_mask_for_step(step: int, t_fault: int, *, drop_duration: int = 1) -> float:
    """Return dataset ``loss_mask`` for a scripted episode step.

    Only the drop-injection frame(s) use ``0.0``. When drop and recovery start on
    the same step (``drop_duration == 1``), that single trigger step is masked out;
    all recovery steps use ``1.0``.
    """
    del drop_duration  # injection is always the first drop-phase step only
    if step < t_fault:
        return 1.0
    if step == t_fault:
        return 0.0
    return 1.0


def _loss_mask_from_fault(*, triggered: bool, drop_injection_step: bool, recovery_active: bool) -> float:
    """Derive ``loss_mask`` from midair_drop fault state when available."""
    del triggered, recovery_active
    if drop_injection_step:
        return 0.0
    return 1.0


def composed_loss_mask(*masks: float) -> float:
    """Combine per-fault masks: zeroed whenever *any* active fault reports ``0.0``."""
    if any(float(m) == 0.0 for m in masks):
        return 0.0
    return 1.0


def run_scripted_episode(
    params: dict[str, Any],
    *,
    output_dir: Path,
    repo_id: str,
    task: str = "pick alphabet soup and place in basket",
    nominal_steps: int | None = None,
    recovery_cap: int = 40,
    policy_fps: int | None = None,
) -> dict[str, Any]:
    """Record one synthetic episode without LIBERO (offline / smoke)."""
    t_fault = int(params["t_fault"])
    t_recovery = int(params["t_recovery_start"])
    if nominal_steps is None:
        nominal_steps = t_fault + 5

    planner = SimpleIKRecoveryPlanner(
        fps=resolve_target_fps(policy_fps),
        speed_multiplier=float(params["speed_multiplier"]),
        waypoint_noise_m=float(params["waypoint_noise_m"]),
        arm_posture_noise_rad=params.get("arm_posture_noise_rad"),
        seed=params.get("seed"),
    )
    eef = np.array([0.0, 0.0, 1.0])
    eef_q = np.array([0.0, 0.0, 0.0, 1.0])
    obj = np.array([0.1, 0.0, 0.9])
    dest = np.array([0.3, 0.2, 0.9])
    recovery_actions = planner.plan(
        eef_pos=eef,
        eef_quat=eef_q,
        object_pos=obj,
        destination_pos=dest,
        gripper_open=True,
    )

    ds_logger = FaultRecoveryDatasetLogger(
        root=output_dir,
        repo_id=repo_id,
        policy_fps=resolve_target_fps(policy_fps),
    )

    total_steps = nominal_steps + min(len(recovery_actions), recovery_cap)
    noise_std = float(params["recovery_action_noise_std"])
    rng = np.random.default_rng(params.get("seed"))

    for step in range(total_steps):
        mask = _loss_mask_for_step(step, t_fault, drop_duration=int(params["drop_duration"]))
        if step < t_fault:
            action = rng.normal(0.0, 0.01, size=7).astype(np.float32)
            phase = "vla"
        elif step < t_recovery:
            action = np.zeros(7, dtype=np.float32)
            phase = "drop"
        else:
            rec_idx = step - t_recovery
            if rec_idx < len(recovery_actions):
                action = recovery_actions[rec_idx].copy()
            else:
                action = recovery_actions[-1].copy()
            if noise_std > 0:
                action[:6] += rng.normal(0.0, noise_std, size=6).astype(np.float32)
                action[:6] = np.clip(action[:6], -1.0, 1.0)
            phase = "recovery"

        ds_logger.log_step(_synthetic_obs(step), action, task, mask, phase=phase)

    ds_logger.end_episode()
    ds_logger.finalize()

    return {
        "output_dir": str(output_dir),
        "total_steps": total_steps,
        "loss_mask_counts": ds_logger.loss_mask_counts,
        "recovery_actions": len(recovery_actions),
        "params": params,
    }


def run_datagen(
    n_episodes: int,
    output_dir: Path | str,
    *,
    mode: str = "scripted",
    config: RandomizationConfig | None = None,
    repo_id: str = "local/fault_recovery_datagen",
    policy_fps: int | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Generate ``n_episodes`` recovery trajectories.

    ``mode='scripted'`` uses synthetic frames (no LIBERO). ``mode='live'`` attempts
    a LIBERO rollout and falls back with a warning on ``ImportError``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or RandomizationConfig()
    seed = cfg.seed if cfg.seed is not None else 0
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        params = sample_episode_params(rng, cfg)
        ep_dir = output_dir / f"episode_{ep:04d}"
        if ep_dir.exists():
            raise FileExistsError(f"Episode output exists: {ep_dir}")

        if mode == "scripted":
            info = run_scripted_episode(
                params,
                output_dir=ep_dir,
                repo_id=f"{repo_id}_ep{ep:04d}",
                policy_fps=policy_fps,
                **kwargs,
            )
        elif mode == "live":
            try:
                info = _run_live_episode(
                    params,
                    output_dir=ep_dir,
                    repo_id=f"{repo_id}_ep{ep:04d}",
                    policy_fps=policy_fps,
                    **kwargs,
                )
            except ImportError as exc:
                warnings.warn(
                    f"LIBERO live datagen unavailable ({exc}); skipping episode {ep}.",
                    stacklevel=2,
                )
                continue
        else:
            raise ValueError(f"Unknown mode {mode!r}; use 'scripted' or 'live'.")

        info["episode_index"] = ep
        results.append(info)
        logger.info("Recorded episode %s -> %s", ep, ep_dir)

    return results



def _run_live_episode(
    params: dict[str, Any],
    *,
    output_dir: Path,
    repo_id: str,
    policy_fps: int | None = None,
    task: str = "pick alphabet soup and place in basket",
    max_steps: int = 200,
) -> dict[str, Any]:
    """Attempt one LIBERO episode with midair_drop + recovery logging."""
    from lerobot.envs.factory import make_env  # noqa: WPS433

    from lerobot.faults.wrappers import DropRecoveryEnvWrapper

    from lerobot.faults.recovery.fps import DEFAULT_LIBERO_CONTROL_FREQ

    target_fps = resolve_target_fps(policy_fps)
    # SmolVLA pick-place fails at control_freq=10; keep LIBERO default 20 Hz and
    # subsample dataset frames to target_fps via recording_stride.
    hook_ok = install_libero_control_freq_hook(DEFAULT_LIBERO_CONTROL_FREQ)

    fault_cfg = FaultInjectionConfig(
        enabled=True,
        type="midair_drop",
        t_min=int(params["t_fault"]),
        t_max=int(params["t_fault"]),
        require_grasp=False,
        impulse_lin_std=0.0,
        impulse_ang_std=0.0,
        recovery_fps=target_fps,
        seed=params.get("seed"),
        waypoint_noise_m=float(params["waypoint_noise_m"]),
        recovery_action_noise_std=float(params["recovery_action_noise_std"]),
        speed_multiplier_min=float(params["speed_multiplier"]),
        speed_multiplier_max=float(params["speed_multiplier"]),
        arm_posture_noise_deg=float(np.rad2deg(np.max(np.abs(params["arm_posture_noise_rad"])))),
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
        root=output_dir,
        repo_id=repo_id,
        policy_fps=target_fps,
    )

    obs, _ = env.reset()
    rs_env = get_robosuite_env(env)
    control_freq = read_control_freq(rs_env)
    model_timestep = read_model_timestep(rs_env)
    assert_control_rate_aligned(control_freq, model_timestep, target_fps)
    stride = recording_stride(control_freq, target_fps)

    for step in range(max_steps):
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

    return {
        "output_dir": str(output_dir),
        "loss_mask_counts": ds_logger.loss_mask_counts,
        "params": params,
        "recording_stride": stride,
        "control_freq_hook": hook_ok,
    }


def generate_recovery_trajectories(
    n_episodes: int,
    output_dir: Path | str,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Public entrypoint mirroring datagen CLI defaults."""
    return run_datagen(n_episodes, output_dir, **kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate mid-air drop recovery trajectories.")
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/datagen/recovery"))
    p.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    p.add_argument("--repo-id", default="local/fault_recovery_datagen")
    p.add_argument("--policy-fps", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--t-min", type=int, default=10)
    p.add_argument("--t-max", type=int, default=30)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    cfg = RandomizationConfig(seed=args.seed, t_min=args.t_min, t_max=args.t_max)
    results = generate_recovery_trajectories(
        args.n_episodes,
        args.output_dir,
        mode=args.mode,
        config=cfg,
        repo_id=args.repo_id,
        policy_fps=args.policy_fps,
    )
    print(f"Recorded {len(results)} episode(s) under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
