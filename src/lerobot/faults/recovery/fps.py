# Copyright 2026 Gangelia. All rights reserved.
"""FPS alignment helpers for LIBERO sim control vs SmolVLA dataset recording."""

from __future__ import annotations

# Dataset / fine-tune recording rate for official lerobot/libero + SmolVLA.
SMOLVLA_LIBERO_TARGET_FPS = 10
# Live LIBERO/robosuite control rate. SmolVLA pick-place FAILS if this is forced
# to 10 Hz — always keep 20 and subsample recordings with recording_stride().
DEFAULT_LIBERO_CONTROL_FREQ = 20
DEFAULT_MUJOCO_MODEL_TIMESTEP = 0.002

# Module-level intent for env hooks / dataset collectors (see configure_libero_control_freq).
_INTENDED_LIBERO_CONTROL_FREQ: int | None = None


def resolve_target_fps(policy_fps: int | None = None) -> int:
    """Return the policy / dataset target FPS (SmolVLA LIBERO default: 10)."""
    if policy_fps is None:
        return SMOLVLA_LIBERO_TARGET_FPS
    fps = int(policy_fps)
    if fps <= 0:
        raise ValueError(f"policy_fps must be positive, got {policy_fps!r}")
    return fps


def configure_libero_control_freq(control_freq: int) -> None:
    """Record intended LIBERO ``control_freq`` for downstream env creation.

    LeRobot ``LiberoEnv`` builds ``OffScreenRenderEnv`` without exposing
    ``control_freq``; pass ``control_freq`` when constructing LIBERO envs
    directly, or read ``get_intended_libero_control_freq()`` from an env hook
    and forward it into ``OffScreenRenderEnv(..., control_freq=...)``.
    """
    freq = int(control_freq)
    if freq <= 0:
        raise ValueError(f"control_freq must be positive, got {control_freq!r}")
    global _INTENDED_LIBERO_CONTROL_FREQ
    _INTENDED_LIBERO_CONTROL_FREQ = freq


def get_intended_libero_control_freq() -> int | None:
    """Return the last ``configure_libero_control_freq`` value, if any."""
    return _INTENDED_LIBERO_CONTROL_FREQ


def assert_control_rate_aligned(
    control_freq: float,
    model_timestep: float,
    target_fps: int,
) -> None:
    """Validate sim control rate vs dataset/policy FPS.

    When ``control_freq == target_fps``, asserts ``1 / control_timestep ≈ target_fps``.
    When ``control_freq == 2 * target_fps`` (LIBERO default 20 Hz vs SmolVLA 10 Hz),
    recording must subsample simulator steps with ``stride=2`` — no error is raised.
    Any other mismatch raises ``ValueError``.
    """
    if model_timestep <= 0:
        raise ValueError(f"model_timestep must be positive, got {model_timestep}")

    control_timestep = 1.0 / float(control_freq)
    effective_hz = 1.0 / control_timestep

    if abs(control_freq - target_fps) < 1e-6:
        if abs(effective_hz - target_fps) >= 1e-3:
            raise ValueError(
                f"control_freq={control_freq} but 1/control_timestep={effective_hz:.4f} "
                f"!= target_fps={target_fps}"
            )
        return

    if abs(control_freq - 2 * target_fps) < 1e-6:
        # LIBERO default 20 Hz control with 10 Hz dataset: subsample every 2nd step.
        return

    raise ValueError(
        f"control_freq={control_freq} is not aligned with target_fps={target_fps}. "
        f"Set control_freq=target_fps ({target_fps}) or use stride="
        f"{int(round(control_freq / target_fps))} subsampling when "
        f"control_freq={DEFAULT_LIBERO_CONTROL_FREQ}."
    )


def recording_stride(control_freq: float, target_fps: int) -> int:
    """Return how many sim steps to skip between dataset frames."""
    assert_control_rate_aligned(control_freq, DEFAULT_MUJOCO_MODEL_TIMESTEP, target_fps)
    if abs(control_freq - target_fps) < 1e-6:
        return 1
    if abs(control_freq - 2 * target_fps) < 1e-6:
        return 2
    return max(1, int(round(control_freq / target_fps)))


def assert_dataset_fps(dataset_fps: int | float, policy_fps: int | float) -> None:
    """Raise ``ValueError`` when recorded dataset FPS differs from policy FPS."""
    ds = float(dataset_fps)
    pol = float(policy_fps)
    if abs(ds - pol) >= 1e-6:
        raise ValueError(
            f"Dataset FPS ({ds}) does not match policy/target FPS ({pol}). "
            "SmolVLA fine-tuning requires matching rates."
        )
