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

"""Configuration for evaluation-time fault injection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_SUPPORTED_TYPES = (
    "action_hold",
    "action_delay",
    "action_jitter",
    "sensor_dropout",
    "visual_occlusion",
    "visual_blur",
    "brightness_drop",
    "obs_latency",
    "object_slip",
    "eef_bump",
    "midair_drop",
)

_BURST_VISUAL_TYPES = frozenset({"visual_occlusion", "visual_blur", "brightness_drop"})
_WINDOW_PHYSICAL_TYPES = frozenset({"object_slip", "eef_bump", "midair_drop"})
_SIM_INJECT_TYPES = frozenset({"object_slip", "eef_bump"})
_RECOVERY_TYPES = frozenset({"midair_drop"})


@dataclass
class FaultInjectionConfig:
    """Controls optional fault injection during ``lerobot-eval`` rollouts.

    When ``enabled`` is False (the default), evaluation behavior is unchanged:
    proposed actions reach ``env.step`` without modification and no fault events
    are logged.
    """

    enabled: bool = False
    type: str = "action_hold"
    # Episode step (0-indexed) at which the fault begins. Must be >= 1 for
    # action_hold. Ignored for action_delay / action_jitter.
    trigger_step: int = 55
    # Number of environment steps to hold the previous action (action_hold) or
    # burst length for observation faults.
    duration: int = 8
    # Probability in [0, 1] that the fault activates when the trigger is reached.
    probability: float = 1.0
    # RNG seed for activation / jitter draws. Independent of eval seed.
    seed: int | None = 42
    # FIFO depth for action_delay: execute the action from delay_steps ago.
    delay_steps: int = 3
    # Gaussian std for action_jitter: N(0, noise_std) per action dim. 0 = identity.
    noise_std: float = 0.05
    # Vector-env indices to apply the fault to. None means all environments.
    env_ids: list[int] | None = None
    # JSONL path for fault events. Relative paths are resolved against the eval
    # ``output_dir`` in ``eval_main`` / :func:`resolve_fault_log_path`.
    log_path: Path | None = None
    # Optional directory for sensor/visual diagnostic PNG frames.
    diag_dir: Path | None = None
    # Optional directory for wrist (camera2) policy GIFs (clean + faulted).
    policy_video_dir: Path | None = None
    # visual_occlusion: random box size as fraction of image H/W.
    occlusion_h_frac_min: float = 0.25
    occlusion_h_frac_max: float = 0.45
    occlusion_w_frac_min: float = 0.25
    occlusion_w_frac_max: float = 0.45
    # visual_blur: box-blur radius (pixels); mapped from blur_sigma.
    blur_sigma: float = 3.0
    # brightness_drop: multiplicative scale applied to image intensities.
    brightness_scale: float = 0.25
    # obs_latency: serve observation from this many steps ago.
    latency_steps: int = 3
    # midair_drop / object_slip / eef_bump: inclusive episode-step window (0-indexed).
    t_min: int = 10
    t_max: int = 30
    # midair_drop / object_slip: LIBERO object body name.
    object_name: str = "alphabet_soup_1"
    # midair_drop: Gaussian std for linear / angular velocity impulse (m/s, rad/s).
    impulse_lin_std: float = 0.5
    impulse_ang_std: float = 0.2
    # midair_drop: mean linear impulse before noise (m/s). Z usually negative.
    impulse_lin_bias: tuple[float, float, float] = (0.0, 0.0, -0.55)
    # midair_drop: sim steps with gripper open before impulse.
    gripper_settle_steps: int = 5
    # After grasp+height gates pass, wait this many env steps before dropping.
    post_grasp_delay_steps: int = 0
    # Drop while object is still this far (XY, meters) from the basket. <= 0 disables.
    min_drop_distance_from_basket_m: float = 0.18
    # midair_drop: recovery planner output FPS.
    recovery_fps: int = 10
    # midair_drop: explicit basket / place target (x, y, z). None = auto from sim.
    recovery_destination: tuple[float, float, float] | None = None
    # midair_drop: basket body name for auto place destination.
    basket_name: str = "basket_1"
    # midair_drop: align gripper across the can's short axis before regrasp.
    side_grasp_enabled: bool = True
    # midair_drop: seat object into basket if already over the rim after release.
    seat_assist_enabled: bool = True
    # midair_drop recovery randomization (fine-tuning trajectory diversity).
    waypoint_noise_m: float = 0.015
    recovery_action_noise_std: float = 0.02
    speed_multiplier_min: float = 0.8
    speed_multiplier_max: float = 1.2
    arm_posture_noise_deg: float = 3.0
    # object_slip: Gaussian std for position / yaw (rad) nudge.
    slip_pos_std: float = 0.02
    slip_yaw_std: float = 0.15
    # eef_bump: Gaussian std for external force components (sim units).
    bump_force_std: float = 40.0
    # Physics substeps after a sim-state disturbance.
    settle_steps: int = 5
    # object_slip: only trigger when the object is grasped.
    require_grasp: bool = True
    # object_slip: require object world-z at/above this height (meters). <= 0 disables.
    min_object_z: float = 0.12

    def __post_init__(self) -> None:
        if isinstance(self.log_path, str):
            self.log_path = Path(self.log_path)
        if isinstance(self.diag_dir, str):
            self.diag_dir = Path(self.diag_dir)
        if isinstance(self.policy_video_dir, str):
            self.policy_video_dir = Path(self.policy_video_dir)
        if self.env_ids is not None:
            self.env_ids = list(self.env_ids)
        self.validate()

    def validate(self, num_envs: int | None = None) -> None:
        """Raise ``ValueError`` if configuration fields are invalid."""
        if self.type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported fault type {self.type!r}. Currently supported: {list(_SUPPORTED_TYPES)}."
            )
        if self.type == "action_hold":
            if self.trigger_step < 1:
                raise ValueError(
                    f"trigger_step must be >= 1 so a previous valid action exists to hold "
                    f"(got {self.trigger_step})."
                )
            if self.duration < 1:
                raise ValueError(f"duration must be >= 1 (got {self.duration}).")
            if not (0.0 <= self.probability <= 1.0):
                raise ValueError(f"probability must be in [0.0, 1.0] (got {self.probability}).")
        elif self.type == "action_delay":
            if self.delay_steps < 1:
                raise ValueError(f"delay_steps must be >= 1 (got {self.delay_steps}).")
        elif self.type == "action_jitter":
            if self.noise_std < 0:
                raise ValueError(f"noise_std must be >= 0 (got {self.noise_std}).")
        elif self.type == "sensor_dropout" or self.type in _BURST_VISUAL_TYPES:
            if self.trigger_step < 0:
                raise ValueError(f"trigger_step must be >= 0 (got {self.trigger_step}).")
            if self.duration < 1:
                raise ValueError(f"duration must be >= 1 (got {self.duration}).")
            if not (0.0 <= self.probability <= 1.0):
                raise ValueError(f"probability must be in [0.0, 1.0] (got {self.probability}).")
            if self.type == "visual_occlusion":
                for name, lo, hi in (
                    ("occlusion_h", self.occlusion_h_frac_min, self.occlusion_h_frac_max),
                    ("occlusion_w", self.occlusion_w_frac_min, self.occlusion_w_frac_max),
                ):
                    if not (0.0 < lo <= hi <= 1.0):
                        raise ValueError(f"{name} frac range invalid: [{lo}, {hi}].")
            if self.type == "visual_blur" and self.blur_sigma <= 0:
                raise ValueError(f"blur_sigma must be > 0 (got {self.blur_sigma}).")
            if self.type == "brightness_drop" and not (0.0 <= self.brightness_scale <= 1.0):
                raise ValueError(
                    f"brightness_scale must be in [0.0, 1.0] (got {self.brightness_scale})."
                )
        elif self.type == "obs_latency":
            if self.latency_steps < 1:
                raise ValueError(f"latency_steps must be >= 1 (got {self.latency_steps}).")
        elif self.type in _WINDOW_PHYSICAL_TYPES:
            if self.t_min < 0:
                raise ValueError(f"t_min must be >= 0 (got {self.t_min}).")
            if self.t_max < self.t_min:
                raise ValueError(f"t_max must be >= t_min (got t_max={self.t_max}, t_min={self.t_min}).")
            if self.settle_steps < 0:
                raise ValueError(f"settle_steps must be >= 0 (got {self.settle_steps}).")
            if not (0.0 <= self.probability <= 1.0):
                raise ValueError(f"probability must be in [0.0, 1.0] (got {self.probability}).")
            if self.type == "object_slip":
                if self.slip_pos_std < 0:
                    raise ValueError(f"slip_pos_std must be >= 0 (got {self.slip_pos_std}).")
                if self.slip_yaw_std < 0:
                    raise ValueError(f"slip_yaw_std must be >= 0 (got {self.slip_yaw_std}).")
                if self.min_object_z < 0:
                    raise ValueError(f"min_object_z must be >= 0 (got {self.min_object_z}).")
                if not self.object_name:
                    raise ValueError("object_name must be non-empty.")
            elif self.type == "eef_bump":
                if self.bump_force_std < 0:
                    raise ValueError(f"bump_force_std must be >= 0 (got {self.bump_force_std}).")
            elif self.type == "midair_drop":
                if self.impulse_lin_std < 0:
                    raise ValueError(f"impulse_lin_std must be >= 0 (got {self.impulse_lin_std}).")
                if self.impulse_ang_std < 0:
                    raise ValueError(f"impulse_ang_std must be >= 0 (got {self.impulse_ang_std}).")
                if len(self.impulse_lin_bias) != 3:
                    raise ValueError(
                        f"impulse_lin_bias must have length 3 (got {self.impulse_lin_bias})."
                    )
                if self.post_grasp_delay_steps < 0:
                    raise ValueError(
                        f"post_grasp_delay_steps must be >= 0 (got {self.post_grasp_delay_steps})."
                    )
                if self.gripper_settle_steps < 0:
                    raise ValueError(
                        f"gripper_settle_steps must be >= 0 (got {self.gripper_settle_steps})."
                    )
                if self.min_object_z < 0:
                    raise ValueError(f"min_object_z must be >= 0 (got {self.min_object_z}).")
                if self.min_drop_distance_from_basket_m < 0:
                    raise ValueError(
                        f"min_drop_distance_from_basket_m must be >= 0 "
                        f"(got {self.min_drop_distance_from_basket_m})."
                    )
                if self.recovery_fps < 1:
                    raise ValueError(f"recovery_fps must be >= 1 (got {self.recovery_fps}).")
                if not self.object_name:
                    raise ValueError("object_name must be non-empty.")
                if not self.basket_name:
                    raise ValueError("basket_name must be non-empty.")
                if self.waypoint_noise_m < 0:
                    raise ValueError(
                        f"waypoint_noise_m must be >= 0 (got {self.waypoint_noise_m})."
                    )
                if self.recovery_action_noise_std < 0:
                    raise ValueError(
                        f"recovery_action_noise_std must be >= 0 "
                        f"(got {self.recovery_action_noise_std})."
                    )
                if self.speed_multiplier_min <= 0 or self.speed_multiplier_max <= 0:
                    raise ValueError("speed_multiplier bounds must be positive.")
                if self.speed_multiplier_max < self.speed_multiplier_min:
                    raise ValueError(
                        f"speed_multiplier_max must be >= speed_multiplier_min "
                        f"(got max={self.speed_multiplier_max}, min={self.speed_multiplier_min})."
                    )
                if self.arm_posture_noise_deg < 0:
                    raise ValueError(
                        f"arm_posture_noise_deg must be >= 0 (got {self.arm_posture_noise_deg})."
                    )
        if self.env_ids is not None:
            if len(self.env_ids) == 0:
                raise ValueError("env_ids must be non-empty when provided (or leave as None for all).")
            if any(i < 0 for i in self.env_ids):
                raise ValueError(f"env_ids must be non-negative (got {self.env_ids}).")
            if len(self.env_ids) != len(set(self.env_ids)):
                raise ValueError(f"env_ids contain duplicates: {self.env_ids}.")
            if num_envs is not None and any(i >= num_envs for i in self.env_ids):
                raise ValueError(f"env_ids out of range for num_envs={num_envs}: {self.env_ids}.")


def resolve_fault_log_path(log_path: Path | str | None, output_dir: Path | str) -> Path:
    """Resolve a fault log path, anchoring relative paths under ``output_dir``.

    - ``None`` → ``<output_dir>/fault_events.jsonl``
    - absolute path → unchanged
    - relative path → ``<output_dir>/<log_path>``
    """
    output = Path(output_dir)
    if log_path is None:
        return output / "fault_events.jsonl"
    path = Path(log_path)
    if path.is_absolute():
        return path
    return output / path


def default_fault_config() -> FaultInjectionConfig:
    """Factory for ``EvalPipelineConfig.fault`` (disabled by default)."""
    return FaultInjectionConfig()
