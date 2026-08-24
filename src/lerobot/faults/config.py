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
)

_BURST_VISUAL_TYPES = frozenset({"visual_occlusion", "visual_blur", "brightness_drop"})


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
