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

"""Factory for fault injectors from configuration."""

from __future__ import annotations

from pathlib import Path

from lerobot.faults.action.delay import ActionDelayFault
from lerobot.faults.action.hold import ActionHoldFault
from lerobot.faults.action.jitter import ActionJitterFault
from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.brightness_drop import BrightnessDropFault
from lerobot.faults.observation.obs_latency import ObsLatencyFault
from lerobot.faults.observation.sensor_dropout import SensorDropoutFault
from lerobot.faults.observation.visual_blur import VisualBlurFault
from lerobot.faults.observation.visual_occlusion import VisualOcclusionFault
from lerobot.faults.sim.eef_bump import EefBumpFault
from lerobot.faults.sim.object_slip import ObjectSlipFault

ActionFaultInjector = ActionHoldFault | ActionDelayFault | ActionJitterFault
ObsFaultInjector = (
    SensorDropoutFault
    | VisualOcclusionFault
    | VisualBlurFault
    | BrightnessDropFault
    | ObsLatencyFault
)
SimInjectFaultInjector = ObjectSlipFault | EefBumpFault

_ACTION_TYPES = frozenset({"action_hold", "action_delay", "action_jitter"})
_OBS_TYPES = frozenset(
    {
        "sensor_dropout",
        "visual_occlusion",
        "visual_blur",
        "brightness_drop",
        "obs_latency",
    }
)
_SIM_INJECT_TYPES = frozenset({"object_slip", "eef_bump"})


def _build_logger(config: FaultInjectionConfig, log_path: str | Path | None) -> FaultEventLogger | None:
    path = log_path if log_path is not None else config.log_path
    return FaultEventLogger(path) if path is not None else None


def make_action_fault_injector(
    config: FaultInjectionConfig | None,
    num_envs: int,
    log_path: str | Path | None = None,
) -> ActionFaultInjector | None:
    """Build an action injector, or ``None`` when disabled / not an action fault."""
    if config is None or not config.enabled or config.type not in _ACTION_TYPES:
        return None
    config.validate(num_envs=num_envs)
    logger = _build_logger(config, log_path)
    if config.type == "action_hold":
        return ActionHoldFault(config=config, num_envs=num_envs, event_logger=logger)
    if config.type == "action_delay":
        return ActionDelayFault(config=config, num_envs=num_envs, event_logger=logger)
    if config.type == "action_jitter":
        return ActionJitterFault(config=config, num_envs=num_envs, event_logger=logger)
    return None


def make_obs_fault_injector(
    config: FaultInjectionConfig | None,
    num_envs: int,
    log_path: str | Path | None = None,
) -> ObsFaultInjector | None:
    """Build an observation injector, or ``None`` when disabled / not an obs fault."""
    if config is None or not config.enabled or config.type not in _OBS_TYPES:
        return None
    config.validate(num_envs=num_envs)
    logger = _build_logger(config, log_path)
    if config.type == "sensor_dropout":
        return SensorDropoutFault(config=config, num_envs=num_envs, event_logger=logger)
    if config.type == "visual_occlusion":
        return VisualOcclusionFault(config=config, num_envs=num_envs, event_logger=logger)
    if config.type == "visual_blur":
        return VisualBlurFault(config=config, num_envs=num_envs, event_logger=logger)
    if config.type == "brightness_drop":
        return BrightnessDropFault(config=config, num_envs=num_envs, event_logger=logger)
    if config.type == "obs_latency":
        return ObsLatencyFault(config=config, num_envs=num_envs, event_logger=logger)
    return None


def make_sim_inject_fault(
    config: FaultInjectionConfig | None,
    num_envs: int,
    log_path: str | Path | None = None,
) -> SimInjectFaultInjector | None:
    """Build an inject-only sim-state fault, or ``None`` when disabled."""
    if config is None or not config.enabled or config.type not in _SIM_INJECT_TYPES:
        return None
    config.validate(num_envs=num_envs)
    logger = _build_logger(config, log_path)
    if config.type == "object_slip":
        return ObjectSlipFault(config=config, num_envs=num_envs, event_logger=logger)
    if config.type == "eef_bump":
        return EefBumpFault(config=config, num_envs=num_envs, event_logger=logger)
    return None


def make_fault_injector(
    config: FaultInjectionConfig | None,
    num_envs: int,
    log_path: str | Path | None = None,
) -> ActionFaultInjector | None:
    """Backward-compatible alias for :func:`make_action_fault_injector`."""
    return make_action_fault_injector(config, num_envs=num_envs, log_path=log_path)
