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

"""Evaluation-time fault injection for LeRobot policies."""

from lerobot.faults.action.delay import ActionDelayFault
from lerobot.faults.action.hold import ActionHoldFault
from lerobot.faults.action.jitter import ActionJitterFault
from lerobot.faults.config import FaultInjectionConfig, default_fault_config, resolve_fault_log_path
from lerobot.faults.factory import (
    make_action_fault_injector,
    make_fault_injector,
    make_obs_fault_injector,
    make_sim_inject_fault,
)
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.brightness_drop import BrightnessDropFault
from lerobot.faults.observation.obs_latency import ObsLatencyFault
from lerobot.faults.observation.sensor_dropout import SensorDropoutFault
from lerobot.faults.observation.visual_blur import VisualBlurFault
from lerobot.faults.observation.visual_occlusion import VisualOcclusionFault
from lerobot.faults.sim.eef_bump import EefBumpFault
from lerobot.faults.sim.object_slip import ObjectSlipFault
from lerobot.faults.wrappers import FaultEnvWrapper, SimFaultEnvWrapper, maybe_wrap_env, maybe_wrap_env_tree

__all__ = [
    "ActionDelayFault",
    "ActionHoldFault",
    "ActionJitterFault",
    "BrightnessDropFault",
    "EefBumpFault",
    "FaultEnvWrapper",
    "FaultEventLogger",
    "FaultInjectionConfig",
    "ObjectSlipFault",
    "ObsLatencyFault",
    "SensorDropoutFault",
    "SimFaultEnvWrapper",
    "VisualBlurFault",
    "VisualOcclusionFault",
    "default_fault_config",
    "make_action_fault_injector",
    "make_fault_injector",
    "make_obs_fault_injector",
    "make_sim_inject_fault",
    "maybe_wrap_env",
    "maybe_wrap_env_tree",
    "resolve_fault_log_path",
]
