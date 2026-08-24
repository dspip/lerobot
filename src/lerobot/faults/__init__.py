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
    make_midair_drop_fault,
    make_obs_fault_injector,
    make_sim_inject_fault,
)
from lerobot.faults.logging import FaultEventLogger
from lerobot.faults.observation.brightness_drop import BrightnessDropFault
from lerobot.faults.observation.obs_latency import ObsLatencyFault
from lerobot.faults.observation.sensor_dropout import SensorDropoutFault
from lerobot.faults.observation.visual_blur import VisualBlurFault
from lerobot.faults.observation.visual_occlusion import VisualOcclusionFault
from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger
from lerobot.faults.recovery.evaluation import evaluate_recovery_episode
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from lerobot.faults.recovery.planner import SimpleIKRecoveryPlanner
from lerobot.faults.sim.eef_bump import EefBumpFault
from lerobot.faults.sim.object_slip import ObjectSlipFault
from lerobot.faults.wrappers import (
    DropRecoveryEnvWrapper,
    FaultEnvWrapper,
    SimFaultEnvWrapper,
    maybe_wrap_env,
    maybe_wrap_env_tree,
)

__all__ = [
    "ActionDelayFault",
    "ActionHoldFault",
    "ActionJitterFault",
    "BrightnessDropFault",
    "DropRecoveryEnvWrapper",
    "EefBumpFault",
    "FaultEnvWrapper",
    "FaultEventLogger",
    "FaultInjectionConfig",
    "FaultRecoveryDatasetLogger",
    "MidAirDropFault",
    "ObjectSlipFault",
    "ObsLatencyFault",
    "SensorDropoutFault",
    "SimpleIKRecoveryPlanner",
    "SimFaultEnvWrapper",
    "VisualBlurFault",
    "VisualOcclusionFault",
    "default_fault_config",
    "evaluate_recovery_episode",
    "make_action_fault_injector",
    "make_fault_injector",
    "make_midair_drop_fault",
    "make_obs_fault_injector",
    "make_sim_inject_fault",
    "maybe_wrap_env",
    "maybe_wrap_env_tree",
    "resolve_fault_log_path",
]
