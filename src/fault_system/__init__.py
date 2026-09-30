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

from fault_system.action.delay import ActionDelayFault
from fault_system.action.hold import ActionHoldFault
from fault_system.action.jitter import ActionJitterFault
from fault_system.annotation import (
    FAILURE_ANNOTATION_FEATURES,
    FailureAnnotator,
    default_failure_frame,
    failure_frame_from_info,
    failure_type_id,
)
from fault_system.config import FaultInjectionConfig, default_fault_config, resolve_fault_log_path
from fault_system.factory import (
    make_action_fault_injector,
    make_fault_injector,
    make_midair_drop_fault,
    make_obs_fault_injector,
    make_sim_inject_fault,
)
from fault_system.logging import FaultEventLogger
from fault_system.observation.brightness_drop import BrightnessDropFault
from fault_system.observation.obs_latency import ObsLatencyFault
from fault_system.observation.sensor_dropout import SensorDropoutFault
from fault_system.observation.visual_blur import VisualBlurFault
from fault_system.observation.visual_occlusion import VisualOcclusionFault
from fault_system.recovery.dataset_logger import FaultRecoveryDatasetLogger
from fault_system.recovery.evaluation import evaluate_recovery_episode
from fault_system.recovery.midair_drop import MidAirDropFault
from fault_system.recovery.planner import SimpleIKRecoveryPlanner
from fault_system.sim.eef_bump import EefBumpFault
from fault_system.sim.object_slip import ObjectSlipFault
from fault_system.wrappers import (
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
    "FAILURE_ANNOTATION_FEATURES",
    "FailureAnnotator",
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
    "default_failure_frame",
    "evaluate_recovery_episode",
    "failure_frame_from_info",
    "failure_type_id",
    "make_action_fault_injector",
    "make_fault_injector",
    "make_midair_drop_fault",
    "make_obs_fault_injector",
    "make_sim_inject_fault",
    "maybe_wrap_env",
    "maybe_wrap_env_tree",
    "resolve_fault_log_path",
]
