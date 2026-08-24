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

"""Mid-air drop recovery: planner, injector, and demo helpers."""

from lerobot.faults.recovery.dataset_logger import FaultRecoveryDatasetLogger
from lerobot.faults.recovery.evaluation import evaluate_recovery_episode
from lerobot.faults.recovery.fps import (
    DEFAULT_LIBERO_CONTROL_FREQ,
    SMOLVLA_LIBERO_TARGET_FPS,
    assert_control_rate_aligned,
    assert_dataset_fps,
    configure_libero_control_freq,
    recording_stride,
    resolve_target_fps,
)
from lerobot.faults.recovery.libero_hook import install_libero_control_freq_hook
from lerobot.faults.recovery.loss_mask import loss_mask_for_step, loss_mask_from_fault
from lerobot.faults.recovery.midair_drop import MidAirDropFault
from lerobot.faults.recovery.planner import SimpleIKRecoveryPlanner

__all__ = [
    "DEFAULT_LIBERO_CONTROL_FREQ",
    "FaultRecoveryDatasetLogger",
    "MidAirDropFault",
    "SMOLVLA_LIBERO_TARGET_FPS",
    "SimpleIKRecoveryPlanner",
    "assert_control_rate_aligned",
    "assert_dataset_fps",
    "configure_libero_control_freq",
    "evaluate_recovery_episode",
    "install_libero_control_freq_hook",
    "loss_mask_for_step",
    "loss_mask_from_fault",
    "recording_stride",
    "resolve_target_fps",
]
