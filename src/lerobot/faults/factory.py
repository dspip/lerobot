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

from lerobot.faults.action.hold import ActionHoldFault
from lerobot.faults.config import FaultInjectionConfig
from lerobot.faults.logging import FaultEventLogger

ActionFaultInjector = ActionHoldFault

_ACTION_TYPES = frozenset({"action_hold"})


def make_action_fault_injector(
    config: FaultInjectionConfig | None,
    num_envs: int,
    log_path: str | Path | None = None,
) -> ActionFaultInjector | None:
    """Build an action injector, or ``None`` when disabled / not an action fault."""
    if config is None or not config.enabled or config.type not in _ACTION_TYPES:
        return None
    config.validate(num_envs=num_envs)
    path = log_path if log_path is not None else config.log_path
    logger = FaultEventLogger(path) if path is not None else None
    if config.type == "action_hold":
        return ActionHoldFault(config=config, num_envs=num_envs, event_logger=logger)
    return None


def make_fault_injector(
    config: FaultInjectionConfig | None,
    num_envs: int,
    log_path: str | Path | None = None,
) -> ActionFaultInjector | None:
    """Backward-compatible alias for :func:`make_action_fault_injector`."""
    return make_action_fault_injector(config, num_envs=num_envs, log_path=log_path)
