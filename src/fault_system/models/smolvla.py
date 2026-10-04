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

"""SmolVLA adapter: raw Gym observation to a LIBERO action vector."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from fault_system.datagen.smolvla_resources import SmolVLAPolicyResources, load_smolvla_policy_resources
from fault_system.models.action_source import as_libero_action
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import ACTION

__all__ = ["SmolVLAActionSource"]


class SmolVLAActionSource:
    """``ActionSource`` over a loaded SmolVLA policy.

    Preprocess, ``select_action``, and postprocess stay in this class. Callers
    pass the raw Gym observation and receive ``float32 (7,)``.
    """

    def __init__(self, resources: SmolVLAPolicyResources, *, task: str) -> None:
        """Store the loaded policy bundle and the default language instruction."""
        self._resources = resources
        self._task = str(task)

    @classmethod
    def from_pretrained(
        cls,
        *,
        policy_path: str,
        device: str,
        env_task: str,
        task_id: int,
        task: str,
    ) -> SmolVLAActionSource:
        """Load SmolVLA weights and return an action source.

        ``env_task`` is the LIBERO suite name used only to size the policy.
        ``task`` is the language string passed into ``act``.
        """
        resources = load_smolvla_policy_resources(
            policy_path=policy_path,
            device=device,
            task=env_task,
            task_id=int(task_id),
        )
        return cls(resources, task=task)

    def reset(self) -> None:
        """Clear the policy action-chunk queue."""
        self._resources.policy.reset()

    def act(self, observation: Mapping[str, Any], *, task: str | None = None) -> np.ndarray:
        """Return a ``(7,)`` float32 action from a raw env observation."""
        language = self._task if task is None else str(task)
        obs_dict = preprocess_observation(dict(observation))
        obs_dict["task"] = [language]
        obs_dict = self._resources.env_preprocessor(obs_dict)
        obs_dict = self._resources.preprocessor(obs_dict)
        with torch.inference_mode():
            action = self._resources.policy.select_action(obs_dict)
        action = self._resources.postprocessor(action)
        action = self._resources.env_postprocessor({ACTION: action})[ACTION]
        return as_libero_action(action)
