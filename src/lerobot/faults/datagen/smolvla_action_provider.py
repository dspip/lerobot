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

"""SmolVLA observation-to-action conversion for hybrid SimpleIK episodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lerobot.envs.utils import preprocess_observation
from lerobot.faults.datagen.smolvla_resources import SmolVLAPolicyResources
from lerobot.utils.constants import ACTION

__all__ = ["SmolVLAActionProvider"]


class SmolVLAActionProvider:
    """Run one SmolVLA ``select_action`` pass on a LIBERO observation dict."""

    def __init__(self, resources: SmolVLAPolicyResources, *, task: str) -> None:
        """Store cached policy resources and the natural-language task string."""
        self._resources = resources
        self._task = str(task)

    def reset(self) -> None:
        """Clear the policy action-chunk queue (post-drop ``reset_then_ik``)."""
        self._resources.policy.reset()

    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a (7,) float32 action from a raw env observation."""
        obs_dict = preprocess_observation(observation)
        obs_dict["task"] = [self._task]
        obs_dict = self._resources.env_preprocessor(obs_dict)
        obs_dict = self._resources.preprocessor(obs_dict)
        with torch.inference_mode():
            action = self._resources.policy.select_action(obs_dict)
        action = self._resources.postprocessor(action)
        action = self._resources.env_postprocessor({ACTION: action})[ACTION]
        action_numpy = np.asarray(action.to("cpu").numpy(), dtype=np.float32)
        if action_numpy.ndim == 2:
            action_numpy = action_numpy[0]
        return np.asarray(action_numpy, dtype=np.float32).reshape(7)
