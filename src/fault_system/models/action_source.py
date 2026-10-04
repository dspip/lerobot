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

"""Action source: raw observation in, env action out.

Implementations own model preprocess and ``select_action``. They do not receive
a simulator and they do not call ``env.step``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

# LIBERO Panda relative control: dx, dy, dz, dax, day, daz, gripper.
LIBERO_PANDA_ACTION_DIM = 7


class ActionSource(Protocol):
    """One env action per control tick.

    ``observation`` is the raw Gym dict (``pixels`` and ``robot_state`` for
    LIBERO). ``task`` is the language instruction. ``act`` returns ``float32``
    of shape ``(action_dim,)``.
    """

    def reset(self) -> None:
        """Clear model state, including any action-chunk queue, at episode start."""

    def act(self, observation: Mapping[str, Any], *, task: str | None = None) -> np.ndarray:
        """Return the next env action. Do not step the simulator."""


def as_libero_action(action: Any, *, action_dim: int = LIBERO_PANDA_ACTION_DIM) -> np.ndarray:
    """Return a ``float32`` vector of shape ``(action_dim,)``."""
    if hasattr(action, "detach"):
        action = action.detach().to("cpu").numpy()
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError(f"Expected a single action row, got shape {array.shape}.")
        array = array[0]
    vector = np.asarray(array, dtype=np.float32).reshape(-1)
    if vector.shape != (action_dim,):
        raise ValueError(f"Expected action shape ({action_dim},), got {vector.shape}.")
    return vector
