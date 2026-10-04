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

"""ActionSource contract: raw observation in, env action out, no simulator."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from fault_system.datagen.smolvla_resources import SmolVLAPolicyResources
from fault_system.models.action_source import LIBERO_PANDA_ACTION_DIM, as_libero_action
from fault_system.models.smolvla import SmolVLAActionSource
from lerobot.utils.constants import ACTION

_RAW_OBS = {
    "pixels": {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "image2": np.zeros((4, 4, 3), dtype=np.uint8),
    },
    "robot_state": np.zeros(8, dtype=np.float32),
}


def _resources(policy: Any, processors: dict[str, Any]) -> SmolVLAPolicyResources:
    return SmolVLAPolicyResources(
        policy_path="lerobot/smolvla_libero",
        device="cpu",
        task="libero_object",
        task_id=0,
        policy_cfg=MagicMock(),
        policy=policy,
        preprocessor=processors["preprocessor"],
        postprocessor=processors["postprocessor"],
        env_preprocessor=processors["env_preprocessor"],
        env_postprocessor=processors["env_postprocessor"],
    )


def test_action_source_module_does_not_import_the_simulator_or_lerobot() -> None:
    path = Path(__file__).resolve().parents[2] / "src" / "fault_system" / "models" / "action_source.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    joined = " ".join(imported)
    assert "lerobot" not in joined
    assert "fault_system.sim" not in joined
    assert "fault_system.datagen" not in joined


def test_as_libero_action_accepts_a_row_and_rejects_the_wrong_width() -> None:
    vector = as_libero_action(np.ones((1, LIBERO_PANDA_ACTION_DIM), dtype=np.float64))
    assert vector.shape == (7,)
    assert vector.dtype == np.float32
    with pytest.raises(ValueError, match="Expected action shape"):
        as_libero_action(np.zeros(3, dtype=np.float32))


def test_smolvla_action_source_preprocesses_inside_the_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def _preprocess(observation: dict[str, Any]) -> dict[str, Any]:
        seen["raw_is_gym"] = "pixels" in observation
        return {"observation.state": observation["robot_state"]}

    monkeypatch.setattr("fault_system.models.smolvla.preprocess_observation", _preprocess)

    def _env_preprocessor(obs: dict[str, Any]) -> dict[str, Any]:
        seen["task"] = obs["task"]
        return obs

    def _preprocessor(obs: dict[str, Any]) -> dict[str, Any]:
        seen["to_policy"] = obs
        return obs

    def _postprocessor(action: torch.Tensor) -> torch.Tensor:
        return action + 1

    def _env_postprocessor(batch: dict[str, Any]) -> dict[str, Any]:
        return {ACTION: batch[ACTION]}

    policy = MagicMock()
    policy.select_action.return_value = torch.zeros(1, LIBERO_PANDA_ACTION_DIM)
    source = SmolVLAActionSource(
        _resources(
            policy,
            {
                "preprocessor": _preprocessor,
                "postprocessor": _postprocessor,
                "env_preprocessor": _env_preprocessor,
                "env_postprocessor": _env_postprocessor,
            },
        ),
        task="pick up the soup",
    )

    action = source.act(_RAW_OBS, task="put the can in the basket")

    assert seen["raw_is_gym"] is True
    assert seen["task"] == ["put the can in the basket"]
    assert "pixels" not in seen["to_policy"]
    policy.select_action.assert_called_once()
    assert policy.select_action.call_args.args[0] is seen["to_policy"]
    np.testing.assert_array_equal(action, np.ones(7, dtype=np.float32))
    source.reset()
    policy.reset.assert_called_once()
