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

"""Cached SmolVLA policy and processor bundles for datagen matrix runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SmolVLAPolicyResources:
    """Reusable SmolVLA policy, processors, and env hooks for multiple episodes."""

    policy_path: str
    device: str
    task: str
    task_id: int
    policy_cfg: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any


def load_smolvla_policy_resources(
    *,
    policy_path: str,
    device: str,
    task: str,
    task_id: int,
) -> SmolVLAPolicyResources:
    """Load SmolVLA policy and pre/post processors once per datagen run."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = device
    if hasattr(policy_cfg, "empty_cameras"):
        policy_cfg.empty_cameras = 1

    env_cfg = LiberoEnv(
        task=task,
        task_ids=[int(task_id)],
        control_mode="relative",
        camera_name_mapping={
            "agentview_image": "camera1",
            "robot0_eye_in_hand_image": "camera2",
        },
    )
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )
    return SmolVLAPolicyResources(
        policy_path=policy_path,
        device=device,
        task=task,
        task_id=int(task_id),
        policy_cfg=policy_cfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
    )
