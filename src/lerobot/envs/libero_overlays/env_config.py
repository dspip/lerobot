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

"""``--env.type=libero_overlay`` adds ``--env.overlay`` without editing stock LIBERO."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.envs.configs import EnvConfig, LiberoEnv, _make_vec_env_cls


@EnvConfig.register_subclass("libero_overlay")
@dataclass
class LiberoOverlayEnvConfig(LiberoEnv):
    """LIBERO env config plus an optional BDDL overlay YAML."""

    # Path to a YAML overlay. Stock BDDL is not edited. None keeps the stock scene.
    overlay: str | None = None

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        """Build overlay LIBERO vector envs. Imports LIBERO only when called."""
        from lerobot.envs.libero_overlays.gym_env import create_overlay_libero_envs

        if self.task is None:
            raise ValueError("LiberoOverlayEnvConfig requires a task to be specified")
        env_cls = _make_vec_env_cls(use_async_envs, n_envs)
        return create_overlay_libero_envs(
            task=self.task,
            n_envs=n_envs,
            camera_name=self.camera_name,
            init_states=self.init_states,
            gym_kwargs=self.gym_kwargs,
            env_cls=env_cls,
            control_mode=self.control_mode,
            episode_length=self.episode_length,
            camera_name_mapping=self.camera_name_mapping,
            is_libero_plus=self.is_libero_plus,
            overlay=self.overlay,
        )
