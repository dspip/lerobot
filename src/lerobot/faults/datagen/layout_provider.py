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

"""Shared episode layout construction for unified drop datagen."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lerobot.faults.datagen.paired_context import PairedEpisodePlan
from lerobot.faults.datagen.recipe import DropDatagenRecipe
from lerobot.faults.datagen.scene import layout_to_serializable, sample_object_layout

__all__ = [
    "LayoutProviderContext",
    "SharedLayoutProvider",
    "libero_init_state_count",
    "libero_shared_layout_provider",
]


@dataclass(frozen=True)
class LayoutProviderContext:
    """Inputs for building the shared object layout across matrix variants."""

    recipe: DropDatagenRecipe
    plan: PairedEpisodePlan
    object_name: str


SharedLayoutProvider = Callable[[LayoutProviderContext], dict[str, dict[str, list[float]]]]


def _vec_env(envs: dict[str, Any]) -> Any:
    suite = next(iter(envs.values()))
    return next(iter(suite.values()))


def libero_init_state_count(recipe: DropDatagenRecipe) -> int:
    """Return the number of LIBERO init states for the recipe task."""
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    from lerobot.faults.sim.libero import unwrap_libero_env

    env_cfg = LiberoEnv(
        task=recipe.task,
        task_ids=[recipe.task_id],
        observation_height=256,
        observation_width=256,
        episode_length=4000,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = _vec_env(envs)
    try:
        libero_env = unwrap_libero_env(vec)
        init_states = getattr(libero_env, "_init_states", None)
        if init_states is None:
            raise RuntimeError("LIBERO env has no _init_states (init_states disabled?)")
        return int(len(init_states))
    finally:
        vec.close()


def libero_shared_layout_provider(context: LayoutProviderContext) -> dict[str, dict[str, list[float]]]:
    """Sample one object layout for the paired episode using LIBERO."""
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env
    from lerobot.faults.sim.libero import get_robosuite_env, unwrap_libero_env

    recipe = context.recipe
    plan = context.plan
    env_cfg = LiberoEnv(
        task=recipe.task,
        task_ids=[recipe.task_id],
        observation_height=256,
        observation_width=256,
        episode_length=4000,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec = _vec_env(envs)
    try:
        libero_env = unwrap_libero_env(vec)
        libero_env.init_state_id = int(plan.init_state_id)
        vec.reset(seed=plan.episode_seed)
        rs_env = get_robosuite_env(vec, 0)
        layout = sample_object_layout(
            rs_env,
            recipe,
            context.object_name,
            np.random.default_rng(plan.layout_seed),
        )
        if layout is None:
            raise RuntimeError("no legal object layout for paired episode")
        return layout_to_serializable(layout)
    finally:
        vec.close()
