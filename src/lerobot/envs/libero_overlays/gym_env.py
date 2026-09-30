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

"""LIBERO gym env that applies an overlay and disables the on-screen renderer.

Stock ``lerobot.envs.libero.LiberoEnv`` is not edited. Async workers import this
module and construct :class:`OverlayLiberoEnv`, so the hook is on the class the
worker unpickles rather than a process-local monkeypatch.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import gymnasium as gym
from libero.libero.envs import OffScreenRenderEnv

from lerobot.envs.libero import LiberoEnv, _get_suite, _select_task_ids
from lerobot.envs.libero_overlays.apply import apply_overlay
from lerobot.envs.libero_overlays.gui import disable_gui_renderer
from lerobot.envs.utils import _LazyAsyncVectorEnv, parse_camera_names


def ensure_env_headless(env: LiberoEnv) -> None:
    """Build the offscreen env and reset it with the GUI renderer already off."""
    if env._env is not None:
        return
    inner = OffScreenRenderEnv(
        bddl_file_name=env._task_bddl_file,
        camera_heights=env.observation_height,
        camera_widths=env.observation_width,
        control_freq=env.control_freq,
        hard_reset=env.hard_reset,
    )
    disable_gui_renderer(inner)
    inner.reset()
    env._env = inner


def install_headless_libero_renderer() -> None:
    """Point stock ``LiberoEnv._ensure_env`` at :func:`ensure_env_headless`.

    Datagen still constructs the stock gym class. Forked worker processes keep
    the patched method. Importing this module does not install the patch;
    callers that use the stock class must call this first.
    """
    current = LiberoEnv._ensure_env
    if getattr(current, "_gangelia_headless", False):
        return
    # Another hook (control-freq) already replaced construction and disables the GUI.
    if getattr(current, "__name__", "") != "_ensure_env":
        return

    def _ensure_env(self: LiberoEnv) -> None:
        ensure_env_headless(self)

    _ensure_env._gangelia_headless = True  # type: ignore[attr-defined]
    LiberoEnv._ensure_env = _ensure_env  # type: ignore[method-assign]


class OverlayLiberoEnv(LiberoEnv):
    """Stock LIBERO env with an optional BDDL overlay and a headless renderer."""

    def __init__(self, *args: Any, overlay: str | None = None, **kwargs: Any) -> None:
        suite_name = kwargs.get("task_suite_name")
        task_id = kwargs.get("task_id")
        super().__init__(*args, **kwargs)
        if not overlay:
            return
        applied = apply_overlay(
            self._task_bddl_file,
            overlay,
            suite_name=suite_name,
            task_id=task_id,
        )
        self._task_bddl_file = str(applied.bddl_path)
        if applied.disable_init_states:
            self.init_states = False
            self._init_states = None
        if applied.overlay.language:
            self.task_description = applied.overlay.language
        if not self.hard_reset and not self.init_states:
            raise ValueError(
                "hard_reset=False requires init_states=True "
                "(overlays with mode='add' disable init_states; keep hard_reset=True)"
            )

    def _ensure_env(self) -> None:
        ensure_env_headless(self)


def _make_overlay_env_fns(
    *,
    suite: Any,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: dict[str, Any],
    control_mode: str,
    camera_name_mapping: dict[str, str] | None,
    is_libero_plus: bool,
    overlay: str | None,
) -> list[Callable[[], OverlayLiberoEnv]]:
    def _make_env(episode_index: int, **kwargs: Any) -> OverlayLiberoEnv:
        local_kwargs = dict(kwargs)
        return OverlayLiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            n_envs=n_envs,
            control_mode=control_mode,
            camera_name_mapping=camera_name_mapping,
            is_libero_plus=is_libero_plus,
            overlay=overlay,
            **local_kwargs,
        )

    return [partial(_make_env, episode_index, **gym_kwargs) for episode_index in range(n_envs)]


def create_overlay_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[..., Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
    camera_name_mapping: dict[str, str] | None = None,
    is_libero_plus: bool = False,
    overlay: str | None = None,
) -> dict[str, dict[int, Any]]:
    """Same shape as ``create_libero_envs``, using :class:`OverlayLiberoEnv`."""
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)
    camera_names = parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    is_async = env_cls is gym.vector.AsyncVectorEnv
    is_sync = env_cls is gym.vector.SyncVectorEnv
    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        selected = _select_task_ids(len(suite.tasks), task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}'.")
        cached_obs_space = None
        cached_act_space = None
        cached_metadata = None
        for tid in selected:
            fns = _make_overlay_env_fns(
                suite=suite,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                episode_length=episode_length,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
                camera_name_mapping=camera_name_mapping,
                is_libero_plus=is_libero_plus,
                overlay=overlay,
            )
            if is_async:
                lazy = _LazyAsyncVectorEnv(fns, cached_obs_space, cached_act_space, cached_metadata)
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                out[suite_name][tid] = lazy
            elif is_sync:
                out[suite_name][tid] = gym.vector.SyncVectorEnv(
                    fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
                )
            else:
                out[suite_name][tid] = env_cls(fns)
    return {suite: dict(task_map) for suite, task_map in out.items()}
