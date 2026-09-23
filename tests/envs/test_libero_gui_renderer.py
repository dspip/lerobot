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

"""LIBERO forces ``has_renderer=True``; we must disable it before the first reset.

Leaving it on makes robosuite tear down an OpenCV GUI window on every hard
reset. With a headless ``cv2`` build that raises, and LIBERO's ``reset()``
retry loop swallows the exception and spins forever at 100% CPU.
"""

from __future__ import annotations

from lerobot.envs.libero import disable_gui_renderer


class _Renderer:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RobosuiteEnv:
    def __init__(self, viewer: _Renderer | None) -> None:
        self.has_renderer = True
        self.has_offscreen_renderer = True
        self.viewer = viewer


class _LiberoEnv:
    def __init__(self, viewer: _Renderer | None) -> None:
        self.env = _RobosuiteEnv(viewer)


def test_disables_renderer_flag_so_reset_never_rebuilds_a_viewer():
    env = _LiberoEnv(_Renderer())

    disable_gui_renderer(env)

    assert env.env.has_renderer is False


def test_clears_viewer_so_reset_never_calls_cv2_destroy_all_windows():
    env = _LiberoEnv(_Renderer())

    disable_gui_renderer(env)

    assert env.env.viewer is None


def test_keeps_offscreen_rendering_enabled():
    env = _LiberoEnv(_Renderer())

    disable_gui_renderer(env)

    assert env.env.has_offscreen_renderer is True


def test_does_not_call_close_on_the_discarded_viewer():
    viewer = _Renderer()
    env = _LiberoEnv(viewer)

    disable_gui_renderer(env)

    assert viewer.closed is False


def test_is_idempotent():
    env = _LiberoEnv(_Renderer())

    disable_gui_renderer(env)
    disable_gui_renderer(env)

    assert env.env.has_renderer is False
    assert env.env.viewer is None


def test_tolerates_env_without_a_viewer():
    env = _LiberoEnv(None)

    disable_gui_renderer(env)

    assert env.env.viewer is None


def test_tolerates_object_without_an_inner_env():
    sentinel = object()

    disable_gui_renderer(sentinel)
