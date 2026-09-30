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

from __future__ import annotations

import numpy as np
import pytest

from lerobot.faults.datagen import episode_preview as preview


class _FakeRenderEnv:
    def call(self, name: str) -> list[np.ndarray]:
        assert name == "render"
        return [np.zeros((64, 64, 3), dtype=np.uint8)]


class _FakeSubEnv:
    def __init__(self) -> None:
        self.render_calls = 0

    def render(self) -> np.ndarray:
        self.render_calls += 1
        return np.full((64, 64, 3), 128, dtype=np.uint8)


class _FakeVecEnv:
    """Gymnasium SyncVectorEnv shape: no ``call``, first sub-env renders."""

    def __init__(self) -> None:
        self.envs = [_FakeSubEnv()]


def test_episode_preview_captures_frames_and_writes_expected_artifacts(tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(
        preview, "_write_mp4", lambda path, frames, fps: written.append((path.name, len(frames), fps))
    )
    monkeypatch.setattr(
        preview, "_write_gif", lambda path, frames, fps: written.append((path.name, len(frames), fps))
    )
    monkeypatch.setattr(preview, "_final_proof_shot", lambda rs, path: str(path))
    recorder = preview.EpisodePreview(tmp_path, control_fps=20)
    recorder.capture(_FakeRenderEnv(), "PHASE: SIMPLE IK", (30, 90, 200))
    artifacts = recorder.finalize(object())
    assert written == [("full_pipeline.gif", 1, 8), ("full_pipeline.mp4", 1, 20)]
    assert artifacts["video_mp4"].endswith("videos/full_pipeline.mp4")
    assert artifacts["video_gif"].endswith("videos/full_pipeline.gif")
    assert artifacts["final_state_multicam"].endswith("final_state_multicam.png")


def test_episode_preview_capture_uses_vecenv_render_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "_write_mp4", lambda path, frames, fps: None)
    monkeypatch.setattr(preview, "_write_gif", lambda path, frames, fps: None)
    monkeypatch.setattr(preview, "_final_proof_shot", lambda rs, path: str(path))
    vec_env = _FakeVecEnv()
    assert not hasattr(vec_env, "call")
    recorder = preview.EpisodePreview(tmp_path, control_fps=20)
    recorder.capture(vec_env, "PHASE: VLA (SmolVLA)", (30, 90, 200))
    assert vec_env.envs[0].render_calls == 1
    assert len(recorder.frames) == 1
    assert recorder.frames[0].shape == (64, 64, 3)
    assert int(recorder.frames[0].mean()) > 0


def test_episode_preview_finalize_raises_without_frames(tmp_path):
    recorder = preview.EpisodePreview(tmp_path, control_fps=20)
    with pytest.raises(RuntimeError, match="cannot finalize an episode preview without frames"):
        recorder.finalize(object())


# ---------------------------------------------------------------------------
# Robust-capture / last-frame-fallback tests (required by spec §"environment
# frame capture with last-frame fallback")
# ---------------------------------------------------------------------------


class _BadRenderEnv:
    """Env whose render() returns a MagicMock — simulates the MagicMock test double."""

    def call(self, name: str):  # noqa: ANN001
        from unittest.mock import MagicMock

        return [MagicMock()]  # np.asarray(MagicMock()) → 0-d array


class _ScalarRenderEnv:
    """Env whose render() returns a 1-D array — wrong ndim, not a valid frame."""

    def call(self, name: str) -> list[np.ndarray]:
        return [np.array([42], dtype=np.uint8)]  # shape (1,) — ndim != 3


def test_episode_preview_capture_falls_back_to_last_frame_on_bad_render(tmp_path):
    """(a) When render is malformed AND a previous frame exists, the last frame
    is duplicated rather than crashing or losing a timeline slot."""
    recorder = preview.EpisodePreview(tmp_path, control_fps=20)
    # Seed one valid frame first.
    recorder.capture(_FakeRenderEnv(), "PHASE: good", (0, 0, 255))
    assert len(recorder.frames) == 1
    good_frame = recorder.frames[0].copy()

    # Now feed a bad render — should not raise, should duplicate the last frame.
    recorder.capture(_BadRenderEnv(), "PHASE: bad-render", (255, 0, 0))
    assert len(recorder.frames) == 2
    np.testing.assert_array_equal(recorder.frames[1], good_frame)

    # Also check with a wrong-ndim env.
    recorder.capture(_ScalarRenderEnv(), "PHASE: scalar-render", (255, 0, 0))
    assert len(recorder.frames) == 3
    np.testing.assert_array_equal(recorder.frames[2], good_frame)


def test_episode_preview_capture_produces_black_frame_on_first_failure(tmp_path, monkeypatch):
    """(b) When the very first capture fails the recorder still appends a valid
    256×256×3 uint8 black frame so finalize() remains possible, and the
    encoding path (mp4 + gif) can complete successfully."""
    written = []
    monkeypatch.setattr(
        preview, "_write_mp4", lambda path, frames, fps: written.append(("mp4", len(frames), fps))
    )
    monkeypatch.setattr(
        preview, "_write_gif", lambda path, frames, fps: written.append(("gif", len(frames), fps))
    )
    monkeypatch.setattr(preview, "_final_proof_shot", lambda rs, path: None)

    recorder = preview.EpisodePreview(tmp_path, control_fps=20)
    assert len(recorder.frames) == 0

    recorder.capture(_BadRenderEnv(), "PHASE: first-bad", (255, 0, 0))

    assert len(recorder.frames) == 1
    frame = recorder.frames[0]
    assert frame.dtype == np.uint8, f"expected uint8, got {frame.dtype}"
    assert frame.shape == (256, 256, 3), f"expected (256,256,3), got {frame.shape}"
    assert frame.sum() == 0, "first-failure fallback frame must be all-black"

    # Encoding path must succeed with the fallback frame.
    artifacts = recorder.finalize(object())
    assert written == [("gif", 1, 8), ("mp4", 1, 20)], f"expected gif+mp4 writes, got {written}"
    assert artifacts["video_mp4"].endswith("videos/full_pipeline.mp4")
    assert artifacts["video_gif"].endswith("videos/full_pipeline.gif")
    assert artifacts["final_state_multicam"] is None  # proof shot is None, that's fine


# ---------------------------------------------------------------------------
# dtype-preservation test (TDD: RED with old coercion, GREEN after fix)
# ---------------------------------------------------------------------------


class _Float32RenderEnv:
    """Env returning a valid float32 H×W×3 frame.

    MuJoCo returns uint8, but capture() must not force-coerce arbitrary dtypes
    before handing the array to _overlay_banner — that decision belongs to the
    caller or to PIL/numpy downstream.
    """

    def call(self, name: str) -> list[np.ndarray]:
        return [np.full((64, 64, 3), 100.0, dtype=np.float32)]


def test_episode_preview_capture_does_not_coerce_dtype(tmp_path, monkeypatch):
    """capture() must not force-convert the raw render to uint8 before overlay.

    A valid H×W×3 float32 array is a legal input; its dtype must be preserved
    when it arrives at _overlay_banner (TDD: RED with ``np.asarray(raw[0],
    dtype=np.uint8)``, GREEN after removing the forced coercion).
    """
    received_dtypes: list = []
    original_overlay = preview._overlay_banner

    def _spy_overlay(frame: np.ndarray, text: str, color: tuple) -> np.ndarray:
        received_dtypes.append(frame.dtype)
        return original_overlay(frame, text, color)

    monkeypatch.setattr(preview, "_overlay_banner", _spy_overlay)

    recorder = preview.EpisodePreview(tmp_path, control_fps=20)
    recorder.capture(_Float32RenderEnv(), "PHASE: float-test", (0, 0, 255))

    assert len(received_dtypes) == 1, "overlay must be called exactly once (normal path, not fallback)"
    assert received_dtypes[0] == np.float32, (
        f"frame dtype must be preserved (expected float32, got {received_dtypes[0]}); "
        "capture() must not coerce to uint8 before overlay"
    )


# ---------------------------------------------------------------------------
# _final_proof_shot channel guard (TDD: RED with ndim-only check, GREEN after
# adding shape[2] != 3 guard)
# ---------------------------------------------------------------------------


class _TwoChannelRS:
    """Fake robosuite env whose render() returns a 2-channel array (H, W, 2).

    ndim==3 so the old guard (``arr.ndim != 3``) passes it through; the new
    guard (``arr.ndim != 3 or arr.shape[2] != 3``) correctly rejects it.
    """

    class sim:
        @staticmethod
        def render(height: int, width: int, camera_name: str) -> np.ndarray:
            return np.zeros((height, width, 2), dtype=np.uint8)


def test_final_proof_shot_rejects_wrong_channel_count(tmp_path, monkeypatch):
    """_final_proof_shot must discard renders that are 3-D but not RGB (channels≠3).

    TDD: with the old ndim-only guard the 2-channel array reaches np.concatenate
    or imageio and either crashes or produces invalid output.  With the new
    ``shape[2] != 3`` guard, shots remains empty and the function returns None.
    """
    # Suppress the basket free-camera path (no MuJoCo in unit test env).
    monkeypatch.setattr(preview, "_render_looking_into_basket", lambda rs, size=384: None)

    result = preview._final_proof_shot(_TwoChannelRS(), tmp_path / "proof.png")
    assert result is None, (
        f"expected None when all renders have wrong channel count, got {result!r}"
    )
