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

from lerobot.faults.datagen import episode_preview as preview


class _FakeRenderEnv:
    def call(self, name: str) -> list[np.ndarray]:
        assert name == "render"
        return [np.zeros((64, 64, 3), dtype=np.uint8)]


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
