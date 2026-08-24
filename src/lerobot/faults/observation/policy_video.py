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

"""Record wrist (camera2) policy videos: clean vs faulted."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.faults.observation.utils import array_to_uint8_hwc, iter_image_entries


def pick_wrist_camera(obs: Any, env_idx: int, num_envs: int) -> np.ndarray | None:
    """Prefer wrist / eye-in-hand / camera2; else first image field."""
    entries = iter_image_entries(obs)
    if not entries:
        return None
    preferred = None
    for key_path, value in entries:
        low = key_path.lower()
        hwc = array_to_uint8_hwc(value, env_idx, num_envs)
        if hwc is None:
            continue
        if (
            "camera2" in low
            or "eye_in_hand" in low
            or "wrist" in low
            or low.endswith("image2")
            or ".image2" in low
        ):
            return hwc
        if preferred is None:
            preferred = hwc
    return preferred


def write_frame_video(
    frames: list[np.ndarray],
    out_path: Path,
    *,
    fps: float = 10.0,
) -> Path | None:
    """Write uint8 HWC frames to GIF (Pillow). Returns path or None."""
    if not frames:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".gif":
        out_path = out_path.with_suffix(".gif")
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        warnings.warn("Pillow missing; cannot write policy camera video.", stacklevel=2)
        return None

    pil_frames = []
    for fr in frames:
        arr = np.asarray(fr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(arr))

    duration_ms = max(40, int(round(1000.0 / max(fps, 1e-6))))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return out_path


def overlay_label(frame: np.ndarray, text: str, *, fault: bool) -> np.ndarray:
    """Copy frame and burn a small status label at the top."""
    out = np.array(frame, copy=True)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return out
    h, w = out.shape[:2]
    banner = max(16, h // 18)
    color = (140, 20, 20) if fault else (20, 90, 30)
    out[:banner] = color
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    draw.text((4, max(1, banner // 5)), text, fill=(255, 255, 0))
    return np.asarray(pil)


class PolicyCameraVideoRecorder:
    """Buffer clean + faulted policy-camera frames; flush two GIFs per episode."""

    def __init__(
        self,
        out_dir: Path,
        *,
        fault_type: str,
        fps: float = 10.0,
        max_frames: int = 300,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fault_type = str(fault_type)
        self.fps = float(fps)
        self.max_frames = int(max_frames)
        self._clean: list[np.ndarray] = []
        self._faulted: list[np.ndarray] = []
        self._episode_idx = 0
        self._paths: list[Path] = []

    def reset_episode(self) -> None:
        self.flush()
        self._clean = []
        self._faulted = []

    def add(
        self,
        clean_hwc: np.ndarray,
        faulted_hwc: np.ndarray,
        *,
        fault_active: bool,
        episode_step: int,
    ) -> None:
        if len(self._clean) >= self.max_frames:
            return
        step = int(episode_step)
        if fault_active:
            clean_lab = f"WRIST CLEAN (ref) step={step}"
            fault_lab = f"WRIST [{self.fault_type}] step={step}"
        else:
            clean_lab = f"WRIST CLEAN step={step}"
            fault_lab = f"WRIST (nominal) step={step}"
        self._clean.append(overlay_label(clean_hwc, clean_lab, fault=False))
        self._faulted.append(overlay_label(faulted_hwc, fault_lab, fault=fault_active))

    def flush(self) -> list[Path]:
        written: list[Path] = []
        if not self._clean:
            return written
        ep = self._episode_idx
        clean_path = write_frame_video(
            self._clean,
            self.out_dir / f"vla_wrist_clean_ep{ep:03d}",
            fps=self.fps,
        )
        fault_path = write_frame_video(
            self._faulted,
            self.out_dir / f"vla_wrist_{self.fault_type}_ep{ep:03d}",
            fps=self.fps,
        )
        self._clean = []
        self._faulted = []
        self._episode_idx += 1
        for p in (clean_path, fault_path):
            if p is not None:
                written.append(p)
                self._paths.append(p)
        return written

    @property
    def written_paths(self) -> list[Path]:
        return list(self._paths)
