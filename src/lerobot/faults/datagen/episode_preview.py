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

from pathlib import Path
from typing import Any

import numpy as np


def _overlay_banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    """Draw a simple top banner (no OpenCV dependency)."""
    out = frame.copy()
    h, w = out.shape[:2]
    banner_h = max(28, h // 12)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.fromarray(out)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        draw.rectangle([0, 0, w, banner_h], fill=color)
        draw.text((8, 6), text, fill=(255, 255, 255), font=font)
        return np.asarray(img)
    except Exception:
        out[:banner_h] = (out[:banner_h].astype(np.float32) * 0.35).astype(np.uint8)
        return out


def _write_gif(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / max(fps, 1))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)


def _write_mp4(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from lerobot.utils.io_utils import write_video

        write_video(str(path), np.stack(frames), fps)
        return
    except (ImportError, OSError, RuntimeError, ValueError):
        pass  # fall back to imageio when lerobot video backend is unavailable
    try:
        import imageio.v2 as imageio

        imageio.mimsave(path, frames, fps=fps)
    except Exception as exc:
        raise RuntimeError(f"Could not write mp4: {exc}") from exc


def _render_looking_into_basket(rs: Any, size: int = 384) -> np.ndarray | None:
    """Render steeply down onto the basket using a free camera.

    Every camera baked into the scene views the basket from the side, and the
    basket rim (~0.15 m) is taller than a can standing on its floor (top ~0.10 m),
    so a correctly placed can is fully occluded and the basket reads as empty.
    Only a view from above can show the outcome.
    """
    try:
        import mujoco

        from lerobot.faults.sim.libero import DEFAULT_BASKET_NAME, _body_xpos

        model = rs.sim.model._model
        data = rs.sim.data._data
        basket = _body_xpos(rs, DEFAULT_BASKET_NAME)
        if basket is None:
            return None

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [float(basket[0]), float(basket[1]), float(basket[2])]
        cam.distance = 0.55
        cam.azimuth = 90.0
        cam.elevation = -75.0  # steep, but not exactly top-down, to keep depth cues

        with mujoco.Renderer(model, height=size, width=size) as renderer:
            renderer.update_scene(data, camera=cam)
            return np.asarray(renderer.render(), dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] basket top-view render failed: {exc}", flush=True)
        return None


def _final_proof_shot(rs: Any, out_path: Path) -> str | None:
    """Save a multi-camera still of the end state.

    The recording camera looks at the basket side-on, so a can resting on the
    basket floor is hidden behind the near wall — indistinguishable from an empty
    basket. Extra viewpoints make the outcome checkable instead of inferred from
    coordinates.
    """
    shots: list[np.ndarray] = []
    top = _render_looking_into_basket(rs)
    if top is not None:
        shots.append(top)
    for camera in ("agentview", "frontview", "birdview", "sideview"):
        try:
            img = rs.sim.render(height=384, width=384, camera_name=camera)
        except Exception:  # nosec B112 — optional proof cameras may be absent in some scenes
            continue
        shots.append(np.asarray(img, dtype=np.uint8)[::-1])
    if not shots:
        return None
    import imageio.v2 as imageio

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, np.concatenate(shots, axis=1))
    return str(out_path)


class EpisodePreview:
    def __init__(self, output_dir: Path, control_fps: int) -> None:
        self.output_dir = Path(output_dir)
        self.control_fps = int(control_fps)
        self.frames: list[np.ndarray] = []

    def capture(self, env: Any, label: str, color: tuple[int, int, int]) -> None:
        raw = env.call("render") if hasattr(env, "call") else [env.envs[0].render()]
        self.frames.append(_overlay_banner(np.asarray(raw[0]), label, color))

    def finalize(self, rs_env: Any) -> dict[str, str | None]:
        if not self.frames:
            raise RuntimeError("cannot finalize an episode preview without frames")
        videos_dir = self.output_dir / "videos"
        gif_path = videos_dir / "full_pipeline.gif"
        mp4_path = videos_dir / "full_pipeline.mp4"
        gif_stride = max(1, len(self.frames) // 100)
        _write_gif(gif_path, self.frames[::gif_stride], fps=8)
        _write_mp4(mp4_path, self.frames, fps=self.control_fps)
        proof = _final_proof_shot(rs_env, self.output_dir / "final_state_multicam.png")
        return {"video_mp4": str(mp4_path), "video_gif": str(gif_path), "final_state_multicam": proof}
