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

"""Tk camera viewer with a persistent, scrollable datagen event log."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lerobot.faults.datagen.display import fit_display_size, hud_font_size
from lerobot.faults.datagen.events import LogEvent, format_event_line, should_autoscroll

DEFAULT_CAMERA_PX = 720


class DatagenViewer:
    """Live camera/HUD and full structured log in one window."""

    def __init__(self, tk: Any, image_tk: Any, camera_px: int = DEFAULT_CAMERA_PX) -> None:
        self._tk = tk
        self._image_tk = image_tk
        self._photo: Any | None = None
        self._camera_px = int(camera_px)
        self._font = _load_hud_font(hud_font_size(self._camera_px))
        self.quit_requested = False
        self.pause = False
        self.next_episode = False
        self.force_q: float | None = None

        self.root = tk.Tk()
        self.root.title("Can SimpleIK datagen")
        self.root.protocol("WM_DELETE_WINDOW", self._request_quit)
        self.root.bind("<q>", lambda _event: self._request_quit())
        self.root.bind("<Escape>", lambda _event: self._request_quit())
        self.root.bind("<space>", lambda _event: self._toggle_pause())

        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)
        tk.Button(controls, text="Next episode", command=self._request_next).pack(
            side=tk.LEFT, padx=3
        )
        self._pause_button = tk.Button(
            controls,
            text="Pause",
            command=self._toggle_pause,
        )
        self._pause_button.pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Next q=0", command=lambda: self._set_force_q(0.0)).pack(
            side=tk.LEFT, padx=3
        )
        tk.Button(controls, text="Next q=1", command=lambda: self._set_force_q(1.0)).pack(
            side=tk.LEFT, padx=3
        )
        tk.Button(controls, text="Clear log", command=self.clear_log).pack(side=tk.LEFT, padx=3)
        tk.Button(controls, text="Quit", command=self._request_quit).pack(side=tk.RIGHT, padx=3)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._image_label = tk.Label(body)
        self._image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        log_frame = tk.Frame(body)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(8, 0))
        scrollbar = tk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._log = tk.Text(
            log_frame,
            width=72,
            height=28,
            wrap=tk.WORD,
            state=tk.DISABLED,
            yscrollcommand=scrollbar.set,
        )
        self._log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.configure(command=self._log.yview)

    @classmethod
    def try_create(cls, camera_px: int = DEFAULT_CAMERA_PX) -> DatagenViewer | None:
        """Create the viewer, returning ``None`` on a headless machine."""
        try:
            import tkinter as tk

            from PIL import ImageTk

            return cls(tk, ImageTk, camera_px=camera_px)
        except (ImportError, RuntimeError, Exception) as exc:
            # TclError is only available after importing tkinter. Avoid importing
            # Tk at module load so headless CPU tests can import this module.
            print(f"[viewer] unavailable; continuing headless: {exc}", flush=True)
            return None

    def _request_quit(self) -> None:
        self.quit_requested = True

    def _request_next(self) -> None:
        self.next_episode = True

    def _toggle_pause(self) -> None:
        self.pause = not self.pause
        self._pause_button.configure(text="Resume" if self.pause else "Pause")

    def _set_force_q(self, q: float) -> None:
        self.force_q = float(q)

    def pump(self) -> None:
        if self.quit_requested:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:
            self.quit_requested = True

    def show(self, frame_rgb: np.ndarray, hud: str) -> None:
        if self.quit_requested:
            return
        image = Image.fromarray(np.asarray(frame_rgb, dtype=np.uint8))
        image = image.resize(
            fit_display_size(image.width, image.height, self._camera_px),
            Image.Resampling.BICUBIC,
        )
        draw = ImageDraw.Draw(image)
        lines = hud.splitlines() or [hud]
        line_h = hud_font_size(self._camera_px) + 5
        banner_h = 10 + line_h * len(lines)
        draw.rectangle((0, 0, image.width, banner_h), fill=(0, 0, 0))
        for index, line in enumerate(lines):
            draw.text((8, 6 + line_h * index), line, fill=(255, 255, 255), font=self._font)
        try:
            self._photo = self._image_tk.PhotoImage(image=image)
            self._image_label.configure(image=self._photo)
        except self._tk.TclError:
            self.quit_requested = True

    def append_log(self, event: LogEvent) -> None:
        if self.quit_requested:
            return
        try:
            follow = should_autoscroll(float(self._log.yview()[1]))
            self._log.configure(state=self._tk.NORMAL)
            self._log.insert(self._tk.END, format_event_line(event) + "\n")
            self._log.configure(state=self._tk.DISABLED)
            if follow:
                self._log.see(self._tk.END)
        except self._tk.TclError:
            self.quit_requested = True

    def clear_log(self) -> None:
        try:
            self._log.configure(state=self._tk.NORMAL)
            self._log.delete("1.0", self._tk.END)
            self._log.configure(state=self._tk.DISABLED)
        except self._tk.TclError:
            self.quit_requested = True

    def destroy(self) -> None:
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except self._tk.TclError:
            pass


def _load_hud_font(size_px: int) -> Any:
    """Size the bitmap fallback too, so old Pillow builds stay legible."""
    try:
        return ImageFont.load_default(size=size_px)
    except TypeError:
        return ImageFont.load_default()
