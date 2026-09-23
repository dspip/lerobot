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

#!/usr/bin/env python3
"""Side-by-side comparison of post-drop recording modes from pipeline episode videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_REL = Path("videos") / "full_pipeline.mp4"
TILE_HEIGHT = 480


def _escape_drawtext(label: str) -> str:
    """Escape a label for ffmpeg drawtext ``text=`` (single-quoted)."""
    return "'" + label.replace("\\", "\\\\").replace(":", r"\:").replace("'", r"'\''") + "'"


def build_filter_graph(labels: list[str]) -> str:
    """Build ffmpeg filter_complex for labeled tiles and horizontal stack."""
    if len(labels) < 2:
        raise ValueError(f"build_filter_graph requires at least 2 labels (got {len(labels)}).")
    chains: list[str] = []
    for i, label in enumerate(labels):
        text = _escape_drawtext(label)
        chains.append(
            # Label sits at the bottom: the pipeline draws its own phase banner
            # across the top of every frame.
            f"[{i}:v]scale=-2:{TILE_HEIGHT},"
            f"drawtext=text={text}:x=10:y=h-th-10:fontsize=24:fontcolor=white:"
            f"box=1:boxcolor=black@0.6[v{i}]"
        )
    stack_inputs = "".join(f"[v{i}]" for i in range(len(labels)))
    chains.append(f"{stack_inputs}hstack=inputs={len(labels)}:shortest=0[outv]")
    return ";".join(chains)


def inject_trailing_tpad(filter_graph: str, pad_seconds: list[float]) -> str:
    """Insert per-input tpad (clone last frame) before hstack for unequal lengths."""
    parts = filter_graph.split(";")
    if len(parts) < 2:
        return filter_graph
    *chains, hstack = parts
    if len(chains) != len(pad_seconds):
        raise ValueError(f"pad_seconds length {len(pad_seconds)} != input chains {len(chains)}")
    new_chains: list[str] = []
    for i, (chain, pad) in enumerate(zip(chains, pad_seconds, strict=True)):
        suffix = f"[v{i}]"
        if not chain.endswith(suffix):
            raise ValueError(f"Expected chain to end with {suffix!r}: {chain!r}")
        if pad > 0:
            chain = chain[: -len(suffix)] + f",tpad=stop_mode=clone:stop_duration={pad:.6f}{suffix}"
        new_chains.append(chain)
    return ";".join([*new_chains, hstack])


def parse_episode_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --episode {raw!r}; expected LABEL=PATH (e.g. immediate_ik=outputs/run_a).")
    label, path_str = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Empty label in --episode {raw!r}.")
    return label, Path(path_str).expanduser().resolve()


def validate_episodes(episodes: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    if len(episodes) < 2:
        raise ValueError(f"At least 2 --episode arguments are required (got {len(episodes)}).")
    validated: list[tuple[str, Path]] = []
    for label, ep_dir in episodes:
        video = ep_dir / VIDEO_REL
        if not video.is_file():
            raise FileNotFoundError(f"Missing pipeline video: {video}")
        validated.append((label, video))
    return validated


def probe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(proc.stdout)
    duration = float(data["format"]["duration"])
    if duration <= 0:
        raise ValueError(f"Could not read duration for {path}")
    return duration


def trailing_pad_seconds(durations: list[float]) -> list[float]:
    longest = max(durations)
    return [max(0.0, longest - d) for d in durations]


def run_ffmpeg(inputs: list[Path], filter_graph: str, output_mp4: Path) -> None:
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = ["ffmpeg", "-y"]
    for path in inputs:
        cmd.extend(["-i", str(path)])
    cmd.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_mp4),
        ]
    )
    subprocess.run(cmd, check=True)


def mp4_to_gif(mp4: Path, gif: Path) -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4),
        "-vf",
        "fps=10,scale=640:-1:flags=lanczos",
        "-loop",
        "0",
        str(gif),
    ]
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a side-by-side MP4/GIF comparing post-drop mode episode videos.",
    )
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Episode output directory (must contain videos/full_pipeline.mp4). Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for post_drop_modes.mp4 and post_drop_modes.gif",
    )
    args = parser.parse_args(argv)

    try:
        parsed = [parse_episode_arg(raw) for raw in args.episode]
    except ValueError as exc:
        print(f"error: {exc}", flush=True)
        return 2

    try:
        validated = validate_episodes(parsed)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", flush=True)
        return 2

    labels = [label for label, _ in validated]
    video_paths = [path for _, path in validated]

    durations = [probe_duration_seconds(p) for p in video_paths]
    pad_secs = trailing_pad_seconds(durations)
    graph = inject_trailing_tpad(build_filter_graph(labels), pad_secs)

    out_dir = args.output.expanduser().resolve()
    mp4_path = out_dir / "post_drop_modes.mp4"
    gif_path = out_dir / "post_drop_modes.gif"

    run_ffmpeg(video_paths, graph, mp4_path)
    mp4_to_gif(mp4_path, gif_path)
    print(f"Wrote {mp4_path}", flush=True)
    print(f"Wrote {gif_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
