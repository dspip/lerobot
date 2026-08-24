# Copyright 2026 Gangelia. All rights reserved.
"""LeRobotDataset episode recorder for VLA → fault → recovery trajectories."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.faults.recovery.fps import assert_dataset_fps, resolve_target_fps

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:  # pragma: no cover - optional at import time
    LeRobotDataset = None  # type: ignore[misc, assignment]

LIBERO_DATASET_FEATURES: dict[str, dict[str, Any]] = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
    "action": {"dtype": "float32", "shape": (7,), "names": None},
    "loss_mask": {"dtype": "float32", "shape": (1,), "names": None},
}


def libero_obs_to_frame(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert raw or preprocessed LIBERO obs to LeRobot frame fields (numpy HWC).

    Accepts:
    - already-final keys: ``observation.images.image`` + ``observation.state``
    - ``preprocess_observation`` output: ``camera1``/``camera2`` + ``robot_state``
    - raw LIBERO gym obs: ``pixels`` + ``robot_state``
    """
    if "observation.images.image" in obs and "observation.state" in obs:
        return _processed_obs_to_frame(obs)

    try:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.processor.env_processor import LiberoProcessorStep
    except ImportError as exc:
        raise ImportError(
            "libero_obs_to_frame requires lerobot for raw LIBERO observations."
        ) from exc

    # Already preprocessed (camera* keys) or still raw gym obs.
    if "observation.images.camera1" in obs or "observation.robot_state" in obs:
        preprocessed = obs
    else:
        preprocessed = preprocess_observation(obs)
    processed = LiberoProcessorStep().observation(preprocessed)
    return _processed_obs_to_frame(processed)


def _processed_obs_to_frame(processed: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "observation.state" not in processed:
        raise KeyError(
            "Missing observation.state after LIBERO processing. "
            f"Keys present: {sorted(processed)}"
        )
    out["observation.state"] = _to_f32_vector(processed["observation.state"], 8)

    # Map policy camera keys → official lerobot/libero dataset keys.
    image_aliases = (
        ("observation.images.image", ("observation.images.image", "observation.images.camera1")),
        ("observation.images.image2", ("observation.images.image2", "observation.images.camera2")),
    )
    for out_key, candidates in image_aliases:
        img = None
        for cand in candidates:
            if cand in processed:
                img = processed[cand]
                break
        if img is None:
            raise KeyError(f"Missing image for {out_key}; looked for {candidates}")
        out[out_key] = _to_hwc_uint8(img)
    return out


def _to_f32_vector(value: Any, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy().astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    arr = arr.reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"Expected state shape ({size},), got {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def _to_hwc_uint8(value: Any, size: int = 256) -> np.ndarray:
    arr = np.asarray(value)
    if hasattr(value, "detach"):
        arr = value.detach().cpu().numpy()
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if float(np.max(arr)) <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HWC image, got {arr.shape}.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.shape[0] != size or arr.shape[1] != size:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to resize LIBERO images to 256x256.") from exc
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return arr


class FaultRecoveryDatasetLogger:
    """Record VLA → fault → recovery episodes into a LeRobot dataset.

    ``loss_mask`` semantics (per frame, float32 scalar in shape ``(1,)``):

    - ``1.0`` — nominal VLA policy steps and all recovery planner steps
    - ``0.0`` — only the mid-air drop injection step (a single frame when drop
      and recovery start on the same env step; recovery frames remain ``1.0``)

    Downstream fine-tuning can mask loss on the injection frame while still
    learning from nominal and recovery segments.
    """

    def __init__(
        self,
        root: Path | str,
        repo_id: str,
        *,
        fps: int | None = None,
        policy_fps: int | None = None,
        robot_type: str = "panda",
        append: bool = False,
    ) -> None:
        if LeRobotDataset is None:
            raise ImportError(
                "FaultRecoveryDatasetLogger requires lerobot. Install LeRobot in your environment."
            )

        self.root = Path(root)
        self.repo_id = repo_id
        self.policy_fps = resolve_target_fps(policy_fps if policy_fps is not None else fps)
        self.fps = self.policy_fps
        assert_dataset_fps(self.fps, self.policy_fps)

        if self.root.exists():
            if not append:
                raise FileExistsError(
                    f"Dataset root {self.root} already exists. "
                    "Choose a fresh output directory or pass append=True."
                )
            info_path = self.root / "meta" / "info.json"
            if not info_path.is_file():
                raise FileExistsError(
                    f"Dataset root {self.root} exists but is missing meta/info.json. "
                    "Cannot append to an invalid dataset root."
                )
            self.dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=self.root,
                image_writer_processes=0,
                image_writer_threads=1,
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=self.fps,
                features=LIBERO_DATASET_FEATURES,
                root=self.root,
                robot_type=robot_type,
                use_videos=True,
                image_writer_processes=0,
                image_writer_threads=1,
            )
        assert_dataset_fps(self.dataset.fps, self.policy_fps)

        self._episode_open = False
        self._loss_mask_counts: dict[float, int] = {0.0: 0, 1.0: 0}

    @property
    def loss_mask_counts(self) -> dict[float, int]:
        """Running counts of ``loss_mask`` values logged in the current build."""
        return dict(self._loss_mask_counts)

    def log_step(
        self,
        observation_dict: dict[str, Any],
        action: np.ndarray | list[float],
        task: str,
        loss_mask: float,
        phase: str | None = None,
    ) -> None:
        """Append one frame. ``observation_dict`` may be LIBERO raw or LeRobot-processed."""
        if not self._episode_open:
            self._episode_open = True

        if "observation.images.image" in observation_dict:
            frame_fields = _processed_obs_to_frame(observation_dict)
        else:
            frame_fields = libero_obs_to_frame(observation_dict)

        mask_val = float(loss_mask)
        if mask_val not in (0.0, 1.0):
            warnings.warn(f"loss_mask={mask_val}; expected 0.0 or 1.0.", stacklevel=2)

        action_arr = np.asarray(action, dtype=np.float32).reshape(7)
        mask_arr = np.array([mask_val], dtype=np.float32)

        frame: dict[str, Any] = {
            **frame_fields,
            "action": action_arr,
            "loss_mask": mask_arr,
            "task": task,
        }
        # ``phase`` is optional caller metadata; not stored in LeRobot feature schema.

        self.dataset.add_frame(frame)
        key = 1.0 if mask_val >= 0.5 else 0.0
        self._loss_mask_counts[key] = self._loss_mask_counts.get(key, 0) + 1

    def end_episode(self) -> None:
        """Flush the current episode buffer to disk."""
        if not self._episode_open:
            return
        # Disable parallel camera encoding: ProcessPool encoding can race with
        # image-path stats (FileNotFoundError on frame PNGs during merge/export).
        self.dataset.save_episode(parallel_encoding=False)
        self._episode_open = False

    def clear_open_episode(self) -> None:
        """Discard buffered frames for a failed episode (do not save)."""
        if hasattr(self.dataset, "clear_episode_buffer"):
            self.dataset.clear_episode_buffer()
        else:
            raise RuntimeError(
                "Installed LeRobotDataset does not support clear_episode_buffer(). "
                "Upgrade lerobot or discard the logger instance."
            )
        self._episode_open = False

    def finalize(self) -> None:
        """Finalize parquet/video writers and assert dataset FPS matches policy FPS."""
        if self._episode_open:
            self.end_episode()
        assert_dataset_fps(self.dataset.fps, self.policy_fps)
        self.dataset.finalize()
