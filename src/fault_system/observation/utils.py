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

"""Helpers for identifying and mutating image-like observation fields."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_IMAGE_KEY_PARTS = ("image", "pixels", "rgb", "camera")


def _torch_tensor_type() -> type | None:
    try:
        import torch
    except ImportError:
        return None
    return torch.Tensor


def is_array_like(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return True
    tensor_type = _torch_tensor_type()
    return tensor_type is not None and isinstance(value, tensor_type)


def _has_spatial_extent(shape: tuple[int, ...], min_size: int = 8) -> bool:
    """True if shape has an HxW-like pair of dimensions at least ``min_size``."""
    if len(shape) >= 3 and shape[-1] in (1, 3, 4):
        # HWC / BHWC
        return min(shape[-3], shape[-2]) >= min_size
    if len(shape) >= 3 and shape[-3] in (1, 3, 4):
        # CHW / BCHW
        return min(shape[-2], shape[-1]) >= min_size
    if len(shape) == 2:
        return min(shape) >= min_size
    return False


def is_image_field(key: str | None, value: Any) -> bool:
    """Return True when ``value`` looks like a camera/image observation."""
    if not is_array_like(value):
        return False

    shape = tuple(int(s) for s in value.shape)
    if len(shape) == 1:
        return False

    # Reject tiny matrices (e.g. 3x3 rotation) even if channel-sized.
    if not _has_spatial_extent(shape):
        return False

    key_lower = (key or "").lower()
    if any(part in key_lower for part in _IMAGE_KEY_PARTS):
        return True

    if len(shape) >= 3 and (shape[-1] in (1, 3, 4) or shape[-3] in (1, 3, 4)):
        return True

    return len(shape) >= 2 and shape[-2] >= 32 and shape[-1] >= 32


def zero_image_slice(value: Any, env_idx: int, num_envs: int) -> None:
    """Zero the image slice for ``env_idx`` in-place."""
    if isinstance(value, np.ndarray):
        target = value[env_idx] if num_envs > 1 and value.shape[0] == num_envs else value
        if target.dtype == np.uint8:
            target[...] = 0
        else:
            target[...] = 0.0
        return

    tensor_type = _torch_tensor_type()
    if tensor_type is not None and isinstance(value, tensor_type):
        target = value[env_idx] if num_envs > 1 and value.shape[0] == num_envs else value
        target.zero_()


def iter_image_entries(
    obs: Any,
    *,
    prefix: str = "",
) -> list[tuple[str, Any]]:
    """Collect ``(key_path, array)`` pairs for image-like leaves in ``obs``."""
    entries: list[tuple[str, Any]] = []

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                _walk(child, child_path)
            return
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes, np.ndarray)):
            for index, child in enumerate(node):
                child_path = f"{path}[{index}]"
                _walk(child, child_path)
            return
        if is_array_like(node):
            leaf_key = path.rsplit(".", 1)[-1] if path else ""
            if is_image_field(leaf_key, node):
                entries.append((path or "image", node))

    _walk(obs, prefix)
    return entries


def copy_obs_tree(obs: Any) -> Any:
    """Shallow-copy dict/list shells; copy numpy/torch arrays that will be mutated."""
    if isinstance(obs, Mapping):
        return {key: copy_obs_tree(value) for key, value in obs.items()}
    if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes, np.ndarray)):
        return type(obs)(copy_obs_tree(value) for value in obs)
    if is_array_like(obs):
        if hasattr(obs, "clone"):
            return obs.clone()
        return obs.copy()
    return obs


def array_to_uint8_hwc(value: Any, env_idx: int, num_envs: int) -> np.ndarray | None:
    """Convert an image array slice to uint8 HWC for PNG export."""
    if isinstance(value, np.ndarray):
        arr = value
    else:
        tensor_type = _torch_tensor_type()
        if tensor_type is None or not isinstance(value, tensor_type):
            return None
        arr = value.detach().cpu().numpy()

    if num_envs > 1 and arr.shape[0] == num_envs:
        arr = arr[env_idx]

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        return None

    # CHW when C is small
    if arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0 if arr.dtype.kind == "f" else 255.0)
        if arr.max() <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def save_png(path: Any, array_hwc: np.ndarray) -> bool:
    try:
        from PIL import Image
    except ImportError:
        warnings.warn(
            "Pillow is not installed; skipping sensor_dropout diagnostic PNG dump.",
            stacklevel=2,
        )
        return False

    Image.fromarray(array_hwc).save(path)
    return True


def _image_slice(value: Any, env_idx: int, num_envs: int) -> Any:
    """Return the writable image view for ``env_idx`` (batched or single).

    Handles common LeRobot/Gymnasium layouts:
    - ``(H, W, C)`` / ``(C, H, W)`` — unbatched
    - ``(N, H, W, C)`` / ``(N, C, H, W)`` — vector batch (including ``N=1``)
    """
    if not hasattr(value, "shape"):
        return value
    shape = tuple(int(s) for s in value.shape)
    if len(shape) == 4 and (shape[0] == num_envs or (num_envs == 1 and shape[0] == 1)):
        return value[env_idx if shape[0] > 1 else 0]
    if num_envs > 1 and len(shape) >= 3 and shape[0] == num_envs:
        return value[env_idx]
    return value


def apply_occlusion_box(
    value: Any,
    env_idx: int,
    num_envs: int,
    *,
    y0: int,
    x0: int,
    y1: int,
    x1: int,
) -> None:
    """Paint a black rectangle onto an image slice (HWC or CHW, batched or not)."""
    target = _image_slice(value, env_idx, num_envs)
    if hasattr(target, "shape") and len(target.shape) == 3 and target.shape[0] in (1, 3, 4) and target.shape[0] < min(
        target.shape[1], target.shape[2]
    ):
        # CHW
        target[:, y0:y1, x0:x1] = 0
    else:
        target[y0:y1, x0:x1, ...] = 0


def apply_box_blur(value: Any, env_idx: int, num_envs: int, *, radius: int) -> None:
    """In-place box blur (numpy-only) approximating a mild lens blur."""
    r = max(1, int(radius))
    target = _image_slice(value, env_idx, num_envs)
    arr = np.asarray(target)
    chw = arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < min(arr.shape[1], arr.shape[2])
    if chw:
        work = np.transpose(arr, (1, 2, 0))
    else:
        work = arr

    k = 2 * r + 1
    padded = np.pad(work.astype(np.float64, copy=False), ((r, r), (r, r), (0, 0)), mode="edge")
    # Integral image for O(1) window sums.
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0)), mode="constant")
    integral = integral.cumsum(axis=0).cumsum(axis=1)
    h, w = work.shape[:2]
    y0 = np.arange(h)
    x0 = np.arange(w)
    yy, xx = np.meshgrid(y0, x0, indexing="ij")
    y1 = yy + k
    x1 = xx + k
    window = (
        integral[y1, x1]
        - integral[yy, x1]
        - integral[y1, xx]
        + integral[yy, xx]
    )
    blurred = window / float(k * k)
    if work.dtype == np.uint8:
        hi = 255.0
    else:
        hi = 1.0 if float(np.nanmax(work)) <= 1.0 + 1e-6 else 255.0
    out = np.clip(blurred, 0.0, hi)
    if work.dtype == np.uint8:
        out = out.astype(np.uint8)
    else:
        out = out.astype(work.dtype, copy=False)

    if chw:
        target[...] = np.transpose(out, (2, 0, 1))
    else:
        target[...] = out


def apply_brightness_scale(value: Any, env_idx: int, num_envs: int, *, scale: float) -> None:
    """Multiply image intensities by ``scale`` and clip to a valid range."""
    target = _image_slice(value, env_idx, num_envs)
    s = float(scale)
    if isinstance(target, np.ndarray):
        if target.dtype == np.uint8:
            target[...] = np.clip(target.astype(np.float32) * s, 0, 255).astype(np.uint8)
        else:
            hi = 1.0 if float(np.nanmax(target)) <= 1.0 + 1e-6 else 255.0
            target[...] = np.clip(target * s, 0.0, hi).astype(target.dtype, copy=False)
        return

    tensor_type = _torch_tensor_type()
    if tensor_type is not None and isinstance(target, tensor_type):
        target.mul_(s)
        target.clamp_(0, 1.0 if float(target.max()) <= 1.0 + 1e-6 else 255.0)
