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

"""Record eval episodes from a vector-env wrapper.

Stock ``rollout`` writes frames with raw env keys. This wrapper writes the
mapped LeRobot keys (``features_map``), optional failure columns, and can drop
failed episodes. ``lerobot-eval-faults`` attaches it and passes
``recording_dir=None`` into upstream ``eval_policy_all`` so the stock recorder
stays off.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_IMAGE
from lerobot_faults.annotation import (
    FAILURE_ANNOTATION_FEATURES,
    default_failure_frame,
    failure_frame_from_info,
)
from lerobot_faults.success_filter import finish_eval_recorded_episode

logger = logging.getLogger(__name__)


def _dataset_feature_key(env_key: str, features_map: dict[str, str] | None) -> str:
    """Map an env feature key to a LeRobot dataset key (no ``/``)."""
    mapped = (features_map or {}).get(env_key, env_key)
    return mapped.replace("/", ".")


def _index_nested_obs(raw_obs: dict, path: list[str], env_idx: int) -> Any:
    """Walk a slash-path through a (possibly batched) gym observation."""
    cur: Any = raw_obs
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing observation path {'/'.join(path)} at {part!r}")
        cur = cur[part]
    if isinstance(cur, np.ndarray) and cur.ndim >= 1:
        return cur[env_idx]
    return cur


def _env_features_to_dataset_features(
    env_features: dict[str, PolicyFeature],
    features_map: dict[str, str] | None = None,
    annotate_failures: bool = False,
) -> dict[str, dict[str, Any]]:
    """Convert EnvConfig.features to the dict format expected by LeRobotDataset.create()."""
    features: dict[str, dict[str, Any]] = {}
    for key, ft in env_features.items():
        ds_key = _dataset_feature_key(key, features_map)
        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            features[ds_key] = {"dtype": "video", "shape": shape, "names": ["height", "width", "channel"]}
        else:
            features[ds_key] = {"dtype": "float32", "shape": shape, "names": None}
    features["next.reward"] = {"dtype": "float32", "shape": (1,), "names": None}
    features["next.success"] = {"dtype": "bool", "shape": (1,), "names": None}
    features["next.done"] = {"dtype": "bool", "shape": (1,), "names": None}
    if annotate_failures:
        features.update(FAILURE_ANNOTATION_FEATURES)
    return features


def _build_raw_frame(
    raw_obs: dict,
    env_idx: int,
    action: np.ndarray,
    reward: float,
    success: bool,
    done: bool,
    task: str,
    env_features: dict,
    info: dict | None = None,
    features_map: dict[str, str] | None = None,
    annotate_failures: bool = False,
) -> dict:
    """Build a dataset frame from raw env observations for one env index.

    Frame keys are the mapped LeRobot names (``features_map``), not raw ``a/b`` env keys.
    """
    frame: dict[str, Any] = {}
    for key, ft in env_features.items():
        if key == ACTION:
            continue
        ds_key = _dataset_feature_key(key, features_map)
        if ds_key.startswith("next.") or ds_key in FAILURE_ANNOTATION_FEATURES:
            continue
        if ft.type is FeatureType.VISUAL and "pixels" in raw_obs:
            pixels = raw_obs["pixels"]
            if isinstance(pixels, dict):
                cam_name = ds_key.rsplit(".", 1)[-1]
                raw_cam = key.split("/", 1)[-1]
                img = pixels.get(cam_name, pixels.get(raw_cam))
                if img is None:
                    raise KeyError(f"No camera image for {ds_key} (tried {cam_name!r}, {raw_cam!r})")
                frame[ds_key] = img[env_idx]
            elif ds_key in (OBS_IMAGE, "pixels"):
                frame[ds_key] = pixels[env_idx]
            continue
        try:
            val = _index_nested_obs(raw_obs, key.split("/"), env_idx)
        except KeyError:
            if key in raw_obs and isinstance(raw_obs[key], np.ndarray):
                val = raw_obs[key][env_idx]
            else:
                continue
        if isinstance(val, np.ndarray) and val.dtype == np.float64:
            val = val.astype(np.float32)
        frame[ds_key] = val
    frame[ACTION] = action
    frame["next.reward"] = np.atleast_1d(np.float32(reward))
    frame["next.success"] = np.atleast_1d(np.bool_(success))
    frame["next.done"] = np.atleast_1d(np.bool_(done))
    if annotate_failures:
        frame.update(failure_frame_from_info(info, env_idx) if info is not None else default_failure_frame())
    frame["task"] = task
    return frame


def _create_recording_datasets(
    recording_dir: Path,
    env: Any,
    env_features: dict,
    recording_repo_id: str | None = None,
    features_map: dict[str, str] | None = None,
    annotate_failures: bool = False,
) -> list[LeRobotDataset]:
    """Create one write-mode dataset per vec-env slot."""
    features = _env_features_to_dataset_features(env_features, features_map, annotate_failures)
    fps = env.unwrapped.metadata.get("render_fps", 30)
    multi_env = env.num_envs > 1
    base_repo_id = recording_repo_id or "eval_recording"
    datasets: list[LeRobotDataset] = []
    for i in range(env.num_envs):
        root = str(recording_dir / f"env_{i}") if multi_env else str(recording_dir)
        repo_id = f"{base_repo_id}_env_{i}" if multi_env else base_repo_id
        datasets.append(
            LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                features=features,
                root=root,
                use_videos=True,
            )
        )
    return datasets


def _successes_from_info(info: dict, n_envs: int) -> list[bool]:
    """Read per-env success flags the same way stock ``rollout`` does."""
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            is_success = final_info.get("is_success", [False] * n_envs)
            if hasattr(is_success, "tolist"):
                return [bool(v) for v in is_success.tolist()]
            return [bool(is_success)] * n_envs
        successes: list[bool] = []
        for item in final_info:
            if isinstance(item, dict) and "is_success" in item:
                successes.append(bool(item["is_success"]))
            else:
                successes.append(False)
        return successes
    if "is_success" in info:
        is_success = info["is_success"]
        if hasattr(is_success, "tolist"):
            return [bool(v) for v in is_success.tolist()]
        return [bool(is_success)] * n_envs
    return [False] * n_envs


class EvalRecordingVecWrapper:
    """Vector env that writes one LeRobot episode per sub-env from reset/step."""

    def __init__(
        self,
        env: Any,
        *,
        recording_dir: Path,
        env_features: dict,
        features_map: dict[str, str] | None,
        recording_repo_id: str | None,
        recording_private: bool,
        success_only: bool,
        annotate_failures: bool,
    ) -> None:
        """Create one dataset per sub-env and remember recording options."""
        self.env = env
        self._features_map = features_map
        self._env_features = env_features
        self._recording_private = recording_private
        self._recording_repo_id = recording_repo_id
        self._success_only = success_only
        self._annotate_failures = annotate_failures
        self._datasets = _create_recording_datasets(
            recording_dir,
            env,
            env_features,
            recording_repo_id,
            features_map,
            annotate_failures,
        )
        self._finalized = False
        self._raw_obs: dict | None = None
        self._done = np.zeros(env.num_envs, dtype=bool)
        self._episode_succeeded = np.zeros(env.num_envs, dtype=bool)
        self._step = 0
        self._max_steps: int | None = None
        self._task_desc = ""

    def __getattr__(self, name: str) -> Any:
        """Forward attributes the rollout loop reads from the inner vector env."""
        return getattr(self.env, name)

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Forward ``env.call`` so rollout can read task text and max steps."""
        return self.env.call(name, *args, **kwargs)

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        """Reset the inner env and start a fresh per-slot episode buffer."""
        observation, info = self.env.reset(**kwargs)
        self._raw_obs = deepcopy(observation)
        self._done = np.zeros(self.env.num_envs, dtype=bool)
        self._episode_succeeded = np.zeros(self.env.num_envs, dtype=bool)
        self._step = 0
        self._task_desc = self._read_task_description()
        return observation, info

    def step(self, action: np.ndarray) -> tuple[Any, Any, Any, Any, dict]:
        """Step, then append the pre-step observation and this action as a frame."""
        if self._raw_obs is None:
            raise RuntimeError("EvalRecordingVecWrapper.step called before reset")
        prev_done = self._done.copy()
        raw_observation = self._raw_obs
        observation, reward, terminated, truncated, info = self.env.step(action)
        successes = _successes_from_info(info, self.env.num_envs)
        action_np = np.asarray(action)
        for env_idx in range(self.env.num_envs):
            if prev_done[env_idx]:
                continue
            self._episode_succeeded[env_idx] = bool(self._episode_succeeded[env_idx] or successes[env_idx])
            frame = _build_raw_frame(
                raw_observation,
                env_idx,
                action_np[env_idx],
                float(reward[env_idx]),
                successes[env_idx],
                bool(terminated[env_idx] or truncated[env_idx]),
                self._task_desc,
                self._env_features,
                info=info,
                features_map=self._features_map,
                annotate_failures=self._annotate_failures,
            )
            self._datasets[env_idx].add_frame(frame)
            if terminated[env_idx] or truncated[env_idx]:
                finish_eval_recorded_episode(
                    self._datasets[env_idx],
                    succeeded=bool(self._episode_succeeded[env_idx]),
                    success_only=self._success_only,
                )
        self._done = np.asarray(terminated) | np.asarray(truncated) | self._done
        self._step += 1
        if self._step == self._max_episode_steps():
            self._done = np.ones_like(self._done, dtype=bool)
            for env_idx in range(self.env.num_envs):
                if self._datasets[env_idx].has_pending_frames():
                    finish_eval_recorded_episode(
                        self._datasets[env_idx],
                        succeeded=bool(self._episode_succeeded[env_idx]),
                        success_only=self._success_only,
                    )
        self._raw_obs = deepcopy(observation)
        return observation, reward, terminated, truncated, info

    def close(self, **kwargs: Any) -> None:
        """Finalize datasets, optionally push them, then close the inner env."""
        self._finalize_datasets()
        close = getattr(self.env, "close", None)
        if close is not None:
            close(**kwargs)

    def _read_task_description(self) -> str:
        try:
            return str(list(self.env.call("task_description"))[0])
        except (AttributeError, NotImplementedError, IndexError):
            return ""

    def _max_episode_steps(self) -> int:
        if self._max_steps is None:
            self._max_steps = int(self.env.call("_max_episode_steps")[0])
        return self._max_steps

    def _finalize_datasets(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        for ds in self._datasets:
            ds.finalize()
            if self._recording_repo_id is not None and "/" in self._recording_repo_id:
                if ds.num_episodes > 0:
                    ds.push_to_hub(private=self._recording_private)
                else:
                    logger.warning("No episodes recorded for %s — skipping push to hub.", ds.repo_id)


def attach_eval_recording(
    envs: dict[str, dict[int, Any]],
    recording_dir: Path,
    env_features: dict,
    *,
    features_map: dict[str, str] | None,
    recording_repo_id: str | None,
    recording_private: bool,
    success_only: bool,
    annotate_failures: bool,
) -> dict[str, dict[int, Any]]:
    """Replace each vec env with a recorder. Directory layout matches stock ``run_one``."""
    for task_group, group in envs.items():
        for task_id, vec in list(group.items()):
            task_dir = recording_dir / f"{task_group}_{task_id}"
            repo_id = (
                f"{recording_repo_id}_{task_group}_{task_id}" if recording_repo_id is not None else None
            )
            group[task_id] = EvalRecordingVecWrapper(
                vec,
                recording_dir=task_dir,
                env_features=env_features,
                features_map=features_map,
                recording_repo_id=repo_id,
                recording_private=recording_private,
                success_only=success_only,
                annotate_failures=annotate_failures,
            )
    return envs
