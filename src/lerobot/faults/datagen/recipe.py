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

"""Typed JSON recipes for drop datagen and legacy SimpleIK generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.faults.config import POST_DROP_MODES

_LAYOUT_SEED_TAG = 0x4C41594F
_DROP_SEED_TAG = 0x44524F50
_CONTROLLER_SEED_TAG = 0x4354524C


class RecipeError(ValueError):
    """Raised when a datagen recipe cannot be loaded or validated."""


class DatagenController(str, Enum):
    SIMPLE_IK = "simple_ik"
    SMOLVLA = "smolvla"


class PostDropMode(str, Enum):
    IMMEDIATE_IK = "immediate_ik"
    CONTINUE_THEN_IK = "continue_then_ik"
    RESET_THEN_IK = "reset_then_ik"


_ALLOWED_CONTROLLER_MODES: frozenset[tuple[DatagenController, PostDropMode]] = frozenset(
    {
        (DatagenController.SIMPLE_IK, PostDropMode.IMMEDIATE_IK),
        (DatagenController.SIMPLE_IK, PostDropMode.CONTINUE_THEN_IK),
        (DatagenController.SMOLVLA, PostDropMode.IMMEDIATE_IK),
        (DatagenController.SMOLVLA, PostDropMode.CONTINUE_THEN_IK),
        (DatagenController.SMOLVLA, PostDropMode.RESET_THEN_IK),
    }
)

_CONTROLLER_MODE_TAGS: dict[tuple[DatagenController, PostDropMode], int] = {
    (DatagenController.SIMPLE_IK, PostDropMode.IMMEDIATE_IK): 1,
    (DatagenController.SIMPLE_IK, PostDropMode.CONTINUE_THEN_IK): 2,
    (DatagenController.SMOLVLA, PostDropMode.IMMEDIATE_IK): 3,
    (DatagenController.SMOLVLA, PostDropMode.CONTINUE_THEN_IK): 4,
    (DatagenController.SMOLVLA, PostDropMode.RESET_THEN_IK): 5,
}


@dataclass(frozen=True)
class PlacementRecipe:
    xy_range_m: float
    min_basket_clearance_m: float
    distractor_basket_clearance_m: float
    min_pairwise_clearance_m: float
    yaw_range_deg: tuple[float, float]
    max_attempts: int


@dataclass(frozen=True)
class DropRecipe:
    eligible_phases: tuple[str, ...]
    min_drop_distance_from_basket_m: float
    hard_keepout_floor_m: float


@dataclass(frozen=True)
class SimpleIKRecipe:
    trajectory_randomization_enabled: bool
    pickup_via_offset_m: float
    transport_via_offset_m: float
    arm_posture_noise_deg: float
    speed_multiplier_range: tuple[float, float]
    waypoint_blend_radius_m: float


@dataclass(frozen=True)
class DropXYBand:
    name: str
    min_m: float
    max_m: float


@dataclass(frozen=True)
class SmolVLARecipe:
    policy_path: str
    post_grasp_delay_steps: int
    drop_xy_bands: tuple[DropXYBand, ...]


@dataclass(frozen=True)
class PostDropRecipe:
    dwell_steps: int


@dataclass(frozen=True)
class RecordingRecipe:
    base_seed: int
    output_dir: str
    dataset_fps: int


@dataclass(frozen=True)
class MatrixVariant:
    controller: DatagenController
    post_drop_mode: PostDropMode
    episodes: int


@dataclass(frozen=True)
class ExpandedMatrixRun:
    controller: DatagenController
    post_drop_mode: PostDropMode
    episode_index: int
    logical_episode_index: int


@dataclass(frozen=True)
class EpisodeSeedManifest:
    controller: DatagenController
    post_drop_mode: PostDropMode
    logical_episode_index: int
    episode_index: int
    episode_seed: int
    layout_seed: int
    drop_seed: int
    controller_seed: int


@dataclass(frozen=True)
class DropDatagenRecipe:
    name: str
    task: str
    task_id: int
    control_hz: int
    q: float
    object_names: tuple[str, ...]
    basket_name: str
    placement: PlacementRecipe
    drop: DropRecipe
    simple_ik: SimpleIKRecipe
    smolvla: SmolVLARecipe
    post_drop: PostDropRecipe
    recording: RecordingRecipe
    experiment_matrix: tuple[MatrixVariant, ...]


@dataclass(frozen=True)
class LegacyPostDropConfig:
    dwell_steps: int
    mode_weights: dict[str, float]


@dataclass(frozen=True)
class DatagenRecipe:
    """Legacy can-only SimpleIK recipe (``object_name`` singular)."""

    q: float
    object_name: str
    basket_name: str
    placement: PlacementRecipe
    drop: DropRecipe
    simple_ik: SimpleIKRecipe


def _required(mapping: dict[str, Any], key: str, *, section: str = "recipe") -> Any:
    if key not in mapping:
        raise RecipeError(f"{section}.{key} is required")
    return mapping[key]


def _parse_controller(value: object, *, section: str) -> DatagenController:
    text = str(value).strip()
    try:
        return DatagenController(text)
    except ValueError as exc:
        raise RecipeError(f"{section} has unknown controller {text!r}") from exc


def _parse_post_drop_mode(value: object, *, section: str) -> PostDropMode:
    text = str(value).strip()
    try:
        return PostDropMode(text)
    except ValueError as exc:
        raise RecipeError(f"{section} has unknown post_drop_mode {text!r}") from exc


def validate_controller_mode_pair(
    controller: DatagenController,
    mode: PostDropMode,
    *,
    section: str = "experiment_matrix",
) -> None:
    pair = (controller, mode)
    if pair not in _ALLOWED_CONTROLLER_MODES:
        if controller is DatagenController.SIMPLE_IK and mode is PostDropMode.RESET_THEN_IK:
            raise RecipeError(
                f"{section}: simple_ik cannot use reset_then_ik (SmolVLA-only mode)"
            )
        raise RecipeError(
            f"{section}: invalid controller/mode pair {controller.value} × {mode.value}"
        )


def effective_post_drop_dwell_steps(
    recipe: DropDatagenRecipe,
    mode: PostDropMode,
) -> int:
    if mode is PostDropMode.IMMEDIATE_IK:
        return 0
    return int(recipe.post_drop.dwell_steps)


def expand_experiment_matrix(recipe: DropDatagenRecipe) -> tuple[ExpandedMatrixRun, ...]:
    runs: list[ExpandedMatrixRun] = []
    for variant in recipe.experiment_matrix:
        for episode_index in range(int(variant.episodes)):
            runs.append(
                ExpandedMatrixRun(
                    controller=variant.controller,
                    post_drop_mode=variant.post_drop_mode,
                    episode_index=episode_index,
                    logical_episode_index=episode_index,
                )
            )
    return tuple(runs)


def _derive_seed(*parts: int) -> int:
    seq = np.random.SeedSequence(list(parts))
    return int(seq.generate_state(1)[0])


def _episode_seed(base_seed: int, logical_episode_index: int) -> int:
    return int(base_seed) + int(logical_episode_index)


def paired_episode_seed_manifests(
    recipe: DropDatagenRecipe,
    logical_episode_index: int,
) -> tuple[EpisodeSeedManifest, ...]:
    base = int(recipe.recording.base_seed)
    ep_seed = _episode_seed(base, logical_episode_index)
    layout_seed = _derive_seed(ep_seed, _LAYOUT_SEED_TAG)
    drop_seed = _derive_seed(ep_seed, _DROP_SEED_TAG)
    manifests: list[EpisodeSeedManifest] = []
    for variant in recipe.experiment_matrix:
        for episode_index in range(int(variant.episodes)):
            if episode_index != logical_episode_index:
                continue
            ctrl_tag = _CONTROLLER_MODE_TAGS[(variant.controller, variant.post_drop_mode)]
            controller_seed = _derive_seed(ep_seed, _CONTROLLER_SEED_TAG, ctrl_tag)
            manifests.append(
                EpisodeSeedManifest(
                    controller=variant.controller,
                    post_drop_mode=variant.post_drop_mode,
                    logical_episode_index=logical_episode_index,
                    episode_index=episode_index,
                    episode_seed=ep_seed,
                    layout_seed=layout_seed,
                    drop_seed=drop_seed,
                    controller_seed=controller_seed,
                )
            )
    return tuple(manifests)


def _parse_placement(placement_raw: dict[str, Any]) -> PlacementRecipe:
    yaw_values = tuple(float(v) for v in _required(placement_raw, "yaw_range_deg", section="placement"))
    if len(yaw_values) != 2 or yaw_values[1] < yaw_values[0]:
        raise RecipeError("placement.yaw_range_deg must be [min, max] with max >= min")
    placement = PlacementRecipe(
        xy_range_m=float(_required(placement_raw, "xy_range_m", section="placement")),
        min_basket_clearance_m=float(
            _required(placement_raw, "min_basket_clearance_m", section="placement")
        ),
        distractor_basket_clearance_m=float(
            _required(placement_raw, "distractor_basket_clearance_m", section="placement")
        ),
        min_pairwise_clearance_m=float(
            _required(placement_raw, "min_pairwise_clearance_m", section="placement")
        ),
        yaw_range_deg=(yaw_values[0], yaw_values[1]),
        max_attempts=int(_required(placement_raw, "max_attempts", section="placement")),
    )
    if placement.xy_range_m < 0:
        raise RecipeError("placement.xy_range_m must be >= 0")
    if (
        placement.min_basket_clearance_m < 0
        or placement.distractor_basket_clearance_m < 0
        or placement.min_pairwise_clearance_m < 0
    ):
        raise RecipeError("placement clearances must be >= 0")
    if placement.distractor_basket_clearance_m > placement.min_basket_clearance_m:
        raise RecipeError(
            "placement.distractor_basket_clearance_m must be <= min_basket_clearance_m"
        )
    if placement.max_attempts < 1:
        raise RecipeError("placement.max_attempts must be >= 1")
    return placement


def _parse_drop(drop_raw: dict[str, Any]) -> DropRecipe:
    phases = tuple(str(v).strip() for v in _required(drop_raw, "eligible_phases", section="drop"))
    if not phases or any(not phase for phase in phases):
        raise RecipeError("drop.eligible_phases must be non-empty")
    drop = DropRecipe(
        eligible_phases=phases,
        min_drop_distance_from_basket_m=float(
            _required(drop_raw, "min_drop_distance_from_basket_m", section="drop")
        ),
        hard_keepout_floor_m=float(
            _required(drop_raw, "hard_keepout_floor_m", section="drop")
        ),
    )
    if drop.min_drop_distance_from_basket_m < 0:
        raise RecipeError("drop.min_drop_distance_from_basket_m must be >= 0")
    if drop.hard_keepout_floor_m < 0:
        raise RecipeError("drop.hard_keepout_floor_m must be >= 0")
    return drop


def _parse_simple_ik(simple_ik_raw: dict[str, Any]) -> SimpleIKRecipe:
    speed_values = tuple(
        float(v)
        for v in _required(simple_ik_raw, "speed_multiplier_range", section="simple_ik")
    )
    if len(speed_values) != 2:
        raise RecipeError("simple_ik.speed_multiplier_range must contain [min, max]")
    simple_ik = SimpleIKRecipe(
        trajectory_randomization_enabled=bool(
            _required(simple_ik_raw, "trajectory_randomization_enabled", section="simple_ik")
        ),
        pickup_via_offset_m=float(
            _required(simple_ik_raw, "pickup_via_offset_m", section="simple_ik")
        ),
        transport_via_offset_m=float(
            _required(simple_ik_raw, "transport_via_offset_m", section="simple_ik")
        ),
        arm_posture_noise_deg=float(
            _required(simple_ik_raw, "arm_posture_noise_deg", section="simple_ik")
        ),
        speed_multiplier_range=(speed_values[0], speed_values[1]),
        waypoint_blend_radius_m=float(
            _required(simple_ik_raw, "waypoint_blend_radius_m", section="simple_ik")
        ),
    )
    for field_name in (
        "pickup_via_offset_m",
        "transport_via_offset_m",
        "arm_posture_noise_deg",
        "waypoint_blend_radius_m",
    ):
        if getattr(simple_ik, field_name) < 0.0:
            raise RecipeError(f"simple_ik.{field_name} must be >= 0")
    speed_lo, speed_hi = simple_ik.speed_multiplier_range
    if speed_lo <= 0.0 or speed_hi <= 0.0:
        raise RecipeError("simple_ik.speed_multiplier_range values must be > 0")
    if speed_hi < speed_lo:
        raise RecipeError("simple_ik.speed_multiplier_range max must be >= min")
    return simple_ik


def _load_json_object(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RecipeError(f"Recipe file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeError(f"Could not load recipe {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RecipeError("recipe must be a JSON object")
    return raw


def load_drop_datagen_recipe(path: Path | str) -> DropDatagenRecipe:
    """Load and strictly validate the unified drop datagen recipe."""
    raw = _load_json_object(Path(path))

    q = float(_required(raw, "q"))
    if not 0.0 <= q <= 1.0:
        raise RecipeError(f"q must be in [0, 1], got {q}")

    object_names_raw = _required(raw, "object_names")
    if not isinstance(object_names_raw, list) or not object_names_raw:
        raise RecipeError("object_names must be a non-empty array")
    object_names = tuple(str(v).strip() for v in object_names_raw)
    if any(not name for name in object_names):
        raise RecipeError("object_names entries must be non-empty strings")

    basket_name = str(_required(raw, "basket_name")).strip()
    if not basket_name:
        raise RecipeError("basket_name must be non-empty")

    placement_raw = _required(raw, "placement")
    drop_raw = _required(raw, "drop")
    simple_ik_raw = _required(raw, "simple_ik")
    smolvla_raw = _required(raw, "smolvla")
    post_drop_raw = _required(raw, "post_drop")
    recording_raw = _required(raw, "recording")
    matrix_raw = _required(raw, "experiment_matrix")
    for key, value in (
        ("placement", placement_raw),
        ("drop", drop_raw),
        ("simple_ik", simple_ik_raw),
        ("smolvla", smolvla_raw),
        ("post_drop", post_drop_raw),
        ("recording", recording_raw),
    ):
        if not isinstance(value, dict):
            raise RecipeError(f"{key} must be a JSON object")
    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise RecipeError("experiment_matrix must be a non-empty array")

    placement = _parse_placement(placement_raw)
    drop = _parse_drop(drop_raw)
    simple_ik = _parse_simple_ik(simple_ik_raw)

    bands_raw = _required(smolvla_raw, "drop_xy_bands", section="smolvla")
    if not isinstance(bands_raw, list) or not bands_raw:
        raise RecipeError("smolvla.drop_xy_bands must be a non-empty array")
    bands: list[DropXYBand] = []
    for idx, band in enumerate(bands_raw):
        if not isinstance(band, dict):
            raise RecipeError(f"smolvla.drop_xy_bands[{idx}] must be an object")
        name = str(_required(band, "name", section=f"smolvla.drop_xy_bands[{idx}]")).strip()
        lo = float(_required(band, "min_m", section=f"smolvla.drop_xy_bands[{idx}]"))
        hi = float(_required(band, "max_m", section=f"smolvla.drop_xy_bands[{idx}]"))
        if lo < 0 or hi < 0 or hi < lo:
            raise RecipeError(f"smolvla.drop_xy_bands[{idx}] must have 0 <= min_m <= max_m")
        bands.append(DropXYBand(name=name, min_m=lo, max_m=hi))

    delay = _required(smolvla_raw, "post_grasp_delay_steps", section="smolvla")
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
        raise RecipeError("smolvla.post_grasp_delay_steps must be an int >= 0")
    policy_path = str(_required(smolvla_raw, "policy_path", section="smolvla")).strip()
    if not policy_path:
        raise RecipeError("smolvla.policy_path must be non-empty")
    smolvla = SmolVLARecipe(
        policy_path=policy_path,
        post_grasp_delay_steps=int(delay),
        drop_xy_bands=tuple(bands),
    )

    dwell = _required(post_drop_raw, "dwell_steps", section="post_drop")
    if not isinstance(dwell, int) or isinstance(dwell, bool) or dwell < 0:
        raise RecipeError("post_drop.dwell_steps must be an int >= 0")
    post_drop = PostDropRecipe(dwell_steps=int(dwell))

    output_dir = str(_required(recording_raw, "output_dir", section="recording")).strip()
    if not output_dir:
        raise RecipeError("recording.output_dir must be non-empty")
    base_seed = _required(recording_raw, "base_seed", section="recording")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise RecipeError("recording.base_seed must be an int")
    dataset_fps = _required(recording_raw, "dataset_fps", section="recording")
    if not isinstance(dataset_fps, int) or isinstance(dataset_fps, bool) or dataset_fps < 1:
        raise RecipeError("recording.dataset_fps must be an int >= 1")
    recording = RecordingRecipe(
        base_seed=int(base_seed),
        output_dir=output_dir,
        dataset_fps=int(dataset_fps),
    )

    control_hz = _required(raw, "control_hz")
    task_id = _required(raw, "task_id")
    if not isinstance(control_hz, int) or isinstance(control_hz, bool) or control_hz < 1:
        raise RecipeError("control_hz must be an int >= 1")
    if not isinstance(task_id, int) or isinstance(task_id, bool) or task_id < 0:
        raise RecipeError("task_id must be an int >= 0")

    matrix: list[MatrixVariant] = []
    for idx, entry in enumerate(matrix_raw):
        section = f"experiment_matrix[{idx}]"
        if not isinstance(entry, dict):
            raise RecipeError(f"{section} must be an object")
        controller = _parse_controller(
            _required(entry, "controller", section=section), section=section
        )
        mode = _parse_post_drop_mode(
            _required(entry, "post_drop_mode", section=section), section=section
        )
        validate_controller_mode_pair(controller, mode, section=section)
        episodes = _required(entry, "episodes", section=section)
        if not isinstance(episodes, int) or isinstance(episodes, bool) or episodes < 1:
            raise RecipeError(f"{section}.episodes must be an int >= 1")
        matrix.append(
            MatrixVariant(
                controller=controller,
                post_drop_mode=mode,
                episodes=int(episodes),
            )
        )

    return DropDatagenRecipe(
        name=str(_required(raw, "name")).strip(),
        task=str(_required(raw, "task")).strip(),
        task_id=int(task_id),
        control_hz=int(control_hz),
        q=q,
        object_names=object_names,
        basket_name=basket_name,
        placement=placement,
        drop=drop,
        simple_ik=simple_ik,
        smolvla=smolvla,
        post_drop=post_drop,
        recording=recording,
        experiment_matrix=tuple(matrix),
    )


def load_legacy_simple_ik_recipe(path: Path | str) -> DatagenRecipe:
    """Load and strictly validate a can-only SimpleIK datagen recipe."""
    raw = _load_json_object(Path(path))

    q = float(_required(raw, "q"))
    if not 0.0 <= q <= 1.0:
        raise RecipeError(f"q must be in [0, 1], got {q}")
    object_name = str(_required(raw, "object_name")).strip()
    basket_name = str(_required(raw, "basket_name")).strip()
    if not object_name:
        raise RecipeError("object_name must be non-empty")
    if not basket_name:
        raise RecipeError("basket_name must be non-empty")

    placement_raw = _required(raw, "placement")
    drop_raw = _required(raw, "drop")
    simple_ik_raw = _required(raw, "simple_ik")
    if not all(isinstance(value, dict) for value in (placement_raw, drop_raw, simple_ik_raw)):
        raise RecipeError("placement, drop, and simple_ik must be JSON objects")

    return DatagenRecipe(
        q=q,
        object_name=object_name,
        basket_name=basket_name,
        placement=_parse_placement(placement_raw),
        drop=_parse_drop(drop_raw),
        simple_ik=_parse_simple_ik(simple_ik_raw),
    )


def load_recipe(path: Path | str) -> DatagenRecipe:
    """Backward-compatible alias for :func:`load_legacy_simple_ik_recipe`."""
    return load_legacy_simple_ik_recipe(path)


def validate_post_drop_mode_weights(weights: dict[str, Any]) -> None:
    total = 0.0
    for key, value in weights.items():
        if key not in POST_DROP_MODES:
            raise RecipeError(f"Unknown post_drop mode {key!r}; expected one of {POST_DROP_MODES}.")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RecipeError(f"Weight for {key!r} must be a number (got {value!r}).")
        w = float(value)
        if w < 0:
            raise RecipeError(f"Weight for {key!r} must be >= 0 (got {w}).")
        total += w
    if total <= 0:
        raise RecipeError("post_drop.mode_weights must sum to a value > 0.")


def load_legacy_post_drop_recipe(path: Path | str) -> LegacyPostDropConfig:
    """Load ``post_drop`` dwell and mode weights from a legacy union recipe JSON."""
    raw = _load_json_object(Path(path))
    post_drop = raw.get("post_drop")
    if not isinstance(post_drop, dict):
        raise RecipeError("Recipe must contain a post_drop object.")
    dwell = post_drop.get("dwell_steps")
    if not isinstance(dwell, int) or isinstance(dwell, bool) or dwell < 0:
        raise RecipeError(f"post_drop.dwell_steps must be an int >= 0 (got {dwell!r}).")
    weights = post_drop.get("mode_weights")
    if not isinstance(weights, dict):
        raise RecipeError("post_drop.mode_weights must be an object.")
    validate_post_drop_mode_weights(weights)
    mode_weights = {str(k): float(v) for k, v in weights.items()}
    return LegacyPostDropConfig(dwell_steps=int(dwell), mode_weights=mode_weights)


def sample_post_drop_mode(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Sample a post-drop mode from non-zero weights."""
    validate_post_drop_mode_weights(weights)
    modes: list[str] = []
    probs: list[float] = []
    for mode in POST_DROP_MODES:
        w = float(weights.get(mode, 0.0))
        if w > 0:
            modes.append(mode)
            probs.append(w)
    total = sum(probs)
    idx = int(rng.choice(len(modes), p=[p / total for p in probs]))
    return modes[idx]
