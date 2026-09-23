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

"""Typed JSON recipes for unified drop datagen."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

_LAYOUT_SEED_TAG = 0x4C41594F
_DROP_SEED_TAG = 0x44524F50
_CONTROLLER_SEED_TAG = 0x4354524C


class RecipeError(ValueError):
    """Raised when a datagen recipe cannot be loaded or validated."""


class DatagenController(StrEnum):
    """Policy/controller backend used for a datagen recipe section."""

    SIMPLE_IK = "simple_ik"
    SMOLVLA = "smolvla"


class PostDropMode(StrEnum):
    """How control resumes after a mid-air drop in datagen."""

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
    """Object placement randomization bounds for scene layout sampling."""

    xy_range_m: float
    min_basket_clearance_m: float
    distractor_basket_clearance_m: float
    min_pairwise_clearance_m: float
    yaw_range_deg: tuple[float, float]
    max_attempts: int


@dataclass(frozen=True)
class DropRecipe:
    """Legacy top-level drop settings (SimpleIK step-index sampling)."""

    eligible_phases: tuple[str, ...]
    min_drop_distance_from_basket_m: float
    hard_keepout_floor_m: float


@dataclass(frozen=True)
class SimpleIKPathDropRecipe:
    """Planned-path drop sampling settings for unified SimpleIK runs."""

    eligible_phases: tuple[str, ...]
    min_drop_distance_from_basket_m: float
    hard_keepout_floor_m: float


@dataclass(frozen=True)
class SimpleIKRecipe:
    """SimpleIK controller settings for unified drop datagen."""

    trajectory_randomization_enabled: bool
    pickup_via_offset_m: float
    transport_via_offset_m: float
    arm_posture_noise_deg: float
    speed_multiplier_range: tuple[float, float]
    waypoint_blend_radius_m: float
    path_drop: SimpleIKPathDropRecipe | None = None


@dataclass(frozen=True)
class DropXYBand:
    """Named basket XY distance interval for SmolVLA drop triggering."""

    name: str
    min_m: float
    max_m: float


@dataclass(frozen=True)
class SmolVLARecipe:
    """SmolVLA policy and drop-band settings for unified datagen."""

    policy_path: str
    post_grasp_delay_steps: int
    min_drop_distance_from_basket_m: float
    drop_xy_bands: tuple[DropXYBand, ...]


@dataclass(frozen=True)
class PostDropRecipe:
    """Post-drop dwell configuration shared across modes that need settling."""

    dwell_steps: int


@dataclass(frozen=True)
class RecordingRecipe:
    """Dataset output location, FPS, and base seed for the matrix run."""

    base_seed: int
    output_dir: str
    dataset_fps: int


@dataclass(frozen=True)
class MatrixVariant:
    """One controller/mode pair and its per-run episode count."""

    controller: DatagenController
    post_drop_mode: PostDropMode
    episodes: int


@dataclass(frozen=True)
class ExpandedMatrixRun:
    """Flattened matrix entry pairing variant with a logical episode index."""

    controller: DatagenController
    post_drop_mode: PostDropMode
    episode_index: int
    logical_episode_index: int


@dataclass(frozen=True)
class EpisodeSeedManifest:
    """Deterministic seeds for one matrix variant episode."""

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
    """Top-level unified drop datagen recipe loaded from JSON."""

    name: str
    task: str
    task_id: int
    control_hz: int
    q: float
    object_names: tuple[str, ...]
    basket_name: str
    placement: PlacementRecipe
    simple_ik: SimpleIKRecipe
    smolvla: SmolVLARecipe
    post_drop: PostDropRecipe
    recording: RecordingRecipe
    experiment_matrix: tuple[MatrixVariant, ...]


def _required(mapping: dict[str, Any], key: str, *, section: str = "recipe") -> Any:
    if key not in mapping:
        raise RecipeError(f"{section}.{key} is required")
    return mapping[key]


def _require_json_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise RecipeError(f"{field} must be a JSON boolean (got {value!r})")
    return value


def _require_json_str(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise RecipeError(f"{field} must be a JSON string (got {value!r})")
    text = value.strip()
    if not allow_empty and not text:
        raise RecipeError(f"{field} must be non-empty")
    return text


def _require_json_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeError(f"{field} must be a JSON number (got {value!r})")
    return float(value)


def _require_json_int(value: object, *, field: str, min_value: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RecipeError(f"{field} must be a JSON int (got {value!r})")
    if min_value is not None and value < min_value:
        raise RecipeError(f"{field} must be >= {min_value}")
    return value


def _parse_controller(value: object, *, section: str, strict: bool) -> DatagenController:
    text = _require_json_str(value, field=f"{section}.controller") if strict else str(value).strip()
    try:
        return DatagenController(text)
    except ValueError as exc:
        raise RecipeError(f"{section} has unknown controller {text!r}") from exc


def _parse_post_drop_mode(value: object, *, section: str, strict: bool) -> PostDropMode:
    text = _require_json_str(value, field=f"{section}.post_drop_mode") if strict else str(value).strip()
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
    """Raise ``RecipeError`` when a controller/post-drop mode pair is not allowed."""
    pair = (controller, mode)
    if pair not in _ALLOWED_CONTROLLER_MODES:
        if controller is DatagenController.SIMPLE_IK and mode is PostDropMode.RESET_THEN_IK:
            raise RecipeError(f"{section}: simple_ik cannot use reset_then_ik (SmolVLA-only mode)")
        raise RecipeError(f"{section}: invalid controller/mode pair {controller.value} × {mode.value}")


def validate_experiment_matrix_entries(matrix: tuple[MatrixVariant, ...]) -> None:
    """Require each approved controller/mode pair exactly once with equal episode counts."""
    expected = len(_ALLOWED_CONTROLLER_MODES)
    if len(matrix) != expected:
        raise RecipeError(f"experiment_matrix must contain exactly {expected} variants (got {len(matrix)})")
    seen: set[tuple[DatagenController, PostDropMode]] = set()
    episode_counts: set[int] = set()
    for variant in matrix:
        pair = (variant.controller, variant.post_drop_mode)
        if pair in seen:
            raise RecipeError(
                "experiment_matrix: duplicate controller/mode pair "
                f"{variant.controller.value} × {variant.post_drop_mode.value}"
            )
        seen.add(pair)
        episode_counts.add(variant.episodes)
    missing = _ALLOWED_CONTROLLER_MODES - seen
    if missing:
        labels = ", ".join(f"{c.value} × {m.value}" for c, m in missing)
        raise RecipeError(f"experiment_matrix: missing controller/mode pair(s): {labels}")
    if len(episode_counts) != 1:
        raise RecipeError("experiment_matrix: all variants must have the same episodes count")


def legacy_drop_recipe(recipe: DropDatagenRecipe) -> DropRecipe:
    """Map unified SimpleIK path-drop settings to the legacy ``DropRecipe`` shape."""
    path_drop = recipe.simple_ik.path_drop
    if path_drop is None:
        raise RecipeError("simple_ik.path_drop is required for unified drop datagen")
    return DropRecipe(
        eligible_phases=path_drop.eligible_phases,
        min_drop_distance_from_basket_m=path_drop.min_drop_distance_from_basket_m,
        hard_keepout_floor_m=path_drop.hard_keepout_floor_m,
    )


def effective_post_drop_dwell_steps(
    recipe: DropDatagenRecipe,
    mode: PostDropMode,
) -> int:
    """Return configured dwell steps for ``mode`` (zero for immediate IK)."""
    if mode is PostDropMode.IMMEDIATE_IK:
        return 0
    return int(recipe.post_drop.dwell_steps)


def expand_experiment_matrix(recipe: DropDatagenRecipe) -> tuple[ExpandedMatrixRun, ...]:
    """List every variant episode slot in the experiment matrix."""
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
    """Build per-variant seed manifests for one shared logical episode index."""
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


def _parse_placement(placement_raw: dict[str, Any], *, strict: bool = False) -> PlacementRecipe:
    yaw_raw = _required(placement_raw, "yaw_range_deg", section="placement")
    if not isinstance(yaw_raw, list) or len(yaw_raw) != 2:
        raise RecipeError("placement.yaw_range_deg must be [min, max]")
    if strict:
        yaw_values = (
            _require_json_number(yaw_raw[0], field="placement.yaw_range_deg[0]"),
            _require_json_number(yaw_raw[1], field="placement.yaw_range_deg[1]"),
        )
        xy_range_m = _require_json_number(
            _required(placement_raw, "xy_range_m", section="placement"),
            field="placement.xy_range_m",
        )
        min_basket_clearance_m = _require_json_number(
            _required(placement_raw, "min_basket_clearance_m", section="placement"),
            field="placement.min_basket_clearance_m",
        )
        distractor_basket_clearance_m = _require_json_number(
            _required(placement_raw, "distractor_basket_clearance_m", section="placement"),
            field="placement.distractor_basket_clearance_m",
        )
        min_pairwise_clearance_m = _require_json_number(
            _required(placement_raw, "min_pairwise_clearance_m", section="placement"),
            field="placement.min_pairwise_clearance_m",
        )
        max_attempts = _require_json_int(
            _required(placement_raw, "max_attempts", section="placement"),
            field="placement.max_attempts",
            min_value=1,
        )
    else:
        yaw_values = tuple(float(v) for v in yaw_raw)
        xy_range_m = float(_required(placement_raw, "xy_range_m", section="placement"))
        min_basket_clearance_m = float(
            _required(placement_raw, "min_basket_clearance_m", section="placement")
        )
        distractor_basket_clearance_m = float(
            _required(placement_raw, "distractor_basket_clearance_m", section="placement")
        )
        min_pairwise_clearance_m = float(
            _required(placement_raw, "min_pairwise_clearance_m", section="placement")
        )
        max_attempts = int(_required(placement_raw, "max_attempts", section="placement"))
    if yaw_values[1] < yaw_values[0]:
        raise RecipeError("placement.yaw_range_deg must be [min, max] with max >= min")
    placement = PlacementRecipe(
        xy_range_m=xy_range_m,
        min_basket_clearance_m=min_basket_clearance_m,
        distractor_basket_clearance_m=distractor_basket_clearance_m,
        min_pairwise_clearance_m=min_pairwise_clearance_m,
        yaw_range_deg=(yaw_values[0], yaw_values[1]),
        max_attempts=max_attempts,
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
        raise RecipeError("placement.distractor_basket_clearance_m must be <= min_basket_clearance_m")
    if not strict and placement.max_attempts < 1:
        raise RecipeError("placement.max_attempts must be >= 1")
    return placement


def _parse_simple_ik_path_drop(path_drop_raw: dict[str, Any]) -> SimpleIKPathDropRecipe:
    section = "simple_ik.path_drop"
    phases_raw = _required(path_drop_raw, "eligible_phases", section=section)
    if not isinstance(phases_raw, list) or not phases_raw:
        raise RecipeError(f"{section}.eligible_phases must be a non-empty array")
    phases = tuple(
        _require_json_str(value, field=f"{section}.eligible_phases[{idx}]")
        for idx, value in enumerate(phases_raw)
    )
    min_dist = _require_json_number(
        _required(path_drop_raw, "min_drop_distance_from_basket_m", section=section),
        field=f"{section}.min_drop_distance_from_basket_m",
    )
    keepout = _require_json_number(
        _required(path_drop_raw, "hard_keepout_floor_m", section=section),
        field=f"{section}.hard_keepout_floor_m",
    )
    if min_dist < 0:
        raise RecipeError(f"{section}.min_drop_distance_from_basket_m must be >= 0")
    if keepout < 0:
        raise RecipeError(f"{section}.hard_keepout_floor_m must be >= 0")
    return SimpleIKPathDropRecipe(
        eligible_phases=phases,
        min_drop_distance_from_basket_m=min_dist,
        hard_keepout_floor_m=keepout,
    )


def _parse_simple_ik(simple_ik_raw: dict[str, Any]) -> SimpleIKRecipe:
    speed_values = tuple(
        float(v) for v in _required(simple_ik_raw, "speed_multiplier_range", section="simple_ik")
    )
    if len(speed_values) != 2:
        raise RecipeError("simple_ik.speed_multiplier_range must contain [min, max]")
    simple_ik = SimpleIKRecipe(
        trajectory_randomization_enabled=bool(
            _required(simple_ik_raw, "trajectory_randomization_enabled", section="simple_ik")
        ),
        pickup_via_offset_m=float(_required(simple_ik_raw, "pickup_via_offset_m", section="simple_ik")),
        transport_via_offset_m=float(_required(simple_ik_raw, "transport_via_offset_m", section="simple_ik")),
        arm_posture_noise_deg=float(_required(simple_ik_raw, "arm_posture_noise_deg", section="simple_ik")),
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


def _parse_simple_ik_unified(simple_ik_raw: dict[str, Any]) -> SimpleIKRecipe:
    speed_raw = _required(simple_ik_raw, "speed_multiplier_range", section="simple_ik")
    if not isinstance(speed_raw, list) or len(speed_raw) != 2:
        raise RecipeError("simple_ik.speed_multiplier_range must contain [min, max]")
    speed_values = (
        _require_json_number(speed_raw[0], field="simple_ik.speed_multiplier_range[0]"),
        _require_json_number(speed_raw[1], field="simple_ik.speed_multiplier_range[1]"),
    )
    path_drop_raw = _required(simple_ik_raw, "path_drop", section="simple_ik")
    if not isinstance(path_drop_raw, dict):
        raise RecipeError("simple_ik.path_drop must be a JSON object")
    simple_ik = SimpleIKRecipe(
        trajectory_randomization_enabled=_require_json_bool(
            _required(simple_ik_raw, "trajectory_randomization_enabled", section="simple_ik"),
            field="simple_ik.trajectory_randomization_enabled",
        ),
        pickup_via_offset_m=_require_json_number(
            _required(simple_ik_raw, "pickup_via_offset_m", section="simple_ik"),
            field="simple_ik.pickup_via_offset_m",
        ),
        transport_via_offset_m=_require_json_number(
            _required(simple_ik_raw, "transport_via_offset_m", section="simple_ik"),
            field="simple_ik.transport_via_offset_m",
        ),
        arm_posture_noise_deg=_require_json_number(
            _required(simple_ik_raw, "arm_posture_noise_deg", section="simple_ik"),
            field="simple_ik.arm_posture_noise_deg",
        ),
        speed_multiplier_range=speed_values,
        waypoint_blend_radius_m=_require_json_number(
            _required(simple_ik_raw, "waypoint_blend_radius_m", section="simple_ik"),
            field="simple_ik.waypoint_blend_radius_m",
        ),
        path_drop=_parse_simple_ik_path_drop(path_drop_raw),
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

    if "drop" in raw:
        raise RecipeError("drop must not appear at the recipe root; use simple_ik.path_drop for SimpleIK")

    q = _require_json_number(_required(raw, "q"), field="q")
    if not 0.0 <= q <= 1.0:
        raise RecipeError(f"q must be in [0, 1], got {q}")

    object_names_raw = _required(raw, "object_names")
    if not isinstance(object_names_raw, list) or not object_names_raw:
        raise RecipeError("object_names must be a non-empty array")
    object_names = tuple(
        _require_json_str(value, field=f"object_names[{idx}]") for idx, value in enumerate(object_names_raw)
    )

    basket_name = _require_json_str(_required(raw, "basket_name"), field="basket_name")

    placement_raw = _required(raw, "placement")
    simple_ik_raw = _required(raw, "simple_ik")
    smolvla_raw = _required(raw, "smolvla")
    post_drop_raw = _required(raw, "post_drop")
    recording_raw = _required(raw, "recording")
    matrix_raw = _required(raw, "experiment_matrix")
    for key, value in (
        ("placement", placement_raw),
        ("simple_ik", simple_ik_raw),
        ("smolvla", smolvla_raw),
        ("post_drop", post_drop_raw),
        ("recording", recording_raw),
    ):
        if not isinstance(value, dict):
            raise RecipeError(f"{key} must be a JSON object")
    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise RecipeError("experiment_matrix must be a non-empty array")

    placement = _parse_placement(placement_raw, strict=True)
    simple_ik = _parse_simple_ik_unified(simple_ik_raw)

    bands_raw = _required(smolvla_raw, "drop_xy_bands", section="smolvla")
    if not isinstance(bands_raw, list) or not bands_raw:
        raise RecipeError("smolvla.drop_xy_bands must be a non-empty array")
    bands: list[DropXYBand] = []
    for idx, band in enumerate(bands_raw):
        if not isinstance(band, dict):
            raise RecipeError(f"smolvla.drop_xy_bands[{idx}] must be an object")
        band_section = f"smolvla.drop_xy_bands[{idx}]"
        name = _require_json_str(
            _required(band, "name", section=band_section),
            field=f"{band_section}.name",
        )
        lo = _require_json_number(
            _required(band, "min_m", section=band_section),
            field=f"{band_section}.min_m",
        )
        hi = _require_json_number(
            _required(band, "max_m", section=band_section),
            field=f"{band_section}.max_m",
        )
        if lo < 0 or hi < 0 or hi < lo:
            raise RecipeError(f"{band_section} must have 0 <= min_m <= max_m")
        bands.append(DropXYBand(name=name, min_m=lo, max_m=hi))

    delay = _require_json_int(
        _required(smolvla_raw, "post_grasp_delay_steps", section="smolvla"),
        field="smolvla.post_grasp_delay_steps",
        min_value=0,
    )
    policy_path = _require_json_str(
        _required(smolvla_raw, "policy_path", section="smolvla"),
        field="smolvla.policy_path",
    )
    smolvla_min_drop = _require_json_number(
        _required(smolvla_raw, "min_drop_distance_from_basket_m", section="smolvla"),
        field="smolvla.min_drop_distance_from_basket_m",
    )
    if smolvla_min_drop < 0:
        raise RecipeError("smolvla.min_drop_distance_from_basket_m must be >= 0")
    smolvla = SmolVLARecipe(
        policy_path=policy_path,
        post_grasp_delay_steps=delay,
        min_drop_distance_from_basket_m=float(smolvla_min_drop),
        drop_xy_bands=tuple(bands),
    )

    dwell = _require_json_int(
        _required(post_drop_raw, "dwell_steps", section="post_drop"),
        field="post_drop.dwell_steps",
        min_value=0,
    )
    post_drop = PostDropRecipe(dwell_steps=dwell)

    output_dir = _require_json_str(
        _required(recording_raw, "output_dir", section="recording"),
        field="recording.output_dir",
    )
    base_seed = _require_json_int(
        _required(recording_raw, "base_seed", section="recording"),
        field="recording.base_seed",
    )
    dataset_fps = _require_json_int(
        _required(recording_raw, "dataset_fps", section="recording"),
        field="recording.dataset_fps",
        min_value=1,
    )
    recording = RecordingRecipe(
        base_seed=base_seed,
        output_dir=output_dir,
        dataset_fps=dataset_fps,
    )

    control_hz = _require_json_int(_required(raw, "control_hz"), field="control_hz", min_value=1)
    task_id = _require_json_int(_required(raw, "task_id"), field="task_id", min_value=0)

    matrix: list[MatrixVariant] = []
    for idx, entry in enumerate(matrix_raw):
        section = f"experiment_matrix[{idx}]"
        if not isinstance(entry, dict):
            raise RecipeError(f"{section} must be an object")
        controller = _parse_controller(
            _required(entry, "controller", section=section),
            section=section,
            strict=True,
        )
        mode = _parse_post_drop_mode(
            _required(entry, "post_drop_mode", section=section),
            section=section,
            strict=True,
        )
        validate_controller_mode_pair(controller, mode, section=section)
        episodes = _require_json_int(
            _required(entry, "episodes", section=section),
            field=f"{section}.episodes",
            min_value=1,
        )
        matrix.append(
            MatrixVariant(
                controller=controller,
                post_drop_mode=mode,
                episodes=episodes,
            )
        )

    experiment_matrix = tuple(matrix)
    validate_experiment_matrix_entries(experiment_matrix)

    return DropDatagenRecipe(
        name=_require_json_str(_required(raw, "name"), field="name"),
        task=_require_json_str(_required(raw, "task"), field="task"),
        task_id=task_id,
        control_hz=control_hz,
        q=q,
        object_names=object_names,
        basket_name=basket_name,
        placement=placement,
        simple_ik=simple_ik,
        smolvla=smolvla,
        post_drop=post_drop,
        recording=recording,
        experiment_matrix=experiment_matrix,
    )
