# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

OverlayMode = Literal["replace", "add"]


@dataclass(frozen=True)
class AddObjectSpec:
    """Declare one object to inject into a LIBERO BDDL scene."""

    category: str
    ranges: tuple[float, float, float, float]
    """Axis-aligned region on the workspace: (xmin, ymin, xmax, ymax)."""
    target: str = "floor"
    """BDDL region target fixture name (e.g. ``floor``, ``kitchen_table``)."""
    instance: str | None = None
    """Instance id. Defaults to ``{category}_1``."""
    region_name: str | None = None
    """Region name without the target prefix. Defaults to ``{category}_init_region``."""
    yaw_rotation: tuple[float, float] | None = None

    def resolved_instance(self) -> str:
        return self.instance or f"{self.category}_1"

    def resolved_region_name(self) -> str:
        return self.region_name or f"{self.category}_init_region"


@dataclass(frozen=True)
class LiberoOverlayConfig:
    """Declarative overlay applied on top of a stock LIBERO BDDL task.

    Stock assets/BDDL under the LIBERO install are never modified. Paths for
    ``objects_module`` are resolved relative to the overlay YAML file.
    """

    mode: OverlayMode
    replacements: dict[str, str] = field(default_factory=dict)
    """Map stock category → custom category (``mode=replace``)."""
    add_objects: tuple[AddObjectSpec, ...] = ()
    """Objects to inject (``mode=add``)."""
    language: str | None = None
    """Optional ``(:language ...)`` override."""
    objects_module: str | None = None
    """Python file or package path that registers custom ``@register_object`` classes."""
    suite: str | None = None
    """Optional suite name used for validation against the active env task."""
    task_id: int | None = None
    """Optional task id used for validation against the active env task."""
    keep_init_states: bool | None = None
    """
    If ``True``, keep LIBERO fixed init states.
    If ``False``, disable them.
    If ``None`` (default): keep for ``replace``, disable for ``add``.
    """
    source_path: Path | None = None
    """Path of the YAML this config was loaded from (set by the loader)."""

    def requires_disable_init_states(self) -> bool:
        if self.keep_init_states is not None:
            return not self.keep_init_states
        return self.mode == "add"

    def validate(self) -> None:
        if self.mode == "replace":
            if not self.replacements:
                raise ValueError("overlay mode 'replace' requires a non-empty 'replacements' map")
            if self.add_objects:
                raise ValueError("overlay mode 'replace' cannot also set 'add_objects'")
            for src, dst in self.replacements.items():
                if not src or not dst:
                    raise ValueError(f"invalid replacement mapping {src!r} -> {dst!r}")
                if src == dst:
                    raise ValueError(f"replacement source and target are identical: {src!r}")
        elif self.mode == "add":
            if not self.add_objects:
                raise ValueError("overlay mode 'add' requires a non-empty 'add_objects' list")
            if self.replacements:
                raise ValueError("overlay mode 'add' cannot also set 'replacements'")
            for spec in self.add_objects:
                if len(spec.ranges) != 4:
                    raise ValueError(f"add_objects ranges must be length 4, got {spec.ranges}")
                xmin, ymin, xmax, ymax = spec.ranges
                if xmin >= xmax or ymin >= ymax:
                    raise ValueError(f"invalid region ranges {spec.ranges}")
        else:
            raise ValueError(f"unknown overlay mode: {self.mode!r}")


def _as_float_quad(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"ranges must be a list of 4 numbers, got {value!r}")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _parse_add_object(raw: dict[str, Any]) -> AddObjectSpec:
    if "category" not in raw:
        raise ValueError("each add_objects entry requires 'category'")
    if "ranges" not in raw:
        raise ValueError(f"add_objects entry for {raw['category']!r} requires 'ranges'")
    yaw = raw.get("yaw_rotation")
    yaw_tuple: tuple[float, float] | None = None
    if yaw is not None:
        if not isinstance(yaw, (list, tuple)) or len(yaw) != 2:
            raise ValueError(f"yaw_rotation must be [min, max], got {yaw!r}")
        yaw_tuple = (float(yaw[0]), float(yaw[1]))
    return AddObjectSpec(
        category=str(raw["category"]),
        ranges=_as_float_quad(raw["ranges"]),
        target=str(raw.get("target", "floor")),
        instance=str(raw["instance"]) if raw.get("instance") is not None else None,
        region_name=str(raw["region_name"]) if raw.get("region_name") is not None else None,
        yaw_rotation=yaw_tuple,
    )


def load_overlay_config(path: str | Path) -> LiberoOverlayConfig:
    """Load and validate an overlay YAML file."""
    overlay_path = Path(path).expanduser().resolve()
    if not overlay_path.is_file():
        raise FileNotFoundError(f"LIBERO overlay not found: {overlay_path}")

    with overlay_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"overlay file is empty: {overlay_path}")
    if not isinstance(data, dict):
        raise ValueError(f"overlay root must be a mapping, got {type(data).__name__}")

    mode = data.get("mode")
    if mode not in ("replace", "add"):
        raise ValueError(f"overlay 'mode' must be 'replace' or 'add', got {mode!r}")

    replacements_raw = data.get("replacements") or {}
    if not isinstance(replacements_raw, dict):
        raise ValueError("'replacements' must be a mapping of category -> category")
    replacements = {str(k): str(v) for k, v in replacements_raw.items()}

    add_raw = data.get("add_objects") or []
    if not isinstance(add_raw, list):
        raise ValueError("'add_objects' must be a list")
    add_objects = tuple(_parse_add_object(item) for item in add_raw)

    keep_init = data.get("keep_init_states", None)
    if keep_init is not None and not isinstance(keep_init, bool):
        raise ValueError("'keep_init_states' must be a boolean when set")

    cfg = LiberoOverlayConfig(
        mode=mode,
        replacements=replacements,
        add_objects=add_objects,
        language=str(data["language"]) if data.get("language") is not None else None,
        objects_module=str(data["objects_module"]) if data.get("objects_module") is not None else None,
        suite=str(data["suite"]) if data.get("suite") is not None else None,
        task_id=int(data["task_id"]) if data.get("task_id") is not None else None,
        keep_init_states=keep_init,
        source_path=overlay_path,
    )
    cfg.validate()
    return cfg
