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

import hashlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .bddl import assert_categories_present, patch_bddl
from .config import LiberoOverlayConfig, load_overlay_config
from .objects import import_objects_module

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppliedOverlay:
    """Result of applying an overlay to a stock BDDL path."""

    bddl_path: Path
    overlay: LiberoOverlayConfig
    disable_init_states: bool
    objects_module_path: Path | None = None


def apply_overlay(
    base_bddl_path: str | Path,
    overlay: str | Path | LiberoOverlayConfig,
    *,
    suite_name: str | None = None,
    task_id: int | None = None,
    output_dir: str | Path | None = None,
) -> AppliedOverlay:
    """Load/register overlay artifacts and write a patched BDDL file.

    The stock ``base_bddl_path`` is never modified.
    """
    base_path = Path(base_bddl_path).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"base BDDL not found: {base_path}")

    cfg = overlay if isinstance(overlay, LiberoOverlayConfig) else load_overlay_config(overlay)

    if cfg.suite is not None and suite_name is not None and cfg.suite != suite_name:
        raise ValueError(f"overlay suite={cfg.suite!r} does not match active suite={suite_name!r}")
    if cfg.task_id is not None and task_id is not None and int(cfg.task_id) != int(task_id):
        raise ValueError(f"overlay task_id={cfg.task_id} does not match active task_id={task_id}")

    objects_module_path: Path | None = None
    if cfg.objects_module:
        relative_to = cfg.source_path.parent if cfg.source_path is not None else None
        objects_module_path = import_objects_module(cfg.objects_module, relative_to=relative_to)

    original = base_path.read_text(encoding="utf-8")
    if cfg.mode == "replace":
        assert_categories_present(original, cfg.replacements.keys())

    patched = patch_bddl(original, cfg)

    if output_dir is None:
        out_root = Path(tempfile.gettempdir()) / "lerobot_libero_overlays"
    else:
        out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha1(patched.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    out_name = f"{base_path.stem}__overlay_{cfg.mode}_{digest}{base_path.suffix}"
    out_path = out_root / out_name
    if not out_path.exists() or out_path.read_text(encoding="utf-8") != patched:
        out_path.write_text(patched, encoding="utf-8")
        logger.info("Wrote overlay BDDL to %s", out_path)

    disable_init = cfg.requires_disable_init_states()
    if disable_init:
        logger.info(
            "Overlay mode=%s disables fixed init_states (set keep_init_states: true to override)",
            cfg.mode,
        )

    return AppliedOverlay(
        bddl_path=out_path,
        overlay=cfg,
        disable_init_states=disable_init,
        objects_module_path=objects_module_path,
    )
