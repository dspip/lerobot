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

"""Stable recipe identity for shard merge validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

__all__ = ["recipe_content_hash_from_json_path", "recipe_content_hash_from_raw"]


def _canonical_recipe_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(raw)
    recording = payload.get("recording")
    if isinstance(recording, dict):
        recording.pop("output_dir", None)
    payload.pop("logical_range", None)
    return payload


def recipe_content_hash_from_raw(raw: dict[str, Any]) -> str:
    """SHA-256 of canonical recipe JSON (excludes output_dir and logical_range)."""
    canonical = json.dumps(_canonical_recipe_payload(raw), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def recipe_content_hash_from_json_path(path: Path) -> str:
    """Hash a recipe file on disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"recipe JSON must be an object: {path}")
    return recipe_content_hash_from_raw(raw)
