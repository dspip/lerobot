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

import re
from typing import Iterable

from .config import AddObjectSpec, LiberoOverlayConfig

_LANGUAGE_RE = re.compile(r"(\(:language\s+)(.*?)(\s*\))", re.DOTALL)


def _replace_category_tokens(text: str, mapping: dict[str, str]) -> str:
    """Replace BDDL category / instance tokens using longest-key-first.

    Instance names follow ``{category}_{index}``. A naive ``\\b`` replace misses
    those because ``_`` is a word character, so we rewrite ``src_<digits>`` first,
    then standalone category tokens.
    """
    out = text
    for src in sorted(mapping, key=len, reverse=True):
        dst = mapping[src]
        # tomato_sauce_1 -> red_cube_1
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}_(?=\d)", f"{dst}_", out)
        # tomato_sauce (type / language word) -> red_cube
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])", dst, out)
    return out


def _format_region(spec: AddObjectSpec) -> str:
    xmin, ymin, xmax, ymax = spec.ranges
    region_name = spec.resolved_region_name()
    lines = [
        f"      ({region_name}",
        f"          (:target {spec.target})",
        "          (:ranges (",
        f"              ({xmin} {ymin} {xmax} {ymax})",
        "            )",
        "          )",
    ]
    if spec.yaw_rotation is not None:
        y0, y1 = spec.yaw_rotation
        lines.extend(
            [
                "          (:yaw_rotation (",
                f"              ({y0} {y1})",
                "            )",
                "          )",
            ]
        )
    lines.append("      )")
    return "\n".join(lines)


def _find_top_level_block(bddl: str, block_name: str) -> tuple[int, int, int]:
    """Return ``(header_end, body_start, close_paren_index)`` for ``(:block_name ...)``.

    Uses parenthesis depth so nested ``(:ranges (...))`` inside ``:regions`` does
    not terminate the outer block early.
    """
    marker = f"(:{block_name}"
    start = bddl.find(marker)
    if start < 0:
        raise ValueError(f"BDDL is missing a :{block_name} block")

    # Skip the opening '(' of the block, then scan until depth returns to 0.
    depth = 0
    header_end = start + len(marker)
    body_start = header_end
    for i in range(start, len(bddl)):
        ch = bddl[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return header_end, body_start, i
    raise ValueError(f"BDDL :{block_name} block is unclosed")


def _append_inside_named_block(bddl: str, block_name: str, insertion: str) -> str:
    _header_end, body_start, close_idx = _find_top_level_block(bddl, block_name)
    body = bddl[body_start:close_idx].rstrip("\n")
    new_body = f"{body}\n{insertion}\n  "
    return bddl[:body_start] + new_body + bddl[close_idx:]


def _set_language(bddl: str, language: str) -> str:
    match = _LANGUAGE_RE.search(bddl)
    if match is None:
        raise ValueError("BDDL is missing a (:language ...) clause")
    return bddl[: match.start()] + match.group(1) + language + match.group(3) + bddl[match.end() :]


def apply_replacements(bddl: str, replacements: dict[str, str], language: str | None = None) -> str:
    """Rewrite categories/instances across the whole BDDL document."""
    patched = _replace_category_tokens(bddl, replacements)
    if language is not None:
        patched = _set_language(patched, language)
    return patched


def apply_add_objects(
    bddl: str,
    add_objects: Iterable[AddObjectSpec],
    language: str | None = None,
) -> str:
    """Inject objects, regions, and On(...) init predicates into a BDDL document."""
    patched = bddl
    specs = list(add_objects)
    if not specs:
        raise ValueError("add_objects must be non-empty")

    object_lines = []
    region_blocks = []
    init_lines = []
    for spec in specs:
        instance = spec.resolved_instance()
        region = spec.resolved_region_name()
        object_lines.append(f"    {instance} - {spec.category}")
        region_blocks.append(_format_region(spec))
        # LIBERO names composed regions as ``{target}_{region_name}``.
        init_lines.append(f"    (On {instance} {spec.target}_{region})")

    patched = _append_inside_named_block(patched, "objects", "\n".join(object_lines))
    patched = _append_inside_named_block(patched, "regions", "\n".join(region_blocks))
    patched = _append_inside_named_block(patched, "init", "\n".join(init_lines))

    if language is not None:
        patched = _set_language(patched, language)
    return patched


def patch_bddl(bddl: str, overlay: LiberoOverlayConfig) -> str:
    """Apply an overlay config to BDDL text and return the patched document."""
    overlay.validate()
    if overlay.mode == "replace":
        return apply_replacements(bddl, overlay.replacements, overlay.language)
    return apply_add_objects(bddl, overlay.add_objects, overlay.language)


def assert_categories_present(bddl: str, categories: Iterable[str]) -> None:
    """Raise if any category token is missing from the BDDL (pre-replace check)."""
    missing = []
    for category in categories:
        pattern = (
            rf"(?<![A-Za-z0-9_]){re.escape(category)}(?![A-Za-z0-9_])|"
            rf"(?<![A-Za-z0-9_]){re.escape(category)}_\d"
        )
        if not re.search(pattern, bddl):
            missing.append(category)
    if missing:
        raise ValueError(f"BDDL does not contain categories to replace: {missing}")


def assert_balanced_parens(bddl: str) -> None:
    """Raise if parentheses are unbalanced."""
    depth = 0
    for ch in bddl:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("BDDL has a closing ')' before any matching '('")
    if depth != 0:
        raise ValueError(f"BDDL has unbalanced parentheses (depth={depth})")
