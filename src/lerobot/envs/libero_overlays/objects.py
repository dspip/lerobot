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

import importlib.util
import sys
from pathlib import Path


def import_objects_module(module_path: str | Path, *, relative_to: Path | None = None) -> Path:
    """Import a user module that registers custom LIBERO objects via ``@register_object``.

    Returns the resolved path that was imported.
    """
    path = Path(module_path).expanduser()
    if not path.is_absolute():
        if relative_to is None:
            raise ValueError(
                f"relative objects_module path {module_path!r} requires an overlay YAML "
                "directory to resolve against"
            )
        path = (relative_to / path).resolve()
    else:
        path = path.resolve()

    if path.is_dir():
        init_py = path / "__init__.py"
        if not init_py.is_file():
            raise FileNotFoundError(
                f"objects_module directory {path} must contain __init__.py (or point to a .py file)"
            )
        path = init_py
    elif not path.is_file():
        raise FileNotFoundError(f"objects_module not found: {path}")

    module_name = f"lerobot_libero_overlay_objects_{abs(hash(path))}"
    if module_name in sys.modules:
        return path

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load objects_module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return path
