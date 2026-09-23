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

"""Human task strings for LIBERO datagen logging."""

from __future__ import annotations

from typing import Any

DEFAULT_LIBERO_OBJECT_TASK_DESCRIPTION = "pick up the alphabet soup and place it in the basket"


def read_libero_task_description(env: Any) -> str:
    """Return the LIBERO ``task_description`` string used by the policy."""
    raw = env.call("task_description")
    if isinstance(raw, (list, tuple)):
        if not raw:
            return DEFAULT_LIBERO_OBJECT_TASK_DESCRIPTION
        return str(raw[0])
    if raw is None:
        return DEFAULT_LIBERO_OBJECT_TASK_DESCRIPTION
    return str(raw)
