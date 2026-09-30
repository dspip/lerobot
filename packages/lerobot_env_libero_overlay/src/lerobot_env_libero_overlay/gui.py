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

"""Turn off the on-screen OpenCV renderer that LIBERO forces on.

``OffScreenRenderEnv`` hardcodes ``has_renderer=True``, so robosuite calls
``cv2.destroyAllWindows()`` on every hard reset. Headless ``cv2`` builds raise
there, and LIBERO's reset retry loop swallows the exception and spins forever.
Observations come from the offscreen renderer, which stays on.

Must run before the first reset. The viewer is dropped rather than closed,
because closing it is the call that raises.
"""

from __future__ import annotations

from typing import Any


def disable_gui_renderer(env: Any) -> None:
    """Disable the robosuite on-screen viewer on a freshly built LIBERO env."""
    inner = getattr(env, "env", None)
    if inner is None:
        return
    inner.has_renderer = False
    inner.viewer = None
