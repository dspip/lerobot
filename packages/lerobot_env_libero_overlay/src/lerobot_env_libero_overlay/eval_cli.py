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

"""Eval entry that registers ``libero_overlay`` before stock ``lerobot-eval`` parses args.

Stock ``lerobot-eval`` never imports this package, so ``--env.type=libero_overlay``
is unknown there. This command registers the subclass, then calls the same main.
"""

from __future__ import annotations

import lerobot_env_libero_overlay.env_config  # noqa: F401  # registers the env type


def main() -> None:
    """CLI entry for ``lerobot-eval-overlay``."""
    from lerobot.scripts.lerobot_eval import main as eval_main

    eval_main()


if __name__ == "__main__":
    main()
