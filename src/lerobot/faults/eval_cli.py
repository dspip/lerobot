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

"""Fault-aware eval entry point.

Stock ``lerobot-eval`` stays on ``EvalPipelineConfig``. This command adds
``--fault.*`` and ``--eval.recording_success_only`` without putting those fields
on the upstream config classes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lerobot.envs.libero_overlays.env_config  # noqa: F401  # registers libero_overlay
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import make_env
from lerobot.faults import maybe_wrap_env_tree, resolve_fault_log_path
from lerobot.faults.config import FaultInjectionConfig, default_fault_config
from lerobot.scripts.lerobot_eval import prepare_eval, run_eval_on_envs
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class FaultEvalConfig(EvalConfig):
    """Eval settings plus success-only recording for nominal baselines."""

    # If true, discard episode buffers that never reported success. Requires recording=true.
    # Failed attempts are not written; n_episodes is still the number of attempts, not kept successes.
    recording_success_only: bool = False

    def __post_init__(self) -> None:
        """Reject success-only recording when recording is off, then apply EvalConfig defaults."""
        super().__post_init__()
        if self.recording_success_only and not self.recording:
            raise ValueError("eval.recording_success_only requires eval.recording=true.")


@dataclass
class FaultEvalPipelineConfig(EvalPipelineConfig):
    """``EvalPipelineConfig`` plus a fault-injection block."""

    eval: FaultEvalConfig = field(default_factory=FaultEvalConfig)
    fault: FaultInjectionConfig = field(default_factory=default_fault_config)


def _prepare_fault_log(cfg: FaultEvalPipelineConfig) -> None:
    fault_cfg = cfg.fault
    if not fault_cfg.enabled:
        return
    if cfg.env.max_parallel_tasks > 1:
        raise ValueError(
            "Fault injection currently requires --env.max_parallel_tasks=1 so "
            "fault event logs stay consistent across tasks."
        )
    if cfg.output_dir is None:
        raise ValueError("EvalPipelineConfig.output_dir is not set.")
    fault_cfg.log_path = resolve_fault_log_path(fault_cfg.log_path, cfg.output_dir)
    fault_cfg.log_path.parent.mkdir(parents=True, exist_ok=True)
    fault_cfg.log_path.write_text("", encoding="utf-8")


@parser.wrap()
def fault_eval_main(cfg: FaultEvalPipelineConfig) -> None:
    """Build envs, wrap them when faults are enabled, and run eval."""
    device = prepare_eval(cfg)
    _prepare_fault_log(cfg)
    if getattr(cfg.env, "type", None) in {"libero", "libero_overlay"}:
        from lerobot.envs.libero_overlays.gym_env import install_headless_libero_renderer

        install_headless_libero_renderer()
    logging.info(f"Making environment (batch_size={cfg.eval.batch_size}, async={cfg.eval.use_async_envs}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    envs = maybe_wrap_env_tree(envs, cfg.fault)
    run_eval_on_envs(
        cfg,
        envs,
        device,
        recording_success_only=cfg.eval.recording_success_only,
        annotate_failures=cfg.fault.enabled,
    )


def main() -> None:
    """CLI entry for ``lerobot-eval-faults``."""
    init_logging()
    register_third_party_plugins()
    fault_eval_main()


if __name__ == "__main__":
    main()
