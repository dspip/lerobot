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

import json
import logging
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import torch
from termcolor import colored

import lerobot_env_libero_overlay.env_config  # noqa: F401  # registers libero_overlay
from fault_system import maybe_wrap_env_tree, resolve_fault_log_path
from fault_system.config import FaultInjectionConfig, default_fault_config
from fault_system.eval_recording import attach_eval_recording
from lerobot.configs import parser
from lerobot.configs.default import EvalConfig
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
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


def _prepare_policy_device(cfg: FaultEvalPipelineConfig) -> torch.device:
    """Validate the eval config, select the torch device, and seed the process."""
    logging.info(pformat(asdict(cfg)))
    if cfg.policy is None:
        raise ValueError(
            "Evaluation requires a policy: pass --policy.path=<pretrained_dir> or --policy.type=<name>."
        )
    if cfg.policy.device is None:
        raise ValueError("Policy config has no device set.")
    if cfg.output_dir is None:
        raise ValueError("EvalPipelineConfig.output_dir is not set.")
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    return device


@parser.wrap()
def fault_eval_main(cfg: FaultEvalPipelineConfig) -> None:
    """Build envs, wrap faults and recording, then call stock ``eval_policy_all``."""
    device = _prepare_policy_device(cfg)
    _prepare_fault_log(cfg)
    if getattr(cfg.env, "type", None) in {"libero", "libero_overlay"}:
        from lerobot_env_libero_overlay.gym_env import install_headless_libero_renderer

        install_headless_libero_renderer()
    logging.info(f"Making environment (batch_size={cfg.eval.batch_size}, async={cfg.eval.use_async_envs}).")
    envs = make_env(
        cfg.env,
        n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs,
        trust_remote_code=cfg.trust_remote_code,
    )
    try:
        envs = maybe_wrap_env_tree(envs, cfg.fault)
        if cfg.eval.recording:
            if cfg.output_dir is None:
                raise ValueError("EvalPipelineConfig.output_dir is not set.")
            attach_eval_recording(
                envs,
                Path(cfg.output_dir) / "recordings",
                cfg.env.features,
                features_map=cfg.env.features_map,
                recording_repo_id=cfg.eval.recording_repo_id,
                recording_private=cfg.eval.recording_private,
                success_only=cfg.eval.recording_success_only,
                annotate_failures=cfg.fault.enabled,
            )
        if cfg.policy is None or cfg.output_dir is None:
            raise ValueError("Policy and output_dir are required.")
        logging.info("Making policy.")
        policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
        policy.eval()
        preprocessor_overrides = {
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg=cfg.env, policy_cfg=cfg.policy
        )
        max_episodes_rendered = 0 if cfg.eval.recording else 10
        videos_dir = None if cfg.eval.recording else Path(cfg.output_dir) / "videos"
        amp = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
        with torch.no_grad(), amp:
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=max_episodes_rendered,
                videos_dir=videos_dir,
                return_episode_data=False,
                start_seed=cfg.seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
                recording_dir=None,
                env_features=None,
            )
            logger.info("Overall Aggregated Metrics:")
            logger.info(info["overall"])
            ci_low, ci_high = info["overall"]["pc_success_ci95"]
            logger.info(
                "Success rate %.1f%% (%d/%d episodes, 95%% Wilson interval %.1f%% to %.1f%%)",
                info["overall"]["pc_success"],
                info["overall"]["n_success"],
                info["overall"]["n_episodes"],
                ci_low,
                ci_high,
            )
            for task_group, task_group_info in info.items():
                logger.info(f"\nAggregated Metrics for {task_group}:")
                logger.info(task_group_info)
        with open(Path(cfg.output_dir) / "eval_info.json", "w") as f:
            json.dump(info, f, indent=2)
        logging.info("End of eval")
    finally:
        close_envs(envs)


def main() -> None:
    """CLI entry for ``lerobot-eval-faults``."""
    init_logging()
    register_third_party_plugins()
    fault_eval_main()


if __name__ == "__main__":
    main()
