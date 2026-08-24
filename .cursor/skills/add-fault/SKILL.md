---
name: add-fault
description: >-
  Use when adding a new LeRobot fault type, injector, failure mode, or extending
  evaluation-time fault injection in the in-tree `src/lerobot/faults/` package.
---

# Add a new fault type

Add one fault that intercepts actions, observations, or sim state at the Gymnasium
env boundary. Faults live **in-tree** under `src/lerobot/faults/` — do not fork
LeRobot and do not add a second eval hook (`maybe_wrap_env_tree` in `eval_main`
is the only integration point).

## Checklist

```text
Fault name: _______________
Kind: action | observation | sim | recovery
Trigger model: always-on | trigger_step+duration | t_min/t_max window
Log event name: _______________
CLI flags: _______________  (draccus nested under --fault.*)
```

## Kind → code path (in-tree)

| Kind | Module location | Factory set | Wrapper |
|------|-----------------|-------------|---------|
| action | `src/lerobot/faults/action/<name>.py` | `_ACTION_TYPES` + `make_action_fault_injector` | `FaultEnvWrapper` |
| observation | `src/lerobot/faults/observation/<name>.py` | `_OBS_TYPES` + `make_obs_fault_injector` | `FaultEnvWrapper` |
| sim-inject | `src/lerobot/faults/sim/<name>.py` | `_SIM_INJECT_TYPES` + `make_sim_inject_fault` | `SimFaultEnvWrapper` |
| recovery | `src/lerobot/faults/recovery/<name>.py` | `_RECOVERY_TYPES` + `make_midair_drop_fault` | `DropRecoveryEnvWrapper` |

Shared helpers:

- LIBERO / MuJoCo: `src/lerobot/faults/sim/libero.py`
- Recovery planner / FPS: `src/lerobot/faults/recovery/planner.py`, `recovery/fps.py`
- Config: `src/lerobot/faults/config.py`
- Factory: `src/lerobot/faults/factory.py`
- Wrappers: `src/lerobot/faults/wrappers.py`
- Tests: `tests/faults/test_<name>_*.py`

## Steps

1. **Design** — start/stop, proposed vs executed, proof (`fault_events.jsonl`, video, diag PNGs).
2. **Config** — `src/lerobot/faults/config.py`: add to `_SUPPORTED_TYPES`, fields, `validate()`.
   Keep `enabled=False` default on `FaultInjectionConfig`.
3. **Injector** — per-env state, `reset()`, `notify_dones()`, never mutate inputs, JSONL via
   `FaultEventLogger`.
4. **Factory** — register type and constructor in `src/lerobot/faults/factory.py`.
5. **Exports** — `src/lerobot/faults/__init__.py` (and subpackage `__init__.py` if needed).
6. **CLI** — flags are automatic via draccus on `EvalPipelineConfig.fault`; use
   `lerobot-eval --fault.<field>=<value>`. **Do not** add a sidecar CLI.
7. **Wrapper dispatch** — extend `maybe_wrap_env` in `wrappers.py` (sim/recovery need their
   own wrapper class). **Do not** patch `lerobot_eval.py` again.
8. **Tests** — `tests/faults/test_<name>_fault.py`, mock sim where possible; no GPU/LIBERO
   required for unit tests.
9. **Docs** — `docs/source/fault_injection.mdx`; example JSONL if schema is new.
10. **Copyright** — new source files: `Copyright 2026 Gangelia. All rights reserved.` + Apache-2.0.
11. **Verify** — `uv run pytest tests/faults -q`

## CLI example

```bash
uv run lerobot-eval \
  --policy.path=lerobot/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="[0]" \
  --eval.n_episodes=1 \
  --env.max_parallel_tasks=1 \
  --fault.enabled=true \
  --fault.type=action_hold \
  --fault.trigger_step=20 \
  --fault.duration=8 \
  --output_dir=outputs/eval/fault_hold
```

## Do not

- Patch LeRobot for the fault itself (beyond the existing `maybe_wrap_env_tree` call).
- Add a second eval hook or monkey-patch `make_env`.
- Combine unrelated fault types in one class.
- Skip JSONL proof for action/sim corruption.
- Use the old `lerobot_faults` sidecar package paths.

## References

- Action: `src/lerobot/faults/action/hold.py`, `delay.py`, `jitter.py`
- Obs: `src/lerobot/faults/observation/sensor_dropout.py`
- Sim-inject: `src/lerobot/faults/sim/object_slip.py`, `eef_bump.py`
- Recovery: `src/lerobot/faults/recovery/midair_drop.py`
- Integration: `src/lerobot/scripts/lerobot_eval.py` (`maybe_wrap_env_tree` only)
- Drop pipeline demos: `examples/faults/run_drop_recovery_demo.py`,
  `examples/faults/run_full_drop_recovery_pipeline.py`
