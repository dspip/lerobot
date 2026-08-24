---
name: remove-fault
description: >-
  Use when removing, disabling, deleting, or retiring a fault type from the
  in-tree `src/lerobot/faults/` package so leftover factory, wrapper, test, or
  doc wiring does not remain.
---

# Remove a fault type

Delete the injector **and** every registration. A leftover `_ACTION_TYPES` entry or
`--fault.type` default will still construct the fault.

## Checklist (all required)

1. Injector module under the correct subpackage:
   - `src/lerobot/faults/action/<name>.py`
   - `src/lerobot/faults/observation/<name>.py`
   - `src/lerobot/faults/sim/<name>.py`
   - `src/lerobot/faults/recovery/<name>.py`
2. Subpackage and package `__init__.py` exports (`src/lerobot/faults/__init__.py`).
3. `src/lerobot/faults/factory.py`: type frozenset, `make_*` branch, type aliases.
4. `src/lerobot/faults/config.py`: `_SUPPORTED_TYPES`, type-specific fields, `validate()` branch.
5. `src/lerobot/faults/wrappers.py`: `maybe_wrap_env` dispatch; delete wrapper class only if
   no remaining types use it (`DropRecoveryEnvWrapper` / `SimFaultEnvWrapper`).
6. Tests: `tests/faults/test_<name>_*.py` and mentions in `test_wrappers.py` or composition tests.
7. Docs: `docs/source/fault_injection.mdx`.
8. Skills: drop the type from add-fault / test-fault tables in `.cursor/skills/`.
9. Smoke: `scripts/run_fault_smoke.sh` mode (if present).
10. Examples: `examples/faults/` scripts that reference the type.

**Do not** remove the single `maybe_wrap_env_tree` call in `lerobot_eval.py` unless removing
the entire fault system.

## Disable without deleting

Set `enabled=False` (default on `FaultInjectionConfig`). Do not leave `enabled=True` in
defaults, smoke, or example configs.

## Verify

```bash
uv run pytest tests/faults -q
rg -n '<fault_type_string>' src/lerobot/faults tests/faults docs .cursor scripts
```

The type string (for example `midair_drop`) must not remain in factory frozensets or
`_SUPPORTED_TYPES`.
