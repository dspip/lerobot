# Recovery request API (Track 2)

## Summary

Added `MidAirDropFault.request_recovery` so callers can start the existing
`SimpleIKRecoveryPlanner` from current sim poses without applying a physics drop
impulse. Planner setup is shared with `_trigger_drop` via `_start_recovery_planner`
so the two paths stay aligned.

No new fault type, no `lerobot_eval.py` changes, no skill updates.

**Design fork:** Not required — Aviya does not need to approve a fork for this slice.

## API

```python
action = fault.request_recovery(env, env_idx, reason="head") -> np.ndarray | None
```

| Case | Behavior |
|------|----------|
| Fault disabled (`config.enabled=False`) | Returns `None`; no state change |
| `env_idx` out of range | Raises `ValueError` |
| Env not selected or episode `finished` | Returns `None` |
| `recovery_active` already true | Returns next recovery action; planner unchanged |
| Otherwise | Sets `recovery_active`, builds planner from current EEF + object pose and destination (config or basket), logs JSONL, returns first recovery action |

**Not set on request path:** `triggered`, `drop_injection_step`, impulse telemetry,
`last_impulse_lin` / `last_impulse_ang`.

**Logging:** `status="recovery_requested"`, `drop_trigger_reason=<reason>` (e.g. `"head"`),
distinct from drop-auto reasons `xy_band` / `delay_elapsed` (`status="triggered"`).

**Follow-up steps:** `on_step` / `apply` continue to replay planner actions while
`recovery_active` is true, same as after a physical drop.

## Files changed

| File | Change |
|------|--------|
| `src/lerobot/faults/recovery/midair_drop.py` | `_start_recovery_planner`, `request_recovery`; `_trigger_drop` delegates planner setup |
| `tests/faults/test_midair_drop_request_recovery.py` | New unit tests (mocks only) |
| `docs/source/fault_injection.mdx` | Short paragraph on request vs impulse path |

## Tests

```bash
uv run pytest tests/faults/test_midair_drop_fault.py tests/faults/test_midair_drop_request_recovery.py -q --maxfail=10
# 24 passed in ~0.15s

uv run pytest tests/faults -q --maxfail=10
# 270 passed in ~1.4s
```

New tests cover: no `midair_drop` impulse call; `on_step` returns planner action;
duplicate request does not reset planner; JSONL reason; disabled fault noop.

Existing drop-trigger tests unchanged and passing.
