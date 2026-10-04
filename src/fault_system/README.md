# Architecture

Three layers. The model never steps the simulator. The simulator never imports a model. The fault system sits between them as a Gym wrapper.

```text
+-------------------------------------------------------+
|                   1. VLA / Model                      |
|       (SmolVLA / R3 / any future model)               |
+-------------------------------------------------------+
|  IN:  raw Gym observation + task string               |
|       pixels: uint8 HWC images                        |
|       robot_state: float32 (8,)                       |
|       task: str                                       |
|  OUT: env action float32 (7,)                         |
|       dx, dy, dz, dax, day, daz, gripper              |
|  The model does not receive an env.                  |
+-------------------------------------------------------+
                           |
                           |  ActionSource.act(observation, task=...)
                           v
+-------------------------------------------------------+
|               2. Fault System Wrapper                 |
|  action / observation / slip / bump / mid-air drop    |
+-------------------------------------------------------+
|  IN from the model:  action float32 (7,)              |
|                      or a batch (num_envs, 7)         |
|  OUT to the simulator: that action, possibly changed  |
|  IN from the simulator: obs, reward, terminated,      |
|                         truncated, info               |
|  OUT to the model: the same Gym result. Observation   |
|                    faults may change the images.      |
|  info also carries privileged labels. They are not    |
|  model inputs: injection_active, failure_type,        |
|  phase, is_failure.                                   |
+-------------------------------------------------------+
                           |
                           |  env.step(action)
                           v
+-------------------------------------------------------+
|                   3. Simulator                        |
|                     (LIBERO)                          |
+-------------------------------------------------------+
|  IN:  action float32 (7,)                             |
|  OUT: observation dict (pixels + robot_state),        |
|       reward, terminated, truncated, info             |
|  Physics faults (slip, bump, drop) call               |
|  fault_system.sim.libero. That module is LIBERO       |
|  only. It is not a generic simulator interface.       |
+-------------------------------------------------------+
```

## Who calls whom

Datagen owns a short loop: `action = source.act(observation, task=...)`, then `env.step(action)` on an env that is already wrapped. SmolVLA's preprocess and `select_action` live in `fault_system.models.smolvla.SmolVLAActionSource`. R3 is another class with the same two methods, `reset` and `act`.

Eval does not use `ActionSource`. `lerobot-eval-faults` builds a LeRobot policy and calls stock `eval_policy_all`. That loop still steps this wrapper, so the action and observation that cross the fault boundary are the same shapes as above.

## What stays out

- Do not pass the env, a policy object, or a CUDA batch into `act`. A later remote model is another `ActionSource` that sends the raw observation and returns the `(7,)` vector.
- Do not put `injection_active` into the observation. The wrapper already writes it on `info`.
- Do not add a second physics API until a second simulator exists.
