# Analytic-MPPI (PMR)

A quick-iteration testbed for **Model Predictive Path Integral (MPPI)** control
and new cost / sampling techniques. Currently codenamed **PMR** while the
controller research happens here before the full system is built.

The framework targets a morphology-agnostic controller (humanoid, quadruped),
where MPC's per-robot tuning burden makes it impractical. MPPI works
out-of-the-box on arbitrary robots, but converges slowly — the research goal
is to fix that with better cost representations.

## Backend: MuJoCo in the loop

PMR runs MuJoCo as its dynamics backend, with sample batches rolled out in
parallel via [`mujoco.rollout.rollout`](https://mujoco.readthedocs.io/) — a
multithreaded CPU primitive. No JAX, no `mjx`, no GPU. The point is to
validate MPPI variants against real(-ish) physics with minimum infrastructure
before moving to a full JAX/GPU stack.

The dynamics interface is small (`Backend` protocol in
[analytic_mppi/controllers/mppi.py](analytic_mppi/controllers/mppi.py)), so an
analytic backend (closed-form LIP, RBD) can be added later behind the same
seam.

## Stack

- Python 3
- NumPy + Matplotlib
- `mujoco` (>= 3.2) Python bindings

## Repo layout

```
analytic_mppi/
  dynamics/       # MujocoBackend (returns states + sensordata from mujoco.rollout)
  tasks/          # Task abstraction + ports (pendulum, walker, cube, g1_standup)
  controllers/    # legacy MPPI + sampling-MPC algorithms (knot-spline, Task-based)
    spline.py
    sampling_base.py
    mppi_v2.py mppi_cma.py cem.py dial.py predictive_sampling.py
    mppi.py                                # legacy full-horizon callable-cost MPPI
  costs/          # legacy cost terms (goal-reach, inv-pen) — used by legacy MPPI
  envs/           # one folder per task with its MJCF + (optional) run script
    pendulum/  walker/  cube/  g1/  unicycle/
  viz/            # matplotlib + viewer/renderer helpers
  run.py          # unified CLI: --task <name> --algo <name> [--fpl ...]
examples/         # runnable scripts wiring backend + cost + controller
tests/            # smoke + shape tests
```

## Getting started

```bash
pip install -r requirements.txt

# Run the unicycle smoke example: drives a planar base from (0,0) to (2,2),
# saves runs/unicycle_goal.png
python examples/unicycle_goal.py

# Unit / smoke tests
pytest -v
```

## First robot: a planar base ("unicycle")

[analytic_mppi/dynamics/unicycle.xml](analytic_mppi/dynamics/unicycle.xml)
is a 3-DOF planar base (`x`, `y`, `theta`) with three world-frame velocity
actuators. It's a holonomic point mass with orientation — *not* a true
nonholonomic unicycle. It exists only as a substrate to validate the MPPI
+ parallel-rollout machinery on a trivial problem before scaling to humanoid
/ quadruped MJCF models.

## Sampling-MPC framework (knot-spline, Task-based)

A second, more general layer sits on top of the MuJoCo backend for the
research work. Pick a **task**, an **algorithm**, and optionally turn on the
**FPL** (Fulfillment Path Length) cost path — all from a single CLI.

### Tasks (`analytic_mppi.tasks.TASKS`)
| name          | DOF (nq/nv/nu) | source MJCF                    | notes |
|---------------|----------------|--------------------------------|-------|
| `pendulum`    | 1 / 1 / 1      | `envs/pendulum/model.xml`      | swing-up to theta=pi |
| `walker`      | 9 / 9 / 6      | `envs/walker/scene.xml`        | planar biped, target forward velocity |
| `cube`        | 23 / 22 / 16   | `envs/cube/scene.xml`          | LEAP hand cube reorientation |
| `g1_standup`  | 36 / 35 / 29   | `envs/g1/scene.xml`            | Unitree G1 standup |

Each task subclasses `analytic_mppi.tasks.base.Task` and exposes:
- `running_cost_terms(qpos, qvel, sensordata, u)` and `terminal_cost_terms(...)` — the **normal** penalty cost,
- `running_cost_terms_f(...)` and `terminal_cost_terms_f(...)` — the **FPL** fulfillment cost in `[0, 1]`.

All cost methods are **batched numpy**: any leading shape (typically `(K, H)`) broadcasts naturally.

### Algorithms (`analytic_mppi.controllers.SAMPLING_CONTROLLERS`)
| name                  | class                | hyperparams                                              |
|-----------------------|----------------------|----------------------------------------------------------|
| `mppi`                | `MPPIv2`             | `noise_level`, `temperature`                             |
| `mppi_cma`            | `MppiCma`            | `initial_noise_level`, `temperature`, `minimum_noise_level`, `covariance_adaptation_rate` |
| `cem`                 | `CEM`                | `num_elites`, `sigma_start`, `sigma_min`, `explore_fraction`         |
| `dial`                | `DIAL`               | `noise_level`, `temperature`, `beta_opt_iter`, `beta_horizon`        |
| `predictive_sampling` | `PredictiveSampling` | `noise_level`                                            |

All algorithms share: `num_samples`, `num_knots`, `plan_horizon` (seconds),
`spline_type` (`zero` | `linear`), `iterations`, `seed`, and the FPL flags
below.

### FPL toggle
The FPL path computes fulfillment terms in `[0, 1]` per timestep, then
aggregates with a power-mean (exponent `p`) over terms and a discounted sum
over time (`gamma`). Two flavours, mutually exclusive:
- `--fpl`              — per-step running fulfillment is power-meaned across terms first, then time-discounted.
- `--fpl-discounted`   — per-term time-discounted first, then power-meaned across terms.
Both use best-sample selection (`argmax(reward)`) in the controller update.

### Run from the CLI
```bash
# pendulum swing-up with each algorithm
python -m analytic_mppi.run --task pendulum --algo mppi \
    --steps 200 --num-samples 512 --num-knots 6 --plan-horizon-sec 1.0 \
    --noise-level 0.5 --temperature 1.0

python -m analytic_mppi.run --task pendulum --algo cem \
    --steps 200 --num-samples 512 --num-knots 6 --plan-horizon-sec 1.0 \
    --sigma-start 1.0 --sigma-min 0.05 --num-elites 32

# walker live viewer
python -m analytic_mppi.run --task walker --algo dial --live \
    --steps 300 --num-samples 256 --num-knots 4 --plan-horizon-sec 0.5 \
    --noise-level 0.3 --temperature 1.0 --beta-opt-iter 3 --beta-horizon 3

# G1 standup with FPL
python -m analytic_mppi.run --task g1_standup --algo mppi --live \
    --steps 200 --num-samples 128 --num-knots 4 --plan-horizon-sec 0.5 \
    --noise-level 0.3 --temperature 1.0 --fpl --fpl-p 0.1 --fpl-gamma 0.99
```

### Use from Python
```python
from analytic_mppi.tasks import make_task
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.controllers import MPPIv2

task    = make_task("pendulum")
backend = MujocoBackend(task.model_path)
ctrl    = MPPIv2(task, backend, num_samples=256, noise_level=0.5,
                 temperature=1.0, num_knots=4, plan_horizon=1.0,
                 use_fpl_cost=False)

state = backend.get_state()
for _ in range(200):
    u = ctrl.act(state)
    state = backend.step(u)
```

### Adding a new task
1. Drop the MJCF under `analytic_mppi/envs/<name>/`.
2. Add `analytic_mppi/tasks/<name>.py` subclassing `Task` with `running_cost_terms` / `terminal_cost_terms` (normal) and `running_cost_terms_f` / `terminal_cost_terms_f` (FPL).
3. Register it in `analytic_mppi/tasks/__init__.py`'s `TASKS` dict.
The CLI picks up the new task automatically via `--task <name>`.

### Adding a new algorithm
1. Add `analytic_mppi/controllers/<name>.py` subclassing
   `SamplingController` from `controllers/sampling_base.py`. Implement
   `sample_knots()` and `update_mean(traj)`.
2. Register it in `controllers/__init__.py`'s `SAMPLING_CONTROLLERS`.
3. If the algorithm has hyperparams that need CLI exposure, add them in
   `analytic_mppi/run.py`'s `build_parser()` and the `_ALGO_PARAMS` table.

### Legacy path
The original full-horizon callable-cost MPPI lives in
`controllers/mppi.py`. The unicycle env still uses it (its goal isn't part
of the MuJoCo model, so it doesn't map cleanly to `Task`).
`python examples/unicycle_goal.py` and `python -m analytic_mppi.envs.unicycle.run`
still work as before.

## Status

Vanilla legacy MPPI, the new sampling-MPC framework (MPPI / MPPI-CMA / CEM /
DIAL / Predictive Sampling), 4 tasks (pendulum, walker, cube, g1_standup),
normal + FPL cost paths, and a unified CLI are in. Smoke tests cover legacy
+ all algorithms + both FPL modes. Next: new cost representations and
domain-randomization hooks.
