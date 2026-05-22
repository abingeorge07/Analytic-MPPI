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
  dynamics/       # MujocoBackend + MJCF model files (unicycle.xml, ...)
  controllers/    # MPPI core (vanilla today; variants TODO)
  costs/          # reusable cost terms (goal-reach today)
  viz/            # matplotlib trajectory + control plots
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

## Status

Vanilla MPPI, MuJoCo backend, goal-reach cost, planar base example, and
smoke tests are in. Next: MPPI variants (new cost representations) and
non-trivial robots (humanoid, quadruped).
