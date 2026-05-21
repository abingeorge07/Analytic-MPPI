# Analytic-MPPI

A lightweight testbed for **Model Predictive Path Integral (MPPI)** control and
new sampling techniques, using **analytic robot models** instead of a physics
simulator. The goal is fast iteration on controller and sampler ideas without
the overhead of MuJoCo / Isaac / Drake.

## Why analytic models?

Physics simulators are great for realism but slow to iterate on when the
research question is about the *controller*, not the dynamics. Closed-form
analytic models give:

- millisecond-scale rollouts → large MPPI batch sizes on a laptop CPU
- deterministic, reproducible behavior
- transparent dynamics (easy to differentiate, perturb, or extend)
- no install/build friction

## Initial models

| Model | State | Control | Use case |
|---|---|---|---|
| Unicycle / Dubins car | `(x, y, θ)` | `(v, ω)` | 2D navigation, obstacle avoidance |
| Linear Inverted Pendulum (LIP) | `(x, ẋ)` per axis | CoP / step location | Humanoid walking, footstep planning |

More models can be added under [analytic_mppi/dynamics/](analytic_mppi/dynamics/)
by implementing the common dynamics interface.

## Stack

- Python 3
- NumPy (core numerics)
- Matplotlib (plots + animation, headless-friendly)

No GPU, no JIT, no simulator dependency.

## Repo layout

```
analytic_mppi/
  dynamics/       # analytic robot models (unicycle, LIP, ...)
  controllers/    # MPPI core + sampling strategies
  costs/          # reusable cost terms (quadratic, obstacle, terminal, ...)
  viz/            # matplotlib plotting / animation helpers
examples/         # runnable scripts wiring a model + controller + cost
tests/            # unit tests for dynamics and controller pieces
```

## Getting started

```bash
pip install -r requirements.txt
python examples/<example>.py
```

(Examples will be added as components land.)

## Status

Scaffolding only. Dynamics, MPPI core, and samplers are not yet implemented.
