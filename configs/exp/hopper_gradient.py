"""Gradient MPC (first-order, MJX autodiff) on hopper -- the phase-3 smoke configs.

Two arms sharing the hopper env: the FPL objective the env ships with, and the plain
quadratic. Both descend the exact objective the corresponding MPPI config ranks
rollouts by (controllers/jax_scoring.py is parity-pinned to the numpy scorers).

Settings notes:
  * noise_level / temperature are reset to their schema defaults: gradient_mpc accepts
    neither, and resolve() errors on any non-default value it cannot pass (that is the
    config system doing its job, not an inconvenience to work around).
  * proposal.num_samples=1 is REQUIRED explicitly (resolve enforces it): one plan, no
    sampling cloud -- the config records what actually runs.
  * run.iterations is the Adam step count per MPC step (same "optimizer iterations"
    meaning it has for every controller).
  * spline "linear": a continuous control gives the gradient a smoother loss landscape
    than ZOH steps; ZOH works too, this is just the kinder default for descent.

Usage:
  python3 -m analytic_mppi.run --config configs/exp/hopper_gradient.py            # FPL arm
  python3 -m analytic_mppi.run --config configs/exp/hopper_gradient.py --index 1  # quadratic arm
"""
from configs.env.hopper import HOPPER

_GRADIENT_BASE = HOPPER.with_(**{
    "run.backend": "mjx",
    "proposal.kind": "gradient",
    "update.rule": "descent",
    "proposal.num_samples": 1,
    "proposal.noise_level": 0.3,     # schema default; documents "no sampling noise here"
    "update.temperature": 0.2,       # schema default; gradient_mpc has no softmax
    "run.iterations": 30,
    "run.spline_type": "linear",
    "update.extra": {"learning_rate": 0.05},
})

GRADIENT_FPL = _GRADIENT_BASE.with_(**{
    "label": "gradient mpc (hopper, fpl_cost)",
})

GRADIENT_VANILLA = _GRADIENT_BASE.with_(**{
    "objective.mode": "normal",
    # The vanilla-cost weights found by the step-2 search (hopper_vanilla_mppi.py) --
    # the same objective its MPPI baseline optimizes, for a like-for-like comparison.
    "task.kwargs": {"target_velocity": 2.0, "height_weight": 10.0,
                    "velocity_weight": 0.15},
    "label": "gradient mpc (hopper, normal cost)",
})


def configs():
    return [GRADIENT_FPL, GRADIENT_VANILLA]
