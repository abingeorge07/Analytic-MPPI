"""Flagship FPL-cost (objective.mode="fpl_cost") MPPI config on hopper, at 8s (400 steps).

Ported from runs/capability/hopper/mppi_fpl_K256_8s/ (produced by
`capability_probe.py --env hopper --sampler mppi --cost fpl --save`, git commit
e74e64a442c6e4282875d8f972ad4c19d7bcd0f2), so this file becomes the canonical, hand-authored
source for that run -- the same role hopper_vanilla_mppi.py plays for the vanilla arm. The
runs/ bundle's video.mp4 / metrics.json / config.json stay put as a fixed provenance snapshot
of exactly what this file reproduces (model files + git commit included); this file doesn't
supersede or require re-running it.

30-seed result (from the bundle's metrics.json): survival 1.00 (Wilson95 lower bound 0.886),
vx 1.665 m/s (83% of target_velocity=2.0) -- clears capability_probe.py's gate (>=50% of
target, >=80% survival lower bound). This is the flagship comparison point for the
vanilla-MPPI config in hopper_vanilla_mppi.py (0.80 survival, 1.07 m/s / 54% of target, same
K=256 / target_velocity=2.0): same env, budget, and difficulty, opposite objective-axis
scalarization (FPL power-mean vs. vanilla quadratic).
"""
from configs.env.hopper import HOPPER

FPL_MPPI = HOPPER.with_(**{
    "objective.mode": "fpl_cost",              # HOPPER's own schema default; set explicitly
                                                # for symmetry with hopper_vanilla_mppi.py's
                                                # override of the same field
    "run.steps": 400,                          # 8.0 s at dt = 0.02 -- matches the ported bundle
    "run.spline_type": "cubic",
    "proposal.num_samples": 256,               # HOPPER leaves this at the schema default (128)
    "proposal.num_knots": 6,
    "proposal.plan_horizon": 0.9,
    "proposal.noise_level": 0.6,
    "update.temperature": 0.1,
    "label": "fpl mppi (hopper, 8s flagship)",
    "notes": (
        "Ported from runs/capability/hopper/mppi_fpl_K256_8s/config.json. 30-seed result: "
        "survival 1.00 (Wilson95 lb 0.886), vx 1.665 m/s (83% of target). See module "
        "docstring for provenance and the vanilla-MPPI comparison point."
    ),
})

CONFIG = FPL_MPPI
