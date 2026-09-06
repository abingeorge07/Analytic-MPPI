"""Step 2 of docs/vanilla_mppi_hyperparams.md: a working vanilla-cost (objective.mode=
"normal") MPPI config on hopper, at the step-3 protocol length (10s / 500 steps).

Search log (see docs/vanilla_mppi_hyperparams.md for the full hyperparameter directory):

  * HOPPER's own baseline weights/settings (tuned for FPL/linear cost, not vanilla) survive
    only ~50-60% of episodes at 150 steps (3.0s), and collapse further -- ~10-20% -- at the
    full 500-step (10.0s) protocol length. The 3s proxy is NOT representative: a config that
    looks fine at 3s can still be accumulating a slow height sag that only produces a fall
    later in a 10s episode. Any vanilla-cost candidate must be checked at the full length.
  * Root cause (matches the FPL-side comment in tasks/hopper.py): vanilla cost's default
    height_weight=10 is small next to orientation_weight=50, so the optimizer can let the
    torso sag ("fold the leg") cheaply while staying upright, chasing velocity_weight=5's
    speed incentive. Lowering velocity_weight so height/orientation dominate the trade-off
    is what actually fixes it -- lowering it just makes the optimizer stop paying for speed
    at the expense of a stable stance.
  * Grid search over height_weight x velocity_weight at 500 steps, 20 seeds/cell, at
    noise_level=0.4 / temperature=0.15, found velocity_weight=0.15, height_weight=10.0 as the
    best cell (80% survival). Confirmed at 30 seeds: survival 0.80 (Wilson95 lower bound
    0.63), vx 1.07 m/s (54% of target_velocity=2.0), mean height 0.83.
  * orientation_weight and control_weight were left at their hopper.py defaults (50.0, 0.3)
    throughout -- the search only touched height_weight/velocity_weight, which were the
    levers that moved survival.

Trade-off, stated plainly: this reaches 2/3 of the *speed* a well-tuned FPL/linear-family
hopper config gets (hopper_pareto_sweep.py's linear family clears >85% survival near the
same target_velocity), because a quadratic cost has no atom-level floor to protect a
minimum-fulfillment guarantee the way FPL's power-mean does -- it can only be pushed toward
caution by turning DOWN the reward for speed. That gap is expected; it is the baseline the
FPL comparison (bullet 4 of the original request) is meant to be run against, not a bug to
re-tune away.
"""
from configs.env.hopper import HOPPER

VANILLA_MPPI = HOPPER.with_(**{
    "objective.mode": "normal",
    "run.steps": 500,                          # 10.0 s at dt = 0.02 (protocol length)
    # HOPPER itself does NOT set proposal.num_samples -- it is left at the schema default
    # (128), not the 256 that hopper_pareto_sweep.py's standalone script constant uses. The
    # whole search behind this config was run and validated at 256; set it explicitly here
    # so the resolved config matches what was actually tested, not the unrelated default.
    "proposal.num_samples": 256,
    "proposal.noise_level": 0.4,
    "update.temperature": 0.15,
    "task.kwargs": {
        "target_velocity": 2.0,
        "height_weight": 10.0,
        "velocity_weight": 0.15,
    },
    "label": "vanilla mppi (hopper, working)",
    "notes": (
        "objective.mode=normal, height_weight=10.0, velocity_weight=0.15, noise=0.4, "
        "temp=0.15. 30-seed check @500 steps: survival 0.80 (Wilson95 lb 0.63), "
        "vx 1.07 m/s (54% of target), h_mean 0.83. See module docstring for search log."
    ),
})

CONFIG = VANILLA_MPPI
