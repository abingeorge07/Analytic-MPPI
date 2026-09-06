"""Step 2 of docs/vanilla_mppi_hyperparams_walker.md: a working vanilla-cost
(objective.mode="normal") MPPI config on walker, at the step-3 protocol length
(10s / 1000 steps -- walker's dt=0.01 is HALF hopper's dt=0.02, so this is 1000 steps,
NOT hopper's 500).

Search log (see docs/vanilla_mppi_hyperparams_walker.md for the full hyperparameter
directory):

  * WALKER's own baseline (flagship settings tuned for FPL, not vanilla;
    target_velocity=5.0 is the "running regime" difficulty -- requires flight-phase
    running, a much harder target than hopper's target_velocity=2.0) gets strong raw
    speed at default weights (velocity_weight=1.0): 72% of target at the full 1000-step
    length -- but only 12% survival (1/8 seeds). Same proxy-length-mismatch pitfall as
    hopper: a short episode looks fine while a slow fall/height-sag failure only shows
    up given enough time, except on walker the collapse is even faster -- default
    weights already show poor survival (12%, 1/8) by just 3s (300 steps), not only at
    the full 10s length. So even the "cheap proxy" here must be long enough (3s used
    below, not hopper's shorter smoke-test lengths) to see the real failure mode.
  * Root cause matches hopper's: velocity_weight=1.0 (default) lets the optimizer trade
    stance/orientation for speed cheaply. UNLIKE hopper, raising height_weight (10 -> 20)
    did NOT help here (worse or unchanged at every velocity_weight tested) -- walker's
    height:velocity weight ratio (10:1) is already 5x more height-protective than
    hopper's original ratio (10:5=2:1) was, so height was never the binding term.
    velocity_weight is the only lever that moves survival, exactly like hopper.
  * Grid search over velocity_weight at a 3s (300-step) proxy, 15 seeds/cell (with an
    earlier 8-seed pass to locate the neighborhood): the region below the flagship
    default (0.05-0.3) is where survival lives; the region hopper's fix landed in
    (roughly default/10, by analogy) is noisy here between 0.35 and 0.7 (survival
    0.47-0.73, no clean optimum) -- so the analogy to hopper's exact relative reduction
    does not transfer number-for-number, only the qualitative direction does.
  * Full 1000-step (10s) validation of the three most promising candidates (12
    seeds/cell): velocity_weight=0.10 -> 100% survival (12/12), vx 2.20 m/s (44% of
    target); =0.15 -> 75% survival, vx 2.56 m/s (51%); =0.20 -> 50% survival, vx 2.78
    m/s (56%). velocity_weight=0.10 is the clear pick: survival does not merely "clear a
    bar", it is the ONLY candidate tested that is robustly safe at both the proxy and
    the full protocol length.
  * 30-seed confirmation at the full 1000-step length: survival 0.97 (29/30, Wilson95
    lower bound 0.833), vx 2.19 m/s (44% of target_velocity=5.0).
  * orientation_weight and control_weight were left at their walker.py defaults (10.0,
    0.001) throughout -- the search only touched height_weight/velocity_weight, and
    height_weight ultimately stayed at its default too (10.0), since raising it never
    helped.

Trade-off, stated plainly: walker's target_velocity=5.0 is a much harder ask than
hopper's 2.0 (it demands flight-phase running, not walking), so 44% of target here is
not directly comparable to hopper's 54% of target at 2.0 -- the absolute bar is higher.
As with hopper, a quadratic cost has no per-atom floor to guarantee a minimum
fulfillment the way FPL's power-mean does, so survival can only be bought by turning
DOWN the reward for speed. That gap is expected; it is the baseline the FPL comparison
(walker_fpl_mppi.py) is meant to be run against, not a bug to re-tune away.
"""
from configs.env.walker import WALKER

VANILLA_MPPI = WALKER.with_(**{
    "objective.mode": "normal",
    "run.steps": 1000,                         # 10.0 s at dt = 0.01 (protocol length)
    "task.kwargs": {
        "target_velocity": 5.0,
        "height_weight": 10.0,
        "velocity_weight": 0.10,
    },
    "label": "vanilla mppi (walker, working)",
    "notes": (
        "objective.mode=normal, height_weight=10.0 (default), velocity_weight=0.10 "
        "(down from default 1.0), noise=0.8, temp=0.2. 30-seed check @1000 steps: "
        "survival 0.97 (29/30, Wilson95 lb 0.833), vx 2.19 m/s (44% of target). See "
        "module docstring for search log."
    ),
})

CONFIG = VANILLA_MPPI