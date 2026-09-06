"""Flagship FPL-cost (objective.mode="fpl_cost") MPPI config on walker, running regime.

Unlike hopper_fpl_mppi.py (ported from a runs/capability/hopper/ bundle produced by
capability_probe.py's exploratory knob-grid), no runs/capability/walker/ bundle existed
before this task -- clean slate. Rather than re-deriving the flagship operating point via
capability_probe's GRID search, this file simply promotes the "running regime" flagship
settings ALREADY established and used identically across
verification/walker_pareto_sweep.py, verification/mismatch_robustness_walker.py, and
verification/_experiment.ENVS["walker"]: num_samples=256, num_knots=6, plan_horizon=0.6,
noise_level=0.8, temperature=0.2, fpl_time_p=-2.0, fpl_p=-1.0, spline_type="zero",
target_velocity=5.0 (the "running regime" difficulty -- requires flight-phase running,
not walking, which is where FPL's edge over linear weighting actually shows up on this
task per walker_pareto_sweep.py's module docstring).

This is configs/env/walker.py's own baseline (WALKER), so this file mainly makes
run.steps explicit at the doc's protocol length (see below) and documents the
validation result -- there is no hyperparameter search here (that is walker's cost
function ISN'T being retuned for the FPL arm; only the vanilla arm needed one, see
walker_vanilla_mppi.py's module docstring).

30-seed result at the FULL 1000-step (10.0s) protocol length (existing
walker_pareto_data.json's FPL row is only at 150 steps / 1.5s -- too short for a
target_velocity=5.0 walker to leave standstill and reach anything close to target speed,
so it is not reused here; this file's own number is measured at the real protocol
length): survival 1.00 (30/30, Wilson95 lower bound 0.89), vx 1.71 m/s (34% of
target_velocity=5.0). This is the flagship comparison point for the vanilla-MPPI config
in walker_vanilla_mppi.py.

NOTABLE, stated plainly: unlike hopper (where FPL clearly beat vanilla on speed at
similar-or-better survival), FPL is SLOWER here than the tuned vanilla config
(walker_vanilla_mppi.py: 2.19 m/s / 44% of target, survival 0.97) despite being
marginally safer (1.00 vs 0.97 survival). At target_velocity=5.0 (flight-phase running,
a much harder ask than hopper's target of 2.0), FPL's min-fulfillment conjunction
appears to be protecting the orientation/height atoms at the cost of the velocity atom
more than vanilla's tuned-down quadratic sum does -- vanilla, once velocity_weight is
turned down enough to survive, can still push speed opportunistically in a way the
conjunction does not reward. This is an empirical result to carry into steps 3/4, not
an assumption to correct for -- see docs/vanilla_mppi_hyperparams_walker.md §5.
"""
from configs.env.walker import WALKER

FPL_MPPI = WALKER.with_(**{
    "objective.mode": "fpl_cost",              # WALKER's own schema default; set
                                                # explicitly for symmetry with
                                                # walker_vanilla_mppi.py's override
    "run.steps": 1000,                         # 10.0 s at dt = 0.01 -- protocol length
    "label": "fpl mppi (walker, running-regime flagship)",
    "notes": (
        "Ported from the flagship settings shared by walker_pareto_sweep.py / "
        "mismatch_robustness_walker.py / _experiment.ENVS['walker']. 30-seed result "
        "@1000 steps (10s): survival 1.00 (30/30, Wilson95 lb 0.89), vx 1.71 m/s (34% "
        "of target) -- SLOWER than the tuned vanilla arm (2.19 m/s / 44%). See module "
        "docstring for the notable comparison."
    ),
})

CONFIG = FPL_MPPI
