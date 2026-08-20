"""Model-mismatch robustness on the WALKER2d (running regime) — generalizes the hopper
traction-robustness result to a second, harder biped.

Same setup as verification/mismatch_robustness_sweep.py (hopper): the controller PLANS on the
nominal model; the true simulator it executes on has REDUCED GROUND FRICTION (traction loss).
No controller is retuned across friction levels — you cannot retune for a drift you don't know.

Claim (replicated on the walker): one FIXED FPL spec holds the highest PRODUCTIVE SPEED
(achieved speed × survival) at every traction level, with survival ~1.0 across a 2.5x traction
reduction, while the linear-weight family has no single fixed weight that stays both fast and
safe — the aggressive weight's survival collapses on slippery ground, the safe weight is slow.

Fairness: identical sampler/atoms/budget/horizon/temporal-weakest-link; only composition
differs (p=1+weights = linear family; p=-1 uniform = FPL). Walker FPL atoms use the hopper
shapes (shared by both families). Running regime knobs (noise=0.8, H=0.6, k=6) from
verification/walker_pareto_sweep.py.

Run:  .venv/bin/python verification/mismatch_robustness_walker.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, make_task
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci  # noqa: E402
from _figs import robustness_figure  # noqa: E402

TV = 5.0
STEPS = 150
N_EPISODES = 30
NUM_SAMPLES = 256
NOISE = 0.8
TIME_P = -2.0
FALL_UPRIGHT = 0.6
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.6, num_knots=6, spline_type="zero")

FRICTIONS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
LINEAR_WV = [4.0, 8.0, 16.0, 32.0]
FPL_P = -1.0


def build_configs():
    lin = [Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=0.2,
                       fpl_weights=[1.0, 1.0, wv, 0.5], fpl_time_p=TIME_P), fpl_p=1.0)
           for wv in LINEAR_WV]
    fpl = Config(f"FPL p={FPL_P:g}", "mppi", "fpl_cost",
                 dict(noise_level=NOISE, temperature=0.2, fpl_time_p=TIME_P), fpl_p=FPL_P)
    return lin + [fpl]


def stats(res, task):
    sd = res["sd"]
    vx = sd[..., task._vel_adr]
    zax = sd[..., task._zax_adr + 2]
    survived = zax.min(axis=1) >= FALL_UPRIGHT
    prod_ep = vx.mean(axis=1) * survived
    prod_m, prod_h = mean_ci(prod_ep)
    p, lo, hi = wilson_ci(int(survived.sum()), int(survived.size))
    return dict(prod=prod_m, prod_ci=prod_h, surv=p, surv_lo=lo, surv_hi=hi)


def main():
    task = make_task("walker", target_velocity=TV)
    per_fr = {}
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        study = run_study("walker", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          task_kwargs=dict(target_velocity=TV),
                          true_perturbation=pert, progress=False, **SHARED)
        per_fr[fr] = {l: stats(r, task) for l, r in study.items()}
        print(f"friction={fr} done")

    fpl_label = f"FPL p={FPL_P:g}"
    print("\n" + "=" * 74)
    print(f"WALKER2d traction-loss robustness | K={NUM_SAMPLES} eps={N_EPISODES} | NO retuning")
    print("=" * 74)
    for fr in FRICTIONS:
        cells = "  ".join(f"{per_fr[fr][l]['prod']:.2f}/{per_fr[fr][l]['surv']:.2f}"
                          for l in per_fr[fr])
        print(f"friction={fr:4.2f}  {cells}   (prod/surv)")

    Path(__file__).resolve().parent.joinpath("mismatch_robustness_walker_data.json").write_text(
        json.dumps({str(fr): per_fr[fr] for fr in FRICTIONS}, indent=2))
    robustness_figure(
        per_fr, fpl_label,
        'Walker2d (running) under traction loss (NO retuning; 30 seeds, 95% CIs):\\none fixed FPL spec delivers the most usable speed and holds survival; the aggressive linear weight collapses',
        Path(__file__).resolve().parent / "mismatch_robustness_walker.png")


if __name__ == "__main__":
    main()
