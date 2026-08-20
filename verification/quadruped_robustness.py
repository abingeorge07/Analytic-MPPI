"""Model-mismatch robustness on the QUADRUPED (Barkour) — extends the un-tunable
traction-robustness result to a whole-body legged robot (third robot after hopper + walker).

Same setup as the hopper/walker robustness studies: the controller PLANS on the nominal model
and EXECUTES on a true simulator with REDUCED GROUND FRICTION (traction loss). No controller is
retuned across friction levels. Claim: one FIXED FPL spec keeps the highest productive speed
(speed x survival) at every traction level while the linear family cannot.

Run:  .venv/bin/python verification/quadruped_robustness.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, init_barkour_stand, make_task
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci  # noqa: E402
from _figs import robustness_figure  # noqa: E402

TV = 2.0
STEPS = 200
N_EPISODES = 30
NUM_SAMPLES = 128
NOISE = 0.5
TIME_P = -2.0
FALL_UP = 0.5
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.3, num_knots=5, spline_type="zero")

FRICTIONS = [1.0, 0.7, 0.5, 0.4, 0.3]
LINEAR_WV = [2.0, 4.0, 8.0, 16.0]
FPL_P = -1.0


def build_configs():
    # atom order: [height, orientation, velocity, heading, posture, control]; sweep the velocity weight.
    lin = [Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=0.2,
                       fpl_weights=[1.0, 1.0, wv, 1.0, 1.0, 0.5], fpl_time_p=TIME_P), fpl_p=1.0)
           for wv in LINEAR_WV]
    fpl = Config(f"FPL p={FPL_P:g}", "mppi", "fpl_cost",
                 dict(noise_level=NOISE, temperature=0.2, fpl_time_p=TIME_P), fpl_p=FPL_P)
    return lin + [fpl]


def stats(res, task):
    sd = res["sd"]
    vx = task._torso_vel_x(sd)
    up = task._torso_up(sd)
    survived = up.min(axis=1) >= FALL_UP
    prod_ep = vx.mean(axis=1) * survived
    prod_m, prod_h = mean_ci(prod_ep)
    p, lo, hi = wilson_ci(int(survived.sum()), int(survived.size))
    return dict(prod=prod_m, prod_ci=prod_h, surv=p, surv_lo=lo, surv_hi=hi)


def main():
    task = make_task("quadruped", target_velocity=TV)
    per_fr = {}
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        study = run_study("quadruped", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_barkour_stand, task_kwargs=dict(target_velocity=TV),
                          true_perturbation=pert, progress=False, **SHARED)
        per_fr[fr] = {l: stats(r, task) for l, r in study.items()}
        print(f"friction={fr} done")

    fpl_label = f"FPL p={FPL_P:g}"
    print("\n" + "=" * 70)
    print(f"QUADRUPED traction-loss robustness | K={NUM_SAMPLES} eps={N_EPISODES} | NO retuning")
    print("=" * 70)
    for fr in FRICTIONS:
        cells = "  ".join(f"{per_fr[fr][l]['prod']:.2f}/{per_fr[fr][l]['surv']:.2f}"
                          for l in per_fr[fr])
        print(f"friction={fr:4.2f}  {cells}   (prod/surv)")

    Path(__file__).resolve().parent.joinpath("quadruped_robustness_data.json").write_text(
        json.dumps({str(fr): per_fr[fr] for fr in FRICTIONS}, indent=2))
    robustness_figure(
        per_fr, fpl_label,
        'Barkour quadruped under traction loss (NO retuning; 30 seeds, 95% CIs):\\none fixed FPL spec delivers the most usable speed and holds survival; the aggressive linear weight collapses',
        Path(__file__).resolve().parent / "quadruped_robustness.png")


if __name__ == "__main__":
    main()
