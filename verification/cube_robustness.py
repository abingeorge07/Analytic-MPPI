"""Model-mismatch robustness on the CUBE (LEAP hand) — extends the un-tunable
grip-loss robustness result to a dexterous manipulator (a fourth robot after the hopper,
walker and quadruped, and the first non-legged one).

Same setup as the legged robustness studies: the controller PLANS on the nominal model and
EXECUTES on a true simulator with REDUCED CONTACT FRICTION (grip loss — a worn/slippery cube
or fingertips, the manipulation analogue of traction loss). No controller is retuned across
friction levels. Claim: one FIXED FPL spec keeps the highest productive rotation
(achieved angle × survival) at every friction level while the linear family cannot — a heavy
orientation weight slips the cube out of the hand, and you cannot retune for a grip you do
not know about.

Run:  .venv/bin/python verification/cube_robustness.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, init_cube, make_task
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci  # noqa: E402
from _figs import robustness_figure  # noqa: E402

TA = 0.8               # rad ≈ 46° forward roll (grip loss removes the safe operating point
                       # entirely at larger angles — tumbling a cube fundamentally needs grip)
STEPS = 200
N_EPISODES = 30
NUM_SAMPLES = 128
NOISE = 0.4
TEMP = 0.1
TIME_P = -2.0
DROP_XY = 0.06
DROP_H = -0.04
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.3, num_knots=4, spline_type="zero")

FRICTIONS = [1.0, 0.9, 0.8, 0.7, 0.6]
LINEAR_WA = [2.0, 4.0, 8.0, 16.0]
FPL_P = -1.0


def build_configs():
    # atom order: [hold, height, alignment, control]; sweep the ALIGNMENT weight.
    lin = [Config(f"lin wa={wa:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=TEMP,
                       fpl_weights=[1.0, 1.0, wa, 0.5], fpl_time_p=TIME_P), fpl_p=1.0)
           for wa in LINEAR_WA]
    fpl = Config(f"FPL p={FPL_P:g}", "mppi", "fpl_cost",
                 dict(noise_level=NOISE, temperature=TEMP, fpl_time_p=TIME_P), fpl_p=FPL_P)
    return lin + [fpl]


def stats(res, task):
    sd = res["sd"]
    xy = task._hold_xy(sd)
    h = task._cube_height(sd)
    align = task._alignment_fulfillment(sd)
    survived = ~((xy.max(axis=1) > DROP_XY) | (h.min(axis=1) < DROP_H))
    deg = np.degrees(task.target_angle)
    prod_ep = deg * align.mean(axis=1) * survived
    prod_m, prod_h = mean_ci(prod_ep)
    p, lo, hi = wilson_ci(int(survived.sum()), int(survived.size))
    return dict(prod=prod_m, prod_ci=prod_h, surv=p, surv_lo=lo, surv_hi=hi)


def main():
    task = make_task("cube", target_angle=TA)
    per_fr = {}
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        study = run_study("cube", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_cube, task_kwargs=dict(target_angle=TA),
                          true_perturbation=pert, progress=False, **SHARED)
        per_fr[fr] = {l: stats(r, task) for l, r in study.items()}
        print(f"friction={fr} done")

    fpl_label = f"FPL p={FPL_P:g}"
    print("\n" + "=" * 70)
    print(f"CUBE grip-loss robustness | K={NUM_SAMPLES} eps={N_EPISODES} | NO retuning")
    print("=" * 70)
    for fr in FRICTIONS:
        cells = "  ".join(f"{per_fr[fr][l]['prod']:.1f}/{per_fr[fr][l]['surv']:.2f}"
                          for l in per_fr[fr])
        print(f"friction={fr:4.2f}  {cells}   (prod°/surv)")

    Path(__file__).resolve().parent.joinpath("cube_robustness_data.json").write_text(
        json.dumps({str(fr): per_fr[fr] for fr in FRICTIONS}, indent=2))
    robustness_figure(
        per_fr, fpl_label,
        'LEAP-hand cube reorientation under grip loss (NO retuning; 30 seeds, 95% CIs):\\none fixed FPL spec keeps the most usable rotation and holds the cube; the aggressive linear weight slips it out',
        Path(__file__).resolve().parent / "cube_robustness.png",
        prod_title="Productive rotation  (achieved angle × survival)",
        prod_ylabel="deg × survival")


if __name__ == "__main__":
    main()
