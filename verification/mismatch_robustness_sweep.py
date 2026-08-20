"""Model-mismatch robustness — the UN-TUNABLE structural FPL win.

Setup (sim-to-real gap): the controller PLANS on the nominal model; the true simulator it
executes on has REDUCED GROUND FRICTION (traction loss — ice / wet / worn surface, the
canonical sim-to-real failure). Crucially you CANNOT retune your cost for a traction drop you
don't know about — so a *fixed* controller is the honest object of study.

Claim: one FIXED FPL spec (p=-1) holds BOTH speed and survival across a 3x traction reduction,
while the linear-weight family has NO single fixed weight that stays both fast and safe — the
weight you'd pick on the nominal model loses speed and/or survival as the ground gets slippery.

Why FPL (mechanism, from the probe): FPL wins specifically when mismatch makes AGGRESSION
DANGEROUS while safe progress remains possible — slippery ground makes hard push-offs slip
(a fall), but gentle hops still work, and FPL's min-fulfillment conjunction (stay upright AND
at height AND moving) structurally finds the safe gait. SCOPE (honest): this does NOT hold for
mismatches that only make the task uniformly harder (added mass, weaker actuators) — there FPL's
speed-floor makes it over-try and it is roughly on the linear frontier. So the robustness claim
is specific to aggression-punishing mismatch (traction), which is the important real-world one.

Fairness: identical sampler/atoms/budget/horizon/temporal-weakest-link; only the objective
composition differs (p=1+weights = linear family; p=-1 uniform = FPL). NO controller is
retuned across friction levels — that is the whole point.

Run:  .venv/bin/python verification/mismatch_robustness_sweep.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, init_hopper_stand, make_task
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci  # noqa: E402
from _figs import robustness_figure  # noqa: E402

TV = 2.5
STEPS = 150
N_EPISODES = 30
NUM_SAMPLES = 256
TIME_P = -2.0
FALL_UPRIGHT = 0.6
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.6, num_knots=4, spline_type="zero")

# Increasing traction loss (1.0 = nominal, smaller = slipperier).
FRICTIONS = [1.0, 0.8, 0.6, 0.45, 0.35, 0.3]
LINEAR_WV = [0.5, 1.0, 2.0, 4.0]
FPL_P = -1.0


def build_configs():
    lin = [Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
                  dict(noise_level=0.3, temperature=0.2,
                       fpl_weights=[1.0, 1.0, wv, 0.5], fpl_time_p=TIME_P), fpl_p=1.0)
           for wv in LINEAR_WV]
    fpl = Config(f"FPL p={FPL_P:g}", "mppi", "fpl_cost",
                 dict(noise_level=0.3, temperature=0.2, fpl_time_p=TIME_P), fpl_p=FPL_P)
    return lin + [fpl]


def stats(res, task):
    sd = res["sd"]
    vx = sd[..., task._vel_adr]
    zax = sd[..., task._zax_adr + 2]
    survived = zax.min(axis=1) >= FALL_UPRIGHT           # (n_ep,)
    prod_ep = vx.mean(axis=1) * survived                  # productive speed per episode
    prod_m, prod_h = mean_ci(prod_ep)
    p, lo, hi = wilson_ci(int(survived.sum()), int(survived.size))
    return dict(prod=prod_m, prod_ci=prod_h, surv=p, surv_lo=lo, surv_hi=hi)


def main():
    task = make_task("hopper", target_velocity=TV)
    per_fr = {}
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        study = run_study("hopper", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_hopper_stand, task_kwargs=dict(target_velocity=TV),
                          true_perturbation=pert, progress=False, **SHARED)
        per_fr[fr] = {l: stats(r, task) for l, r in study.items()}
        print(f"friction={fr} done")

    fpl_label = f"FPL p={FPL_P:g}"
    print("\n" + "=" * 74)
    print(f"HOPPER traction-loss robustness | K={NUM_SAMPLES} eps={N_EPISODES} | NO retuning")
    print("=" * 74)
    print(f"{'friction':>9s}  " + "  ".join(f"{l:>12s}" for l in per_fr[FRICTIONS[0]]))
    for fr in FRICTIONS:
        cells = "  ".join(f"{per_fr[fr][l]['prod']:.2f}/{per_fr[fr][l]['surv']:.2f}"
                          for l in per_fr[fr])
        print(f"{fr:9.2f}  {cells}   (prod/surv)")

    Path(__file__).resolve().parent.joinpath("mismatch_robustness_data.json").write_text(
        json.dumps({str(fr): per_fr[fr] for fr in FRICTIONS}, indent=2))
    robustness_figure(
        per_fr, fpl_label,
        'Hopper under traction loss (planner uses nominal model, reality is slippery — NO retuning; 30 seeds, 95% CIs):\\none fixed FPL spec holds speed AND survival; no fixed linear weight does',
        Path(__file__).resolve().parent / "mismatch_robustness.png")


if __name__ == "__main__":
    main()
