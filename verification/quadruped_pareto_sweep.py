"""Quadruped (Barkour) flagship — FPL Pareto-dominates the linear-weight family under dynamic
competition, on a 12-DoF whole-body legged robot.

This lifts the hopper/walker result to a quadruped in the regime of Alvarez-Padilla et al.,
"Real-Time Whole-Body Control of Legged Robots with MPPI" (arXiv:2409.10469): MuJoCo-parallel
sampling MPC on a whole-body legged robot. Same fairness protocol as the other studies:
identical sampler / atoms / budget / horizon / temporal weakest-link; only the objective
composition differs (p=1+weights = linear family; p=-1 uniform = FPL).

Dynamic competition: to go faster the quadruped must break into a gait that risks toppling.
The aggressive linear weight sprints and inverts (100% falls); one fixed FPL spec keeps
~0% falls at competitive speed. No single linear weight is both fast and safe.

Run:  .venv/bin/python verification/quadruped_pareto_sweep.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, init_barkour_stand, make_task
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci  # noqa: E402
from _figs import pareto_figure  # noqa: E402

STEPS = 200            # ~0.8 s at dt=0.004
N_EPISODES = 30
NUM_SAMPLES = 128
NOISE = 0.5
TEMP = 0.2
TIME_P = -2.0
FALL_UP = 0.5          # min torso uprightness over an episode below this = a fall
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.3, num_knots=5, spline_type="zero")

TARGET_VELS = [2.0, 2.5, 3.0]
LINEAR_WV = [1.0, 2.0, 4.0, 8.0, 16.0]
FPL_P = -1.0


def build_configs():
    # atom order: [height, orientation, velocity, heading, posture, control]; sweep the velocity weight.
    lin = [Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=TEMP,
                       fpl_weights=[1.0, 1.0, wv, 1.0, 1.0, 0.5], fpl_time_p=TIME_P), fpl_p=1.0)
           for wv in LINEAR_WV]
    fpl = Config(f"FPL p={FPL_P:g}", "mppi", "fpl_cost",
                 dict(noise_level=NOISE, temperature=TEMP, fpl_time_p=TIME_P), fpl_p=FPL_P)
    return lin + [fpl]


def episode_stats(res, task):
    sd = res["sd"]
    vx = task._torso_vel_x(sd)
    up = task._torso_up(sd)
    fell = up.min(axis=1) < FALL_UP
    vx_ep = vx.mean(axis=1)
    vx_m, vx_h = mean_ci(vx_ep)
    p, lo, hi = wilson_ci(int((~fell).sum()), int(fell.size))
    return dict(vx=vx_m, vx_ci=vx_h, fall=float(fell.mean()),
                surv=p, surv_lo=lo, surv_hi=hi)


def main():
    per_tv = {}
    for tv in TARGET_VELS:
        task = make_task("quadruped", target_velocity=tv)
        study = run_study("quadruped", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_barkour_stand, task_kwargs=dict(target_velocity=tv),
                          progress=False, **SHARED)
        per_tv[tv] = {l: episode_stats(r, task) for l, r in study.items()}
        print(f"tv={tv} done")

    fpl_label = f"FPL p={FPL_P:g}"
    print("\n" + "=" * 72)
    print(f"QUADRUPED (Barkour) linear-family vs FPL | K={NUM_SAMPLES} eps={N_EPISODES}")
    print("=" * 72)
    for tv in TARGET_VELS:
        print(f"\n-- target_velocity = {tv} " + "-" * 36)
        print(f"{'config':12s}{'vx±CI':>12s}{'fall':>7s}{'surv':>7s}")
        for l, s in per_tv[tv].items():
            print(f"{l:12s}{s['vx']:7.3f}±{s['vx_ci']:.2f}{s['fall']:7.2f}{s['surv']:7.2f}")

    Path(__file__).resolve().parent.joinpath("quadruped_pareto_data.json").write_text(
        json.dumps({str(tv): per_tv[tv] for tv in TARGET_VELS}, indent=2))
    pareto_figure(
        per_tv, fpl_label,
        'Barkour quadruped (straight-line walk): one fixed FPL spec dominates the linear-weight family at every speed — no retuning\\n(identical sampler / atoms / budget; 30 seeds, 95% CIs; only the objective composition differs)',
        Path(__file__).resolve().parent / "quadruped_pareto.png")


if __name__ == "__main__":
    main()
