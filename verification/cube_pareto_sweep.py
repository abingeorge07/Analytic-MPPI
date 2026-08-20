"""In-hand cube reorientation (LEAP hand) — FPL Pareto-dominates the linear-weight family
under dynamic competition, on a 16-DoF dexterous manipulator.

This is the manipulation member of the study (after the hopper / walker / quadruped legged
robots). Same fairness protocol: identical sampler / atoms / budget / horizon / temporal
weakest-link; only the objective composition differs (p=1+weights = linear family;
p=-1 uniform = FPL).

Dynamic competition: the hand cradles the cube palm-up and must tumble it forward (roll
about x) to a commanded angle. The only way to rotate faster is to push harder with the
fingertips, which risks rolling the cube off the palm — a drop. The aggressive linear weight
rotates far but drops the cube; the safe linear weight never drops but barely turns it; one
fixed FPL spec rotates far AND keeps the cube in hand. No single linear weight is both.

Run:  .venv/bin/python verification/cube_pareto_sweep.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, init_cube, make_task
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci  # noqa: E402
from _figs import pareto_figure  # noqa: E402

STEPS = 200            # 2.0 s at dt=0.01
N_EPISODES = 30
NUM_SAMPLES = 128
NOISE = 0.4
TEMP = 0.1
TIME_P = -2.0
DROP_XY = 0.06         # cube xy off the palm centre above this = a drop
DROP_H = -0.04         # cube height (rel. grasp site) below this = a drop
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.3, num_knots=4, spline_type="zero")

TARGET_ANGLES = [0.8, 1.2, 1.8]     # rad ≈ 46° / 69° / 103° forward roll
LINEAR_WA = [1.0, 2.0, 4.0, 8.0, 16.0]
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


def episode_stats(res, task):
    sd = res["sd"]                                   # (ep, T, nsd)
    xy = task._hold_xy(sd)                           # (ep, T)
    h = task._cube_height(sd)
    align = task._alignment_fulfillment(sd)          # (ep, T) fraction of target achieved
    dropped = (xy.max(axis=1) > DROP_XY) | (h.min(axis=1) < DROP_H)
    # "achieved rotation" (the speed analogue): degrees of the commanded roll realized,
    # averaged over the episode — dragged toward 0 by a drop (cube tumbles away, align→0).
    deg = np.degrees(task.target_angle)
    achieved_ep = deg * align.mean(axis=1)
    vx_m, vx_h = mean_ci(achieved_ep)
    p, lo, hi = wilson_ci(int((~dropped).sum()), int(dropped.size))
    return dict(vx=vx_m, vx_ci=vx_h, fall=float(dropped.mean()),
                surv=p, surv_lo=lo, surv_hi=hi)


def main():
    per_ta = {}
    for ta in TARGET_ANGLES:
        task = make_task("cube", target_angle=ta)
        study = run_study("cube", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_cube, task_kwargs=dict(target_angle=ta),
                          progress=False, **SHARED)
        per_ta[ta] = {l: episode_stats(r, task) for l, r in study.items()}
        print(f"target_angle={ta} ({np.degrees(ta):.0f}°) done")

    fpl_label = f"FPL p={FPL_P:g}"
    print("\n" + "=" * 74)
    print(f"CUBE reorientation (LEAP hand) linear-family vs FPL | K={NUM_SAMPLES} eps={N_EPISODES}")
    print("=" * 74)
    for ta in TARGET_ANGLES:
        print(f"\n-- target_angle = {ta} rad ({np.degrees(ta):.0f}°) " + "-" * 28)
        print(f"{'config':12s}{'achieved°±CI':>16s}{'drop':>7s}{'surv':>7s}")
        for l, s in per_ta[ta].items():
            print(f"{l:12s}{s['vx']:9.1f}±{s['vx_ci']:.1f}{s['fall']:9.2f}{s['surv']:7.2f}")

    Path(__file__).resolve().parent.joinpath("cube_pareto_data.json").write_text(
        json.dumps({str(ta): per_ta[ta] for ta in TARGET_ANGLES}, indent=2))
    # keys of per_ta are radians; relabel panels in degrees for the figure.
    per_deg = {int(round(np.degrees(ta))): per_ta[ta] for ta in TARGET_ANGLES}
    pareto_figure(
        per_deg, fpl_label,
        'In-hand cube reorientation (LEAP hand, forward roll): one fixed FPL spec dominates the linear-weight family at every commanded angle — no retuning\\n(identical sampler / atoms / budget; 30 seeds, 95% CIs; only the objective composition differs)',
        Path(__file__).resolve().parent / "cube_pareto.png",
        xlabel="achieved rotation  (deg toward goal)", gain_word="rotation",
        panel_title="commanded roll = {tv}°")


if __name__ == "__main__":
    main()
