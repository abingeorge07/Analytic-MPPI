"""Quadruped CoM under HARD mismatch — stress survival to collapse and test the faint signal.

The moderate-weight mismatch test (quadruped_com_mismatch.py) never broke survival, so the
safety-collapse regime that drives FPL's documented robustness win could not appear — yet a
faint positive interaction showed at the harshest slip (FPL's edge over best linear grew with
CoM). This test pushes into the collapse regime: higher target speed (3.0), the AGGRESSIVE
linear weight (wv=16, documented to flip), and lower friction. Question: does the CoM atom act
as a stability FLOOR that keeps FPL SURVIVING where aggressive linear+CoM falls — more than in
the base arm? If CoM+FPL holds survival while CoM+linear collapses (and base does not show that
gap), the anti-bob atom finally earns an FPL win under mismatch.

Outputs: quadruped_com_stress.png + .json.
Run: python verification/com_study/quadruped_com_stress.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, init_barkour_stand, make_task

TV = 3.0
STEPS = 200
N_EPISODES = 12
SHARED = dict(num_samples=128, plan_horizon=0.3, num_knots=5, spline_type="zero")
NOISE, TEMP, TIME_P = 0.5, 0.2, -2.0
FALL_UP = 0.5
FRICTIONS = [0.5, 0.35, 0.25, 0.15]
LIN_WV = [8.0, 16.0]
BOB = 4.0


def configs(com: bool):
    def w(wv):
        return [1.0, 1.0, wv, 1.0, 1.0, BOB, 0.5] if com else [1.0, 1.0, wv, 1.0, 1.0, 0.5]
    lin = [Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=TEMP, fpl_weights=w(wv), fpl_time_p=TIME_P),
                  fpl_p=1.0) for wv in LIN_WV]
    return lin + [Config("FPL", "mppi", "fpl_cost",
                         dict(noise_level=NOISE, temperature=TEMP, fpl_time_p=TIME_P), fpl_p=-1.0)]


def stats(res, task):
    sd = res["sd"]
    vx = task._torso_vel_x(sd)
    surv = task._torso_up(sd).min(axis=1) >= FALL_UP
    return dict(prod=float((vx.mean(axis=1) * surv).mean()), surv=float(surv.mean()))


def run_arm(task_name, com):
    task = make_task(task_name, target_velocity=TV)
    per_fr = {}
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        study = run_study(task_name, configs(com), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_barkour_stand, task_kwargs=dict(target_velocity=TV),
                          true_perturbation=pert, progress=False, **SHARED)
        per_fr[fr] = {l: stats(r, task) for l, r in study.items()}
    return per_fr


def main():
    arms = {"base": run_arm("quadruped", False), "com": run_arm("quadruped_com", True)}
    for arm_name, per_fr in arms.items():
        print(f"\n=== {arm_name} arm ===   TV={TV}   (prod = vx x survived ; surv)")
        labels = list(next(iter(per_fr.values())).keys())
        print(f"{'friction':10s}" + "".join(f"{l:>16s}" for l in labels))
        for fr in FRICTIONS:
            print(f"{fr:<10.2f}" + "".join(f"{per_fr[fr][l]['prod']:6.2f}/{per_fr[fr][l]['surv']:.2f}   "
                                            for l in labels))

    out_dir = Path(__file__).resolve().parent
    out_dir.joinpath("quadruped_com_stress_data.json").write_text(
        json.dumps({a: {str(fr): v[fr] for fr in FRICTIONS} for a, v in arms.items()}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col, (arm_name, per_fr) in enumerate(arms.items()):
        labels = list(next(iter(per_fr.values())).keys())
        for row, key in enumerate(["surv", "prod"]):
            ax = axes[row][col]
            for l in labels:
                fpl = l == "FPL"
                ys = [per_fr[fr][l][key] for fr in FRICTIONS]
                ax.plot(FRICTIONS, ys, ("*-" if fpl else "o--"), lw=(2.4 if fpl else 1.4),
                        ms=(14 if fpl else 7), color=("#d62728" if fpl else None),
                        label=l, zorder=(3 if fpl else 2))
            ax.invert_xaxis(); ax.grid(alpha=0.2)
            if row == 0:
                ax.set_title(f"{arm_name} arm"); ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel("survival" if key == "surv" else "productive speed")
            if row == 1:
                ax.set_xlabel("friction scale (->slippery)")
            if col == 1 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Quadruped HARD mismatch (TV={TV}, aggressive linear, NO retuning, {N_EPISODES} seeds):\n"
                 "does the CoM atom keep FPL SURVIVING where aggressive linear collapses?", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "quadruped_com_stress.png", dpi=130)
    print(f"\nsaved -> {out_dir / 'quadruped_com_stress.png'}")


if __name__ == "__main__":
    main()
