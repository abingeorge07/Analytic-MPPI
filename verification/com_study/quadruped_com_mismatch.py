"""Quadruped CoM under MISMATCH — the one regime where the anti-bob atom could still win.

At nominal conditions the CoM anti-bob atom is a tunable stabilizer (see
quadruped_com_speed_sweep.py: a fixed linear b=4 matches/beats FPL everywhere). But FPL's
documented wins are all under NO-RETUNING traction loss (fpl-mismatch-robustness). So the
decisive question: does ADDING the anti-bob atom WIDEN FPL's robustness margin over the
linear family under slip — i.e. does it act as a never-sacrificed stability FLOOR that keeps
FPL surviving/productive when a fixed linear weight collapses?

Setup mirrors verification/quadruped_robustness.py: PLAN on nominal, EXECUTE on reduced ground
friction, NO retuning across friction. Two arms — base (6-atom) and +CoM (7-atom) — each with
FPL (uniform) and a couple of fixed linear weights. Metric: productive speed (vx x survived)
and survival vs friction. INTERACTION = is (FPL - linear) margin larger in the +CoM arm?

Outputs: quadruped_com_mismatch.png + .json.
Run: python verification/com_study/quadruped_com_mismatch.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import Config, run_study, init_barkour_stand, make_task

TV = 2.0
STEPS = 200
N_EPISODES = 12
SHARED = dict(num_samples=128, plan_horizon=0.3, num_knots=5, spline_type="zero")
NOISE, TEMP, TIME_P = 0.5, 0.2, -2.0
FALL_UP = 0.5
FRICTIONS = [1.0, 0.5, 0.35, 0.25]
LIN_WV = [4.0, 8.0]
BOB = 4.0                       # linear bob-weight in the +CoM arm (the good nominal weight)


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
    prod = (vx.mean(axis=1) * surv)
    return dict(prod=float(prod.mean()), surv=float(surv.mean()))


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
        print(f"\n=== {arm_name} arm ===   (prod = vx x survived ; surv)")
        labels = list(next(iter(per_fr.values())).keys())
        print(f"{'friction':10s}" + "".join(f"{l:>16s}" for l in labels))
        for fr in FRICTIONS:
            print(f"{fr:<10.2f}" + "".join(f"{per_fr[fr][l]['prod']:6.2f}/{per_fr[fr][l]['surv']:.2f}   "
                                            for l in labels))

    out_dir = Path(__file__).resolve().parent
    out_dir.joinpath("quadruped_com_mismatch_data.json").write_text(
        json.dumps({a: {str(fr): v[fr] for fr in FRICTIONS} for a, v in arms.items()}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (arm_name, per_fr) in zip(axes, arms.items()):
        labels = list(next(iter(per_fr.values())).keys())
        for l in labels:
            fpl = l == "FPL"
            ys = [per_fr[fr][l]["prod"] for fr in FRICTIONS]
            ax.plot(FRICTIONS, ys, ("*-" if fpl else "o--"),
                    lw=(2.4 if fpl else 1.4), ms=(14 if fpl else 7),
                    color=("#d62728" if fpl else None), label=l, zorder=(3 if fpl else 2))
        ax.set_xlabel("ground friction scale  (1.0 = nominal, ->slippery)")
        ax.set_title(f"{arm_name} arm")
        ax.invert_xaxis(); ax.grid(alpha=0.2); ax.legend(fontsize=8)
    axes[0].set_ylabel("productive speed  (vx x survived)")
    fig.suptitle("Quadruped under traction loss (NO retuning, 12 seeds): does +CoM widen FPL's "
                 "robustness margin?\nInteraction = FPL-minus-linear gap larger in the +CoM (right) panel",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "quadruped_com_mismatch.png", dpi=130)
    print(f"\nsaved -> {out_dir / 'quadruped_com_mismatch.png'}")


if __name__ == "__main__":
    main()
