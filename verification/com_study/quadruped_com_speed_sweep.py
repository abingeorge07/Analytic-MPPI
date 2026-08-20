"""Quadruped CoM — the ROBUSTNESS test (the regime where FPL actually wins).

The single-speed 2x2 (quadruped_com_2x2.py) showed that at tv=2.0 a TUNED linear bob-weight
(b=4) matches/beats FPL: at one gentle, tunable operating point linear is fine, as the thesis
predicts. FPL's edge is no-retuning across CONDITIONS. So here we ask the real question:

  Does ONE fixed FPL+CoM spec (uniform) stay fast + smooth + upright across a RANGE of target
  speeds, while any ONE fixed linear+CoM bob-weight falls off somewhere — too bouncy/unsafe
  when pushed fast, or needlessly slow when not?

For each target_velocity in {2.0, 2.5, 3.0} we run FPL+CoM (uniform) and a sweep of fixed
linear+CoM bob-weights, and record (speed vx, bounce |vz|, fall%). The win condition: no
single linear bob-weight is on the speed/smoothness/safety frontier at EVERY speed, but the
one fixed FPL spec is. (Fairness protocol identical to quadruped_pareto_sweep.py.)

Outputs: quadruped_com_speed_sweep.png + .json.
Run: python verification/com_study/quadruped_com_speed_sweep.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import run_study, Config, init_barkour_stand, make_task

TVS = [2.0, 2.5, 3.0]
STEPS = 250
N_EPISODES = 8
SHARED = dict(num_samples=128, plan_horizon=0.3, num_knots=5, spline_type="zero")
NOISE, TEMP, TIME_P = 0.5, 0.2, -2.0
FALL_UP = 0.5
VW = 4.0
LIN_BOB = [0.0, 2.0, 4.0, 8.0]     # fixed linear bob-weights (must pick ONE for all speeds)


def configs():
    lin = [Config(f"lin b={b:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=TEMP,
                       fpl_weights=[1.0, 1.0, VW, 1.0, 1.0, b, 0.5], fpl_time_p=TIME_P),
                  fpl_p=1.0) for b in LIN_BOB]
    return lin + [Config("FPL", "mppi", "fpl_cost",
                         dict(noise_level=NOISE, temperature=TEMP, fpl_time_p=TIME_P),
                         fpl_p=-1.0)]


def stats(res, task):
    sd = res["sd"]; vadr = task._vel_adr; g = slice(60, None)
    vx = sd[:, g, vadr].mean(axis=1)
    bob = np.abs(sd[:, g, vadr + 2]).mean(axis=1)
    fell = task._torso_up(sd).min(axis=1) < FALL_UP
    return dict(vx=float(vx.mean()), bob=float(bob.mean()), fall=float(fell.mean()))


def main():
    per_tv = {}
    for tv in TVS:
        task = make_task("quadruped_com", target_velocity=tv)
        study = run_study("quadruped_com", configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_barkour_stand, task_kwargs=dict(target_velocity=tv),
                          progress=False, **SHARED)
        per_tv[tv] = {label: stats(r, task) for label, r in study.items()}
        print(f"tv={tv} done")

    print("\n" + "=" * 60)
    for tv in TVS:
        print(f"\n-- target_velocity = {tv} --")
        print(f"{'config':10s}{'vx':>8s}{'bob|vz|':>10s}{'fall':>7s}")
        for label, s in per_tv[tv].items():
            print(f"{label:10s}{s['vx']:8.2f}{s['bob']:10.3f}{s['fall']:7.0%}")

    out_dir = Path(__file__).resolve().parent
    out_dir.joinpath("quadruped_com_speed_sweep_data.json").write_text(
        json.dumps({str(tv): per_tv[tv] for tv in TVS}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"lin b={b:g}" for b in LIN_BOB] + ["FPL"]
    cmap = {lab: c for lab, c in zip(labels, plt.cm.viridis(np.linspace(0, 0.8, len(LIN_BOB))).tolist() + [(0.84, 0.15, 0.16, 1.0)])}
    fig, axes = plt.subplots(1, len(TVS), figsize=(5 * len(TVS), 4.6), sharey=True)
    for ax, tv in zip(axes, TVS):
        for label, s in per_tv[tv].items():
            fpl = label == "FPL"
            ax.scatter([s["vx"]], [s["bob"]], s=(300 if fpl else 90),
                       marker=("*" if fpl else "o"), color=cmap[label],
                       edgecolor="k", zorder=(3 if fpl else 2), label=label)
            if s["fall"] > 0:
                ax.annotate(f"{s['fall']:.0%} fall", (s["vx"], s["bob"]), fontsize=6,
                            xytext=(3, 4), textcoords="offset points")
        ax.set_title(f"target_velocity = {tv}")
        ax.set_xlabel("speed vx  ->better"); ax.grid(alpha=0.2)
    axes[0].set_ylabel("CoM bounce |vz|  better<-")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Quadruped +CoM across speeds: is ONE fixed spec fast+smooth+safe everywhere?\n"
                 "FPL is fixed (uniform); each linear bob-weight is fixed across all 3 panels — "
                 "the winner holds the lower-right corner at every speed", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "quadruped_com_speed_sweep.png", dpi=130)
    print(f"\nsaved -> {out_dir / 'quadruped_com_speed_sweep.png'}")


if __name__ == "__main__":
    main()
