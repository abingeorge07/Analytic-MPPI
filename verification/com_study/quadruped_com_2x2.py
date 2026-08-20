"""Quadruped CoM interaction study: does adding a CoM-straightness (vertical anti-bob) atom
let FPL beat the linear-weight family MORE than it does without it?

Framing (mirrors verification/quadruped_pareto_sweep.py's fairness protocol: identical
sampler / budget / horizon / temporal weakest-link; only the objective COMPOSITION differs):

  * -CoM (base 6-atom quadruped): the smoothness objective is ABSENT. FPL's only edge here is
    safety-at-speed. Sweep the linear velocity weight; FPL is one uniform point. Control.
  * +CoM (7-atom quadruped_com): the anti-bob atom is PRESENT and competes with speed (fast
    gaits pogo). Sweep the linear BOB weight to trace linear's speed-vs-bob front; FPL is one
    uniform point. If FPL+CoM sits BELOW/RIGHT of the linear+CoM front (lower bob at equal
    speed, or faster at equal bob), no single linear weight matches it -> FPL resolves the
    new competition better. That growth of FPL's edge when CoM is added IS the interaction.

Metric plane: forward speed vx (x, higher better) vs CoM bounce |vz| (y, LOWER better); fall
rate annotated. Linear = fpl_cost p=1 + weights; FPL = fpl_cost p=-1 uniform.

Outputs: quadruped_com_2x2.png (+ .json). Run: python verification/com_study/quadruped_com_2x2.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from analytic_mppi.eval import run_study, Config, init_barkour_stand, make_task

TV = 2.0
STEPS = 250
N_EPISODES = 8
SHARED = dict(num_samples=128, plan_horizon=0.3, num_knots=5, spline_type="zero")
NOISE, TEMP, TIME_P = 0.5, 0.2, -2.0
FALL_UP = 0.5
VW = 4.0                       # fixed velocity weight for the +CoM linear sweep
LIN_VW = [1.0, 4.0, 16.0]      # -CoM linear velocity-weight sweep
LIN_BOB = [0.0, 2.0, 4.0, 8.0, 16.0]   # +CoM linear bob-weight sweep


def _lin(weights):
    return dict(noise_level=NOISE, temperature=TEMP, fpl_weights=weights, fpl_time_p=TIME_P)


def _fpl():
    return dict(noise_level=NOISE, temperature=TEMP, fpl_time_p=TIME_P)


def configs_base():
    # atom order (6): [height, orient, vel, heading, posture, control]
    lin = [Config(f"lin vw={vw:g}", "mppi", "fpl_cost",
                  _lin([1.0, 1.0, vw, 1.0, 1.0, 0.5]), fpl_p=1.0) for vw in LIN_VW]
    return lin + [Config("FPL", "mppi", "fpl_cost", _fpl(), fpl_p=-1.0)]


def configs_com():
    # atom order (7): [height, orient, vel, heading, posture, com_bob, control]
    lin = [Config(f"lin b={b:g}", "mppi", "fpl_cost",
                  _lin([1.0, 1.0, VW, 1.0, 1.0, b, 0.5]), fpl_p=1.0) for b in LIN_BOB]
    return lin + [Config("FPL", "mppi", "fpl_cost", _fpl(), fpl_p=-1.0)]


def stats(res, task):
    sd = res["sd"]
    vadr = task._vel_adr
    g = slice(60, None)
    vx = sd[:, g, vadr].mean(axis=1)                 # per-episode forward speed
    bob = np.abs(sd[:, g, vadr + 2]).mean(axis=1)    # per-episode CoM bounce
    fell = task._torso_up(sd).min(axis=1) < FALL_UP
    return dict(vx=float(vx.mean()), vx_sd=float(vx.std()),
                bob=float(bob.mean()), bob_sd=float(bob.std()), fall=float(fell.mean()))


def run_arm(task_name, cfgs):
    task = make_task(task_name, target_velocity=TV)
    study = run_study(task_name, cfgs, steps=STEPS, n_episodes=N_EPISODES,
                      init_fn=init_barkour_stand, task_kwargs=dict(target_velocity=TV),
                      progress=False, **SHARED)
    return {label: stats(r, task) for label, r in study.items()}


def main():
    base = run_arm("quadruped", configs_base())
    com = run_arm("quadruped_com", configs_com())

    for name, arm in [("-CoM (base 6-atom)", base), ("+CoM (7-atom)", com)]:
        print(f"\n=== {name} ===   target_velocity={TV}")
        print(f"{'config':12s}{'vx':>8s}{'bob|vz|':>10s}{'fall':>7s}")
        for label, s in arm.items():
            print(f"{label:12s}{s['vx']:8.2f}{s['bob']:10.3f}{s['fall']:7.0%}")

    out_dir = Path(__file__).resolve().parent
    out_dir.joinpath("quadruped_com_2x2_data.json").write_text(
        json.dumps({"base": base, "com": com}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (title, arm) in zip(axes, [("-CoM (base): no smoothness objective", base),
                                        ("+CoM: anti-bob competes with speed", com)]):
        lin = {k: v for k, v in arm.items() if k != "FPL"}
        f = arm["FPL"]
        lx = [v["vx"] for v in lin.values()]; ly = [v["bob"] for v in lin.values()]
        order = np.argsort(lx)
        ax.plot(np.array(lx)[order], np.array(ly)[order], "o-", color="#1f77b4",
                label="linear family", zorder=2)
        for label, v in lin.items():
            ax.annotate(label, (v["vx"], v["bob"]), fontsize=7, color="#1f77b4",
                        xytext=(3, 3), textcoords="offset points")
        ax.scatter([f["vx"]], [f["bob"]], marker="*", s=320, color="#d62728",
                   edgecolor="k", zorder=3, label="FPL (uniform)")
        ax.annotate(f"fall {f['fall']:.0%}", (f["vx"], f["bob"]), fontsize=7,
                    color="#d62728", xytext=(5, -10), textcoords="offset points")
        ax.set_xlabel("forward speed vx (m/s)  ->better")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("CoM bounce |vz| (m/s)  better<-")
    fig.suptitle("Quadruped CoM interaction: FPL vs linear family in the speed-smoothness plane "
                 f"({N_EPISODES} seeds)\nlower-right = fast & smooth; FPL beyond the linear front on +CoM = the win",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "quadruped_com_2x2.png", dpi=130)
    print(f"\nsaved -> {out_dir / 'quadruped_com_2x2.png'}")


if __name__ == "__main__":
    main()
