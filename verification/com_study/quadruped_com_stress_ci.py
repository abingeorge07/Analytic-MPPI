"""Confirmatory high-seed + CI run of the ONE positive signal from quadruped_com_stress.py:
under SEVERE traction loss, does FPL's productive-speed edge over the best fixed linear weight
grow with the CoM atom (and is it real vs 12-seed noise)?

Focused on the severe-slip regime where the signal appeared. 24 seeds, 95% CIs (paired per
seed for the FPL-minus-best-linear margin, so the interaction can be judged against noise).
TV=3.0, aggressive + conservative linear weights, both arms.

Outputs: quadruped_com_stress_ci.png + .json. Run: python verification/com_study/quadruped_com_stress_ci.py
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

from analytic_mppi.eval import Config, run_study, init_barkour_stand, make_task
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _ci import mean_ci, wilson_ci  # noqa: E402

TV = 3.0
STEPS = 200
N_EPISODES = 24
SHARED = dict(num_samples=128, plan_horizon=0.3, num_knots=5, spline_type="zero")
NOISE, TEMP, TIME_P = 0.5, 0.2, -2.0
FALL_UP = 0.5
FRICTIONS = [0.2, 0.15, 0.1]
LIN_WV = [8.0, 16.0]
BOB = 4.0


def configs(com):
    def w(wv):
        return [1.0, 1.0, wv, 1.0, 1.0, BOB, 0.5] if com else [1.0, 1.0, wv, 1.0, 1.0, 0.5]
    lin = [Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=TEMP, fpl_weights=w(wv), fpl_time_p=TIME_P),
                  fpl_p=1.0) for wv in LIN_WV]
    return lin + [Config("FPL", "mppi", "fpl_cost",
                         dict(noise_level=NOISE, temperature=TEMP, fpl_time_p=TIME_P), fpl_p=-1.0)]


def prod_per_seed(res, task):
    sd = res["sd"]
    surv = task._torso_up(sd).min(axis=1) >= FALL_UP
    return task._torso_vel_x(sd).mean(axis=1) * surv, surv


def run_arm(task_name, com):
    task = make_task(task_name, target_velocity=TV)
    labels = [c.label for c in configs(com)]
    out = {}
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        study = run_study(task_name, configs(com), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_barkour_stand, task_kwargs=dict(target_velocity=TV),
                          true_perturbation=pert, progress=False, **SHARED)
        prods = {l: prod_per_seed(r, task) for l, r in study.items()}
        # best fixed linear per-seed is ambiguous; compare FPL vs EACH linear, paired.
        fpl_p = prods["FPL"][0]
        cell = {}
        for l in labels:
            p, s = prods[l]
            m, h = mean_ci(p)
            sp, slo, shi = wilson_ci(int(s.sum()), int(s.size))
            cell[l] = dict(prod=float(m), prod_ci=float(h), surv=float(sp))
        # paired margin FPL - best-linear-mean (per seed, best chosen by mean prod)
        best_lin = max(LIN_WV, key=lambda wv: prods[f"lin wv={wv:g}"][0].mean())
        dif = fpl_p - prods[f"lin wv={best_lin:g}"][0]
        dm, dh = mean_ci(dif)
        cell["_margin"] = dict(vs=f"lin wv={best_lin:g}", d=float(dm), ci=float(dh))
        out[fr] = cell
    return out


def main():
    arms = {"base": run_arm("quadruped", False), "com": run_arm("quadruped_com", True)}
    for arm, per_fr in arms.items():
        print(f"\n=== {arm} arm (TV={TV}, {N_EPISODES} seeds, 95% CI) ===")
        for fr in FRICTIONS:
            c = per_fr[fr]
            cells = "  ".join(f"{l.split()[-1] if l!='FPL' else 'FPL'}={c[l]['prod']:.2f}±{c[l]['prod_ci']:.2f}/{c[l]['surv']:.2f}"
                              for l in c if l != "_margin")
            m = c["_margin"]
            print(f" fr={fr:.2f}  {cells}   FPL-{m['vs']} margin={m['d']:+.2f}±{m['ci']:.2f}")

    out_dir = Path(__file__).resolve().parent
    out_dir.joinpath("quadruped_com_stress_ci_data.json").write_text(json.dumps(arms, indent=2, default=str))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for arm, style in [("base", dict(ls="--", marker="s")), ("com", dict(ls="-", marker="*"))]:
        d = [arms[arm][fr]["_margin"]["d"] for fr in FRICTIONS]
        e = [arms[arm][fr]["_margin"]["ci"] for fr in FRICTIONS]
        ax.errorbar(FRICTIONS, d, yerr=e, capsize=4, lw=2, ms=10,
                    label=f"{arm} arm  (FPL - best linear)", **style)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.invert_xaxis(); ax.grid(alpha=0.2)
    ax.set_xlabel("friction scale (->slippery)")
    ax.set_ylabel("FPL - best-linear productive speed  (>0 = FPL better)")
    ax.set_title(f"Is the severe-mismatch CoM interaction real? ({N_EPISODES} seeds, 95% CI)\n"
                 "com-arm margin above base-arm margin (both >0) = FPL edge grows with CoM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "quadruped_com_stress_ci.png", dpi=130)
    print(f"\nsaved -> {out_dir / 'quadruped_com_stress_ci.png'}")


if __name__ == "__main__":
    main()
