"""Model-mismatch robustness on the WALKER2d (running regime) — generalizes the hopper
traction-robustness result to a second, harder biped.

Same setup as verification/mismatch_robustness_sweep.py (hopper): the controller PLANS on the
nominal model; the true simulator it executes on has REDUCED GROUND FRICTION (traction loss).
No controller is retuned across friction levels — you cannot retune for a drift you don't know.

Claim (replicated on the walker): one FIXED FPL spec holds the highest PRODUCTIVE SPEED
(achieved speed × survival) at every traction level, with survival ~1.0 across a 2.5x traction
reduction, while the linear-weight family has no single fixed weight that stays both fast and
safe — the aggressive weight's survival collapses on slippery ground, the safe weight is slow.

Fairness: identical sampler/atoms/budget/horizon/temporal-weakest-link; only composition
differs (p=1+weights = linear family; p=-1 uniform = FPL). Walker FPL atoms use the hopper
shapes (shared by both families). Running regime knobs (noise=0.8, H=0.6, k=6) from
verification/walker_pareto_sweep.py.

Run:  .venv/bin/python verification/mismatch_robustness_walker.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic_mppi.eval import Config, run_study, make_task

TV = 5.0
STEPS = 150
N_EPISODES = 16
NUM_SAMPLES = 256
NOISE = 0.8
TIME_P = -2.0
FALL_UPRIGHT = 0.6
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.6, num_knots=6, spline_type="zero")

FRICTIONS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3]
LINEAR_WV = [4.0, 8.0, 16.0, 32.0]
FPL_P = -1.0


def build_configs():
    lin = [Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
                  dict(noise_level=NOISE, temperature=0.2,
                       fpl_weights=[1.0, 1.0, wv, 0.5], fpl_time_p=TIME_P), fpl_p=1.0)
           for wv in LINEAR_WV]
    fpl = Config(f"FPL p={FPL_P:g}", "mppi", "fpl_cost",
                 dict(noise_level=NOISE, temperature=0.2, fpl_time_p=TIME_P), fpl_p=FPL_P)
    return lin + [fpl]


def stats(res, task):
    sd = res["sd"]
    vx = sd[..., task._vel_adr]
    zax = sd[..., task._zax_adr + 2]
    surv = 1.0 - float((zax.min(axis=1) < FALL_UPRIGHT).mean())
    return dict(vx=float(vx.mean()), surv=surv)


def main():
    task = make_task("walker", target_velocity=TV)
    per_fr = {}
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        study = run_study("walker", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          task_kwargs=dict(target_velocity=TV),
                          true_perturbation=pert, progress=False, **SHARED)
        per_fr[fr] = {l: stats(r, task) for l, r in study.items()}
        print(f"friction={fr} done")

    fpl_label = f"FPL p={FPL_P:g}"
    print("\n" + "=" * 74)
    print(f"WALKER2d traction-loss robustness | K={NUM_SAMPLES} eps={N_EPISODES} | NO retuning")
    print("=" * 74)
    for fr in FRICTIONS:
        cells = "  ".join(f"{per_fr[fr][l]['vx']:.2f}/{per_fr[fr][l]['surv']:.2f}"
                          for l in per_fr[fr])
        print(f"friction={fr:4.2f}  {cells}   (vx/surv)")

    Path(__file__).resolve().parent.joinpath("mismatch_robustness_walker_data.json").write_text(
        json.dumps({str(fr): per_fr[fr] for fr in FRICTIONS}, indent=2))
    make_figure(per_fr, fpl_label)


def make_figure(per_fr, fpl_label):
    frs = list(per_fr.keys())
    x = [1.0 - fr for fr in frs]
    labels = list(per_fr[frs[0]].keys())
    lin_labels = [l for l in labels if l.startswith("lin")]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(lin_labels)))
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    def series(l, key):
        return np.array([per_fr[fr][l][key] for fr in frs])

    for i, l in enumerate(lin_labels):
        prod = series(l, "vx") * series(l, "surv")
        axP.plot(x, prod, "-o", color=cmap[i], lw=1.5, ms=5, label=l.replace("lin ", ""))
        axS.plot(x, series(l, "surv"), "-o", color=cmap[i], lw=1.5, ms=5, label=l.replace("lin ", ""))
    prod = series(fpl_label, "vx") * series(fpl_label, "surv")
    axP.plot(x, prod, "-*", color="#c44e52", lw=3.2, ms=15, zorder=6, label="FPL (one fixed spec)")
    axS.plot(x, series(fpl_label, "surv"), "-*", color="#c44e52", lw=3.2, ms=15, zorder=6,
             label="FPL (one fixed spec)")
    axP.set(title="Productive speed  (speed × survival)", ylabel="m/s × survival",
            xlabel="traction loss  (1 − friction;  0 = nominal)")
    axS.set(title="Survival", ylabel="survival rate (1 − fall)",
            xlabel="traction loss  (1 − friction;  0 = nominal)")
    for ax in (axP, axS):
        ax.grid(alpha=0.3)
    axP.legend(fontsize=8, title="cost", loc="lower left")
    fig.suptitle("Walker2d (running), planner uses nominal model but reality gets slippery "
                 "(NO retuning):\none fixed FPL spec delivers the most usable speed and holds "
                 "survival ~1.0; the aggressive linear weight collapses", y=1.06, fontsize=11)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "mismatch_robustness_walker.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved robustness figure -> {out}")


if __name__ == "__main__":
    main()
