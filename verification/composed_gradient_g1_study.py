"""Headline comparison for ComposedGradientMPPI on g1_standup (layered FPL, J=3 groups).

Falsifiable claim (plan): under a fixed sample budget, worst-objective-first gradient
composition holds the WORST group's fulfillment (min_j s_j — esp. the orientation floor)
higher than the scalar-composite softmax-FPL baseline, at equal-or-higher ESS, without
regressing standup success (upright, height).

The per-group fulfillment metric here is recomputed from the realized trajectory via the
task's grouped-atom method, so it is INDEPENDENT of each controller's internals and
directly comparable across configs.

Run:  python verification/composed_gradient_g1_study.py
"""
import numpy as np

from analytic_mppi.eval import Config, run_study, g1_metrics, init_g1_stand
from analytic_mppi.tasks import make_task
from analytic_mppi.tasks.base import power_mean

TASK = "g1_standup"
COST_MODE = "fpl_layered"
FPL_P = 0.1
NOISE = 0.3
TEMP = 0.1          # baseline softmax temperature (composed power_p ignores it)

SHARED = dict(num_samples=128, plan_horizon=0.5, num_knots=4, spline_type="zero")
STEPS = 300
N_EPISODES = 5

CONFIGS = [
    Config("mppi softmax-FPL (baseline)", "mppi", COST_MODE,
           dict(noise_level=NOISE, temperature=TEMP, fpl_weighting="softmax"), fpl_p=FPL_P),
    Config("composed worst_first power_p", "composed_grad", COST_MODE,
           dict(noise_level=NOISE, temperature=TEMP, compose="worst_first",
                alpha_mode="power_p"), fpl_p=FPL_P),
    Config("composed uniform (ablation)", "composed_grad", COST_MODE,
           dict(noise_level=NOISE, temperature=TEMP, compose="uniform"), fpl_p=FPL_P),
    Config("composed worst_first softmax-b", "composed_grad", COST_MODE,
           dict(noise_level=NOISE, temperature=TEMP, compose="worst_first",
                alpha_mode="softmax", compose_temp=0.1), fpl_p=FPL_P),
]


def grouped_fulfillment(res, task, p=FPL_P):
    """Per-group [0,1] fulfillment at each realized state (controller-independent).

    Returns dict: per-group series (N_EP, T) under task.fpl_group_names, plus 'min'
    (worst group per step) and 'mean'.
    """
    states, sd, ctrls = res["states"], res["sd"], res["ctrls"]
    qpos, qvel = task.qpos_of(states)[:, :-1], task.qvel_of(states)[:, :-1]
    atoms = task.running_cost_terms_f_grouped(qpos, qvel, sd, ctrls)   # (N_EP, T, n_atoms)
    groups = [power_mean(atoms[..., idx], p) for idx in task.fpl_groups]  # each (N_EP, T)
    out = {name: g for name, g in zip(task.fpl_group_names, groups)}
    stacked = np.stack(groups, axis=-1)                                # (N_EP, T, n_groups)
    out["min"] = stacked.min(axis=-1)
    out["mean"] = stacked.mean(axis=-1)
    return out


def main():
    task = make_task(TASK)
    study = run_study(TASK, CONFIGS, steps=STEPS, n_episodes=N_EPISODES,
                      init_fn=init_g1_stand, **SHARED)

    gnames = task.fpl_group_names
    cols = ["min_fulfil", *gnames, "upright", "height_err", "ess"]
    print("\n" + "=" * 96)
    print(f"g1_standup | layered FPL J={len(gnames)} {gnames} | "
          f"K={SHARED['num_samples']} steps={STEPS} eps={N_EPISODES}")
    print("=" * 96)
    print(f"{'config':32s}" + "".join(f"{c:>12s}" for c in cols))
    for label, res in study.items():
        gf = grouped_fulfillment(res, task)
        gm = g1_metrics(res, task)
        row = [gf["min"].mean(), *[gf[n].mean() for n in gnames],
               gm["upright"].mean(), gm["height_err"].mean(),
               np.nanmean(res["ess"])]
        print(f"{label:32s}" + "".join(f"{v:12.4f}" for v in row))
    print("=" * 96)
    print("Claim holds if 'composed worst_first power_p' has higher min_fulfil (esp. the\n"
          "worst group) than baseline at >= baseline ess, with upright~1 / low height_err.")


if __name__ == "__main__":
    main()
