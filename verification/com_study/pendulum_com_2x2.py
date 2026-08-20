"""±CoM x ±FPL 2x2 on the pendulum: a sanity check for the interaction-effect harness.

Four cells, run at TWO commanded angles from hang-down:

              linear (normal)      FPL (fpl_discounted, p<0)
  angle-cost  (-CoM, -FPL)         (-CoM, +FPL)     task = "pendulum"
  CoM-cost    (+CoM, -FPL)         (+CoM, +FPL)     task = "pendulum_com"

  * CoM axis  = objective encoded in ANGLE space (original task) vs CoM-POSITION space.
  * FPL axis  = linear scalarization (p=1 / normal) vs power-mean CONJUNCTION (p<0).

Two targets, which (with a correctly-configured FPL) separate FPL's two REGIMES rather than
"null vs not". FPL here uses the FLOOR-shaped control atom (PendulumTask._control_fulfillment)
so every atom reads ~1 at the held pose, and a temperature matched to the [0,1] fulfillment
scale (a normal-cost temperature makes the -log(reward) softmax ~uniform -> optimizer stalls).
With that:
  * UPRIGHT (180 deg): the com objective is FAR from satisfied at the hang-down start
    (com_fulfillment ~ 0 -> full reward headroom). FPL drives hard and REACHES -> FPL ~= linear,
    the clean NULL (validates the harness: no manufactured win on a non-competing task).
  * HOLDABLE (20 deg): the com objective is ALREADY ~0.83 satisfied at the start (headroom only
    ~0.17). To reach it the controller must take a control transient whose fulfillment dip
    exceeds that tiny headroom, so the min-conjunction won't "pay" for it -> FPL underperforms.
    This is intrinsic bounded-reward SATURATION, not misconfiguration (goal-state atoms = [1,1]).
    It is the same reason FPL washes on easy / nearly-satisfied tasks and wins only when the
    objective is genuinely unsatisfied (the thesis), reproduced on the simplest system.

Common yardstick (task-independent): settle error = wrap-aware mean |theta - theta*| over
the last 0.5 s, in degrees. Improvement := err(linear) - err(FPL) per CoM condition (+ve ->
FPL settles closer). Interaction := improvement(+CoM) - improvement(-CoM).

The harness is validated if it reads ~0 interaction on the holdable (no-competition) target
and a real, signed effect on the swing-up target. The genuinely non-zero, FPL-FAVOURING
interaction belongs on the quadruped, where "go fast" competes with "keep the CoM straight".

Outputs: verification/pendulum_com_2x2.png (+ .mp4 of the upright CoM swing-up).
Run:     python verification/pendulum_com_2x2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from analytic_mppi.eval import run_episode, render_video, init_hang_down


TARGETS = [("holdable (20 deg)", np.deg2rad(20.0)),
           ("upright (180 deg)", np.pi)]
N_SEEDS = 6
STEPS = 200             # 200 * 0.02 s = 4 s
DT = 0.02
TAIL = 25               # last 0.5 s used to judge the settle

# Sampler settings shared across cells. FPL scores are -log(reward) (a different scale from
# the quadratic normal cost), so each mode gets its own temperature — otherwise the softmax
# is unfairly sharp/flat for one of them. A fairness choice, not a tuned advantage. FPL temp
# is set to the fulfillment scale (0.05); at the normal-cost temp (1.0) the FPL softmax is
# ~uniform (ESS ~ K) and the optimizer stalls — see the guardrail memo.
COMMON = dict(num_samples=256, num_knots=8, plan_horizon=1.0, spline_type="zero",
              noise_level=0.6)
FPL = dict(cost_mode="fpl_discounted", fpl_p=-2.0, fpl_gamma=0.99, temperature=0.05)
LIN = dict(cost_mode="normal", temperature=0.1)

CELLS = [
    dict(key="linear . angle", task="pendulum",     com=False, **LIN),
    dict(key="FPL . angle",    task="pendulum",     com=False, **FPL),
    dict(key="linear . CoM",   task="pendulum_com", com=True,  **LIN),
    dict(key="FPL . CoM",      task="pendulum_com", com=True,  **FPL),
]
COLORS = {"linear . angle": "#1f77b4", "FPL . angle": "#ff7f0e",
          "linear . CoM": "#2ca02c", "FPL . CoM": "#d62728"}


def _wrap_err(theta: np.ndarray, target: float) -> np.ndarray:
    """Wrap-aware |theta - target| in radians."""
    return np.abs(((theta - target + np.pi) % (2.0 * np.pi)) - np.pi)


def _run_cell(cell: dict, target: float) -> dict:
    build = {k: v for k, v in cell.items() if k not in ("key", "task", "com", "cost_mode")}
    task_kwargs = dict(target_angle=float(target))
    errs_t = []          # wrap-error time series per seed (deg)
    for seed in range(N_SEEDS):
        res = run_episode(
            cell["task"], "mppi", steps=STEPS, seed=seed, cost_mode=cell["cost_mode"],
            init_fn=init_hang_down, task_kwargs=task_kwargs, **COMMON, **build,
        )
        errs_t.append(np.rad2deg(_wrap_err(res["states"][:, 1], target)))
    errs_t = np.asarray(errs_t)                        # (seeds, T+1)
    settle = errs_t[:, -TAIL:].mean(axis=1)            # (seeds,)
    return dict(key=cell["key"], com=cell["com"], errs_t=errs_t,
                err_mean=float(settle.mean()), err_std=float(settle.std()),
                success=float((settle < 10.0).mean()))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(__file__).resolve().parent
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(12, 4.2 * len(TARGETS)))
    t = np.arange(STEPS + 1) * DT

    for row, (tname, target) in enumerate(TARGETS):
        results = {c["key"]: _run_cell(c, target) for c in CELLS}
        err = {k: r["err_mean"] for k, r in results.items()}
        imp_noCoM = err["linear . angle"] - err["FPL . angle"]
        imp_CoM = err["linear . CoM"] - err["FPL . CoM"]
        interaction = imp_CoM - imp_noCoM

        print(f"\n=== target: {tname} ===")
        print(f"{'cell':16s} {'settle err (deg)':>18s} {'success':>9s}")
        print("-" * 46)
        for k, r in results.items():
            print(f"{k:16s} {r['err_mean']:10.1f} +/-{r['err_std']:4.1f} {r['success']:9.0%}")
        print(f"improvement (FPL-linear) no-CoM: {imp_noCoM:+.1f} | +CoM: {imp_CoM:+.1f} "
              f"| INTERACTION: {interaction:+.1f} deg")

        # left: wrap-error over time (unambiguous — mean-theta hides the +/- swing-up split)
        axL, axR = axes[row]
        for k, r in results.items():
            mu, sd = r["errs_t"].mean(axis=0), r["errs_t"].std(axis=0)
            axL.plot(t, mu, color=COLORS[k], lw=1.6, label=k)
            axL.fill_between(t, np.clip(mu - sd, 0, None), mu + sd, color=COLORS[k],
                             alpha=0.12, lw=0)
        axL.axhline(0.0, ls="--", color="k", lw=1.0, alpha=0.5)
        axL.set_xlabel("time (s)"); axL.set_ylabel("|theta - theta*| (deg)")
        axL.set_title(f"{tname}: error to target (mean +/- std)")
        if row == 0:
            axL.legend(fontsize=8, loc="upper right")

        # right: settle-error bars
        keys = list(results.keys())
        xs = np.arange(len(keys))
        axR.bar(xs, [results[k]["err_mean"] for k in keys],
                yerr=[results[k]["err_std"] for k in keys],
                color=[COLORS[k] for k in keys], alpha=0.85, capsize=4)
        axR.set_xticks(xs); axR.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
        axR.set_ylabel("settle error (deg)")
        axR.set_title(f"{tname}: interaction = {interaction:+.1f} deg")

    fig.tight_layout()
    fig_path = out_dir / "pendulum_com_2x2.png"
    fig.savefig(fig_path, dpi=130)
    print(f"\nsaved figure -> {fig_path}")

    # video: the CoM cell reaching UPRIGHT (the swing-up the linear-CoM controller solves).
    vid_path = out_dir / "pendulum_com_upright.mp4"
    render_video(
        "pendulum_com", "mppi", steps=STEPS, out_path=vid_path, seed=0,
        cost_mode="normal", init_fn=init_hang_down, camera="camera",
        task_kwargs=dict(target_angle=float(np.pi)), temperature=0.1, **COMMON,
    )
    print(f"saved video  -> {vid_path}  (cell: linear . CoM @ upright)")


if __name__ == "__main__":
    main()
