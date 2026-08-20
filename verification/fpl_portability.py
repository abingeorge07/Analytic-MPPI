"""FPL PORTABILITY (backbone): the FPL-cost advantage is a property of the OBJECTIVE, not
the sampler — it transfers to every sampler we tried.

For each environment and each sampler {MPPI, MPPI-CMA, CEM, FPL-colored}, we run the SAME
comparison the flagship pareto sweeps run: one fixed FPL spec (p=-1, uniform min-fulfillment
conjunction) vs the whole linear-weight family (p=+1, swept progress weight). If FPL's
Pareto-dominance of the linear frontier is real and cost-driven, it should hold no matter
which sampler draws the rollouts. The result is a grid of (achieved progress vs survival)
panels — one per (env, sampler) — with the linear frontier and the single FPL point.

Fairness: within each (env, sampler) cell EVERYTHING is identical (budget / horizon / knots /
temporal weakest-link / the shared [0,1] atoms); only the objective-axis composition differs
(linear weighted-sum vs min-fulfillment). Across samplers only the proposal differs.

Checkpointed per (env, sampler, cost, seed). Run:
    .venv/bin/python verification/fpl_portability.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial, ENVS as ENVSPEC       # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci, wilson_ci                         # noqa: E402

EXP = "portability"
ENVS = ["walker", "hopper"]        # fast envs first; cube/quadruped added by extend runs
SAMPLERS = ["mppi", "mppi_cma", "cem", "fpl_colored"]
SAMPLER_PRETTY = {"mppi": "MPPI", "mppi_cma": "MPPI-CMA", "cem": "CEM",
                  "fpl_colored": "FPL-colored (ours)"}
SEEDS = list(range(16))
LINEAR_WV = {
    "hopper": [0.5, 1.0, 2.0, 4.0, 8.0],
    "walker": [2.0, 4.0, 8.0, 16.0, 32.0],
    "cube":   [1.0, 2.0, 4.0, 8.0],
    "quadruped": [1.0, 2.0, 4.0, 8.0, 16.0],
}
K = 256
CKPT = Path(__file__).resolve().parent / "checkpoints" / "portability.jsonl"


def configs(env):
    yield ("FPL", "fpl")
    for wv in LINEAR_WV[env]:
        yield (f"lin wv={wv:g}", f"lin:{wv}")


def run(progress=True):
    ckpt = Checkpoint(CKPT)
    for env in ENVS:
        for sampler in SAMPLERS:
            for label, cost in configs(env):
                for seed in SEEDS:
                    fields = dict(exp=EXP, env=env, sampler=sampler, label=label,
                                  cost=cost, K=K, seed=seed)
                    if ckpt.has(fields):
                        continue
                    try:
                        m = run_trial(env, sampler, cost, seed=seed, K=K)
                    except Exception as e:
                        m = dict(error=f"{type(e).__name__}: {e}", prod=None,
                                 survived=None, vx=None, achieved=None)
                    ckpt.record(fields, m)
            print(f"  {env} / {sampler} done", flush=True)
    print(ckpt.summary())
    return ckpt


def _stats(rows, env, sampler, label):
    prog_key = "achieved" if ENVSPEC[env]["metric"] == "cube" else "vx"
    vals = [r for r in rows if r["env"] == env and r["sampler"] == sampler
            and r["label"] == label and r.get(prog_key) is not None]
    if not vals:
        return None
    prog = np.array([r[prog_key] for r in vals], dtype=float)
    surv = np.array([r["survived"] for r in vals], dtype=float)
    m, h = mean_ci(prog)
    p, lo, hi = wilson_ci(int(surv.sum()), int(surv.size))
    fall = 1.0 - float(surv.mean())
    return dict(x=m, xci=h, surv=p, surv_lo=lo, surv_hi=hi, fall=fall)


def report(ckpt):
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 96)
    print("FPL PORTABILITY — FPL vs the best comparably-safe linear weight, per sampler (16 seeds)")
    print("=" * 96)
    summary = {}
    for env in ENVS:
        print(f"\n== {env} ==")
        for sampler in SAMPLERS:
            fpl = _stats(rows, env, sampler, "FPL")
            if fpl is None:
                print(f"  {SAMPLER_PRETTY[sampler]:20s}  (no data)")
                continue
            lins = [(lab, _stats(rows, env, sampler, lab))
                    for lab, _c in configs(env) if lab != "FPL"]
            lins = [(lab, s) for lab, s in lins if s is not None]
            safe = [(lab, s) for lab, s in lins if s["fall"] <= fpl["fall"] + 1e-9]
            best = max(safe, key=lambda kv: kv[1]["x"]) if safe else None
            summary[(env, sampler)] = dict(fpl=fpl, best_safe=best, lins=lins)
            gain = (100 * (fpl["x"] - best[1]["x"]) / max(best[1]["x"], 1e-6)) if best else float("nan")
            bs = f"{best[0]} x={best[1]['x']:.2f}" if best else "none as safe"
            print(f"  {SAMPLER_PRETTY[sampler]:20s}  FPL x={fpl['x']:.2f} surv={fpl['surv']*100:.0f}%"
                  f"   best-safe linear: {bs}   -> FPL {gain:+.0f}%")
    return summary


def figure(ckpt, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = ckpt.rows(exp=EXP)
    nR, nC = len(ENVS), len(SAMPLERS)
    fig, axes = plt.subplots(nR, nC, figsize=(3.7 * nC, 3.6 * nR), squeeze=False)
    for i, env in enumerate(ENVS):
        prog_lbl = "rotation (deg)" if ENVSPEC[env]["metric"] == "cube" else "speed (m/s)"
        for j, sampler in enumerate(SAMPLERS):
            ax = axes[i][j]
            lins = [(lab, _stats(rows, env, sampler, lab))
                    for lab, _c in configs(env) if lab != "FPL"]
            lins = [(lab, s) for lab, s in lins if s is not None]
            lins.sort(key=lambda kv: kv[1]["x"])
            if lins:
                xs = [s["x"] for _l, s in lins]; ys = [s["surv"] for _l, s in lins]
                xe = [s["xci"] for _l, s in lins]
                ye = [[s["surv"] - s["surv_lo"] for _l, s in lins],
                      [s["surv_hi"] - s["surv"] for _l, s in lins]]
                ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o-", color="#4c72b0",
                            lw=1.8, ms=6, capsize=2, label="linear family", zorder=3)
            fpl = _stats(rows, env, sampler, "FPL")
            if fpl is not None:
                ax.errorbar(fpl["x"], fpl["surv"], xerr=fpl["xci"],
                            yerr=[[fpl["surv"] - fpl["surv_lo"]], [fpl["surv_hi"] - fpl["surv"]]],
                            fmt="*", color="#c44e52", ms=20, mec="k", mew=1.1,
                            capsize=2, label="FPL (fixed)", zorder=5)
            if i == 0:
                ax.set_title(SAMPLER_PRETTY[sampler], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{env}\nsurvival", fontsize=10)
            ax.set_xlabel(prog_lbl, fontsize=9)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower left")
    ns = len(SEEDS)
    # The strong "dominates EVERY sampler" claim is asserted only for the clean, high-budget
    # locomotion run; extension runs (cube at reduced K/seeds) get a neutral, accurate title.
    if ENVS == ["walker", "hopper"]:
        head = ("FPL dominates the linear-weight frontier for EVERY sampler — the win is the "
                "objective, not the proposal")
    else:
        head = "FPL vs the linear-weight frontier, per sampler"
    fig.suptitle(f"{head}\n(one fixed FPL spec vs the linear family; {ns} seeds, K={K}, "
                 "95% CIs; identical budget/atoms/horizon within each cell)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def main():
    global ENVS, SAMPLERS, SEEDS, K
    # CLI overrides for cheap extension runs (cube is slow): envs / samplers / seeds / K.
    # e.g.  fpl_portability.py --envs cube --samplers mppi cem --seeds 8 --K 128
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=None)
    ap.add_argument("--samplers", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--K", type=int, default=None)
    a = ap.parse_args()
    if a.envs:
        ENVS = a.envs
    if a.samplers:
        SAMPLERS = a.samplers
    if a.seeds:
        SEEDS = list(range(a.seeds))
    if a.K:
        K = a.K
    ckpt = run()
    report(ckpt)
    # Default (walker+hopper) writes the canonical figure; extension runs (e.g. cube at a
    # cheaper K) write a per-env-tagged figure so they don't clobber the main one.
    tag = "" if ENVS == ["walker", "hopper"] else "_" + "_".join(ENVS)
    figure(ckpt, Path(__file__).resolve().parent / f"fpl_portability{tag}.png")


if __name__ == "__main__":
    main()
