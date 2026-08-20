"""FPL Shielded MPPI — does a hard lexicographic safety filter beat the soft min-fulfillment cost?

The distinctly-FPL idea (#2): use the per-objective bounded fulfillment to enforce SAFETY as a
hard constraint (min over-time safety ≥ τ, optional terminal safety ≥ τ_T) and optimize among the
certified-safe rollouts — something a scalar weighted cost cannot express. Tested where it should
pay off: pushing performance aggressively (nominal, rising target speed) and under model mismatch
(traction loss). Compared against plain soft FPL (min-fulfillment, the campaign's win) on the
hopper, no retuning, 16 seeds. Checkpointed.

Shield variants:
  shield-perf   : maximize the performance atom among the safe set (greedy — boundary-seeking)
  shield-bal    : keep the soft min-fulfillment composite among the safe set (hard safety, no
                  boundary-seeking) — the sensible form.

Run:  .venv/bin/python verification/fpl_shield_study.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci, wilson_ci                         # noqa: E402

EXP = "shield"
ENV = "hopper"
K = 256
SEEDS = list(range(16))
S = dict(safety_indices=[0, 1], perf_indices=[2])          # hopper: safety=height,orient; perf=vel
CONFIGS = [
    ("plain FPL", "mppi", {}),
    ("shield-bal τ.3", "fpl_shielded", dict(**S, safety_floor=0.3, perf_mode="balanced")),
    ("shield-bal τ.4 t.3", "fpl_shielded",
     dict(**S, safety_floor=0.4, terminal_safety_floor=0.3, perf_mode="balanced")),
    ("shield-perf τ.4 t.4", "fpl_shielded",
     dict(**S, safety_floor=0.4, terminal_safety_floor=0.4, perf_mode="performance")),
]
# regime A: nominal, rising aggression (target speed). regime B: fixed tv=2.5, rising traction loss.
AGGR = [2.5, 3.0, 3.5]
FRICT = [1.0, 0.6, 0.45, 0.35]
TV_MISMATCH = 2.5
CKPT = Path(__file__).resolve().parent / "checkpoints" / "shield.jsonl"


def run():
    ckpt = Checkpoint(CKPT)
    trials = ([("aggr", tv, None) for tv in AGGR]
              + [("mismatch", TV_MISMATCH, fr) for fr in FRICT])
    for regime, tv, fr in trials:
        pert = None if not fr else dict(friction_scale=fr)
        for label, name, extra in CONFIGS:
            for seed in SEEDS:
                fields = dict(exp=EXP, regime=regime, tv=tv, fr=(fr or 1.0), label=label, seed=seed)
                if ckpt.has(fields):
                    continue
                try:
                    m = run_trial(ENV, name, "fpl", seed=seed, K=K, difficulty=tv,
                                  true_perturbation=pert, extra=dict(extra))
                except Exception as e:
                    m = dict(error=f"{type(e).__name__}: {e}", prod=None, survived=None)
                ckpt.record(fields, m)
            print(f"  {regime} tv={tv} fr={fr} {label} done", flush=True)
    print(ckpt.summary())
    return ckpt


def _cell(rows, **filt):
    v = [r for r in rows if all(r.get(k) == val for k, val in filt.items())
         and r.get("prod") is not None]
    if not v:
        return None
    prod = np.array([r["prod"] for r in v]); surv = np.array([r["survived"] for r in v])
    vx = np.array([r["vx"] for r in v])
    m, h = mean_ci(prod); p, lo, hi = wilson_ci(int(surv.sum()), int(surv.size))
    return dict(prod=m, prod_ci=h, surv=p, surv_lo=lo, surv_hi=hi, vx=float(vx.mean()))


def report_and_figure(ckpt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 90)
    print("FPL SHIELD — productive speed / survival (hopper, 16 seeds, no retuning)")
    print("=" * 90)
    for regime, xs, xkey, xlabel in [("aggr", AGGR, "tv", "target speed"),
                                     ("mismatch", FRICT, "fr", "friction")]:
        print(f"\n-- {regime} --   ({xlabel}: {xs})")
        for label, _n, _e in CONFIGS:
            cells = []
            for x in xs:
                filt = dict(regime=regime, label=label)
                filt[xkey] = x if regime == "aggr" else (x or 1.0)
                c = _cell(rows, **filt)
                cells.append(f"{c['prod']:.2f}/{c['surv']*100:3.0f}%" if c else "  -  ")
            print(f"  {label:20s} " + "  ".join(f"{c:>11s}" for c in cells))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {"plain FPL": "#4c72b0", "shield-bal τ.3": "#c44e52",
              "shield-bal τ.4 t.3": "#dd8452", "shield-perf τ.4 t.4": "#8172b3"}
    # panel A: aggression (x=target speed), panel B: mismatch (x=traction loss)
    for ax, regime, xs, xkey, xlab in [(axes[0], "aggr", AGGR, "tv", "target speed (m/s)"),
                                       (axes[1], "mismatch", FRICT, "fr", "traction loss (1−friction)")]:
        for label, _n, _e in CONFIGS:
            ys, es = [], []
            for x in xs:
                filt = dict(regime=regime, label=label)
                filt[xkey] = x if regime == "aggr" else (x or 1.0)
                c = _cell(rows, **filt)
                ys.append(c["prod"] if c else np.nan); es.append(c["prod_ci"] if c else 0)
            xv = xs if regime == "aggr" else [1.0 - f for f in xs]
            sh = label != "plain FPL"
            ax.errorbar(xv, ys, yerr=es, fmt="-*" if sh else "-o", color=colors[label],
                        lw=2.4 if sh else 1.8, ms=11 if sh else 6, capsize=2, label=label)
        ax.set(xlabel=xlab, ylabel="productive speed (m/s × survival)")
        ax.grid(alpha=0.3)
    axes[0].set_title("Rising aggression (nominal)")
    axes[1].set_title("Rising traction loss (tv=2.5)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("FPL hard-safety SHIELD vs soft min-fulfillment (hopper): does a lexicographic "
                 "safety filter\nbeat the soft cost when pushed hard or under mismatch?", y=1.03,
                 fontsize=10)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fpl_shield.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out}")


def main():
    ckpt = run()
    report_and_figure(ckpt)


if __name__ == "__main__":
    main()
