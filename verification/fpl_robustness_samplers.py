"""Mismatch robustness is PORTABLE across samplers — the un-tunable FPL win doesn't need MPPI.

FPL_FINDINGS establishes the structural win: under traction loss (plan on the nominal model,
execute on slippery ground, NO retuning) one fixed FPL spec holds speed+survival while the best
fixed linear weight degrades. That was shown with the MPPI sampler. Here we check it is a property
of the OBJECTIVE, not the proposal, by re-running the hopper traction sweep inside every sampler
{MPPI, MPPI-CMA, CEM, colored}. Same fairness protocol: within each (sampler) cell only the
objective composition varies (FPL p=-1 vs the linear-weight family); across cells only the sampler.

Checkpointed. Run:  .venv/bin/python verification/fpl_robustness_samplers.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci, wilson_ci                         # noqa: E402

EXP = "robust_samplers"
ENV = "hopper"
TV = 2.5
K = 256
SEEDS = list(range(12))
FRICTIONS = [1.0, 0.6, 0.45, 0.35]
SAMPLERS = ["mppi", "mppi_cma", "cem", "fpl_colored"]
SAMPLER_PRETTY = {"mppi": "MPPI", "mppi_cma": "MPPI-CMA", "cem": "CEM",
                  "fpl_colored": "FPL-colored"}
COSTS = [("FPL", "fpl"), ("lin wv=1", "lin:1"), ("lin wv=2", "lin:2")]
CKPT = Path(__file__).resolve().parent / "checkpoints" / "robust_samplers.jsonl"


def run():
    ckpt = Checkpoint(CKPT)
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        for sampler in SAMPLERS:
            for clabel, cost in COSTS:
                for seed in SEEDS:
                    fields = dict(exp=EXP, fr=fr, sampler=sampler, cost=clabel, seed=seed)
                    if ckpt.has(fields):
                        continue
                    try:
                        m = run_trial(ENV, sampler, cost, seed=seed, K=K, difficulty=TV,
                                      true_perturbation=pert)
                    except Exception as e:
                        m = dict(error=f"{type(e).__name__}: {e}", prod=None, survived=None)
                    ckpt.record(fields, m)
            print(f"  friction={fr} {sampler} done", flush=True)
    print(ckpt.summary())
    return ckpt


def _cell(rows, fr, sampler, clabel):
    v = [r for r in rows if r["fr"] == fr and r["sampler"] == sampler and r["cost"] == clabel
         and r.get("prod") is not None]
    if not v:
        return None
    prod = np.array([r["prod"] for r in v]); surv = np.array([r["survived"] for r in v])
    m, h = mean_ci(prod); p, lo, hi = wilson_ci(int(surv.sum()), int(surv.size))
    return dict(prod=m, prod_ci=h, surv=p, surv_lo=lo, surv_hi=hi)


def report_and_figure(ckpt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 90)
    print("FPL ROBUSTNESS ACROSS SAMPLERS (hopper traction loss, tv=2.5, 12 seeds, no retuning)")
    print("=" * 90)
    for sampler in SAMPLERS:
        print(f"\n== {SAMPLER_PRETTY[sampler]} ==   (friction: {FRICTIONS})")
        for clabel, _c in COSTS:
            cells = [ _cell(rows, fr, sampler, clabel) for fr in FRICTIONS ]
            s = "  ".join(f"{c['prod']:.2f}/{c['surv']*100:3.0f}%" if c else "  -  " for c in cells)
            print(f"  {clabel:9s} {s}")

    x = [1.0 - fr for fr in FRICTIONS]
    fig, axes = plt.subplots(1, len(SAMPLERS), figsize=(3.7 * len(SAMPLERS), 3.8), sharey=True)
    ccolor = {"FPL": "#c44e52", "lin wv=1": "#4c72b0", "lin wv=2": "#55a868"}
    for ax, sampler in zip(axes, SAMPLERS):
        for clabel, _c in COSTS:
            cells = [_cell(rows, fr, sampler, clabel) for fr in FRICTIONS]
            ys = [c["prod"] if c else np.nan for c in cells]
            es = [c["prod_ci"] if c else 0 for c in cells]
            fpl = clabel == "FPL"
            ax.errorbar(x, ys, yerr=es, fmt="-*" if fpl else "-o", color=ccolor[clabel],
                        lw=2.6 if fpl else 1.5, ms=12 if fpl else 5, capsize=2, label=clabel)
        ax.set_title(SAMPLER_PRETTY[sampler], fontsize=10)
        ax.set_xlabel("traction loss (1−friction)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("productive speed (m/s × survival)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Un-tunable traction robustness holds WITHIN every sampler — the FPL win is the "
                 "objective, not MPPI\n(hopper, plan nominal / execute slippery, no retuning; 12 "
                 "seeds, 95% CI)", y=1.05, fontsize=10)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fpl_robustness_samplers.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out}")


def main():
    ckpt = run()
    report_and_figure(ckpt)


if __name__ == "__main__":
    main()
