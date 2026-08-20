"""Does the FPL-NATIVE weighting help under model mismatch? (the decisive attribution test)

The weighting study showed ESS-targeting beats a fixed temperature on the noisy hopper — but
plain ESS-targeting is generic (scale-invariant, not FPL-specific). The genuinely FPL-native
levers need the ABSOLUTE fulfillment scale a scalar cost lacks:
  * adaptive_ess + fpl_calibrated : modulate the target ESS by the absolute best fulfillment
    u_best — when the situation gets HARD (mismatch lowers achievable fulfillment) it HEDGES
    (raises ESS, refuses to over-commit to a rollout that only looks good relatively).
  * feasibility_gate               : zero-weight any rollout whose composite fulfillment is below
    an absolute floor τ — drop the near-failing rollouts outright instead of averaging them in.

The paper's strongest axis is un-tunable robustness to aggression-punishing mismatch (traction
loss). If the FPL-native weightings hold survival + productive speed better than a fixed λ AND
than generic ESS-targeting as the ground gets slippery — with NO retuning — that is a real
second FPL win, in the weighting stage. Held fixed: FPL cost + plain MPPI proposal; the planner
uses the nominal model, reality is slippery. Checkpointed.

Run:  .venv/bin/python verification/fpl_weighting_robustness.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci, wilson_ci                         # noqa: E402

EXP = "weighting_robust"
ENV = "hopper"
TV = 2.5
K = 256
SEEDS = list(range(16))
FRICTIONS = [1.0, 0.6, 0.45, 0.35]
# (label, extra weighting kwargs)
CONFIGS = [
    ("relative λ=0.05", dict(weight_mode="relative", temperature=0.05)),
    ("adaptEss f=0.08 (generic)", dict(weight_mode="adaptive_ess", fpl_calibrated=False,
                                       ess_frac_lo=0.08, ess_frac_hi=0.08)),
    ("adaptEss calib (FPL)", dict(weight_mode="adaptive_ess", fpl_calibrated=True,
                                  ess_frac_lo=0.03, ess_frac_hi=0.25)),
    ("feas-gate τ=0.25 (FPL)", dict(weight_mode="feasibility_gate", feas_floor=0.25,
                                    temperature=0.05)),
]
CKPT = Path(__file__).resolve().parent / "checkpoints" / "weighting_robust.jsonl"


def run():
    ckpt = Checkpoint(CKPT)
    for fr in FRICTIONS:
        pert = None if fr == 1.0 else dict(friction_scale=fr)
        for label, extra in CONFIGS:
            for seed in SEEDS:
                fields = dict(exp=EXP, fr=fr, label=label, seed=seed)
                if ckpt.has(fields):
                    continue
                try:
                    m = run_trial(ENV, "fpl_tempered", "fpl", seed=seed, K=K, difficulty=TV,
                                  true_perturbation=pert, extra=dict(extra))
                except Exception as e:
                    m = dict(error=f"{type(e).__name__}: {e}", prod=None, survived=None)
                ckpt.record(fields, m)
            print(f"  friction={fr} {label} done", flush=True)
    print(ckpt.summary())
    return ckpt


def _cell(rows, fr, label):
    v = [r for r in rows if r["fr"] == fr and r["label"] == label and r.get("prod") is not None]
    if not v:
        return None
    prod = np.array([r["prod"] for r in v]); surv = np.array([r["survived"] for r in v])
    m, h = mean_ci(prod)
    p, lo, hi = wilson_ci(int(surv.sum()), int(surv.size))
    return dict(prod=m, prod_ci=h, surv=p, surv_lo=lo, surv_hi=hi)


def report_and_figure(ckpt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 84)
    print(f"WEIGHTING ROBUSTNESS (hopper traction loss, tv={TV}, NO retuning, 16 seeds)")
    print("=" * 84)
    header = "  ".join(f"{l.split(' (')[0]:>22s}" for l, _ in CONFIGS)
    print(f"{'friction':>9s}  {header}")
    data = {l: [] for l, _ in CONFIGS}
    for fr in FRICTIONS:
        cells = []
        for label, _ in CONFIGS:
            c = _cell(rows, fr, label)
            data[label].append(c)
            cells.append(f"{c['prod']:.2f}/{c['surv']*100:3.0f}%" if c else "   -   ")
        print(f"{fr:9.2f}  " + "  ".join(f"{x:>22s}" for x in cells))

    x = [1.0 - fr for fr in FRICTIONS]
    colors = {"relative λ=0.05": "#4c72b0", "adaptEss f=0.08 (generic)": "#55a868",
              "adaptEss calib (FPL)": "#c44e52", "feas-gate τ=0.25 (FPL)": "#dd8452"}
    fig, (axP, axS) = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, _ in CONFIGS:
        cs = data[label]
        fpl = "FPL" in label
        prod = [c["prod"] for c in cs]; pci = [c["prod_ci"] for c in cs]
        axP.errorbar(x, prod, yerr=pci, fmt="-*" if fpl else "-o", color=colors[label],
                     lw=3 if fpl else 1.6, ms=13 if fpl else 6, capsize=2, label=label)
        surv = [c["surv"] for c in cs]
        lo = [c["surv"] - c["surv_lo"] for c in cs]; hi = [c["surv_hi"] - c["surv"] for c in cs]
        axS.errorbar(x, surv, yerr=[lo, hi], fmt="-*" if fpl else "-o", color=colors[label],
                     lw=3 if fpl else 1.6, ms=13 if fpl else 6, capsize=2, label=label)
    axP.set(title="Productive speed (m/s × survival)", ylabel="m/s × survival",
            xlabel="traction loss  (1 − friction)")
    axS.set(title="Survival", ylabel="survival rate", xlabel="traction loss  (1 − friction)")
    for ax in (axP, axS):
        ax.grid(alpha=0.3)
    axP.legend(fontsize=8, loc="lower left")
    fig.suptitle("FPL-native WEIGHTING under traction loss (hopper, no retuning): does calibrating "
                 "the importance weights\nby the absolute fulfillment scale hold up better than a "
                 "fixed temperature / generic ESS-targeting?", y=1.04, fontsize=10)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fpl_weighting_robustness.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out}")


def main():
    ckpt = run()
    report_and_figure(ckpt)


if __name__ == "__main__":
    main()
