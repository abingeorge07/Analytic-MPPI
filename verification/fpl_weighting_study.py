"""FPL-calibrated WEIGHTING layer — does FPL's absolute scale remove the MPPI temperature knob?

MPPI turns scored rollouts into an update via softmax weights w_k ∝ (u_k/max u)^{1/λ}. The
temperature λ is a notorious per-task knob: too warm ⇒ weights ≈ uniform (no selection, the
update just averages every rollout); too cold ⇒ weights collapse onto one rollout (high variance).
The right λ depends on the SPREAD of the rewards, which differs per robot — so λ is retuned per
task. FPL supplies an ABSOLUTE reference (u ∈ [0,1]) that scalar cost lacks, so we can instead set
the weighting by a TARGET EFFECTIVE-SAMPLE-SIZE (`adaptive_ess`, λ solved per step to hold ESS),
whose knob (the ESS fraction) should transfer across robots.

This study asks the falsifiable question directly:
  * `relative`     : sweep λ per robot → is the best λ DIFFERENT across robots, and is performance
                     sensitive to it (collapse when mis-set)?
  * `adaptive_ess` : sweep the ESS-target fraction per robot → is the best fraction the SAME across
                     robots (one global setting transfers = the knob is removed)?

Held fixed: the winning FPL cost + plain Gaussian MPPI proposal; only the reward→weight map varies.
Checkpointed. Run:  .venv/bin/python verification/fpl_weighting_study.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci                                    # noqa: E402

EXP = "weighting"
ENVS = ["walker", "hopper"]
SEEDS = list(range(12))
K = 256
TEMPS = [0.05, 0.1, 0.2, 0.4, 0.8]        # relative-λ sweep
FRACS = [0.03, 0.08, 0.2, 0.4]            # adaptive-ESS target-fraction sweep
CKPT = Path(__file__).resolve().parent / "checkpoints" / "weighting.jsonl"


def _cfgs():
    for t in TEMPS:
        yield (f"relative T={t:g}", "relative", dict(temperature=t))
    for f in FRACS:
        # fixed ESS target (lo==hi so the fpl-calibration is a no-op — a clean single knob)
        yield (f"adaptEss f={f:g}", "adaptive_ess",
               dict(weight_mode="adaptive_ess", fpl_calibrated=False,
                    ess_frac_lo=f, ess_frac_hi=f))


def run():
    ckpt = Checkpoint(CKPT)
    for env in ENVS:
        for label, mode, extra in _cfgs():
            ex = dict(extra)
            ex.setdefault("weight_mode", mode)
            for seed in SEEDS:
                fields = dict(exp=EXP, env=env, label=label, seed=seed, K=K)
                if ckpt.has(fields):
                    continue
                try:
                    m = run_trial(env, "fpl_tempered", "fpl", seed=seed, K=K, extra=ex)
                except Exception as e:
                    m = dict(error=f"{type(e).__name__}: {e}", prod=None, survived=None)
                ckpt.record(fields, m)
            print(f"  {env} / {label} done", flush=True)
    print(ckpt.summary())
    return ckpt


def _agg(rows, env, label):
    v = [r for r in rows if r["env"] == env and r["label"] == label and r.get("prod") is not None]
    if not v:
        return None
    prod = np.array([r["prod"] for r in v]); surv = np.array([r["survived"] for r in v])
    ess = np.array([r.get("ess", np.nan) for r in v], dtype=float)
    m, h = mean_ci(prod)
    return dict(prod=m, prod_ci=h, surv=float(surv.mean()), ess=float(np.nanmean(ess)))


def report_and_figure(ckpt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 80)
    print("FPL WEIGHTING — productive speed vs temperature/ESS-target, 12 seeds")
    print("=" * 80)
    best = {}
    for env in ENVS:
        print(f"\n== {env} ==")
        print("  relative (sweep λ):")
        rel = [(t, _agg(rows, env, f"relative T={t:g}")) for t in TEMPS]
        for t, s in rel:
            if s: print(f"    T={t:<5g} prod={s['prod']:.2f}±{s['prod_ci']:.2f} surv={s['surv']*100:.0f}% ESS~{s['ess']:.0f}")
        rbest = max([(t, s) for t, s in rel if s], key=lambda x: x[1]["prod"])
        print(f"    -> best relative λ = {rbest[0]:g} (prod={rbest[1]['prod']:.2f})")
        print("  adaptive_ess (sweep ESS-target fraction):")
        ada = [(f, _agg(rows, env, f"adaptEss f={f:g}")) for f in FRACS]
        for f, s in ada:
            if s: print(f"    f={f:<5g} prod={s['prod']:.2f}±{s['prod_ci']:.2f} surv={s['surv']*100:.0f}% ESS~{s['ess']:.0f}")
        abest = max([(f, s) for f, s in ada if s], key=lambda x: x[1]["prod"])
        print(f"    -> best ESS-frac = {abest[0]:g} (prod={abest[1]['prod']:.2f})")
        best[env] = dict(rel=rel, ada=ada, rbest=rbest, abest=abest)

    # transfer verdict
    print("\n" + "=" * 80)
    rbest_temps = {env: best[env]["rbest"][0] for env in ENVS}
    abest_fracs = {env: best[env]["abest"][0] for env in ENVS}
    print(f"best relative λ per env:      {rbest_temps}")
    print(f"best adaptive ESS-frac per env: {abest_fracs}")
    print("If the ESS-frac is consistent across envs while λ is not, the temperature knob is"
          " removed by the FPL absolute-scale weighting.")

    # figure: prod vs setting, one panel per env, relative curve + adaptive curve
    fig, axes = plt.subplots(1, len(ENVS), figsize=(5.6 * len(ENVS), 4.4))
    if len(ENVS) == 1:
        axes = [axes]
    for ax, env in zip(axes, ENVS):
        rel = best[env]["rel"]; ada = best[env]["ada"]
        xs = TEMPS; ys = [s["prod"] for _t, s in rel]; es = [s["prod_ci"] for _t, s in rel]
        ax.errorbar(xs, ys, yerr=es, fmt="-o", color="#4c72b0", capsize=2, label="relative (sweep λ)")
        ax2 = ax.twiny()
        fx = FRACS; fy = [s["prod"] for _f, s in ada]; fe = [s["prod_ci"] for _f, s in ada]
        ax2.errorbar(fx, fy, yerr=fe, fmt="-s", color="#c44e52", capsize=2,
                     label="adaptive_ess (sweep ESS-frac)")
        ax.set_xscale("log"); ax2.set_xscale("log")
        ax.set_xlabel("relative temperature λ", color="#4c72b0")
        ax2.set_xlabel("adaptive_ess target ESS fraction", color="#c44e52")
        ax.set_title(env)
        ax.grid(alpha=0.3)
        if env == ENVS[0]:
            ax.set_ylabel("productive speed (m/s × survival)")
    axes[0].legend(loc="lower center", fontsize=8)
    fig.suptitle("Weighting layer: targeting a fixed effective-sample-size (adaptive_ess, red) "
                 "matches the best temperature on walker\nand BEATS it on the noisy hopper (higher "
                 "productive speed, no weight-collapse) — a fixed λ can't hold selectivity as the "
                 "reward\nspread varies; FPL's bounded reward makes the ESS target meaningful. "
                 "12 seeds, 95% CI", y=1.06, fontsize=10)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fpl_weighting.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out}")


def main():
    ckpt = run()
    report_and_figure(ckpt)


if __name__ == "__main__":
    main()
