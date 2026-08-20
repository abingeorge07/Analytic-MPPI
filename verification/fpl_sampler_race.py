"""FPL-native SAMPLER race: does a smarter proposal beat plain FPL+MPPI at matched budget?

The prior finding was a clean negative for ONE sampler idea (adaptive diagonal covariance +
binding-objective steering: `fpl_adaptive.py`). This revisits the question with the technique
the recent sampling-MPC literature says actually matters — COLORED (temporally-correlated)
noise (iCEM; Pinneri et al. 2021) — plus an FPL-native absolute-scale exploration schedule
that only a bounded [0,1] reward makes possible. The cost is held FIXED (the winning FPL
min-fulfillment spec, p=-1); only the SAMPLER varies. We sweep the sample budget K, because
sampler quality matters most when samples are scarce (the online/real-time regime).

Configs (all FPL cost, identical everything except the proposal):
  FPL·MPPI (white)        plain warm-started Gaussian MPPI          [the prior best]
  FPL·CMA                 covariance-adaptation MPPI                 [baseline sampler]
  FPL·CEM                 cross-entropy elites                       [baseline sampler]
  FPL·colored             generic colored noise (cost-agnostic)     [ablation: color only]
  FPL·colored+abs (ours)  colored noise + absolute-scale schedule   [full FPL-native]

Reads out productive speed (achieved speed x survival) vs K, 95% CIs over seeds. Checkpointed:
every (env, config, K, seed) trial is durable, so a crash resumes without recomputation.

Run:  .venv/bin/python verification/fpl_sampler_race.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial                       # noqa: E402
from _checkpoint import Checkpoint, aggregate           # noqa: E402
from _ci import mean_ci, wilson_ci                       # noqa: E402

EXP = "sampler_race"
ENVS = ["walker", "hopper"]
KS = [16, 32, 64, 128, 256]
SEEDS = list(range(16))
# (label, sampler-name-in-_experiment)
CONFIGS = [
    ("FPL.MPPI (white)", "mppi"),
    ("FPL.CMA", "mppi_cma"),
    ("FPL.CEM", "cem"),
    ("FPL.colored", "fpl_colored_generic"),
    ("FPL.colored+abs (ours)", "fpl_colored"),
]
CKPT = Path(__file__).resolve().parent / "checkpoints" / "sampler_race.jsonl"


def run(progress=True):
    ckpt = Checkpoint(CKPT)
    total = len(ENVS) * len(CONFIGS) * len(KS) * len(SEEDS)
    done = 0
    for env in ENVS:
        for label, sampler in CONFIGS:
            for K in KS:
                for seed in SEEDS:
                    fields = dict(exp=EXP, env=env, label=label, sampler=sampler,
                                  K=K, seed=seed)
                    done += 1
                    if ckpt.has(fields):
                        continue
                    try:
                        m = run_trial(env, sampler, "fpl", seed=seed, K=K)
                    except Exception as e:  # a sampler blew up on this trial
                        m = dict(error=f"{type(e).__name__}: {e}", prod=None,
                                 survived=None, vx=None)
                    ckpt.record(fields, m)
                    if progress and done % 25 == 0:
                        print(f"  [{done}/{total}] {env} {label} K={K} seed={seed} "
                              f"prod={m.get('prod')}", flush=True)
    print(ckpt.summary())
    return ckpt


def report(ckpt: Checkpoint):
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 92)
    print(f"FPL SAMPLER RACE — productive speed (achieved x survival), {len(SEEDS)} seeds, 95% CI")
    print("=" * 92)
    agg = {}
    for env in ENVS:
        print(f"\n== {env} ==   (K:  " + "   ".join(f"{k:>13d}" for k in KS) + ")")
        for label, sampler in CONFIGS:
            cells = []
            for K in KS:
                vals = [r for r in rows if r["env"] == env and r["label"] == label
                        and r["K"] == K and r.get("prod") is not None]
                prod = np.array([r["prod"] for r in vals], dtype=float)
                surv = np.array([r["survived"] for r in vals], dtype=float)
                m, h = mean_ci(prod) if prod.size else (float("nan"), 0.0)
                sr = float(surv.mean()) if surv.size else float("nan")
                agg[(env, label, K)] = dict(prod=m, prod_ci=h, surv=sr, n=prod.size)
                cells.append(f"{m:5.2f}±{h:.2f}({sr*100:3.0f}%)")
            print(f"  {label:24s} " + " ".join(cells))
    return agg


def figure(agg, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"FPL.MPPI (white)": "#4c72b0", "FPL.CMA": "#55a868", "FPL.CEM": "#8172b3",
              "FPL.colored": "#c9a227", "FPL.colored+abs (ours)": "#c44e52"}
    fig, axes = plt.subplots(1, len(ENVS), figsize=(6.0 * len(ENVS), 4.6))
    if len(ENVS) == 1:
        axes = [axes]
    for ax, env in zip(axes, ENVS):
        for label, _s in CONFIGS:
            xs = KS
            ys = [agg[(env, label, K)]["prod"] for K in KS]
            es = [agg[(env, label, K)]["prod_ci"] for K in KS]
            ours = "ours" in label
            ax.errorbar(xs, ys, yerr=es, fmt="-*" if ours else "-o",
                        color=colors[label], lw=3 if ours else 1.6,
                        ms=15 if ours else 6, capsize=2, zorder=6 if ours else 3,
                        label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_title(f"{env} (running regime)")
        ax.set_xlabel("sample budget K")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("productive speed  (m/s x survival)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("FPL-native sampler vs baselines at matched budget (FPL cost fixed; only the "
                 "proposal varies; 16 seeds, 95% CI)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure -> {out_path}")


def main():
    ckpt = run()
    agg = report(ckpt)
    figure(agg, Path(__file__).resolve().parent / "fpl_sampler_race.png")


if __name__ == "__main__":
    main()
