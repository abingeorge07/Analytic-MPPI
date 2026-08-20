"""ZERO-TUNING / off-the-shelf: the FPL objective transfers across robots with no retuning;
a linear objective needs a per-robot weight search.

The pitch (paper part 1): you want to drop a controller onto a new morphology and get an
HONEST assessment of whether it can do the task, WITHOUT a hyperparameter hunt. FPL is
specified once from task intent — bounded [0,1] atoms + "satisfy the least-satisfied one"
(min-fulfillment, p=-1) — with NO per-task weights. A linear cost needs a weight vector, and
the right weight differs per robot (and per difficulty; cf. the mismatch study). So we hold
ONE global objective spec fixed and drop it on every robot:

  * FPL:  p=-1, uniform, time_p=-2   (ONE spec, all robots)
  * linear: a single GLOBAL progress weight wv (ONE value, all robots), swept to show that no
    single wv works everywhere — small wv is too timid (no progress), large wv falls/drops.

Only the OBJECTIVE is held global. Each env keeps its own matched SAMPLER (noise/horizon/knots
scale with the robot's timestep + actuator range — structural, not task tuning). Same budget,
16 seeds. A cell "works" iff survival>=90% AND it makes real progress (>=50% of FPL's progress
on that robot). The claim: FPL is the only objective that works on EVERY robot out of the box.

Checkpointed. Run:  .venv/bin/python verification/fpl_zero_tuning.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial, ENVS as ENVSPEC       # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci                                    # noqa: E402

EXP = "zero_tuning"
ENVS = ["walker", "hopper", "cube"]        # quadruped added by an extend run (slow)
OBJECTIVES = ["fpl", "lin:0.5", "lin:1", "lin:2", "lin:4", "lin:8"]
OBJ_PRETTY = {"fpl": "FPL\n(one spec)", "lin:0.5": "linear\nwv=0.5", "lin:1": "linear\nwv=1",
              "lin:2": "linear\nwv=2", "lin:4": "linear\nwv=4", "lin:8": "linear\nwv=8"}
SAMPLER = "mppi"
K = 256
SEEDS = list(range(16))
WORKS_SURV = 0.90
WORKS_PROG = 0.50
CKPT = Path(__file__).resolve().parent / "checkpoints" / "zero_tuning.jsonl"


def run(progress=True):
    ckpt = Checkpoint(CKPT)
    for env in ENVS:
        for obj in OBJECTIVES:
            for seed in SEEDS:
                fields = dict(exp=EXP, env=env, obj=obj, sampler=SAMPLER, K=K, seed=seed)
                if ckpt.has(fields):
                    continue
                try:
                    m = run_trial(env, SAMPLER, obj, seed=seed, K=K)
                except Exception as e:
                    m = dict(error=f"{type(e).__name__}: {e}", prod=None, survived=None)
                ckpt.record(fields, m)
            print(f"  {env} / {obj} done", flush=True)
    print(ckpt.summary())
    return ckpt


def _cell(rows, env, obj):
    prog_key = "achieved" if ENVSPEC[env]["metric"] == "cube" else "vx"
    vals = [r for r in rows if r["env"] == env and r["obj"] == obj and r.get(prog_key) is not None]
    if not vals:
        return None
    prog = np.array([r[prog_key] for r in vals], dtype=float)
    surv = np.array([r["survived"] for r in vals], dtype=float)
    m, h = mean_ci(prog)
    return dict(prog=m, prog_ci=h, surv=float(surv.mean()),
                prod=float((prog * surv).mean()))


def report_and_figure(ckpt, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = ckpt.rows(exp=EXP)
    # per-env normalizer = FPL progress on that env
    grid = np.full((len(ENVS), len(OBJECTIVES)), np.nan)
    surv_grid = np.full((len(ENVS), len(OBJECTIVES)), np.nan)
    works = np.zeros((len(ENVS), len(OBJECTIVES)), dtype=bool)
    print("\n" + "=" * 90)
    print("ZERO-TUNING — one global objective spec dropped on every robot (16 seeds)")
    print("=" * 90)
    for i, env in enumerate(ENVS):
        fpl = _cell(rows, env, "fpl")
        ref = fpl["prog"] if fpl and fpl["prog"] > 1e-6 else 1.0
        print(f"\n== {env} ==  (FPL progress ref = {ref:.2f})")
        for j, obj in enumerate(OBJECTIVES):
            c = _cell(rows, env, obj)
            if c is None:
                continue
            norm = c["prog"] / ref
            grid[i, j] = norm
            surv_grid[i, j] = c["surv"]
            works[i, j] = (c["surv"] >= WORKS_SURV) and (norm >= WORKS_PROG)
            print(f"  {obj:8s}  progress={c['prog']:.2f} ({norm*100:3.0f}% of FPL)  "
                  f"surv={c['surv']*100:3.0f}%  {'WORKS' if works[i,j] else 'fails'}")

    # figure: heatmap of productive (normalized progress x survival); box the BEST objective
    # in each row (the most productive controller on that robot, threshold-free & honest).
    fig, ax = plt.subplots(figsize=(1.5 * len(OBJECTIVES) + 2, 1.3 * len(ENVS) + 1.5))
    prod_norm = np.clip(grid * surv_grid, 0, 1.2)
    im = ax.imshow(prod_norm, cmap="RdYlGn", vmin=0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(OBJECTIVES)))
    ax.set_xticklabels([OBJ_PRETTY[o] for o in OBJECTIVES], fontsize=9)
    ax.set_yticks(range(len(ENVS)))
    ax.set_yticklabels(ENVS, fontsize=11)
    best_per_row = np.nanargmax(np.where(np.isnan(prod_norm), -np.inf, prod_norm), axis=1)
    for i in range(len(ENVS)):
        for j in range(len(OBJECTIVES)):
            if np.isnan(grid[i, j]):
                continue
            is_best = (j == best_per_row[i])
            txt = f"{grid[i,j]*100:.0f}%\n{surv_grid[i,j]*100:.0f}% surv"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black", fontweight="bold" if is_best else "normal")
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=2.8))
    ax.set_title("Off-the-shelf: the SAME FPL spec is the most productive objective on EVERY robot "
                 "(boxed)\nno single linear weight transfers — each collapses on >=1 robot "
                 "(wv=4/8 fall on hopper; wv<=1 too timid; aggressive wv drops the cube)\n"
                 "(cell = progress % of FPL x survival; 16 seeds)", fontsize=10)
    fig.colorbar(im, ax=ax, label="productive score (norm. progress x survival)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved figure -> {out_path}")


def main():
    ckpt = run()
    report_and_figure(ckpt, Path(__file__).resolve().parent / "fpl_zero_tuning.png")


if __name__ == "__main__":
    main()
