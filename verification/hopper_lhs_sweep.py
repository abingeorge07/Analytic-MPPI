"""Step 4 of docs/vanilla_mppi_hyperparams.md: a Latin Hypercube Sampling (LHS) sweep
over all 6 shared §1 hyperparameters (num_samples, num_knots, plan_horizon, noise_level,
temperature, spline_type) for BOTH the vanilla and FPL arms, sharing ONE 6-D design so
each LHS sample i puts both arms at the exact same param combination -- a paired
comparison of joint effects, as opposed to step 3's one-at-a-time (OAT) sweep
(hopper_sensitivity_sweep.py / hopper_fpl_vs_vanilla_sweep.py).

Each arm is still evaluated around its OWN baseline config for everything NOT swept
here (task.kwargs weights, objective.p, etc.) -- vanilla: hopper_vanilla_mppi.py,
FPL: hopper_fpl_mppi.py -- exactly like the step-3 extension. Ranges for the 6 swept
dims are pulled from hopper_sensitivity_sweep.SWEEPS (min/max of each OAT grid), not
re-hardcoded. scipy is not installed in this repo's venv, so the LHS design is
hand-rolled with numpy only (stratified + jittered columns, one column per dimension --
see `build_lhs_design`).

Protocol (mirrors step 3, doc §4):
  * run.steps = 500 (10.0 s), for BOTH arms -- same GOTCHA as step 3: do NOT shorten the
    episode. FPL's flagship file is validated at 400 steps/8s; overridden to 500 here via
    `.with_(**{"run.steps": 500})` at sweep time (not by editing hopper_fpl_mppi.py) --
    see hopper_fpl_vs_vanilla_sweep.py's module docstring for the survival spot-check
    justifying that override.
  * 5 seeds per point per arm (doc §4 default), using run.seeds-style integer seeds
    independent of the LHS design's own RNG stream (a fresh `np.random.default_rng`
    used only to build the param grid; episode seeds are plain ints 0..n_seeds-1 fed to
    run_episode/the controller's own seeding -- two independent seed spaces, not one
    reused for both purposes).
  * Metrics per seed: the 4 fulfillment atoms (height/orientation/velocity/control, via
    HopperTask.running_cost_terms_f, valid post-hoc regardless of cost_mode -- same as
    step 3) PLUS mean forward velocity (vx) and survival (1 if the episode's minimum
    torso-zaxis-z never drops below 0.6, else 0) -- matching verification/_experiment.py's
    `_metrics()` convention for locomotion envs exactly (fall line 0.6).

Output: results/sens_mppi/hopper/vanilla_vs_fpl_lhs/
  lhs_design.json              all 64 points' raw param values + design rng seed
  results.json                 per arm, per point: params + per-seed metrics + aggregates
  sensitivity_summary.json/png param x metric x arm summary -- Spearman rank correlation
                                (numpy rankdata + corrcoef, no scipy) for the 5 numeric
                                params, per-category means for categorical spline_type
  lhs_scatter.png               2x3 grid, one panel per param: raw param value (x) vs
                                each of the 4 fulfillment atoms (y), one marker per arm --
                                the joint-effects view (all 6 dims vary simultaneously, so
                                expect visible spread around any trend)
  vanilla_config.resolved.json / fpl_config.resolved.json / provenance.json

Run:  a-mppi/bin/python verification/hopper_lhs_sweep.py
      a-mppi/bin/python verification/hopper_lhs_sweep.py --n-points 4 --seeds 1   (smoke test)
      a-mppi/bin/python verification/hopper_lhs_sweep.py --append --n-points 128 --design-seed 1
          (extend an existing design/results with a NEW batch of points at a NEW design
          seed, reusing the previously computed points instead of re-running them --
          this is how the design was grown from 64 to 192 points)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytic_mppi.config import load_config_file, resolve  # noqa: E402
from analytic_mppi.eval import run_episode, make_task  # noqa: E402
from _ci import mean_ci  # noqa: E402
from hopper_sensitivity_sweep import ATOM_NAMES, ATOM_COLORS, STEPS, SWEEPS, _fmt, _get_path  # noqa: E402
from hopper_fpl_vs_vanilla_sweep import (  # noqa: E402
    VANILLA_CONFIG_PATH, FPL_CONFIG_PATH, CONTROLLERS, CTRL_LABEL, CTRL_STYLE, CTRL_HATCH,
    _git_sha, _git_dirty,
)

OUT_DIR = REPO / "results" / "sens_mppi" / "hopper" / "vanilla_vs_fpl_lhs"

N_POINTS_DEFAULT = 64
N_SEEDS_DEFAULT = 5
DESIGN_SEED_DEFAULT = 0

# Locomotion fall line -- matches verification/_experiment.py:ENVS["hopper"]["fall"] and
# HopperTask's own orientation_floor default (both 0.6).
FALL_LINE = 0.6

METRIC_NAMES = ATOM_NAMES + ["vx", "survival"]

SWEEP_BY_NAME = {s["name"]: s for s in SWEEPS}

# The 6 shared dims, in LHS-column order. Ranges pulled from SWEEPS at runtime (min/max
# of each OAT grid) rather than re-hardcoded literals.
DIMENSIONS: list[dict[str, Any]] = [
    dict(name="num_samples",  field="proposal.num_samples",  sweep="num_samples",  kind="log_int"),
    dict(name="num_knots",    field="proposal.num_knots",    sweep="num_knots",    kind="linear_int"),
    dict(name="plan_horizon", field="proposal.plan_horizon", sweep="plan_horizon", kind="linear"),
    dict(name="noise_level",  field="proposal.noise_level",  sweep="noise_level",  kind="linear"),
    dict(name="temperature",  field="update.temperature",    sweep="temperature",  kind="linear"),
    dict(name="spline_type",  field="run.spline_type",       sweep="spline_type",  kind="categorical"),
]
DIM_FIELD = {d["name"]: d["field"] for d in DIMENSIONS}


def _lo_hi(sweep_name: str) -> tuple[float, float]:
    grid = SWEEP_BY_NAME[sweep_name]["grid"]
    return min(grid), max(grid)


def build_lhs_design(n_points: int, seed: int) -> dict:
    """Stratified + jittered LHS: each of the 6 columns independently permuted into
    n_points equal-probability bins, one sample per bin (real LHS marginals; joint
    combination randomized per column -- see module docstring / doc §4 GOTCHA)."""
    rng = np.random.default_rng(seed)
    d = len(DIMENSIONS)
    U = np.empty((n_points, d))
    for j in range(d):
        perm = rng.permutation(n_points)
        jitter = rng.uniform(size=n_points)
        U[:, j] = (perm + jitter) / n_points

    points = []
    for i in range(n_points):
        params: dict[str, Any] = {}
        u_row: dict[str, float] = {}
        for j, dim in enumerate(DIMENSIONS):
            u = float(U[i, j])
            u_row[dim["name"]] = u
            if dim["kind"] == "categorical":
                cats = SWEEP_BY_NAME[dim["sweep"]]["grid"]
                idx = min(len(cats) - 1, int(u * len(cats)))
                params[dim["name"]] = cats[idx]
            else:
                lo, hi = _lo_hi(dim["sweep"])
                if dim["kind"] == "log_int":
                    val = 2.0 ** (np.log2(lo) + u * (np.log2(hi) - np.log2(lo)))
                    params[dim["name"]] = int(round(val))
                elif dim["kind"] == "linear_int":
                    params[dim["name"]] = int(round(lo + u * (hi - lo)))
                else:
                    params[dim["name"]] = float(lo + u * (hi - lo))
        points.append(dict(index=i, params=params, u=u_row))
    return dict(design_seed=seed, n_points=n_points,
                dimensions=[dict(name=dd["name"], field=dd["field"], kind=dd["kind"])
                            for dd in DIMENSIONS],
                points=points)


def eval_point(base_cfg, params: dict, seeds: range, task) -> np.ndarray:
    """Override all 6 dims at once; return (n_seeds, len(METRIC_NAMES)) per-episode means."""
    overrides = {DIM_FIELD[name]: value for name, value in params.items()}
    cfg = base_cfg.with_(**overrides)
    r = resolve(cfg)
    out = np.empty((len(seeds), len(METRIC_NAMES)), dtype=np.float64)
    for i, seed in enumerate(seeds):
        res = run_episode(r.task_name, r.controller, steps=r.steps, seed=seed,
                          init_fn=r.init_fn, task_kwargs=r.task_kwargs, **r.build)
        states, ctrls, sd = res["states"], res["ctrls"], res["sd"]
        qpos = task.qpos_of(states[:-1])
        qvel = task.qvel_of(states[:-1])
        atoms = task.running_cost_terms_f(qpos, qvel, sd, ctrls)   # (T, 4)
        out[i, :4] = atoms.mean(axis=0)
        out[i, 4] = float(sd[..., task._vel_adr].mean())
        out[i, 5] = 1.0 if float(sd[..., task._zax_adr + 2].min()) >= FALL_LINE else 0.0
    return out


def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    """Average-rank ranking (ties share the mean rank of their positions), numpy only."""
    x = np.asarray(x, dtype=float)
    sorter = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[sorter] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rankdata_avg(x), _rankdata_avg(y)
    if rx.std() == 0.0 or ry.std() == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_sensitivity_summary(results: dict) -> dict:
    numeric = [d for d in DIMENSIONS if d["kind"] != "categorical"]
    categorical = [d for d in DIMENSIONS if d["kind"] == "categorical"]
    correlation: dict[str, Any] = {}
    for d in numeric:
        correlation[d["name"]] = {}
        for metric in METRIC_NAMES:
            correlation[d["name"]][metric] = {}
            for ctrl in CONTROLLERS:
                pts = results["controllers"][ctrl]["points"]
                xs = np.array([p["params"][d["name"]] for p in pts], dtype=float)
                ys = np.array([p["metrics"][metric]["mean"] for p in pts], dtype=float)
                correlation[d["name"]][metric][ctrl] = spearman(xs, ys)
    cat_summary: dict[str, Any] = {}
    for d in categorical:
        cats = SWEEP_BY_NAME[d["sweep"]]["grid"]
        cat_summary[d["name"]] = {}
        for metric in METRIC_NAMES:
            cat_summary[d["name"]][metric] = {}
            for ctrl in CONTROLLERS:
                pts = results["controllers"][ctrl]["points"]
                per_cat = {}
                for c in cats:
                    vals = [p["metrics"][metric]["mean"] for p in pts if p["params"][d["name"]] == c]
                    per_cat[c] = float(np.mean(vals)) if vals else float("nan")
                cat_summary[d["name"]][metric][ctrl] = per_cat
    return dict(correlation=correlation, categorical=cat_summary)


def plot_sensitivity_summary(summary: dict, out_path: Path) -> None:
    numeric_names = [d["name"] for d in DIMENSIONS if d["kind"] != "categorical"]
    cat_dims = [d for d in DIMENSIONS if d["kind"] == "categorical"]
    metric_labels = [m.replace("_fulfillment", "") for m in METRIC_NAMES]

    fig = plt.figure(figsize=(16.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0])

    for k, ctrl in enumerate(CONTROLLERS):
        ax = fig.add_subplot(gs[0, k])
        mat = np.array([[summary["correlation"][p][m][ctrl] for m in METRIC_NAMES]
                        for p in numeric_names])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(METRIC_NAMES)))
        ax.set_xticklabels(metric_labels, rotation=40, ha="right")
        ax.set_yticks(range(len(numeric_names)))
        ax.set_yticklabels(numeric_names)
        for yi in range(mat.shape[0]):
            for xi in range(mat.shape[1]):
                ax.text(xi, yi, f"{mat[yi, xi]:.2f}", ha="center", va="center", fontsize=7,
                       color="white" if abs(mat[yi, xi]) > 0.55 else "black")
        ax.set_title(f"{CTRL_LABEL[ctrl]}\nSpearman rho (param x metric)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 2])
    d = cat_dims[0]
    cats = SWEEP_BY_NAME[d["sweep"]]["grid"]
    cat_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:len(cats)]
    n_metric = len(METRIC_NAMES)
    n_bars = len(cats) * len(CONTROLLERS)
    total_width = 0.82
    width = total_width / n_bars
    xpos = np.arange(n_metric)
    bar_i = 0
    for ctrl in CONTROLLERS:
        for ci, c in enumerate(cats):
            vals = [summary["categorical"][d["name"]][m][ctrl][c] for m in METRIC_NAMES]
            xs = xpos - total_width / 2 + (bar_i + 0.5) * width
            ax2.bar(xs, vals, width=width, color=cat_colors[ci], hatch=CTRL_HATCH[ctrl],
                    edgecolor="black", linewidth=0.4,
                    label=f"{ctrl}:{c}")
            bar_i += 1
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(metric_labels, rotation=40, ha="right")
    ax2.set_ylabel("metric mean")
    ax2.set_title(f"{d['name']} (categorical)\nper-category mean -- "
                  f"plain=vanilla, hatched=FPL")
    ax2.legend(fontsize=6, ncol=2, loc="best")
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("Hopper LHS sensitivity summary: param x metric x arm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_lhs_scatter(results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0))
    jitter_rng = np.random.default_rng(12345)   # visualization-only jitter, independent
                                                 # of both the LHS design rng and episode seeds

    for ax, dim in zip(axes.flat, DIMENSIONS):
        name = dim["name"]
        for ctrl in CONTROLLERS:
            pts = results["controllers"][ctrl]["points"]
            raw = [p["params"][name] for p in pts]
            if dim["kind"] == "categorical":
                cats = SWEEP_BY_NAME[dim["sweep"]]["grid"]
                cat_idx = {c: i for i, c in enumerate(cats)}
                side = -0.15 if ctrl == CONTROLLERS[0] else 0.15
                xs = np.array([cat_idx[v] for v in raw], dtype=float) + side
                xs = xs + jitter_rng.uniform(-0.05, 0.05, size=len(xs))
            else:
                xs = np.array(raw, dtype=float)
            for j, atom in enumerate(ATOM_NAMES):
                ys = [p["metrics"][atom]["mean"] for p in pts]
                ax.scatter(xs, ys, color=ATOM_COLORS[j], marker=CTRL_STYLE[ctrl]["marker"],
                          alpha=0.55, s=20, linewidths=0)
        if dim["kind"] == "log_int":
            ax.set_xscale("log")
        if dim["kind"] == "categorical":
            cats = SWEEP_BY_NAME[dim["sweep"]]["grid"]
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats)
        ax.set_xlabel(f"{name}  (raw value)")
        ax.set_ylabel("fulfillment atom  [0,1]")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(name)
        ax.grid(alpha=0.25)

    atom_handles = [plt.Line2D([0], [0], marker="s", ls="", color=ATOM_COLORS[j],
                               label=name.replace("_fulfillment", ""))
                    for j, name in enumerate(ATOM_NAMES)]
    ctrl_handles = [plt.Line2D([0], [0], marker=CTRL_STYLE[ctrl]["marker"], ls="",
                               color="0.3", label=CTRL_LABEL[ctrl]) for ctrl in CONTROLLERS]
    fig.legend(handles=atom_handles + ctrl_handles, fontsize=8, loc="lower center",
              ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Hopper LHS sweep: joint effects (all 6 dims vary simultaneously per point)\n"
                f"{len(results['controllers'][CONTROLLERS[0]]['points'])} LHS samples/arm, "
                f"{results['n_seeds']} seeds/point, {results['steps']} steps (10s)")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-points", type=int, default=N_POINTS_DEFAULT,
                   help=f"LHS design points (default {N_POINTS_DEFAULT}); cheap to raise")
    p.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT,
                   help=f"seeds per point per arm (protocol default {N_SEEDS_DEFAULT}). "
                        "Lower this or --n-points to trade wall-clock, NOT run.steps.")
    p.add_argument("--design-seed", type=int, default=DESIGN_SEED_DEFAULT,
                   help=f"rng seed for the LHS design (default {DESIGN_SEED_DEFAULT})")
    p.add_argument("--append", action="store_true",
                   help="extend the existing lhs_design.json/results.json in OUT_DIR with "
                        "a NEW batch of --n-points points at a NEW --design-seed, instead "
                        "of starting over -- reuses the previously computed points rather "
                        "than re-running them. --seeds must match the existing protocol.")
    args = p.parse_args()

    seeds = range(args.seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    design_path, results_path = OUT_DIR / "lhs_design.json", OUT_DIR / "results.json"
    existing_design = existing_results = None
    prior_batches: list[dict] = []
    start_index = 0
    if args.append:
        if not design_path.exists() or not results_path.exists():
            raise SystemExit(f"--append requires an existing {design_path} and {results_path}")
        existing_design = json.loads(design_path.read_text())
        existing_results = json.loads(results_path.read_text())
        if "batches" in existing_design:
            prior_batches = existing_design["batches"]
        else:
            prior_batches = [dict(design_seed=existing_design["design_seed"],
                                  n_points=existing_design["n_points"])]
        used_seeds = {b["design_seed"] for b in prior_batches}
        if args.design_seed in used_seeds:
            raise SystemExit(
                f"--design-seed {args.design_seed} already used in this design "
                f"({sorted(used_seeds)}); pick a new one so the new batch's points are "
                f"independently drawn from the existing ones.")
        if args.seeds != existing_results["n_seeds"]:
            raise SystemExit(
                f"--seeds {args.seeds} != existing protocol n_seeds "
                f"{existing_results['n_seeds']}; keep the seed protocol identical across "
                f"batches so per-point aggregates stay comparable.")
        start_index = existing_design["n_points"]

    vanilla_cfg = load_config_file(str(VANILLA_CONFIG_PATH))
    assert vanilla_cfg.run.steps == STEPS, (
        f"vanilla config run.steps={vanilla_cfg.run.steps} != protocol length {STEPS}")

    fpl_cfg_raw = load_config_file(str(FPL_CONFIG_PATH))
    print(f"FPL flagship config validated at run.steps={fpl_cfg_raw.run.steps} "
         f"({fpl_cfg_raw.run.steps * 0.02:.1f}s); overriding to {STEPS} "
         f"({STEPS * 0.02:.1f}s) to match the shared protocol length.")
    fpl_cfg = fpl_cfg_raw.with_(**{"run.steps": STEPS})

    cfgs = dict(vanilla=vanilla_cfg, fpl=fpl_cfg)
    tasks = {}
    for ctrl, cfg in cfgs.items():
        r = resolve(cfg)
        tasks[ctrl] = make_task(r.task_name, **r.task_kwargs)
        (OUT_DIR / f"{ctrl}_config.resolved.json").write_text(cfg.to_json())

    batches = prior_batches + [dict(design_seed=args.design_seed, n_points=args.n_points)]

    (OUT_DIR / "provenance.json").write_text(json.dumps(dict(
        sha=_git_sha(), dirty=_git_dirty(),
        config_hash=dict(vanilla=vanilla_cfg.hash(), fpl=fpl_cfg.hash()),
        design_batches=batches,
        note="fpl config_hash reflects run.steps overridden to match the shared protocol "
             f"length ({STEPS}), not the {fpl_cfg_raw.run.steps}-step flagship file on disk. "
             "Step 4: Latin Hypercube sweep over the 6 shared hyperparameters "
             "(docs/vanilla_mppi_hyperparams.md step 4); each arm keeps its own baseline "
             "for everything not swept. design_batches lists each independently-drawn LHS "
             "batch (design_seed, n_points) unioned into the full design -- see --append.",
    ), indent=2))

    new_design = build_lhs_design(args.n_points, args.design_seed)
    for pt in new_design["points"]:
        pt["index"] += start_index
    total_points = start_index + args.n_points
    merged_points = (existing_design["points"] if existing_design else []) + new_design["points"]
    design_out = dict(batches=batches, n_points=total_points,
                      dimensions=new_design["dimensions"], points=merged_points)
    (OUT_DIR / "lhs_design.json").write_text(json.dumps(design_out, indent=2))

    n_new_episodes = args.n_points * args.seeds * len(CONTROLLERS)
    print(f"hopper LHS sweep: {'appending' if args.append else 'running'} "
         f"{args.n_points} new points (design_seed={args.design_seed}) x {args.seeds} seeds x "
         f"{len(CONTROLLERS)} arms = {n_new_episodes} new episodes, {STEPS} steps "
         f"({STEPS * 0.02:.1f}s) each" +
         (f"; {start_index} points already computed, reused as-is -> {total_points} total"
          if args.append else ""))
    for ctrl in CONTROLLERS:
        print(f"  {ctrl} baseline: " + ", ".join(
            f"{d['name']}={_fmt(_get_path(cfgs[ctrl], d['field']))}" for d in DIMENSIONS))

    t_start = time.time()
    per_ctrl_points: dict[str, list] = {}
    for ctrl in CONTROLLERS:
        cfg, task = cfgs[ctrl], tasks[ctrl]
        pts_out = list(existing_results["controllers"][ctrl]["points"]) if args.append else []
        for pt in new_design["points"]:
            t0 = time.time()
            per_seed = eval_point(cfg, pt["params"], seeds, task)   # (n_seeds, 6)
            elapsed = time.time() - t0
            metrics = {}
            for j, name in enumerate(METRIC_NAMES):
                m, ci = mean_ci(per_seed[:, j])
                metrics[name] = dict(mean=m, ci95=ci, per_seed=per_seed[:, j].tolist())
            pts_out.append(dict(index=pt["index"], params=pt["params"], metrics=metrics))
            p_str = " ".join(f"{d['name']}={_fmt(pt['params'][d['name']]):>7s}" for d in DIMENSIONS)
            print(f"  [{ctrl:7s}] pt {pt['index']+1:>3d}/{total_points}  {p_str}"
                 f"  height={metrics['height_fulfillment']['mean']:.2f}"
                 f"  vel={metrics['velocity_fulfillment']['mean']:.2f}"
                 f"  vx={metrics['vx']['mean']:.2f}"
                 f"  surv={metrics['survival']['mean']:.2f}"
                 f"  ({elapsed:.1f}s)")
        per_ctrl_points[ctrl] = pts_out

    results = dict(
        batches=batches, n_points=total_points, n_seeds=args.seeds,
        seeds=list(seeds), steps=STEPS, metric_names=METRIC_NAMES,
        controllers={ctrl: dict(points=per_ctrl_points[ctrl]) for ctrl in CONTROLLERS},
    )
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"-> {(OUT_DIR / 'results.json').relative_to(REPO)}")

    summary = compute_sensitivity_summary(results)
    (OUT_DIR / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    plot_sensitivity_summary(summary, OUT_DIR / "sensitivity_summary.png")
    print(f"-> {(OUT_DIR / 'sensitivity_summary.png').relative_to(REPO)}")

    plot_lhs_scatter(results, OUT_DIR / "lhs_scatter.png")
    print(f"-> {(OUT_DIR / 'lhs_scatter.png').relative_to(REPO)}")

    print(f"\ndone in {time.time() - t_start:.1f}s -> {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
