"""Step 3 of docs/vanilla_mppi_hyperparams.md: one-at-a-time (OAT) sensitivity sweep
for vanilla-cost (objective.mode="normal") MPPI on hopper.

For each of the 6 sweep params in the doc's §1 table, vary ONLY that param across its
grid while holding every other param -- including the other 5 swept params -- at the
step-2 working config (configs/exp/hopper_vanilla_mppi.py). Metric: the 4 fulfillment
atoms (height/orientation/velocity/control, each in [0,1]) from HopperTask's
running_cost_terms_f, applied post-hoc to the vanilla-cost rollout -- this is valid
regardless of the "normal" cost mode actually driving the controller (see doc §4).

Protocol (doc §4, non-negotiable -- see the module's GOTCHA note below):
  * run.steps = 500 (10.0 s at dt=0.02). NOT the 150-step (3.0 s) proxy length used
    during the step-2 search: doc §5 documents a config that looked fine at 150 steps
    (85% survival) collapsing to 10% survival at the real 500-step length, because
    height sag accumulates gradually and only produces a fall late in a longer
    episode. Shortening the episode here would silently reproduce that mismatch.
  * 5 seeds per grid point (run.seeds in the doc's protocol), aggregated with
    verification/_ci.py's mean_ci (95% CI on the per-episode mean).
  * x-axis: each param's own grid min-max normalized to [0, 1]
    (x_norm = (x - x_min) / (x_max - x_min)) so every plot shares one x-axis scale;
    raw values stay as tick labels. spline_type is categorical -- plotted as a grouped
    bar chart over its 3 categories instead.

Output: results/sens_mppi/hopper/vanilla_mppi/{param}.png + {param}.json (raw numbers,
per-seed atom means included) for each of the 6 params, plus one shared
config.resolved.json / provenance.json for the baseline config (config.py:save_resolved).

Run:  a-mppi/bin/python verification/hopper_sensitivity_sweep.py
      a-mppi/bin/python verification/hopper_sensitivity_sweep.py --seeds 2   (faster, smoke test)
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

from analytic_mppi.config import load_config_file, resolve, save_resolved  # noqa: E402
from analytic_mppi.eval import run_episode, make_task  # noqa: E402
from _ci import mean_ci  # noqa: E402

BASELINE_CONFIG_PATH = REPO / "configs" / "exp" / "hopper_vanilla_mppi.py"
OUT_DIR = REPO / "results" / "sens_mppi" / "hopper" / "vanilla_mppi"

ATOM_NAMES = ["height_fulfillment", "orientation_fulfillment",
              "velocity_fulfillment", "control_fulfillment"]
ATOM_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

STEPS = 500       # 10.0 s at dt=0.02 -- the real protocol length, NOT a proxy (see module
                   # docstring GOTCHA). Every sweep point below runs the full length.
N_SEEDS_DEFAULT = 5

# One entry per doc §1 swept param: (name, dotted config field, baseline value, grid,
# categorical). Baselines are read back off the loaded config at runtime (asserted to
# match this table) rather than hardcoded twice.
SWEEPS: list[dict[str, Any]] = [
    dict(name="num_samples", field="proposal.num_samples",
         grid=[64, 128, 256, 512, 1024], categorical=False),
    dict(name="num_knots", field="proposal.num_knots",
         grid=[2, 3, 4, 5, 6, 7, 8], categorical=False),
    dict(name="plan_horizon", field="proposal.plan_horizon",
         grid=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], categorical=False),
    dict(name="noise_level", field="proposal.noise_level",
         grid=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8], categorical=False),
    dict(name="temperature", field="update.temperature",
         grid=[0.05, 0.1, 0.15, 0.2], categorical=False),
    dict(name="spline_type", field="run.spline_type",
         grid=["zero", "linear", "cubic"], categorical=True),
]


def _get_path(cfg, path: str):
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def run_grid_point(base_cfg, field: str, value, seeds: range, task) -> np.ndarray:
    """Vary ONLY `field`; return (n_seeds, 4) per-episode mean fulfillment atoms."""
    cfg = base_cfg.with_(**{field: value})
    r = resolve(cfg)
    out = np.empty((len(seeds), len(ATOM_NAMES)), dtype=np.float64)
    for i, seed in enumerate(seeds):
        res = run_episode(r.task_name, r.controller, steps=r.steps, seed=seed,
                          init_fn=r.init_fn, task_kwargs=r.task_kwargs, **r.build)
        states, ctrls, sd = res["states"], res["ctrls"], res["sd"]
        qpos = task.qpos_of(states[:-1])   # (T, nq) -- align with ctrls/sd (T,)
        qvel = task.qvel_of(states[:-1])   # (T, nv)
        atoms = task.running_cost_terms_f(qpos, qvel, sd, ctrls)   # (T, 4)
        out[i] = atoms.mean(axis=0)
    return out


def sweep_param(base_cfg, spec: dict, seeds: range, task) -> dict:
    baseline = _get_path(base_cfg, spec["field"])
    if baseline not in spec["grid"]:
        raise AssertionError(
            f"{spec['name']}: baseline {baseline!r} not in grid {spec['grid']} -- "
            f"grid must include the step-2 working value.")
    points = []
    lo, hi = (min(spec["grid"]), max(spec["grid"])) if not spec["categorical"] else (None, None)
    for value in spec["grid"]:
        t0 = time.time()
        per_seed = run_grid_point(base_cfg, spec["field"], value, seeds, task)   # (n_seeds, 4)
        elapsed = time.time() - t0
        x_norm = None if spec["categorical"] else (
            0.0 if hi == lo else (value - lo) / (hi - lo))
        atoms = {}
        for j, name in enumerate(ATOM_NAMES):
            m, ci = mean_ci(per_seed[:, j])
            atoms[name] = dict(mean=m, ci95=ci, per_seed=per_seed[:, j].tolist())
        points.append(dict(value=value, x_norm=x_norm, atoms=atoms))
        print(f"  {spec['name']}={_fmt(value):>8s}"
              f"  height={atoms['height_fulfillment']['mean']:.3f}"
              f"  orient={atoms['orientation_fulfillment']['mean']:.3f}"
              f"  vel={atoms['velocity_fulfillment']['mean']:.3f}"
              f"  ctrl={atoms['control_fulfillment']['mean']:.3f}"
              f"  ({elapsed:.1f}s)")
    return dict(
        param=spec["name"], field=spec["field"], baseline=baseline, grid=spec["grid"],
        categorical=spec["categorical"], seeds=list(seeds), steps=STEPS,
        atom_names=ATOM_NAMES, points=points,
    )


def plot_numeric(result: dict, out_path: Path, title: str | None = None,
                 baseline_label: str = "step-2 baseline") -> None:
    points = result["points"]
    x = np.array([p["x_norm"] for p in points])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for j, name in enumerate(result["atom_names"]):
        means = np.array([p["atoms"][name]["mean"] for p in points])
        cis = np.array([p["atoms"][name]["ci95"] for p in points])
        ax.errorbar(x, means, yerr=cis, fmt="o-", color=ATOM_COLORS[j], lw=1.6, ms=5,
                    capsize=3, ecolor=ATOM_COLORS[j], elinewidth=1,
                    label=name.replace("_fulfillment", ""))
    lo, hi = min(result["grid"]), max(result["grid"])
    b_x = 0.0 if hi == lo else (result["baseline"] - lo) / (hi - lo)
    ax.axvline(b_x, color="0.4", ls="--", lw=1, alpha=0.8,
               label=f"{baseline_label} ({_fmt(result['baseline'])})")
    ax.set_xticks(x)
    ax.set_xticklabels([_fmt(v) for v in result["grid"]])
    ax.set_xlabel(f"{result['param']}  (raw value; x-axis min-max normalized to [0,1])")
    ax.set_ylabel("fulfillment atom, mean over episode  [0,1]")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title or (
        f"Hopper vanilla-MPPI sensitivity: {result['param']} ({result['field']})\n"
        f"{len(result['seeds'])} seeds/point, {result['steps']} steps (10s), 95% CI"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_categorical(result: dict, out_path: Path) -> None:
    points = result["points"]
    cats = [p["value"] for p in points]
    n_cat, n_atom = len(cats), len(result["atom_names"])
    width = 0.8 / n_atom
    xpos = np.arange(n_cat)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for j, name in enumerate(result["atom_names"]):
        means = [p["atoms"][name]["mean"] for p in points]
        cis = [p["atoms"][name]["ci95"] for p in points]
        ax.bar(xpos + (j - (n_atom - 1) / 2) * width, means, width=width,
               yerr=cis, capsize=3, color=ATOM_COLORS[j],
               label=name.replace("_fulfillment", ""))
    ax.set_xticks(xpos)
    labels = ax.set_xticklabels(cats)
    for lbl, c in zip(labels, cats):
        if c == result["baseline"]:
            lbl.set_fontweight("bold")
    ax.set_xlabel(f"{result['param']}  ({result['field']}, categorical) -- "
                  f"bold = step-2 baseline")
    ax.set_ylabel("fulfillment atom, mean over episode  [0,1]")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Hopper vanilla-MPPI sensitivity: {result['param']} ({result['field']})\n"
                f"{len(result['seeds'])} seeds/point, {result['steps']} steps (10s), 95% CI")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT,
                   help=f"seeds per grid point (protocol default: {N_SEEDS_DEFAULT}). "
                        "Lower this to trade statistical power for wall-clock, NOT "
                        "episode length -- run.steps stays fixed at 500 regardless.")
    p.add_argument("--only", default=None,
                   help="comma-separated subset of param names to run (default: all 6)")
    args = p.parse_args()

    seeds = range(args.seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_config_file(str(BASELINE_CONFIG_PATH))
    assert base_cfg.run.steps == STEPS, (
        f"baseline config run.steps={base_cfg.run.steps} != protocol length {STEPS}; "
        f"refusing to run the sensitivity sweep at a shorter (proxy) length -- see "
        f"module docstring GOTCHA / doc §5.")
    save_resolved(base_cfg, OUT_DIR)

    r0 = resolve(base_cfg)
    task = make_task(r0.task_name, **r0.task_kwargs)

    specs = SWEEPS if args.only is None else [
        s for s in SWEEPS if s["name"] in set(args.only.split(","))]
    if not specs:
        raise SystemExit(f"--only matched nothing; known: {[s['name'] for s in SWEEPS]}")

    print(f"hopper sensitivity sweep: {len(specs)} params, {args.seeds} seeds/point, "
          f"{STEPS} steps ({STEPS * 0.02:.1f}s) per episode")
    t_start = time.time()
    for spec in specs:
        print(f"\n[{spec['name']}] baseline={_fmt(_get_path(base_cfg, spec['field']))}  "
              f"grid={[_fmt(v) for v in spec['grid']]}")
        result = sweep_param(base_cfg, spec, seeds, task)
        (OUT_DIR / f"{spec['name']}.json").write_text(json.dumps(result, indent=2))
        out_png = OUT_DIR / f"{spec['name']}.png"
        (plot_categorical if spec["categorical"] else plot_numeric)(result, out_png)
        print(f"  -> {out_png.relative_to(REPO)}")

    print(f"\ndone in {time.time() - t_start:.1f}s -> {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
