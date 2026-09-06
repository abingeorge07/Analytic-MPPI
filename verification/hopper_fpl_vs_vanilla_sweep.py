"""Extension of hopper_sensitivity_sweep.py (step 3): overlay vanilla-cost and FPL-cost
MPPI sensitivity on the same 6 §1 hyperparameters, plus a 7th param -- the FPL power-mean
exponent `objective.p` -- that only exists for the FPL arm.

Both cost modes share the exact sampler/optimizer hyperparameter axes (proposal.*,
update.temperature, run.spline_type), so the two OAT sweeps overlay directly: one figure
per param, 8 lines (4 fulfillment atoms x {vanilla, FPL}). Per-controller design (agreed
2026-08-28):

  * Each arm is swept AROUND ITS OWN step-2/flagship working config
    (configs/exp/hopper_vanilla_mppi.py / hopper_fpl_mppi.py) -- NOT a shared baseline --
    since the two searches independently found different working points for 4 of the 6
    params (spline_type, num_knots, plan_horizon, noise_level, temperature all differ;
    only num_samples=256 matches). Each plot carries two baseline markers.
  * Grids are IDENTICAL to hopper_sensitivity_sweep.SWEEPS (imported, not duplicated) so
    the shared min-max-normalized x-axis means the same thing for both arms.
  * Episode length: BOTH arms run the full 500-step (10s) protocol length, even though
    the FPL flagship config (hopper_fpl_mppi.py) was validated at 400 steps (8s). Spot
    check (10 seeds, module load time): FPL survives 10/10 at 500 steps with vx=1.65 m/s,
    matching its 8s flagship number (1.665 m/s) -- FPL's min-fulfillment conjunction is
    specifically designed against the late-episode sag that made vanilla-cost length-
    sensitive (doc §5's GOTCHA), so extending 2s does not reproduce that failure mode.
    Both arms are measured on the SAME length so the overlay is apples-to-apples.
  * p_value sweep (FPL only, no vanilla counterpart -- objective.p is a no-op in
    "normal" cost mode): grid [-0.5, -1, -2, -4, -8], centered on the FPL config's own
    p=-1.0 (Objective schema default). Matches the range already explored in
    verification/fpl_binding_probe.py / fpl_monotonicity.py's P_SWEEP.

Output: results/sens_mppi/hopper/vanilla_vs_fpl/{param}.png + .json for the 6 shared
params (8-line overlay, or hatched grouped-bar for spline_type) + p_value.png/.json
(FPL-only, single-mode). One folder -- this is meant to grow into "every MPPI variant vs.
its FPL counterpart", not a per-controller-pair directory tree. Plus per-arm
`{vanilla,fpl}_config.resolved.json` and a shared `provenance.json` (git sha/hash of both).

Run:  a-mppi/bin/python verification/hopper_fpl_vs_vanilla_sweep.py
      a-mppi/bin/python verification/hopper_fpl_vs_vanilla_sweep.py --seeds 2   (smoke test)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analytic_mppi.config import load_config_file, resolve  # noqa: E402
from analytic_mppi.eval import make_task  # noqa: E402
from hopper_sensitivity_sweep import (  # noqa: E402
    ATOM_NAMES, ATOM_COLORS, STEPS, SWEEPS, _fmt, _get_path, sweep_param, plot_numeric,
)

VANILLA_CONFIG_PATH = REPO / "configs" / "exp" / "hopper_vanilla_mppi.py"
FPL_CONFIG_PATH = REPO / "configs" / "exp" / "hopper_fpl_mppi.py"
OUT_DIR = REPO / "results" / "sens_mppi" / "hopper" / "vanilla_vs_fpl"

CONTROLLERS = ["vanilla", "fpl"]
CTRL_LABEL = dict(vanilla="vanilla (normal cost)", fpl="FPL (fpl_cost)")
CTRL_STYLE = dict(vanilla=dict(ls="-", marker="o"), fpl=dict(ls="--", marker="^"))
CTRL_HATCH = dict(vanilla=None, fpl="//")
CTRL_VLINE = dict(vanilla=dict(color="0.35", ls="--"), fpl=dict(color="black", ls=":"))

# FPL-only: the power-mean exponent that selects the min-fulfillment conjunction.
# No vanilla counterpart -- objective.p is inert under cost_mode="normal".
P_VALUE_SWEEP = dict(name="p_value", field="objective.p",
                     grid=[-0.5, -1.0, -2.0, -4.0, -8.0], categorical=False)


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception:
        return ""


def _git_dirty() -> bool:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=5)
        return bool(out.stdout.strip())
    except Exception:
        return False


def plot_overlay_numeric(result: dict, out_path: Path) -> None:
    grid = result["grid"]
    lo, hi = min(grid), max(grid)
    x = np.array([0.0 if hi == lo else (v - lo) / (hi - lo) for v in grid])
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for ctrl in CONTROLLERS:
        cres = result["controllers"][ctrl]
        for j, name in enumerate(ATOM_NAMES):
            means = np.array([p["atoms"][name]["mean"] for p in cres["points"]])
            cis = np.array([p["atoms"][name]["ci95"] for p in cres["points"]])
            ax.errorbar(x, means, yerr=cis, color=ATOM_COLORS[j], lw=1.6, ms=5,
                       capsize=3, ecolor=ATOM_COLORS[j], elinewidth=1,
                       label=f"{ctrl}: {name.replace('_fulfillment', '')}",
                       **CTRL_STYLE[ctrl])
        b = cres["baseline"]
        bx = 0.0 if hi == lo else (b - lo) / (hi - lo)
        ax.axvline(bx, lw=1.2, alpha=0.85,
                  label=f"{CTRL_LABEL[ctrl]} baseline ({_fmt(b)})", **CTRL_VLINE[ctrl])
    ax.set_xticks(x)
    ax.set_xticklabels([_fmt(v) for v in grid])
    ax.set_xlabel(f"{result['param']}  (raw value; x-axis min-max normalized to [0,1])")
    ax.set_ylabel("fulfillment atom, mean over episode  [0,1]")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Hopper vanilla vs FPL MPPI sensitivity: {result['param']} ({result['field']})\n"
        f"solid/● = vanilla, dashed/▲ = FPL   |   "
        f"{len(result['seeds'])} seeds/point, {result['steps']} steps (10s), 95% CI")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.8, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_categorical(result: dict, out_path: Path) -> None:
    cats = result["grid"]
    n_cat, n_atom = len(cats), len(ATOM_NAMES)
    n_bars = n_atom * len(CONTROLLERS)
    total_width = 0.82
    width = total_width / n_bars
    xpos = np.arange(n_cat)
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    bar_i = 0
    for ctrl in CONTROLLERS:
        cres = result["controllers"][ctrl]
        by_val = {p["value"]: p for p in cres["points"]}
        for j, name in enumerate(ATOM_NAMES):
            means = [by_val[c]["atoms"][name]["mean"] for c in cats]
            cis = [by_val[c]["atoms"][name]["ci95"] for c in cats]
            xs = xpos - total_width / 2 + (bar_i + 0.5) * width
            ax.bar(xs, means, width=width, yerr=cis, capsize=2, color=ATOM_COLORS[j],
                  hatch=CTRL_HATCH[ctrl], edgecolor="black", linewidth=0.4)
            bar_i += 1
    labels = []
    for c in cats:
        tags = [CTRL_LABEL[ctrl].split(" ")[0] for ctrl in CONTROLLERS
               if result["controllers"][ctrl]["baseline"] == c]
        labels.append(c + ("\n(" + "+".join(tags) + " baseline)" if tags else ""))
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    atom_handles = [Patch(facecolor=ATOM_COLORS[j], edgecolor="black", linewidth=0.4,
                          label=name.replace("_fulfillment", ""))
                    for j, name in enumerate(ATOM_NAMES)]
    ctrl_handles = [Patch(facecolor="white", edgecolor="black", hatch=CTRL_HATCH[ctrl],
                          label=CTRL_LABEL[ctrl]) for ctrl in CONTROLLERS]
    ax.legend(handles=atom_handles + ctrl_handles, fontsize=7.5, loc="best", ncol=2)
    ax.set_xlabel(f"{result['param']}  ({result['field']}, categorical)")
    ax.set_ylabel("fulfillment atom, mean over episode  [0,1]")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"Hopper vanilla vs FPL MPPI sensitivity: {result['param']} ({result['field']})\n"
        f"plain = vanilla, hatched = FPL   |   "
        f"{len(result['seeds'])} seeds/point, {result['steps']} steps (10s), 95% CI")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5,
                   help="seeds per grid point (protocol default: 5)")
    p.add_argument("--only", default=None,
                   help="comma-separated subset of param names, incl. 'p_value' "
                        "(default: all 6 shared params + p_value)")
    args = p.parse_args()

    seeds = range(args.seeds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    vanilla_cfg = load_config_file(str(VANILLA_CONFIG_PATH))
    assert vanilla_cfg.run.steps == STEPS, (
        f"vanilla config run.steps={vanilla_cfg.run.steps} != protocol length {STEPS}")

    fpl_cfg_raw = load_config_file(str(FPL_CONFIG_PATH))
    print(f"FPL flagship config validated at run.steps={fpl_cfg_raw.run.steps} "
         f"({fpl_cfg_raw.run.steps * 0.02:.1f}s); overriding to {STEPS} "
         f"({STEPS * 0.02:.1f}s) to match the shared protocol length (see module "
         f"docstring for the survival spot-check justifying this).")
    fpl_cfg = fpl_cfg_raw.with_(**{"run.steps": STEPS})

    cfgs = dict(vanilla=vanilla_cfg, fpl=fpl_cfg)
    tasks = {}
    for ctrl, cfg in cfgs.items():
        r = resolve(cfg)
        tasks[ctrl] = make_task(r.task_name, **r.task_kwargs)
        (OUT_DIR / f"{ctrl}_config.resolved.json").write_text(cfg.to_json())

    (OUT_DIR / "provenance.json").write_text(json.dumps(dict(
        sha=_git_sha(), dirty=_git_dirty(),
        config_hash=dict(vanilla=vanilla_cfg.hash(), fpl=fpl_cfg.hash()),
        note="fpl config_hash reflects run.steps overridden to match the shared "
             f"protocol length ({STEPS}), not the {fpl_cfg_raw.run.steps}-step flagship "
             "file on disk.",
    ), indent=2))

    all_specs = SWEEPS + [P_VALUE_SWEEP]
    specs = all_specs if args.only is None else [
        s for s in all_specs if s["name"] in set(args.only.split(","))]
    if not specs:
        raise SystemExit(f"--only matched nothing; known: {[s['name'] for s in all_specs]}")

    print(f"hopper vanilla-vs-FPL sensitivity sweep: {len(specs)} params, "
         f"{args.seeds} seeds/point/arm, {STEPS} steps ({STEPS * 0.02:.1f}s) per episode")
    t_start = time.time()
    for spec in specs:
        if spec["name"] == "p_value":
            print(f"\n[p_value] (FPL only) baseline={_fmt(_get_path(fpl_cfg, spec['field']))}"
                 f"  grid={[_fmt(v) for v in spec['grid']]}")
            result = sweep_param(fpl_cfg, spec, seeds, tasks["fpl"])
            (OUT_DIR / "p_value.json").write_text(json.dumps(result, indent=2))
            plot_numeric(
                result, OUT_DIR / "p_value.png", baseline_label="FPL config baseline",
                title=(
                    "Hopper FPL-MPPI sensitivity: objective.p (power-mean exponent)\n"
                    f"{len(result['seeds'])} seeds/point, {result['steps']} steps (10s), "
                    "95% CI"))
            print(f"  -> {(OUT_DIR / 'p_value.png').relative_to(REPO)}")
            continue

        print(f"\n[{spec['name']}] "
             f"vanilla baseline={_fmt(_get_path(vanilla_cfg, spec['field']))}  "
             f"FPL baseline={_fmt(_get_path(fpl_cfg, spec['field']))}  "
             f"grid={[_fmt(v) for v in spec['grid']]}")
        per_ctrl = {}
        for ctrl in CONTROLLERS:
            print(f"  -- {ctrl} --")
            per_ctrl[ctrl] = sweep_param(cfgs[ctrl], spec, seeds, tasks[ctrl])
        result = dict(
            param=spec["name"], field=spec["field"], grid=spec["grid"],
            categorical=spec["categorical"], seeds=list(seeds), steps=STEPS,
            atom_names=ATOM_NAMES,
            controllers={
                ctrl: dict(baseline=per_ctrl[ctrl]["baseline"], points=per_ctrl[ctrl]["points"])
                for ctrl in CONTROLLERS
            },
        )
        (OUT_DIR / f"{spec['name']}.json").write_text(json.dumps(result, indent=2))
        out_png = OUT_DIR / f"{spec['name']}.png"
        (plot_overlay_categorical if spec["categorical"] else plot_overlay_numeric)(
            result, out_png)
        print(f"  -> {out_png.relative_to(REPO)}")

    print(f"\ndone in {time.time() - t_start:.1f}s -> {OUT_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
