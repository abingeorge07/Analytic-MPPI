"""Capability probe — establish what a controller can actually DO on an env, before
any cost-representation ablation runs on top of it.

Why this exists
---------------
A cost comparison is only meaningful in the regime where the optimizer has enough
capability to discriminate. If both cost arms fail (floor) or both saturate (ceiling),
whatever separates them is noise. `g1_walk` is the in-repo example: both controllers
collapse, which is a task-capability gap, not a method result.

So this script answers one question per (env, sampler): *is the robot doing a good job,
and where is the ceiling?* It sweeps the knobs that plausibly move absolute performance
at a FIXED sample budget — spline parameterization, temperature, injected noise — reports
absolute metrics against the task's own target, and records a pass/fail against a
capability gate declared at the top of this file.

The gate is deliberately a code artifact, fixed BEFORE the cost comparison runs and
applied identically to every env, so that excluding an env is a pre-registered decision
rather than a post-hoc one. Envs that fail the gate are reported, not silently dropped.

Artifacts
---------
`--save` writes a self-describing bundle per config so a video can always be traced back
to the exact model and settings that produced it:

    runs/capability/<env>/<tag>/
        video.mp4          closed-loop episode
        config.json        sampler, cost, budget, every build kwarg, seed, env spec
        metrics.json       per-seed and aggregate outcome metrics
        model/scene.xml    the entry-point MJCF, verbatim
        model/INFO.json    source path, sha256, mujoco version, git commit

Meshes are NOT vendored (barkour/cube/g1 carry 9-35 MB of them); the git commit plus the
scene hash pins them instead.

Usage
-----
    python verification/capability_probe.py --env hopper --sampler mppi           # sweep
    python verification/capability_probe.py --env hopper --sampler mppi --save    # + video
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Offscreen rendering needs a GL context before mujoco is imported anywhere. This box has
# no working EGL device and no OSMesa, but it does have an X server on :1, so glfw against
# that display is the backend that works. Set only if the caller hasn't chosen already.
os.environ.setdefault("DISPLAY", ":1")
os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Run as a script from anywhere: the repo root must be importable for `analytic_mppi`,
# and this directory for the sibling `_experiment` helpers.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import ENVS, run_trial  # noqa: E402

RUNS = REPO / "runs" / "capability"


# --- capability gate (PRE-REGISTERED: fix before the cost comparison, do not revisit) ---
# An env enters the cost-representation comparison only if the BEST configuration found
# here clears both bars. `frac_of_target` is measured against the env's own difficulty
# setting (target velocity / target angle), so the bar means the same thing across
# morphologies. Envs that fail are reported as capability-gated, with the failure treated
# as a scoped boundary condition rather than as evidence for or against any cost.
GATE = dict(frac_of_target=0.50, survival=0.80)

# Preferred rendering camera per env. These are the MJCF's own `mode="trackcom"` cameras,
# which follow the robot's centre of mass from a fixed side offset — so a locomoting robot
# stays framed instead of walking out of shot, which the default free camera lets it do.
# An env with no entry (or a name the model doesn't define) falls back to the free camera.
CAMERAS = dict(hopper="track", walker="floating", quadruped="track")

# Knob grid. These are the axes that move absolute capability at a fixed budget: the
# action parameterization, the softmax selectivity, and the exploration magnitude.
GRID = dict(
    spline_type=["zero", "cubic"],
    temperature=[0.05, 0.1, 0.2],
    noise_level=[0.3, 0.6],
)


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_dt(env: str) -> float:
    """Control period of the env's backend, for converting seconds <-> steps."""
    from analytic_mppi.dynamics import MujocoBackend
    from analytic_mppi.eval import make_task
    return float(MujocoBackend(make_task(ENVS[env]["task"]).model_path).dt)


def evaluate(env: str, sampler: str, cost: str, K: int, seeds: int,
             extra: Dict[str, Any], steps: Optional[int] = None) -> Dict[str, Any]:
    """Run `seeds` episodes of one configuration; return per-seed and aggregate metrics."""
    rows: List[Dict[str, Any]] = []
    for s in range(seeds):
        try:
            rows.append(run_trial(env, sampler, cost, seed=s, K=K, steps=steps,
                                  extra=dict(extra)))
        except Exception as exc:  # a broken cell is data, not a reason to abort the sweep
            return dict(error=f"{type(exc).__name__}: {exc}", extra=extra, rows=[])

    def agg(key: str) -> float:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    # `prod` is speed x survival for locomotion, achieved-angle x survival for the cube:
    # the env's own productive-outcome metric, defined in _experiment._metrics.
    out = dict(extra=extra, rows=rows, n_seeds=len(rows),
               prod=agg("prod"), survived=agg("survived"), ess=agg("ess"))
    out["primary"] = agg("vx") if "vx" in rows[0] else agg("achieved")
    out["survival_lb"] = wilson_lb(int(round(out["survived"] * len(rows))), len(rows))
    return out


def sweep(env: str, sampler: str, cost: str, K: int, seeds: int,
          steps: Optional[int] = None,
          fixed: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Sweep GRID, with `fixed` merged into every cell (structural settings such as
    plan_horizon / num_knots that are held constant rather than swept)."""
    keys = list(GRID)
    results = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        extra = dict(fixed or {})
        extra.update(zip(keys, combo))
        results.append(evaluate(env, sampler, cost, K, seeds, extra, steps=steps))
    return results


def wilson_lb(k: int, n: int, z: float = 1.96) -> float:
    """Lower bound of the Wilson 95% interval for k successes in n trials."""
    if n == 0:
        return float("nan")
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half


def gate_verdict(env: str, best: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the pre-registered gate. Target is the env's own difficulty setting.

    Survival is judged on the LOWER BOUND of its Wilson interval, not the point estimate:
    27/30 and 30/30 both read as ">= 0.80" pointwise, but only the latter is distinguishable
    from a controller that fails one run in five at this sample size.
    """
    target = float(ENVS[env]["difficulty"])
    frac = (best["primary"] / target) if target else float("nan")
    lb = best.get("survival_lb", float("nan"))
    passed = bool(frac >= GATE["frac_of_target"] and lb >= GATE["survival"])
    return dict(target=target, frac_of_target=frac, survival=best["survived"],
                survival_lb=lb, n_seeds=best.get("n_seeds"), gate=dict(GATE),
                passed=passed)


def save_bundle(env: str, sampler: str, cost: str, K: int, seeds: int,
                best: Dict[str, Any], verdict: Dict[str, Any], tag: str,
                steps: Optional[int] = None, camera: Optional[str] = None) -> Path:
    """Write video + config + metrics + model provenance into one clean directory."""
    from analytic_mppi.eval import make_task, render_video
    from _experiment import cost_kwargs, sampler_kwargs

    spec = ENVS[env]
    steps = spec["steps"] if steps is None else steps
    out_dir = RUNS / env / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "model").mkdir(parents=True)

    build = dict(num_samples=K, plan_horizon=spec["horizon"], num_knots=spec["knots"],
                 spline_type="zero",
                 **sampler_kwargs(sampler, env, K), **cost_kwargs(cost, env))
    build.update(best["extra"])

    task = make_task(spec["task"], **{spec["difficulty_key"]: spec["difficulty"]})
    src = Path(task.model_path)
    shutil.copy2(src, out_dir / "model" / src.name)

    import mujoco
    (out_dir / "model" / "INFO.json").write_text(json.dumps(dict(
        source_path=str(src), sha256=_sha256(src), mujoco_version=mujoco.__version__,
        git_commit=_git_commit(),
        note="Entry-point MJCF only; meshes are pinned by git_commit, not vendored.",
    ), indent=2))

    (out_dir / "config.json").write_text(json.dumps(dict(
        env=env, task=spec["task"], sampler=sampler, cost=cost, budget_K=K,
        steps=steps, seeds=seeds, camera=(CAMERAS.get(env) if camera is None else camera),
        difficulty={spec["difficulty_key"]: spec["difficulty"]},
        build_kwargs={k: v for k, v in build.items()}, env_spec={
            k: v for k, v in spec.items() if k != "init"},
        git_commit=_git_commit(),
    ), indent=2, default=str))

    (out_dir / "metrics.json").write_text(json.dumps(dict(
        best_config=best["extra"], primary=best["primary"], prod=best["prod"],
        survived=best["survived"], ess=best["ess"], per_seed=best["rows"],
        verdict=verdict,
    ), indent=2, default=str))

    # Rendering needs an offscreen GL context, which a headless machine may not have.
    # The measurement bundle is the artifact that matters, so a missing context degrades
    # to a recorded note instead of losing the whole run.
    cam = CAMERAS.get(env) if camera is None else camera
    try:
        render_video(spec["task"], sampler, steps=steps, out_path=out_dir / "video.mp4",
                     seed=0, cost_mode=("normal" if cost == "normal" else "fpl_cost"),
                     init_fn=spec["init"], camera=cam,
                     task_kwargs={spec["difficulty_key"]: spec["difficulty"]}, **build)
    except Exception as exc:
        (out_dir / "VIDEO_UNAVAILABLE.txt").write_text(
            f"{type(exc).__name__}: {exc}\n\n"
            "No usable GL context. The bundle's config.json fully specifies the run, so\n"
            "it can be re-rendered later. This machine's working combination is\n"
            "DISPLAY=:1 with MUJOCO_GL=glfw (the module-level default); EGL has no\n"
            "device here and OSMesa is not installed.\n"
        )
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="hopper", choices=sorted(ENVS))
    p.add_argument("--sampler", default="mppi")
    p.add_argument("--cost", default="fpl")
    p.add_argument("-K", type=int, default=256)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--seconds", type=float, default=None,
                   help="episode length in seconds (converted via the env's control "
                        "period); applies to BOTH the sweep and the video so the "
                        "measured config is the one you watch")
    p.add_argument("--steps", type=int, default=None,
                   help="episode length in control steps; overrides --seconds")
    p.add_argument("--fixed", default=None, metavar="JSON",
                   help='build kwargs held constant across every grid cell, e.g. '
                        '\'{"plan_horizon": 0.9, "num_knots": 6}\'')
    p.add_argument("--save", action="store_true", help="write video + artifact bundle")
    p.add_argument("--tag", default=None, help="bundle directory name")
    p.add_argument("--camera", default=None,
                   help="MJCF camera name for the video; default is the env's tracking "
                        "camera (see CAMERAS), 'free' forces the static default view")
    args = p.parse_args()

    steps = args.steps
    if steps is None and args.seconds is not None:
        steps = max(1, int(round(args.seconds / env_dt(args.env))))
    eff_steps = ENVS[args.env]["steps"] if steps is None else steps

    fixed = json.loads(args.fixed) if args.fixed else None
    results = sweep(args.env, args.sampler, args.cost, args.K, args.seeds, steps=steps,
                    fixed=fixed)
    ok = [r for r in results if "error" not in r]

    print(f"\n{args.env} / {args.sampler} / {args.cost} / K={args.K} / {args.seeds} seeds "
          f"/ {eff_steps} steps ({eff_steps * env_dt(args.env):.1f}s)"
          + (f" / fixed={fixed}" if fixed else ""))
    print(f"{'spline':7s} {'temp':>5s} {'noise':>5s} {'primary':>8s} {'surv':>5s} "
          f"{'surv_lb':>7s} {'prod':>7s} {'ess':>6s}")
    for r in sorted(results, key=lambda x: -(x.get("prod") or -np.inf)):
        e = r["extra"]
        if "error" in r:
            print(f"{e.get('spline_type',''):7s} {e.get('temperature',''):>5} "
                  f"{e.get('noise_level',''):>5}   ERROR  {r['error'][:60]}")
            continue
        print(f"{e['spline_type']:7s} {e['temperature']:>5} {e['noise_level']:>5} "
              f"{r['primary']:>+8.3f} {r['survived']:>5.2f} {r['survival_lb']:>7.2f} "
              f"{r['prod']:>+7.3f} {r['ess']:>6.1f}")

    if not ok:
        print("\nAll cells errored — nothing to gate.")
        return

    # Selection is feasibility-first, not composite-first. Ranking by `prod` alone lets a
    # config that falls 1 run in 10 outrank a config that never falls, on a ~2% speed edge
    # -- the composite trades survival for speed at a fixed exchange rate nobody chose.
    # So: keep only configs whose survival lower bound clears the gate, then maximise prod
    # among those. If none clear it, fall back to max prod and say so, since the honest
    # report is "nothing passed" rather than a silently relabelled winner.
    feasible = [r for r in ok if r["survival_lb"] >= GATE["survival"]]
    if feasible:
        best = max(feasible, key=lambda r: r["prod"])
    else:
        best = max(ok, key=lambda r: r["prod"])
        print("\n[no config's survival lower bound clears the gate; showing best prod]")
    verdict = gate_verdict(args.env, best)
    print(f"\nbest: {best['extra']}")
    print(f"  primary {best['primary']:+.3f} vs target {verdict['target']:.2f} "
          f"= {verdict['frac_of_target']:.0%} of target, survival {best['survived']:.2f} "
          f"(Wilson95 lower bound {best['survival_lb']:.2f}, n={best['n_seeds']})")
    print(f"  gate (>={GATE['frac_of_target']:.0%} of target, >={GATE['survival']:.0%} "
          f"survival lower bound): "
          f"{'PASS' if verdict['passed'] else 'FAIL — capability-gated'}")

    if args.save:
        tag = args.tag or f"{args.sampler}_{args.cost}_K{args.K}"
        # "free" is the escape hatch for the static default view; anything else is passed
        # through to the model and silently falls back if the name isn't defined.
        cam = None if args.camera is None else ("" if args.camera == "free" else args.camera)
        out = save_bundle(args.env, args.sampler, args.cost, args.K, args.seeds,
                          best, verdict, tag, steps=steps, camera=cam)
        print(f"\nbundle: {out.relative_to(REPO)}")
        for f in sorted(out.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(out)}  ({f.stat().st_size:,} B)")


if __name__ == "__main__":
    main()
