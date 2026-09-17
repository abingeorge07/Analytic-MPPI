"""Headless MuJoCo recorder for the Phase 3 iLQR arm (and its baselines).

Mirrors ``scripts/live_demo.py`` but renders to an mp4 via ``mujoco.Renderer`` +
``imageio``. The closed loop is CPU-stepped in every case (invariant 11.2 — the
loop is always CPU MuJoCo, mjx is planning-only).

Examples
--------
    # pendulum swing-up with iLQR (normal quadratic-style cost, the G8 setup)
    ./a-mppi/bin/python scripts/record_demo.py --task pendulum --algo ilqr

    # walker with the S12 published-result controller (WarmStartedILQR + FPL)
    ./a-mppi/bin/python scripts/record_demo.py --task walker --algo warm_ilqr

    # hopper baseline sampling MPPI, for a sanity mp4 next to the iLQR clip
    ./a-mppi/bin/python scripts/record_demo.py --task hopper --algo mppi

First iLQR run compiles ~5-10 min. `runs/demos/{task}_{algo}_{arm}.mp4` unless
--out is given.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# _experiment lives under verification/ — same trick as live_demo.py.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "verification"))
import _experiment                                                    # noqa: E402

from analytic_mppi.controllers import MPPIv2                          # noqa: E402
from analytic_mppi.dynamics import MujocoBackend                      # noqa: E402
from analytic_mppi.tasks import make_task                             # noqa: E402


# Pendulum is NOT in _experiment.ENVS (that spec targets locomotion tasks).
# Hand-tuned iLQR-friendly spec: horizon long enough for a full swing-up under
# torque limit, few knots, linear spline (invariant per ILQRMPC docstring).
PENDULUM_SPEC = dict(
    task="pendulum",
    horizon=1.0,
    knots=5,
    noise=0.5,       # for MPPI comparison only
    temp=1.0,        # for MPPI comparison only
    steps=250,       # 2.5 s at dt=0.01
    difficulty_key=None,
    difficulty=None,
    time_p=None,     # unused with use_fpl_cost=False
)


def _spec(task_name: str) -> dict:
    if task_name == "pendulum":
        return PENDULUM_SPEC
    return _experiment.ENVS[task_name]


def _cost_kwargs(task_name: str, arm: str, floor: float) -> dict:
    """Match live_demo.cost_kwargs but add a `normal` arm for pendulum."""
    if task_name == "pendulum":
        if arm != "normal":
            print(f"[demo] pendulum uses the normal (non-FPL) quadratic-style cost; "
                  f"ignoring --arm {arm}", flush=True)
        return dict(use_fpl_cost=False)

    spec = _experiment.ENVS[task_name]
    base = dict(use_fpl_cost=True, fpl_time_p=spec["time_p"], fpl_atom_floor=floor)
    if arm == "fpl":
        return dict(base, fpl_p=-1.0)
    if arm == "linear":
        return dict(base, fpl_p=1.0, fpl_weights=_experiment.linear_weights(task_name, 1.0))
    raise ValueError(f"arm={arm!r} not valid for task={task_name!r}")


def _build(args):
    spec = _spec(args.task)
    if args.task == "pendulum":
        task = make_task("pendulum")
    else:
        task = make_task(spec["task"], **{spec["difficulty_key"]: spec["difficulty"]})

    cpu = MujocoBackend(task.model_path)
    grid = dict(num_knots=spec["knots"], plan_horizon=spec["horizon"])
    cost = _cost_kwargs(args.task, args.arm, args.floor)

    if args.algo == "mppi":
        K = args.K or 256
        noise = spec.get("noise", 0.5)
        temp = spec.get("temp", 1.0)
        ctrl = MPPIv2(task, cpu, num_samples=K, noise_level=noise, temperature=temp,
                      seed=args.seed, iterations=1, spline_type="zero",
                      **grid, **cost)
        return task, cpu, ctrl

    # iLQR paths — MJX backend for planning, CPU backend for the closed loop.
    from analytic_mppi.controllers.ilqr import ILQRMPC, WarmStartedILQR
    from analytic_mppi.dynamics.mjx_backend import MJXBackend
    mjxb = MJXBackend(task.model_path)
    print(f"[demo] compiling iLQR ({args.ilqr_iters} iters) — 5-10 min first time...",
          flush=True)
    t0 = time.perf_counter()
    ilqr = ILQRMPC(task, mjxb, iterations=args.ilqr_iters, warmup=True,
                   spline_type="linear", **grid, **cost)
    print(f"[demo] iLQR warmup {time.perf_counter() - t0:.1f}s", flush=True)

    if args.algo == "ilqr":
        return task, cpu, ilqr

    # warm_ilqr — PredictiveSampling warm-starts ILQR (S11 setup).
    K = args.K or 128
    from analytic_mppi.controllers import PredictiveSampling
    sampler = PredictiveSampling(task, cpu, num_samples=K, noise_level=spec["noise"],
                                 seed=args.seed, iterations=1,
                                 spline_type="linear", **grid, **cost)
    return task, cpu, WarmStartedILQR(sampler, ilqr)


def _diag_line(step: int, dt_plan: float, ctrl) -> str:
    parts = [f"[{step:4d}] plan {1e3 * dt_plan:6.1f} ms"]
    if hasattr(ctrl, "ilqr"):                # WarmStartedILQR
        i = ctrl.ilqr
        parts.append(f"acc {i.last_accept_rate:.2f} alpha {i.last_alpha_used:.3f} "
                     f"lam {i.last_lambda_trace[-1]:.1e} Jpost {i.last_J_post:.4f} "
                     f"resid {i.last_proj_resid:.3f}")
    elif hasattr(ctrl, "last_J_post"):       # ILQRMPC
        parts.append(f"acc {ctrl.last_accept_rate:.2f} alpha {ctrl.last_alpha_used:.3f} "
                     f"lam {ctrl.last_lambda_trace[-1]:.1e} Jpost {ctrl.last_J_post:.4f} "
                     f"resid {ctrl.last_proj_resid:.3f}")
    else:                                    # MPPI
        parts.append(f"ess {ctrl.last_ess:.1f}")
    return "  ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="pendulum",
                   choices=["pendulum", "hopper", "walker"])
    p.add_argument("--algo", default="ilqr",
                   choices=["mppi", "ilqr", "warm_ilqr"])
    p.add_argument("--arm", default="fpl",
                   choices=["fpl", "linear", "normal"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ilqr-iters", type=int, default=6,
                   help="iLQR iterations per act() (default 6; 4 in live_demo)")
    p.add_argument("--K", type=int, default=None,
                   help="samples per step; default 256 (mppi) / 128 (warm_ilqr)")
    p.add_argument("--steps", type=int, default=None,
                   help="max steps; default = spec's episode length")
    p.add_argument("--floor", type=float, default=1e-3,
                   help="atom floor (default 1e-3, the S12 campaign of record)")
    p.add_argument("--out", type=str, default=None,
                   help="output mp4 path (default: runs/demos/{task}_{algo}_{arm}.mp4)")
    p.add_argument("--camera", type=str, default=None,
                   help="mujoco camera name or index; default = free camera (-1)")
    p.add_argument("--width", type=int, default=1280,
                   help="frame width in pixels (default 1280; model offwidth "
                        "is overridden at runtime to fit)")
    p.add_argument("--height", type=int, default=720,
                   help="frame height in pixels (default 720; model offheight "
                        "is overridden at runtime to fit)")
    args = p.parse_args()

    print(f"[demo] {args.algo} on {args.task} / {args.arm} arm (floor {args.floor})",
          flush=True)
    task, backend, ctrl = _build(args)
    spec = _spec(args.task)
    if spec.get("init") is not None:
        spec["init"](backend)
    state = backend.get_state()
    steps = args.steps or spec["steps"]

    import mujoco
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise RuntimeError('Recording needs imageio. Install with: pip install -e ".[viz]"') from e

    tag = args.arm if args.task != "pendulum" else "normal"
    default_out = _REPO / "runs" / "demos" / f"{args.task}_{args.algo}_{tag}.mp4"
    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)

    fps = max(1, int(round(1.0 / backend.dt)))
    cam = args.camera
    if cam is not None:
        try:
            cam = int(cam)
        except ValueError:
            pass  # a named camera; renderer accepts str too
    else:
        cam = -1

    # MuJoCo's default offscreen framebuffer is 640x480. Bump the loaded model's
    # <visual><global offwidth=... offheight=.../> to match the requested render
    # size so `mujoco.Renderer` doesn't clamp/error on our resolution request.
    backend.model.vis.global_.offwidth = int(args.width)
    backend.model.vis.global_.offheight = int(args.height)

    frames = []
    renderer = mujoco.Renderer(backend.model, width=args.width, height=args.height)
    try:
        for step in range(steps):
            tic = time.perf_counter()
            u = ctrl.act(state)
            dt_plan = time.perf_counter() - tic
            state = backend.step(u)
            renderer.update_scene(backend.data, camera=cam)
            frames.append(renderer.render().copy())
            if step % 10 == 0:
                print(_diag_line(step, dt_plan, ctrl), flush=True)
    finally:
        renderer.close()

    imageio.mimsave(out, frames, fps=fps, codec="libx264", quality=8,
                    macro_block_size=None)
    print(f"[demo] saved {out} ({len(frames)} frames @ {fps} fps)", flush=True)


if __name__ == "__main__":
    main()
