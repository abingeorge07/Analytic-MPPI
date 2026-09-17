"""Live MuJoCo viewer for the new iLQR arm (and its baselines), for eyeball-level
verification that the controllers are running the way the tests say they are.

  ./a-mppi/bin/python scripts/live_demo.py --algo mppi          # fast, CPU, no compile
  ./a-mppi/bin/python scripts/live_demo.py --algo ilqr          # needs GPU jax, ~5-10 min first compile
  ./a-mppi/bin/python scripts/live_demo.py --algo warm_ilqr     # MPPI-warm-started iLQR (S11)

  --task hopper (default) | walker
  --arm  fpl (default)    | linear         (which cost the controller optimises)

Uses the *published* study spec (horizon/knots/noise/time_p per env, atom floor 1e-3),
CPU-stepped closed loop even for iLQR (invariant 11.2 — the loop is always CPU MuJoCo,
mjx is planning-only). Prints planner wall time and, for iLQR, its instrumentation
(acceptance rate, final alpha, lambda, J_post) each step so you can see the gates in
action live.

Ctrl-C or close the viewer window to stop. This script is not on the test/gate path;
it exists purely to make the abstract "the iLQR controller works" observable.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, "verification")
import _experiment                                                # noqa: E402

from analytic_mppi.controllers import MPPIv2                       # noqa: E402
from analytic_mppi.dynamics import MujocoBackend                   # noqa: E402
from analytic_mppi.tasks import make_task                          # noqa: E402


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="hopper", choices=["hopper", "walker"])
    p.add_argument("--algo", default="mppi", choices=["mppi", "ilqr", "warm_ilqr"])
    p.add_argument("--arm", default="fpl", choices=["fpl", "linear"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ilqr-iters", type=int, default=4,
                   help="iLQR iterations per act() (default 4)")
    p.add_argument("--K", type=int, default=None,
                   help="samples per step; default 256 (mppi) / 128 (warm_ilqr)")
    p.add_argument("--steps", type=int, default=None,
                   help="max steps; default = published episode length")
    p.add_argument("--floor", type=float, default=1e-3,
                   help="atom floor (default 1e-3, the campaign of record)")
    return p.parse_args()


def cost_kwargs(env: str, arm: str, floor: float) -> dict:
    spec = _experiment.ENVS[env]
    base = dict(use_fpl_cost=True, fpl_time_p=spec["time_p"], fpl_atom_floor=floor)
    if arm == "fpl":
        return dict(base, fpl_p=-1.0)
    return dict(base, fpl_p=1.0, fpl_weights=_experiment.linear_weights(env, 1.0))


def build_controller(args):
    spec = _experiment.ENVS[args.task]
    task = make_task(spec["task"], **{spec["difficulty_key"]: spec["difficulty"]})
    cpu = MujocoBackend(task.model_path)
    grid = dict(num_knots=spec["knots"], plan_horizon=spec["horizon"])
    cost = cost_kwargs(args.task, args.arm, args.floor)

    if args.algo == "mppi":
        # Zero-order spline is the campaign's default; the iLQR arms need "linear"
        # because their spline projection would overshoot on cubic and the fix is a
        # post-projection clip (confound). Kept per-arm honest.
        K = args.K or 256
        ctrl = MPPIv2(task, cpu, num_samples=K, noise_level=spec["noise"],
                      temperature=spec["temp"], seed=args.seed, iterations=1,
                      spline_type="zero", **grid, **cost)
        return task, cpu, ctrl

    # iLQR paths -- MJX backend for planning, CPU backend for the closed loop.
    from analytic_mppi.controllers.ilqr import ILQRMPC, WarmStartedILQR
    from analytic_mppi.dynamics.mjx_backend import MJXBackend
    mjxb = MJXBackend(task.model_path)
    print(f"[demo] compiling iLQR ({args.ilqr_iters} iters) -- 5-10 min first time...",
          flush=True)
    t0 = time.perf_counter()
    ilqr = ILQRMPC(task, mjxb, iterations=args.ilqr_iters, warmup=True,
                   spline_type="linear", **grid, **cost)
    print(f"[demo] iLQR warmup {time.perf_counter() - t0:.1f}s", flush=True)

    if args.algo == "ilqr":
        return task, cpu, ilqr

    K = args.K or 128
    from analytic_mppi.controllers import PredictiveSampling
    sampler = PredictiveSampling(task, cpu, num_samples=K, noise_level=spec["noise"],
                                 seed=args.seed, iterations=1,
                                 spline_type="linear", **grid, **cost)
    return task, cpu, WarmStartedILQR(sampler, ilqr)


def diag_line(step: int, dt_plan: float, ctrl) -> str:
    parts = [f"[{step:4d}] plan {1e3 * dt_plan:6.1f} ms"]
    if hasattr(ctrl, "ilqr"):
        i = ctrl.ilqr
        parts.append(f"acc {i.last_accept_rate:.2f} alpha {i.last_alpha_used:.3f} "
                     f"lam {i.last_lambda_trace[-1]:.1e} "
                     f"Jpost {i.last_J_post:.4f} resid {i.last_proj_resid:.3f}")
    elif hasattr(ctrl, "last_J_post"):
        parts.append(f"acc {ctrl.last_accept_rate:.2f} alpha {ctrl.last_alpha_used:.3f} "
                     f"lam {ctrl.last_lambda_trace[-1]:.1e} "
                     f"Jpost {ctrl.last_J_post:.4f} resid {ctrl.last_proj_resid:.3f}")
    else:
        parts.append(f"ess {ctrl.last_ess:.1f}")
    return "  ".join(parts)


def main() -> None:
    args = build_args()
    spec = _experiment.ENVS[args.task]
    print(f"[demo] {args.algo} on {args.task} / {args.arm} arm (floor {args.floor})",
          flush=True)

    task, backend, ctrl = build_controller(args)
    if spec["init"] is not None:
        spec["init"](backend)
    state = backend.get_state()
    steps = args.steps or spec["steps"]
    dt = float(backend.dt)

    import mujoco.viewer as mj_viewer
    viewer = mj_viewer.launch_passive(backend.model, backend.data,
                                      show_left_ui=False, show_right_ui=False)
    try:
        for step in range(steps):
            if not viewer.is_running():
                break
            tic = time.perf_counter()
            u = ctrl.act(state)
            dt_plan = time.perf_counter() - tic
            with viewer.lock():
                state = backend.step(u)
            if step % 10 == 0:
                print(diag_line(step, dt_plan, ctrl), flush=True)
            viewer.sync()
            wall = time.perf_counter() - tic
            if dt - wall > 0:
                time.sleep(dt - wall)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            viewer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
