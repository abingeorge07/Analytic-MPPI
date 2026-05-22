"""Vanilla MPPI on the planar 'unicycle' (3-DOF base) in MuJoCo.

Drives the base from the origin to a goal in (x, y). Modes:

    python examples/unicycle_goal.py              # headless, save trajectory PNG
    python examples/unicycle_goal.py --live       # interactive viewer
    python examples/unicycle_goal.py --live --rollout
                                                  # ... with MPPI sample overlays
    python examples/unicycle_goal.py --record     # save runs/unicycle_goal.mp4
    python examples/unicycle_goal.py --record --rollout
                                                  # ... with sample overlays in video
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from analytic_mppi.controllers import MPPI
from analytic_mppi.costs import GoalReachCost
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.viz import plot_run, run_live, run_record


REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "analytic_mppi" / "dynamics" / "unicycle.xml"


def build():
    backend = MujocoBackend(MODEL_PATH)
    goal = np.array([2.0, 2.0])
    cost = GoalReachCost(
        goal=goal,
        state_xy_idx=(1, 2),
        Q=1.0,
        R=[0.01, 0.01, 0.001],
        terminal_weight=20.0,
    )
    ctl = MPPI(
        horizon=25,
        n_samples=512,
        nu=backend.nu,
        sigma=np.array([0.8, 0.8, 1.2]),
        lambda_=1.0,
        u_min=np.array([-2.0, -2.0, -3.0]),
        u_max=np.array([ 2.0,  2.0,  3.0]),
        seed=0,
    )
    return backend, cost, ctl, goal


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",    action="store_true",
                   help="open interactive MuJoCo viewer (real-time paced)")
    p.add_argument("--record",  action="store_true",
                   help="render headlessly and save mp4 to runs/")
    p.add_argument("--rollout", action="store_true",
                   help="overlay MPPI sample trajectories (requires --live or --record)")
    p.add_argument("--steps",   type=int, default=150)
    p.add_argument("--camera",  default="topdown",
                   help="MJCF camera name for --record")
    p.add_argument("--out",     default=str(REPO / "runs" / "unicycle_goal.mp4"),
                   help="output path for --record")
    args = p.parse_args()

    if args.live and args.record:
        p.error("choose one of --live or --record (cannot be combined)")
    if args.rollout and not (args.live or args.record):
        p.error("--rollout requires --live or --record")

    backend, cost, ctl, goal = build()
    print(f"Loaded {MODEL_PATH.name}: nq={backend.nq}, nv={backend.nv}, nu={backend.nu}, "
          f"nstate={backend.nstate}, dt={backend.dt}, threads={backend.nthread}")

    if args.live:
        print(f"Opening live viewer (steps={args.steps}, rollout={args.rollout})")
        run_live(backend, ctl, cost, goal, args.steps,
                 xy_idx=(1, 2), draw_rollouts=args.rollout)
        final_xy = backend.get_state()[[1, 2]]
        print(f"Final distance : {float(np.linalg.norm(final_xy - goal)):.4f}")
        return

    if args.record:
        out = Path(args.out)
        print(f"Recording {args.steps} steps to {out} (camera={args.camera}, "
              f"rollout={args.rollout}) ...")
        run_record(backend, ctl, cost, goal, args.steps, out,
                   xy_idx=(1, 2), draw_rollouts=args.rollout, camera=args.camera)
        final_xy = backend.get_state()[[1, 2]]
        print(f"Saved {out}")
        print(f"Final distance : {float(np.linalg.norm(final_xy - goal)):.4f}")
        return

    # default: headless run + matplotlib trajectory plot
    state = backend.get_state()
    states_hist = [state.copy()]
    ctrls_hist = []
    plan_times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        u = ctl.act(state, cost, backend)
        plan_times.append(time.perf_counter() - t0)
        state = backend.step(u)
        states_hist.append(state.copy())
        ctrls_hist.append(u.copy())

    states_hist = np.array(states_hist)
    ctrls_hist = np.array(ctrls_hist)
    final_xy = states_hist[-1, [1, 2]]
    dist = float(np.linalg.norm(final_xy - goal))

    print(f"Final position : ({final_xy[0]:+.3f}, {final_xy[1]:+.3f})   goal: "
          f"({goal[0]:+.3f}, {goal[1]:+.3f})")
    print(f"Final distance : {dist:.4f}")
    print(f"Planning time  : {1e3 * np.mean(plan_times):6.2f} ms / step  "
          f"(K={ctl.n_samples}, H={ctl.horizon}, threads={backend.nthread})")

    out = REPO / "runs" / "unicycle_goal.png"
    plot_run(states_hist, ctrls_hist, goal, xy_idx=(1, 2), theta_idx=3, save_path=out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
