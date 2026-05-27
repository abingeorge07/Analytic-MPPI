"""Unicycle env demo: vanilla MPPI driving the planar base to a goal.

    python -m analytic_mppi.envs.unicycle.run               # headless, save PNG
    python -m analytic_mppi.envs.unicycle.run --live        # interactive viewer
    python -m analytic_mppi.envs.unicycle.run --live --rollout
    python -m analytic_mppi.envs.unicycle.run --record      # save mp4 to runs/
    python -m analytic_mppi.envs.unicycle.run --record --rollout
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from analytic_mppi.controllers import list_controllers
from analytic_mppi.envs.unicycle import make_env
from analytic_mppi.viz import plot_run, run_live, run_record


# analytic_mppi/envs/unicycle/run.py -> parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO / "runs"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",    action="store_true",
                   help="open interactive MuJoCo viewer (real-time paced)")
    p.add_argument("--record",  action="store_true",
                   help="render headlessly and save mp4 to --out")
    p.add_argument("--rollout", action="store_true",
                   help="overlay MPPI sample trajectories (requires --live or --record)")
    p.add_argument("--variant", default="vanilla", choices=list_controllers(),
                   help="MPPI variant from analytic_mppi.controllers.CONTROLLERS")
    p.add_argument("--steps",   type=int, default=150)
    p.add_argument("--camera",  default="topdown",
                   help="MJCF camera name for --record")
    p.add_argument("--out",     default=str(DEFAULT_RUNS / "unicycle_goal.mp4"),
                   help="output path for --record")
    args = p.parse_args()

    if args.live and args.record:
        p.error("choose one of --live or --record (cannot be combined)")
    if args.rollout and not (args.live or args.record):
        p.error("--rollout requires --live or --record")

    env = make_env(variant=args.variant)
    print(f"unicycle env: nq={env.backend.nq}, nv={env.backend.nv}, nu={env.backend.nu}, "
          f"nstate={env.backend.nstate}, dt={env.backend.dt}, threads={env.backend.nthread}")
    print(f"controller   : variant={args.variant} -> {type(env.controller).__name__}")

    if args.live:
        print(f"Opening live viewer (steps={args.steps}, rollout={args.rollout})")
        run_live(env.backend, env.controller, env.cost, env.goal, args.steps,
                 xy_idx=env.xy_idx, draw_rollouts=args.rollout)
        final_xy = env.backend.get_state()[list(env.xy_idx)]
        print(f"Final distance : {float(np.linalg.norm(final_xy - env.goal)):.4f}")
        # Skip Python teardown -- launch_passive on Linux segfaults during
        # GL / MjModel destructor ordering at interpreter exit.
        os._exit(0)

    if args.record:
        out = Path(args.out)
        print(f"Recording {args.steps} steps to {out} "
              f"(camera={args.camera}, rollout={args.rollout}) ...")
        run_record(env.backend, env.controller, env.cost, env.goal, args.steps, out,
                   xy_idx=env.xy_idx, draw_rollouts=args.rollout, camera=args.camera)
        final_xy = env.backend.get_state()[list(env.xy_idx)]
        print(f"Saved {out}")
        print(f"Final distance : {float(np.linalg.norm(final_xy - env.goal)):.4f}")
        return

    # ---- default: headless run + matplotlib trajectory plot ----
    state = env.backend.get_state()
    states_hist = [state.copy()]
    ctrls_hist = []
    plan_times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        u = env.controller.act(state, env.cost, env.backend)
        plan_times.append(time.perf_counter() - t0)
        state = env.backend.step(u)
        states_hist.append(state.copy())
        ctrls_hist.append(u.copy())

    states_hist = np.array(states_hist)
    ctrls_hist = np.array(ctrls_hist)
    final_xy = states_hist[-1, list(env.xy_idx)]
    dist = float(np.linalg.norm(final_xy - env.goal))

    print(f"Final position : ({final_xy[0]:+.3f}, {final_xy[1]:+.3f})   "
          f"goal: ({env.goal[0]:+.3f}, {env.goal[1]:+.3f})")
    print(f"Final distance : {dist:.4f}")
    print(f"Planning time  : {1e3 * np.mean(plan_times):6.2f} ms / step  "
          f"(K={env.controller.n_samples}, H={env.controller.horizon})")

    out = DEFAULT_RUNS / "unicycle_goal.png"
    plot_run(states_hist, ctrls_hist, env.goal,
             xy_idx=env.xy_idx, theta_idx=env.theta_idx, save_path=out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
