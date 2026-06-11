"""Hopper demo: vanilla MPPI hopping the planar one-legged robot forward.

    python -m analytic_mppi.envs.hopper.run               # headless, save PNG
    python -m analytic_mppi.envs.hopper.run --live        # interactive viewer
    python -m analytic_mppi.envs.hopper.run --record      # save mp4 to runs/
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from analytic_mppi.controllers import list_controllers
from analytic_mppi.envs.hopper import make_env
from analytic_mppi.viz import run_live, run_record


# analytic_mppi/envs/hopper/run.py -> parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO / "runs"


def _plot_hopper_run(
    states_history: np.ndarray,
    controls_history: np.ndarray,
    *,
    x_idx: int,
    height_idx: int,
    orient_idx: int,
    vx_idx: int,
    target_height: float,
    dt: float,
    save_path: Path,
) -> None:
    """Forward position / forward velocity / height / pitch / control over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_state = dt * np.arange(states_history.shape[0])
    t_ctrl = dt * np.arange(controls_history.shape[0])
    x = states_history[:, x_idx]
    height = states_history[:, height_idx]
    pitch = states_history[:, orient_idx]
    vx = states_history[:, vx_idx]

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 9.5), sharex=True)

    ax = axes[0]
    ax.plot(t_state, x, lw=1.6, color="tab:blue", label="x (forward)")
    ax.plot(t_state, vx, lw=1.4, color="tab:orange", label="vx")
    ax.set_ylabel("forward")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    ax.set_title("Hopper forward hopping")

    ax = axes[1]
    ax.plot(t_state, height, lw=1.6, color="tab:green", label="height")
    ax.axhline(target_height, color="tab:red", ls="--", lw=1.0,
               label=f"target ({target_height:g})")
    ax.set_ylabel("height [m]")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t_state, pitch, lw=1.4, color="tab:purple")
    ax.axhline(0.0, color="tab:red", ls="--", lw=1.0, label="upright (0)")
    ax.set_ylabel("pitch [rad]")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    ax = axes[3]
    mean_ctrl = controls_history.mean(axis=1)
    ax.plot(t_ctrl, mean_ctrl, lw=1.4, color="tab:blue", label="mean over joints")
    ax.set_xlabel("time [s]"); ax.set_ylabel("control")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",    action="store_true",
                   help="open interactive MuJoCo viewer (real-time paced)")
    p.add_argument("--record",  action="store_true",
                   help="render headlessly and save mp4 to --out")
    p.add_argument("--variant", default="vanilla", choices=list_controllers(),
                   help="MPPI variant from analytic_mppi.controllers.CONTROLLERS")
    p.add_argument("--steps",   type=int, default=300)
    p.add_argument("--camera",  default="track",
                   help="MJCF camera name for --record")
    p.add_argument("--out",     default=str(DEFAULT_RUNS / "hopper_hop.mp4"),
                   help="output path for --record")
    p.add_argument("--rollout", action="store_true",
                   help="overlay MPPI sample trajectories (requires --record)")
    args = p.parse_args()

    if args.live and args.record:
        p.error("choose one of --live or --record (cannot be combined)")
    if args.rollout and not args.record:
        p.error("--rollout requires --record")

    env = make_env(variant=args.variant)
    print(f"hopper env  : nq={env.backend.nq}, nv={env.backend.nv}, nu={env.backend.nu}, "
          f"nstate={env.backend.nstate}, dt={env.backend.dt}, threads={env.backend.nthread}")
    print(f"controller   : variant={args.variant} -> {type(env.controller).__name__}  "
          f"(K={env.controller.n_samples}, H={env.controller.horizon})")

    # The viewer helpers want a goal_xy for marker drawing; the hopper has no
    # xy goal, so park the marker at the world origin.
    marker_xy = (0.0, 0.0)

    def _report_final(state: np.ndarray) -> None:
        print(f"Final x        : {float(state[env.x_idx]):+.4f}")
        print(f"Final height   : {float(state[env.height_idx]):+.4f}  "
              f"(target = {env.cost.target_height:+.4f})")
        print(f"Final vx       : {float(state[env.vx_idx]):+.4f}")

    if args.live:
        print(f"Opening live viewer (steps={args.steps})")
        run_live(env.backend, env.controller, env.cost, marker_xy, args.steps,
                 draw_rollouts=False)
        _report_final(env.backend.get_state())
        # Skip Python teardown -- launch_passive on Linux segfaults during
        # GL / MjModel destructor ordering at interpreter exit.
        os._exit(0)

    if args.record:
        out = Path(args.out)
        print(f"Recording {args.steps} steps to {out} "
              f"(camera={args.camera}, rollout={args.rollout}) ...")
        run_record(env.backend, env.controller, env.cost, marker_xy, args.steps, out,
                   camera=args.camera, draw_rollouts=args.rollout)
        print(f"Saved {out}")
        _report_final(env.backend.get_state())
        return

    # ---- default: headless run + matplotlib trace plot ----
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

    _report_final(states_hist[-1])
    print(f"Planning time  : {1e3 * np.mean(plan_times):6.2f} ms / step")

    out = DEFAULT_RUNS / "hopper_hop.png"
    _plot_hopper_run(
        states_hist, ctrls_hist,
        x_idx=env.x_idx, height_idx=env.height_idx,
        orient_idx=env.orient_idx, vx_idx=env.vx_idx,
        target_height=env.cost.target_height,
        dt=env.backend.dt, save_path=out,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
