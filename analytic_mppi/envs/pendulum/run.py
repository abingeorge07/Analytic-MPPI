"""Pendulum swing-up demo: vanilla MPPI swinging the link up to theta=pi.

    python -m analytic_mppi.envs.pendulum.run               # headless, save PNG
    python -m analytic_mppi.envs.pendulum.run --live        # interactive viewer
    python -m analytic_mppi.envs.pendulum.run --record      # save mp4 to runs/
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from analytic_mppi.controllers import list_controllers
from analytic_mppi.envs.pendulum import make_env
from analytic_mppi.viz import run_live, run_record


# analytic_mppi/envs/pendulum/run.py -> parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO / "runs"


def _plot_pendulum_run(
    states_history: np.ndarray,
    controls_history: np.ndarray,
    *,
    theta_idx: int,
    theta_dot_idx: int,
    dt: float,
    save_path: Path,
) -> None:
    """theta / theta_dot / control over time, with the upright target marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_state = dt * np.arange(states_history.shape[0])
    t_ctrl = dt * np.arange(controls_history.shape[0])
    theta = states_history[:, theta_idx]
    theta_dot = states_history[:, theta_dot_idx]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(t_state, theta, lw=1.6, color="tab:blue", label="theta")
    ax.axhline(np.pi, color="tab:red", ls="--", lw=1.0, label="upright (pi)")
    ax.set_ylabel("theta [rad]")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    ax.set_title("Pendulum swing-up")

    ax = axes[1]
    ax.plot(t_state, theta_dot, lw=1.4, color="tab:green")
    ax.set_ylabel("theta_dot [rad/s]")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for i in range(controls_history.shape[1]):
        ax.plot(t_ctrl, controls_history[:, i], lw=1.2, label=f"u[{i}]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("control")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _distance_to_upright(theta: float) -> float:
    """Same metric as InvPenCost._distance_to_upright (scalar)."""
    e = theta - np.pi
    return float((np.cos(e) - 1.0) ** 2 + np.sin(e) ** 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--live",    action="store_true",
                   help="open interactive MuJoCo viewer (real-time paced)")
    p.add_argument("--record",  action="store_true",
                   help="render headlessly and save mp4 to --out")
    p.add_argument("--variant", default="vanilla", choices=list_controllers(),
                   help="MPPI variant from analytic_mppi.controllers.CONTROLLERS")
    p.add_argument("--steps",   type=int, default=200)
    p.add_argument("--camera",  default="camera",
                   help="MJCF camera name for --record")
    p.add_argument("--out",     default=str(DEFAULT_RUNS / "pendulum_swingup.mp4"),
                   help="output path for --record")
    
    p.add_argument("--rollout",    action="store_true",
                   help="Overlays with the rollout")
    args = p.parse_args()

    if args.live and args.record:
        p.error("choose one of --live or --record (cannot be combined)")

    env = make_env(variant=args.variant)
    print(f"pendulum env: nq={env.backend.nq}, nv={env.backend.nv}, nu={env.backend.nu}, "
          f"nstate={env.backend.nstate}, dt={env.backend.dt}, threads={env.backend.nthread}")
    print(f"controller   : variant={args.variant} -> {type(env.controller).__name__}  "
          f"(K={env.controller.n_samples}, H={env.controller.horizon})")

    # The viewer helpers want a goal_xy for marker drawing; pendulum has no
    # xy goal, so park the marker at the world origin (under the base).
    marker_xy = (0.0, 0.0)

    if args.live:
        print(f"Opening live viewer (steps={args.steps})")
        run_live(env.backend, env.controller, env.cost, marker_xy, args.steps)
        final_theta = float(env.backend.get_state()[env.theta_idx])
        print(f"Final theta    : {final_theta:+.4f}  "
              f"(upright = {np.pi:+.4f}, dist^2 = {_distance_to_upright(final_theta):.4f})")
        # Skip Python teardown -- launch_passive on Linux segfaults during
        # GL / MjModel destructor ordering at interpreter exit.
        os._exit(0)

    if args.record:
        out = Path(args.out)
        print(f"Recording {args.steps} steps to {out} (camera={args.camera}) ...")
        run_record(env.backend, env.controller, env.cost, marker_xy, args.steps, out,
                   camera=args.camera, draw_rollouts=args.rollout)
        final_theta = float(env.backend.get_state()[env.theta_idx])
        print(f"Saved {out}")
        print(f"Final theta    : {final_theta:+.4f}  "
              f"(upright = {np.pi:+.4f}, dist^2 = {_distance_to_upright(final_theta):.4f})")
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
    final_theta = float(states_hist[-1, env.theta_idx])
    final_thetadot = float(states_hist[-1, env.theta_dot_idx])

    print(f"Final theta    : {final_theta:+.4f}  "
          f"(upright = {np.pi:+.4f}, dist^2 = {_distance_to_upright(final_theta):.4f})")
    print(f"Final theta_dot: {final_thetadot:+.4f}")
    print(f"Planning time  : {1e3 * np.mean(plan_times):6.2f} ms / step")

    out = DEFAULT_RUNS / "pendulum_swingup.png"
    _plot_pendulum_run(
        states_hist, ctrls_hist,
        theta_idx=env.theta_idx, theta_dot_idx=env.theta_dot_idx,
        dt=env.backend.dt, save_path=out,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
