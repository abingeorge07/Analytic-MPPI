"""Hopper demo: vanilla MPPI hopping the planar one-legged robot forward.

    python -m analytic_mppi.envs.hopper.run               # headless, save PNG
    python -m analytic_mppi.envs.hopper.run --live        # interactive viewer
    python -m analytic_mppi.envs.hopper.run --record      # save mp4 to runs/
    python -m analytic_mppi.envs.hopper.run --refine-steps 3   # + gradient refinement
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from analytic_mppi.controllers import list_controllers
from analytic_mppi.costs import HopperCost
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.envs.hopper import make_env
from analytic_mppi.viz import run_live, run_record


# analytic_mppi/envs/hopper/run.py -> parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO / "runs"


def _make_cost_grads(cost: HopperCost, backend: MujocoBackend):
    """Analytic gradients of HopperCost in the ordering `MPPI.refine_best` expects.

    refine_best wants gradients in mjd_transitionFD state ordering
    [qpos-tangent (nv), qvel (nv), act (na)], evaluated at a MuJoCo `data`.
    HopperCost indexes the FULLPHYSICS state ([time, qpos, qvel, ...]); we map
    those indices into the FD ordering below. This mapping (qpos-tangent ==
    d/dqpos) is valid because the hopper is hinge/slide only, so nq == nv.

    The objective mirrors HopperCost's per-step structure: running state +
    control cost at each step, with `terminal_weight` scaling the terminal
    state cost -- the natural smooth surrogate for refinement.
    """
    nq, nv = backend.nq, backend.nv
    nstate = 2 * nv + int(backend.model.na)

    # FULLPHYSICS index -> FD-state index, and -> the qpos/qvel slot the cost reads.
    h_fd, h_q = cost.height_idx - 1, cost.height_idx - 1   # height = qpos[1]
    o_fd, o_q = cost.orient_idx - 1, cost.orient_idx - 1   # orient = qpos[2]
    v_fd = nv + (cost.vx_idx - 1 - nq)                     # vx = qvel[0]; cost is linear in vx

    def grad_running_x(data) -> np.ndarray:
        g = np.zeros(nstate)
        g[h_fd] = 2.0 * cost.height_weight * (data.qpos[h_q] - cost.target_height)
        g[o_fd] = 2.0 * cost.orient_weight * data.qpos[o_q]
        g[v_fd] = -cost.forward_weight                              # d(-w*vx)/dvx
        return g

    def grad_running_u(u: np.ndarray) -> np.ndarray:
        return 2.0 * cost.control_weight * np.asarray(u, dtype=np.float64)

    def grad_terminal_x(data) -> np.ndarray:
        return cost.terminal_weight * grad_running_x(data)

    return grad_running_x, grad_running_u, grad_terminal_x


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
    p.add_argument("--refine-steps", type=int, default=0,
                   help="gradient-refine the best sampled rollout this many steps "
                        "each control step (0 = off; headless run only)")
    p.add_argument("--refine-lr", type=float, default=0.2,
                   help="gradient-refinement step size (MPPI.refine_best lr)")
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
    if args.refine_steps > 0 and (args.live or args.record):
        print("note: --refine-steps only affects the headless run; ignored for "
              "--live/--record")

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
    do_refine = args.refine_steps > 0
    if do_refine:
        # refine_best needs the stored samples (to pick the best one) plus a
        # MuJoCo model + a scratch data object and the cost's analytic gradients.
        env.controller.store_samples = True
        model = env.backend.model
        scratch = mujoco.MjData(model)
        grad_x, grad_u, grad_tx = _make_cost_grads(env.cost, env.backend)
        print(f"refinement   : {args.refine_steps} grad step(s)/control step, "
              f"lr={args.refine_lr}")

    state = env.backend.get_state()
    states_hist = [state.copy()]
    ctrls_hist = []
    plan_times = []
    refine_times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        u = env.controller.act(state, env.cost, env.backend)
        plan_times.append(time.perf_counter() - t0)
        if do_refine:
            tr = time.perf_counter()
            U_ref = env.controller.refine_best(
                model, scratch,
                state[env.backend.qpos_slice], state[env.backend.qvel_slice],
                grad_x, grad_u, grad_tx,
                n_steps=args.refine_steps, lr=args.refine_lr,
                apply_to_nominal=True,   # warm-start next step from the refined plan
            )
            refine_times.append(time.perf_counter() - tr)
            u = U_ref[0].copy()          # execute the refined first control
        state = env.backend.step(u)
        states_hist.append(state.copy())
        ctrls_hist.append(u.copy())

    states_hist = np.array(states_hist)
    ctrls_hist = np.array(ctrls_hist)

    _report_final(states_hist[-1])
    print(f"Planning time  : {1e3 * np.mean(plan_times):6.2f} ms / step")
    if do_refine:
        print(f"Refine time    : {1e3 * np.mean(refine_times):6.2f} ms / step")

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
