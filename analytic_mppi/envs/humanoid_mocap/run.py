"""G1 mocap walking demo: MPPI tracking a retargeted motion-capture clip.

    python -m analytic_mppi.envs.humanoid_mocap.run             # headless, save PNG
    python -m analytic_mppi.envs.humanoid_mocap.run --live      # interactive viewer
    python -m analytic_mppi.envs.humanoid_mocap.run --record    # save mp4 to runs/

The G1 uses PD position actuators (ctrl = target joint angle), so by default the
MPPI nominal is warm-started each step from the reference joint targets
(feedforward) and MPPI adds the feedback correction. Pass --no-feedforward to
sample from a zero nominal instead (much harder; useful for comparison).
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from analytic_mppi.controllers import list_controllers
from analytic_mppi.envs.humanoid_mocap import make_env
from analytic_mppi.viz import run_live, run_record


# analytic_mppi/envs/humanoid_mocap/run.py -> parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO / "runs"


class _ReferenceFeedforward:
    """Wrap a controller so each act() warm-starts its nominal from the reference
    joint-angle trajectory over the horizon (feedforward), then runs MPPI.

    Delegates all other attribute access to the wrapped controller so the viz
    helpers (which read e.g. `last_samples`, `horizon`) keep working.
    """

    def __init__(self, inner, cost, dt: float):
        self._inner = inner
        self._cost = cost
        self._dt = float(dt)

    def act(self, state, cost_fn, backend):
        H = self._inner.horizon
        t0 = float(state[0])
        ts = t0 + self._dt * np.arange(H)
        self._inner.nominal = self._cost.reference_action(ts).copy()   # (H, nu)
        return self._inner.act(state, cost_fn, backend)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _camera_arg(value: str) -> str | int:
    v = value.strip()
    return int(v) if v.lstrip("-").isdigit() else v


def _reference_qpos_at(cost, t: np.ndarray) -> np.ndarray:
    i = cost._frame_index(np.asarray(t))
    return cost.reference_qpos[i]


def _plot_mocap_run(states_hist, ctrls_hist, cost, *, qpos_slice, dt, save_path):
    """Actual vs reference base path/height, configuration error, mean control."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_state = states_hist[:, 0]                       # sim time recorded per step
    qpos = states_hist[:, qpos_slice]                 # (T, nq)
    ref = _reference_qpos_at(cost, t_state)           # (T, nq)

    base_xy, ref_xy = qpos[:, 0:2], ref[:, 0:2]
    height, ref_h = qpos[:, 2], ref[:, 2]
    config_err = np.linalg.norm(qpos - ref, axis=-1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.plot(ref_xy[:, 0], ref_xy[:, 1], "--", color="tab:red", lw=1.4, label="reference")
    ax.plot(base_xy[:, 0], base_xy[:, 1], color="tab:blue", lw=1.6, label="actual")
    ax.set_xlabel("base x [m]"); ax.set_ylabel("base y [m]")
    ax.set_title("Base path (top-down)"); ax.axis("equal")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_state, ref_h, "--", color="tab:red", lw=1.2, label="reference")
    ax.plot(t_state, height, color="tab:green", lw=1.6, label="actual")
    ax.set_xlabel("time [s]"); ax.set_ylabel("base height [m]")
    ax.set_title("Base height"); ax.legend(loc="best"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_state, config_err, color="tab:purple", lw=1.4)
    ax.set_xlabel("time [s]"); ax.set_ylabel("||qpos - ref||")
    ax.set_title("Configuration tracking error"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    t_ctrl = dt * np.arange(ctrls_hist.shape[0])
    ax.plot(t_ctrl, ctrls_hist.mean(axis=1), color="tab:blue", lw=1.3)
    ax.set_xlabel("time [s]"); ax.set_ylabel("mean ctrl (joint target)")
    ax.set_title("Mean control"); ax.grid(alpha=0.3)

    fig.suptitle("G1 mocap walking")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--live", action="store_true",
                   help="open interactive MuJoCo viewer (real-time paced)")
    p.add_argument("--record", action="store_true",
                   help="render headlessly and save mp4 to --out")
    p.add_argument("--variant", default="vanilla", choices=list_controllers(),
                   help="MPPI variant from analytic_mppi.controllers.CONTROLLERS")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--start-frame", type=int, default=0,
                   help="reference frame to start from. walk1 stands ~in place for "
                        "its first ~2.5s (~100 frames) then walks; starting mid-stride "
                        "(e.g. 150) is much harder and may fall without further MPPI tuning")
    p.add_argument("--no-feedforward", dest="feedforward", action="store_false",
                   help="sample from a zero nominal instead of the reference feedforward")
    p.add_argument("--camera", default="-1",
                   help="MJCF camera name or numeric id for --record (-1 = free camera)")
    p.add_argument("--out", default=str(DEFAULT_RUNS / "humanoid_mocap.mp4"),
                   help="output path for --record")
    p.add_argument("--rollout", action="store_true",
                   help="overlay MPPI sample trajectories (requires --record)")
    args = p.parse_args()

    if args.live and args.record:
        p.error("choose one of --live or --record (cannot be combined)")
    if args.rollout and not args.record:
        p.error("--rollout requires --record")

    env = make_env(variant=args.variant, start_frame=args.start_frame)
    print(f"g1 mocap env : nq={env.backend.nq}, nv={env.backend.nv}, nu={env.backend.nu}, "
          f"nstate={env.backend.nstate}, dt={env.backend.dt}, threads={env.backend.nthread}")
    print(f"controller   : variant={args.variant} -> {type(env.controller).__name__}  "
          f"(K={env.controller.n_samples}, H={env.controller.horizon})")
    print(f"reference    : {env.cost.reference_qpos.shape[0]} frames @ {env.cost.fps:g} Hz, "
          f"start_frame={args.start_frame}, feedforward={'on' if args.feedforward else 'off'}")

    controller = _ReferenceFeedforward(env.controller, env.cost, env.backend.dt) \
        if args.feedforward else env.controller

    # Follow the base with the top-down marker at the world origin (no xy goal).
    marker_xy = (0.0, 0.0)

    if args.live:
        print(f"Opening live viewer (steps={args.steps})")
        run_live(env.backend, controller, env.cost, marker_xy, args.steps,
                 draw_rollouts=False)
        # Skip Python teardown -- launch_passive on Linux segfaults during
        # GL / MjModel destructor ordering at interpreter exit.
        os._exit(0)

    if args.record:
        out = Path(args.out)
        cam = _camera_arg(args.camera)
        print(f"Recording {args.steps} steps to {out} (camera={cam}, rollout={args.rollout}) ...")
        run_record(env.backend, controller, env.cost, marker_xy, args.steps, out,
                   camera=cam, draw_rollouts=args.rollout)
        print(f"Saved {out}")
        return

    # ---- default: headless run + matplotlib trace plot ----
    state = env.backend.get_state()
    states_hist = [state.copy()]
    ctrls_hist = []
    plan_times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        u = controller.act(state, env.cost, env.backend)
        plan_times.append(time.perf_counter() - t0)
        state = env.backend.step(u)
        states_hist.append(state.copy())
        ctrls_hist.append(u.copy())

    states_hist = np.array(states_hist)
    ctrls_hist = np.array(ctrls_hist)

    qpos = states_hist[:, env.backend.qpos_slice]
    ref = _reference_qpos_at(env.cost, states_hist[:, 0])
    final_err = float(np.linalg.norm(qpos[-1] - ref[-1]))
    print(f"Final config error : {final_err:.4f}  (||qpos - ref||)")
    print(f"Planning time      : {1e3 * np.mean(plan_times):6.2f} ms / step")

    out = DEFAULT_RUNS / "humanoid_mocap.png"
    _plot_mocap_run(states_hist, ctrls_hist, env.cost,
                    qpos_slice=env.backend.qpos_slice, dt=env.backend.dt, save_path=out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
