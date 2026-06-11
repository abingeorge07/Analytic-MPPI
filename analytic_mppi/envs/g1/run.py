"""G1 standup demo: vanilla MPPI raising the humanoid to a standing pose.

    python -m analytic_mppi.envs.g1.run               # headless, save PNG
    python -m analytic_mppi.envs.g1.run --live        # interactive viewer
    python -m analytic_mppi.envs.g1.run --record      # save mp4 to runs/
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from analytic_mppi.controllers import list_controllers
from analytic_mppi.costs.g1 import _quat_rotate
from analytic_mppi.envs.g1 import make_env
from analytic_mppi.viz import run_live, run_record


# analytic_mppi/envs/g1/run.py -> parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO / "runs"


def _up_z(quat: np.ndarray) -> np.ndarray:
    """World-frame z-component of the up-axis for a wxyz quat (1 = upright)."""
    return _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))[..., 2]


def _camera_arg(value: str) -> str | int:
    """Allow numeric camera ids (e.g. '-1' free camera) or a named MJCF camera."""
    v = value.strip()
    return int(v) if v.lstrip("-").isdigit() else v


def _plot_g1_run(
    states_history: np.ndarray,
    controls_history: np.ndarray,
    torso_quat_history: np.ndarray,
    *,
    height_idx: int,
    qstand_joints: np.ndarray,
    joint_slice: slice,
    target_height: float,
    dt: float,
    save_path: Path,
) -> None:
    """Height / uprightness / nominal joint deviation / mean control over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_state = dt * np.arange(states_history.shape[0])
    t_ctrl = dt * np.arange(controls_history.shape[0])
    height = states_history[:, height_idx]
    up_z = _up_z(torso_quat_history)
    nominal_dev = ((states_history[:, joint_slice] - qstand_joints) ** 2).sum(axis=-1)

    fig, axes = plt.subplots(4, 1, figsize=(7.5, 9.5), sharex=True)

    ax = axes[0]
    ax.plot(t_state, height, lw=1.6, color="tab:green", label="torso height")
    ax.axhline(target_height, color="tab:red", ls="--", lw=1.0,
               label=f"target ({target_height:g})")
    ax.set_ylabel("height [m]")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    ax.set_title("G1 standup")

    ax = axes[1]
    ax.plot(t_state, up_z, lw=1.6, color="tab:blue")
    ax.axhline(1.0, color="tab:red", ls="--", lw=1.0, label="upright (1)")
    ax.set_ylabel("up_z")
    ax.legend(loc="best"); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t_state, nominal_dev, lw=1.4, color="tab:purple")
    ax.set_ylabel("nominal dev")
    ax.grid(alpha=0.3)

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
    p.add_argument("--steps",   type=int, default=200)
    p.add_argument("--camera",  default="-1",
                   help="MJCF camera name or numeric id for --record "
                        "(scene.xml has no cameras; -1 = free camera)")
    p.add_argument("--out",     default=str(DEFAULT_RUNS / "g1_standup.mp4"),
                   help="output path for --record")
    p.add_argument("--rollout", action="store_true",
                   help="overlay MPPI sample trajectories (requires --record)")
    args = p.parse_args()

    if args.live and args.record:
        p.error("choose one of --live or --record (cannot be combined)")
    if args.rollout and not args.record:
        p.error("--rollout requires --record")

    env = make_env(variant=args.variant)
    print(f"g1 env      : nq={env.backend.nq}, nv={env.backend.nv}, nu={env.backend.nu}, "
          f"nstate={env.backend.nstate}, dt={env.backend.dt}, threads={env.backend.nthread}")
    print(f"controller   : variant={args.variant} -> {type(env.controller).__name__}  "
          f"(K={env.controller.n_samples}, H={env.controller.horizon})")

    # The viewer helpers want a goal_xy for marker drawing; the G1 has no xy
    # goal, so park the marker at the world origin.
    marker_xy = (0.0, 0.0)

    adr = env.cost.orientation_sensor_adr

    def _torso_quat() -> np.ndarray:
        """Current torso quat from the live sim's imu_in_torso_quat sensor."""
        return env.backend.data.sensordata[adr : adr + 4].copy()

    def _report_final(state: np.ndarray) -> None:
        print(f"Final height   : {float(state[env.height_idx]):+.4f}  "
              f"(target = {env.cost.target_height:+.4f})")
        print(f"Final up_z     : {float(_up_z(_torso_quat())):+.4f}  (upright = +1.0000)")

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
        cam = _camera_arg(args.camera)
        print(f"Recording {args.steps} steps to {out} "
              f"(camera={cam}, rollout={args.rollout}) ...")
        run_record(env.backend, env.controller, env.cost, marker_xy, args.steps, out,
                   camera=cam, draw_rollouts=args.rollout)
        print(f"Saved {out}")
        _report_final(env.backend.get_state())
        return

    # ---- default: headless run + matplotlib trace plot ----
    state = env.backend.get_state()
    states_hist = [state.copy()]
    torso_quats = [_torso_quat()]   # torso orientation tracks the imu_in_torso sensor
    ctrls_hist = []
    plan_times = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        u = env.controller.act(state, env.cost, env.backend)
        plan_times.append(time.perf_counter() - t0)
        state = env.backend.step(u)
        states_hist.append(state.copy())
        torso_quats.append(_torso_quat())
        ctrls_hist.append(u.copy())

    states_hist = np.array(states_hist)
    ctrls_hist = np.array(ctrls_hist)
    torso_quats = np.array(torso_quats)

    _report_final(states_hist[-1])
    print(f"Planning time  : {1e3 * np.mean(plan_times):6.2f} ms / step")

    out = DEFAULT_RUNS / "g1_standup.png"
    _plot_g1_run(
        states_hist, ctrls_hist, torso_quats,
        height_idx=env.height_idx,
        qstand_joints=env.cost._qstand, joint_slice=env.cost.joint_slice,
        target_height=env.cost.target_height,
        dt=env.backend.dt, save_path=out,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
