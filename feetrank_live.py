"""Live G1 standup with FeetRankMPPI and runtime-tweakable hyperparameters.

Usage:
    .venv/bin/python feetrank_live.py [--samples 256] [--iterations 2] ...

The scripted collapse plays first, then FeetRankMPPI stands the robot up at the
feet end-reference (orange sphere) and balances it until the window is closed.

Keys (viewer window focused; values echo in the terminal):
    [ / ]    select previous / next hyperparameter
    - / =    decrease / increase the selected hyperparameter
    0        reset all hyperparameters to launch values
    R        re-collapse the robot and restart the standup
    Space    pause / resume

Planning at the default 256 samples x 2 iterations runs ~2x slower than real
time; lower `samples`/`iterations` live to trade robustness for speed.
"""
from __future__ import annotations

import argparse
import os
import time

import mujoco
import numpy as np

from analytic_mppi.tasks.g1_standup import G1StandupTask
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.controllers import MPPIv2
from analytic_mppi.controllers.sampling_base import Trajectory

R_FEET = 0.3
FOLD_T, SETTLE_T = 3.0, 2.0


# ---------------------------------------------------------------------------
# Same classes as humanoid_standup_feetref_mppi.ipynb (see FeetRankMPPI_README.md)
# ---------------------------------------------------------------------------

class ModelBackend(MujocoBackend):
    """MujocoBackend built from an already-compiled model instead of a path."""

    def __init__(self, model, nthread=None):
        self.model_path = None
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        self.nthread = nthread if nthread is not None else max(1, os.cpu_count() or 1)
        self._thread_data = [mujoco.MjData(model) for _ in range(self.nthread)]
        self.nq = int(model.nq)
        self.nv = int(model.nv)
        self.nu = int(model.nu)
        self.nsensordata = int(model.nsensordata)
        self.nstate = int(mujoco.mj_stateSize(model, int(self.state_spec)))
        self.dt = float(model.opt.timestep)
        self.qpos_slice = slice(1, 1 + self.nq)
        self.qvel_slice = slice(1 + self.nq, 1 + self.nq + self.nv)


class G1FeetRefTask(G1StandupTask):
    cost_term_names = G1StandupTask.cost_term_names + ["feet_cost"]

    def __init__(self, model, feet_ref, *, feet_radius, w_feet):
        super().__init__()
        # swap in the augmented model (same dynamics; extra sensors + visual sites)
        self.mj_model = model
        self.nsensordata = int(model.nsensordata)
        self._lf_adr = int(model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_pos")])
        self._rf_adr = int(model.sensor_adr[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_pos")])
        self.feet_ref = np.asarray(feet_ref, dtype=np.float64)
        self.feet_radius = float(feet_radius)
        self.w_feet = float(w_feet)

    def feet_mid(self, sensordata):
        l = sensordata[..., self._lf_adr:self._lf_adr + 3]
        r = sensordata[..., self._rf_adr:self._rf_adr + 3]
        return 0.5 * (l + r)

    def feet_dist(self, sensordata):
        return np.linalg.norm(self.feet_mid(sensordata) - self.feet_ref, axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u):
        base = super().running_cost_terms(qpos, qvel, sensordata, u)
        feet = self.w_feet * self.feet_dist(sensordata) ** 2
        return np.concatenate([base, feet[..., None]], axis=-1)

    def terminal_cost_terms(self, qpos, qvel, sensordata):
        base = super().terminal_cost_terms(qpos, qvel, sensordata)
        feet = self.w_feet * self.feet_dist(sensordata) ** 2
        return np.concatenate([base, feet[..., None]], axis=-1)


class FeetRankMPPI(MPPIv2):
    def __init__(self, *args, tie_tol=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.tie_tol = float(tie_tol)
        self.last_rank = None
        self.last_mode = None

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        dist = self.task.feet_dist(traj.sensordata[:, -1])   # terminal feet midpoint
        rank = np.clip(1.0 - dist / self.task.feet_radius, 0.0, 1.0)
        self.last_rank = rank

        r_max = rank.max()
        if r_max <= 0.0:                    # no rollout's feet end inside the sphere
            self.last_mode = "softmax-fallback"
            return super().update_mean(traj)

        ties = np.flatnonzero(rank >= r_max - self.tie_tol)
        if ties.size == 1:
            self.last_mode = "best"
            return traj.knots[ties[0]].copy()

        self.last_mode = f"tie-softmax({ties.size})"
        s = traj.scores[ties]
        z = -(s - s.min()) / self.temperature
        w = np.exp(z)
        w /= w.sum()
        return (w[:, None, None] * traj.knots[ties]).sum(axis=0)


# ---------------------------------------------------------------------------
# Environment assembly
# ---------------------------------------------------------------------------

def fold_ctrl(nu):
    fold = np.zeros(nu)
    fold[[0, 6]] = -1.8      # hip pitch
    fold[[3, 9]] = 2.6       # knees
    fold[[4, 10]] = -0.6     # ankle pitch
    return fold


def run_collapse(backend, viewer=None):
    """Scripted fold + settle; returns the collapsed state. Live if viewer given."""
    fold = fold_ctrl(backend.nu)
    n_fold = round(FOLD_T / backend.dt)
    n_settle = round(SETTLE_T / backend.dt)
    for step in range(n_fold + n_settle):
        backend.step(fold if step < n_fold else np.zeros(backend.nu))
        if viewer is not None:
            viewer.sync()
            time.sleep(backend.dt)
    return backend.get_state().copy()


def build(args):
    plain = G1StandupTask()

    # collapse once on the plain model to find where the feet end up
    backend0 = MujocoBackend(plain.model_path)
    run_collapse(backend0)
    lf = mujoco.mj_name2id(backend0.model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rf = mujoco.mj_name2id(backend0.model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")
    feet_ref = (backend0.data.site_xpos[lf] + backend0.data.site_xpos[rf]) / 2.0

    # augmented model: feet framepos sensors + visual reference sphere
    spec = mujoco.MjSpec.from_file(str(plain.model_path))
    for nm, site in (("left_foot_pos", "left_foot"), ("right_foot_pos", "right_foot")):
        s = spec.add_sensor()
        s.name = nm
        s.type = mujoco.mjtSensor.mjSENS_FRAMEPOS
        s.objtype = mujoco.mjtObj.mjOBJ_SITE
        s.objname = site
    spec.worldbody.add_site(name="feet_ref", pos=feet_ref.tolist(), size=[R_FEET] * 3,
                            rgba=[0.9, 0.35, 0.1, 0.3])
    spec.worldbody.add_site(name="feet_ref_center", pos=feet_ref.tolist(),
                            size=[0.04] * 3, rgba=[0.9, 0.35, 0.1, 1.0])
    model = spec.compile()

    task = G1FeetRefTask(model, feet_ref, feet_radius=R_FEET, w_feet=args.w_feet)
    backend = ModelBackend(model)
    ctrl = FeetRankMPPI(task=task, backend=backend, num_samples=args.samples,
                        noise_level=args.noise, temperature=args.temperature,
                        num_knots=args.knots, plan_horizon=args.horizon,
                        spline_type="cubic", iterations=args.iterations,
                        seed=args.seed, tie_tol=args.tie_tol)
    return task, backend, ctrl, feet_ref


# ---------------------------------------------------------------------------
# Live hyperparameter tweaking
# ---------------------------------------------------------------------------

class ParamPanel:
    """Terminal-echoed hyperparameter editor driven by viewer key presses.

    Each entry: (label, getter, setter, step, lo, hi). All targeted attributes
    are safe to mutate between act() calls.
    """

    def __init__(self, task, ctrl):
        self.entries = [
            ("noise_level", lambda: ctrl.noise_level,
             lambda v: setattr(ctrl, "noise_level", v), 0.05, 0.05, 1.0),
            ("temperature", lambda: ctrl.temperature,
             lambda v: setattr(ctrl, "temperature", v), 0.01, 0.01, 1.0),
            ("tie_tol", lambda: ctrl.tie_tol,
             lambda v: setattr(ctrl, "tie_tol", v), 0.05, 0.0, 1.0),
            ("w_feet", lambda: task.w_feet,
             lambda v: setattr(task, "w_feet", v), 1.0, 0.0, 50.0),
            ("num_samples", lambda: ctrl.num_samples,
             lambda v: setattr(ctrl, "num_samples", int(v)), 32, 32, 1024),
            ("iterations", lambda: ctrl.iterations,
             lambda v: setattr(ctrl, "iterations", int(v)), 1, 1, 4),
            ("target_height", lambda: task.target_height,
             lambda v: setattr(task, "target_height", v), 0.05, 0.3, 1.1),
        ]
        self.selected = 0
        self.defaults = [g() for _, g, *_ in self.entries]

    def show(self):
        parts = []
        for i, (name, get, *_rest) in enumerate(self.entries):
            v = get()
            vs = f"{v:d}" if isinstance(v, (int, np.integer)) else f"{v:.2f}"
            parts.append(f"{'>' if i == self.selected else ' '}{name}={vs}")
        print("  " + "  ".join(parts))

    def select(self, delta):
        self.selected = (self.selected + delta) % len(self.entries)
        self.show()

    def nudge(self, direction):
        name, get, set_, step, lo, hi = self.entries[self.selected]
        set_(float(np.clip(get() + direction * step, lo, hi)))
        self.show()

    def reset(self):
        for (name, get, set_, *_), v in zip(self.entries, self.defaults):
            set_(v)
        print("  hyperparameters reset to launch values")
        self.show()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--knots", type=int, default=6)
    p.add_argument("--horizon", type=float, default=0.8)
    p.add_argument("--noise", type=float, default=0.15)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--tie-tol", type=float, default=0.15)
    p.add_argument("--w-feet", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", type=int, default=0, metavar="N",
                   help="headless self-test: run N MPPI steps and exit")
    args = p.parse_args()

    task, backend, ctrl, feet_ref = build(args)
    print(f"end reference (feet midpoint): {np.round(feet_ref, 3)}  R={R_FEET}")

    if args.smoke:
        state0 = run_collapse(backend)
        backend.set_state(state0)
        ctrl.reset()
        for _ in range(args.smoke):
            backend.step(ctrl.act(backend.get_state()))
        h = float(backend.data.sensordata[task._torso_pos_adr + 2])
        print(f"smoke ok: {args.smoke} steps, torso z={h:.3f}, mode={ctrl.last_mode}")
        return

    import mujoco.viewer

    panel = ParamPanel(task, ctrl)
    flags = {"paused": False, "recollapse": False}

    def key_callback(keycode):
        if keycode == ord("["):
            panel.select(-1)
        elif keycode == ord("]"):
            panel.select(+1)
        elif keycode == ord("-"):
            panel.nudge(-1)
        elif keycode == ord("="):
            panel.nudge(+1)
        elif keycode == ord("0"):
            panel.reset()
        elif keycode == ord("R"):
            flags["recollapse"] = True
        elif keycode == ord(" "):
            flags["paused"] = not flags["paused"]
            print("  paused" if flags["paused"] else "  resumed")

    print("Keys" + __doc__.split("Keys", 1)[1].split("Planning", 1)[0])
    panel.show()

    with mujoco.viewer.launch_passive(backend.model, backend.data,
                                      key_callback=key_callback) as viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.0, -15.0, 135.0
        viewer.cam.lookat[:] = [feet_ref[0], feet_ref[1], 0.6]

        print("scripted collapse ...")
        run_collapse(backend, viewer)
        ctrl.reset()
        print("FeetRankMPPI running - close the window to quit")

        last_report = time.time()
        while viewer.is_running():
            t0 = time.time()
            if flags["recollapse"]:
                flags["recollapse"] = False
                print("re-collapsing ...")
                run_collapse(backend, viewer)
                ctrl.reset()
            if flags["paused"]:
                viewer.sync()
                time.sleep(0.05)
                continue
            u0 = ctrl.act(backend.get_state())
            backend.step(u0)
            viewer.sync()
            if time.time() - last_report > 5.0:
                h = float(backend.data.sensordata[task._torso_pos_adr + 2])
                fd = float(task.feet_dist(backend.data.sensordata))
                print(f"  torso z={h:.2f}  feet dist={fd:.2f}  mode={ctrl.last_mode}  "
                      f"plan {time.time() - t0:.2f}s/step")
                last_report = time.time()
            time.sleep(max(0.0, backend.dt - (time.time() - t0)))

    # launch_passive on Linux can segfault in GL/MjModel destructor ordering at
    # interpreter exit (see envs/g1/run.py) - skip Python teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
