"""FEASIBILITY GATE for the jump-over-obstacle idea: can Barkour leave the ground under
sampling MPC at all? (g1_walk warns that sampling-MPC can just collapse on hard maneuvers.)

Minimal jump objective: drive the CoM up to jump_h while staying upright. Longer horizon
(0.5 s) so the controller can 'see' the flight apex payoff of a push-off. We measure peak
torso height (standing ~0.21, walking ~0.28) and the vertical-velocity arc. Airborne signal:
peak height well above standing-tall AND a clean ballistic vz arc (+ then -g decel).

Run: python verification/com_study/quadruped_jump_probe.py
"""
from __future__ import annotations

import numpy as np

from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.controllers import MPPIv2
from analytic_mppi.tasks.quadruped import QuadrupedTask
from analytic_mppi.eval import init_barkour_stand


class JumpProbe(QuadrupedTask):
    cost_term_names = ["height_cost", "orientation_cost", "control_cost"]

    def __init__(self, jump_h: float = 0.5, **kw):
        super().__init__(**kw)
        self.jump_h = float(jump_h)

    def terminal_cost_terms(self, qpos, qvel, sd):
        h = self._torso_height(sd)
        up = self._torso_up(sd)
        height_cost = 120.0 * np.clip(self.jump_h - h, 0.0, None) ** 2   # one-sided: reward going UP
        orient_cost = 20.0 * (1.0 - up)                                   # stay upright (land safe)
        return np.stack([height_cost, orient_cost, np.zeros_like(h)], axis=-1)

    def running_cost_terms(self, qpos, qvel, sd, u):
        out = self.terminal_cost_terms(qpos, qvel, sd).copy()
        out[..., -1] = 0.3 * self._control_effort(u)
        return out


def main():
    task = JumpProbe(jump_h=0.5)
    backend = MujocoBackend(task.model_path, nthread=8)
    ctrl = MPPIv2(task, backend, num_samples=384, num_knots=8, plan_horizon=0.5,
                  spline_type="zero", noise_level=0.9, temperature=0.08, seed=0)
    init_barkour_stand(backend)
    state = backend.get_state()
    h0 = task._torso_height(np.asarray(backend.data.sensordata))
    print(f"standing torso height = {h0:.3f}")

    hs, vzs = [], []
    for _ in range(260):                       # ~0.52 s at dt=0.002
        u = ctrl.act(state)
        state = backend.step(u)
        sd = np.asarray(backend.data.sensordata)
        hs.append(float(task._torso_height(sd)))
        vzs.append(float(sd[task._vel_adr + 2]))
    hs, vzs = np.asarray(hs), np.asarray(vzs)

    peak = hs.max()
    print(f"peak torso height = {peak:.3f}  (rise {peak - h0:+.3f} m above standing)")
    print(f"max upward CoM vel = {vzs.max():.2f} m/s   min (landing) = {vzs.min():.2f} m/s")
    print(f"airborne-ish (peak > 0.40)? {'YES' if peak > 0.40 else 'no'}")
    # crude arc report: height at t = launch..apex..land
    idx = np.linspace(0, len(hs) - 1, 12).astype(int)
    print("height trace: " + " ".join(f"{hs[i]:.2f}" for i in idx))
    print("vz trace:     " + " ".join(f"{vzs[i]:+.1f}" for i in idx))


if __name__ == "__main__":
    main()
