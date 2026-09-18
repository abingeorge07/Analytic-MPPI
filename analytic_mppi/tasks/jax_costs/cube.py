"""jnp mirror of tasks/cube.py (CubeRotationTask costs). Parity-pinned at 1e-6."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _quat_sub_jnp(q, q_ref):
    """Hamiltonian quaternion 'subtraction' returning the 3-vec rotation error.
    Mirrors tasks/cube.py:_quat_sub exactly. q:(...,4), q_ref:(4,) both wxyz.
    """
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # conj of ref
    w2 = q_ref[0]; x2 = -q_ref[1]; y2 = -q_ref[2]; z2 = -q_ref[3]
    vx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    vy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    vz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    real = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    sign = jnp.where(real < 0, -1.0, 1.0)
    return jnp.stack([vx, vy, vz], axis=-1) * sign[..., None]


class CubeJaxCosts:
    """Built from a live CubeRotationTask; constants become jit-time constants."""

    def __init__(self, task):
        self._pos_adr = int(task._pos_adr)
        self._ori_adr = int(task._ori_adr)
        self.target_angle = float(task.target_angle)
        self.hold_xy_full = float(task.hold_xy_full)
        self.hold_xy_floor = float(task.hold_xy_floor)
        self.height_full = float(task.height_full)
        self.height_floor = float(task.height_floor)
        self.ctrl_cost_weight = float(task.ctrl_cost_weight)
        self.goal_quat = jnp.asarray(np.asarray(task.goal_quat, dtype=np.float32))
        self.u_ref = jnp.asarray(np.asarray(task.u_ref, dtype=np.float32))
        self.u_half_range = jnp.asarray(np.asarray(task.u_half_range, dtype=np.float32))

    # ---- state decode ----

    def _cube_pos_err(self, sd):
        return sd[..., self._pos_adr:self._pos_adr + 3]

    def _cube_quat(self, sd):
        return sd[..., self._ori_adr:self._ori_adr + 4]

    def _hold_xy(self, sd):
        pe = self._cube_pos_err(sd)
        return jnp.sqrt(jnp.sum(pe[..., 0:2] ** 2, axis=-1))

    def _cube_height(self, sd):
        return self._cube_pos_err(sd)[..., 2]

    def _angle_err(self, sd):
        v = _quat_sub_jnp(self._cube_quat(sd), self.goal_quat)
        # jnp.arcsin can be NaN-sensitive at boundaries; clip norm to [0,1] first.
        return 2.0 * jnp.arcsin(jnp.clip(jnp.linalg.norm(v, axis=-1), 0.0, 1.0))

    def _control_effort(self, u):
        u_norm = (u - self.u_ref) / self.u_half_range
        return jnp.mean(u_norm ** 2, axis=-1)

    # ---- normal (quadratic) cost ----

    def terminal_cost_terms(self, qpos, qvel, sensordata):
        xy = self._hold_xy(sensordata)
        h = self._cube_height(sensordata)
        ang = self._angle_err(sensordata)
        hold_cost = 10.0 * xy ** 2 + 100.0 * jnp.maximum(xy - self.hold_xy_floor, 0.0) ** 2
        height_cost = 50.0 * jnp.clip(self.height_full - h, 0.0, None) ** 2
        alignment_cost = 5.0 * ang ** 2
        zero = jnp.zeros_like(hold_cost)
        return jnp.stack([hold_cost, height_cost, alignment_cost, zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u):
        terms = self.terminal_cost_terms(qpos, qvel, sensordata)
        control_cost = self.ctrl_cost_weight * self._control_effort(u)
        return terms.at[..., -1].set(control_cost)

    # ---- FPL fulfillment atoms ----

    def _hold_fulfillment(self, sd):
        xy = self._hold_xy(sd)
        return jnp.clip((self.hold_xy_floor - xy) / (self.hold_xy_floor - self.hold_xy_full),
                       0.0, 1.0)

    def _height_fulfillment(self, sd):
        h = self._cube_height(sd)
        return jnp.clip((h - self.height_floor) / (self.height_full - self.height_floor),
                       0.0, 1.0)

    def _alignment_fulfillment(self, sd):
        ang = self._angle_err(sd)
        return jnp.clip((self.target_angle - ang) / self.target_angle, 0.0, 1.0)

    def _control_fulfillment(self, u):
        return jnp.exp(-self._control_effort(u))

    def running_cost_terms_f(self, qpos, qvel, sensordata, u):
        return jnp.stack(
            [
                self._hold_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._alignment_fulfillment(sensordata),
                self._control_fulfillment(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata):
        return jnp.stack(
            [
                self._hold_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._alignment_fulfillment(sensordata),
            ],
            axis=-1,
        )
