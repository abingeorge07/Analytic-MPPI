"""jnp mirrors of tasks/g1_standup.py + tasks/g1_walk.py costs. Parity-pinned at 1e-6.

The flat (non-grouped) FPL surface only -- layered/grouped FPL is CPU-only by scope.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def _quat_rotate(q, v: np.ndarray):
    """Rotate constant v by batched quaternions q (wxyz). Mirrors g1_standup.py:20-37."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    ax = y * vz - z * vy + w * vx
    ay = z * vx - x * vz + w * vy
    az = x * vy - y * vx + w * vz
    rx = vx + 2.0 * (y * az - z * ay)
    ry = vy + 2.0 * (z * ax - x * az)
    rz = vz + 2.0 * (x * ay - y * ax)
    return jnp.stack([rx, ry, rz], axis=-1)


def _apply_offset(f, delta):
    """FPL priority offset (paper Eq. 7). Mirrors g1_standup.py:135-141."""
    return jnp.clip((f + jnp.maximum(delta, 0.0)) / (1.0 + delta), 0.0, 1.0)


class G1StandupJaxCosts:
    """Built from a live G1StandupTask. Term columns match cost_term_names[_f]:
    [orientation, height, nominal, control, stillness, wobble]."""

    def __init__(self, task):
        self._quat_adr = task._quat_adr
        self._torso_pos_adr = task._torso_pos_adr
        self._linvel_adr = task._linvel_adr
        self._angvel_adr = task._angvel_adr
        self.nu = task.nu
        self.target_height = task.target_height
        self.qstand_j = jnp.asarray(task.qstand[7:])
        self.joint_max_dev = jnp.asarray(task.joint_max_dev)
        self.ctrl_cost_weight = task.ctrl_cost_weight
        self.u_ref = jnp.asarray(task.u_ref)
        self.u_half_range = jnp.asarray(task.u_half_range)
        self.stillness_weight = task.stillness_weight
        self.wobble_weight = task.wobble_weight
        self.lin_vel_ref = task.lin_vel_ref
        self.ang_vel_ref = task.ang_vel_ref
        self.fpl_offsets = jnp.asarray(task.fpl_offsets)

    # ---- sensor decoding ----

    def _torso_orientation(self, sd):
        quat = sd[..., self._quat_adr : self._quat_adr + 4]
        return _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))

    def _torso_height(self, sd):
        return sd[..., self._torso_pos_adr + 2]

    def _torso_linvel(self, sd):
        return sd[..., self._linvel_adr : self._linvel_adr + 3]

    def _torso_angvel(self, sd):
        return sd[..., self._angvel_adr : self._angvel_adr + 3]

    # ---- efforts (shared by normal + FPL) ----

    def _control_effort(self, u):
        u_norm = (u - self.u_ref) / self.u_half_range
        return jnp.mean(u_norm ** 2, axis=-1)

    def _stillness_effort(self, sd):
        v = self._torso_linvel(sd)
        return jnp.sum(v ** 2, axis=-1) / (self.lin_vel_ref ** 2)

    def _wobble_effort(self, sd):
        w = self._torso_angvel(sd)
        return jnp.sum(w ** 2, axis=-1) / (self.ang_vel_ref ** 2)

    # ---- normal cost ----

    def running_cost_terms(self, qpos, qvel, sensordata, u):
        rz = self._torso_orientation(sensordata)[..., 2]
        orientation_cost = 10.0 * (1.0 - rz)
        height_cost = 10.0 * (self._torso_height(sensordata) - self.target_height) ** 2
        nominal_cost = 0.1 * jnp.sum((qpos[..., 7:] - self.qstand_j) ** 2, axis=-1)
        control_cost = self.ctrl_cost_weight * self._control_effort(u)
        stillness_cost = self.stillness_weight * self._stillness_effort(sensordata)
        wobble_cost = self.wobble_weight * self._wobble_effort(sensordata)
        return jnp.stack([orientation_cost, height_cost, nominal_cost, control_cost,
                          stillness_cost, wobble_cost], axis=-1)

    def terminal_cost_terms(self, qpos, qvel, sensordata):
        return self.running_cost_terms(qpos, qvel, sensordata,
                                       jnp.zeros_like(qpos[..., :self.nu]))

    # ---- FPL atoms ----

    def _height_fulfillment(self, sd):
        h = self._torso_height(sd)
        delta = 0.9
        err = -jnp.abs(h - self.target_height)
        err = jnp.clip(err, -delta, 0.0)
        return err / delta + 1.0

    def _orientation_fulfillment(self, sd):
        rz = self._torso_orientation(sd)[..., 2]
        rz_full, rz_floor = 0.95, 0.5
        return jnp.clip((rz - rz_floor) / (rz_full - rz_floor), 0.0, 1.0)

    def _joint_fulfillments(self, qpos):
        diff = jnp.abs(qpos[..., 7:] - self.qstand_j)
        return jnp.clip(1.0 - diff / self.joint_max_dev, 1e-8, 1.0)

    def _nominal_fulfillment(self, qpos):
        frac = self._joint_fulfillments(qpos)
        return jnp.exp(jnp.mean(jnp.log(frac), axis=-1))

    def _control_fulfillment(self, u):
        return jnp.exp(-self._control_effort(u))

    def _stillness_fulfillment(self, sd):
        return jnp.exp(-self._stillness_effort(sd))

    def _wobble_fulfillment(self, sd):
        return jnp.exp(-self._wobble_effort(sd))

    def running_cost_terms_f(self, qpos, qvel, sensordata, u):
        atoms = jnp.stack(
            [
                self._orientation_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._nominal_fulfillment(qpos),
                self._control_fulfillment(u),
                self._stillness_fulfillment(sensordata),
                self._wobble_fulfillment(sensordata),
            ],
            axis=-1,
        )
        return _apply_offset(atoms, self.fpl_offsets)

    def terminal_cost_terms_f(self, qpos, qvel, sensordata):
        return self.running_cost_terms_f(qpos, qvel, sensordata,
                                         jnp.zeros_like(qpos[..., :self.nu]))


class G1WalkJaxCosts(G1StandupJaxCosts):
    """Mirror of G1WalkTask: stillness atom replaced by forward-speed tracking.
    Columns: [orientation, height, nominal, control, forward, wobble]."""

    def __init__(self, task):
        super().__init__(task)
        self.target_velocity = task.target_velocity
        self.vel_cost_weight = task.vel_cost_weight

    def _forward_speed(self, sd):
        return self._torso_linvel(sd)[..., 0]

    def _forward_fulfillment(self, sd):
        return jnp.clip(self._forward_speed(sd) / self.target_velocity, 0.0, 1.0)

    def running_cost_terms(self, qpos, qvel, sensordata, u):
        rz = self._torso_orientation(sensordata)[..., 2]
        orientation_cost = 10.0 * (1.0 - rz)
        height_cost = 10.0 * (self._torso_height(sensordata) - self.target_height) ** 2
        nominal_cost = 0.1 * jnp.sum((qpos[..., 7:] - self.qstand_j) ** 2, axis=-1)
        control_cost = self.ctrl_cost_weight * self._control_effort(u)
        vx = self._forward_speed(sensordata)
        forward_cost = self.vel_cost_weight * jnp.clip(self.target_velocity - vx, 0.0, None) ** 2
        wobble_cost = self.wobble_weight * self._wobble_effort(sensordata)
        return jnp.stack([orientation_cost, height_cost, nominal_cost, control_cost,
                          forward_cost, wobble_cost], axis=-1)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u):
        atoms = jnp.stack(
            [
                self._orientation_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._nominal_fulfillment(qpos),
                self._control_fulfillment(u),
                self._forward_fulfillment(sensordata),
                self._wobble_fulfillment(sensordata),
            ],
            axis=-1,
        )
        return _apply_offset(atoms, self.fpl_offsets)
