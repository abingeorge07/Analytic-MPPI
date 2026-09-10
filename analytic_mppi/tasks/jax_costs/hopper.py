"""jnp mirror of tasks/hopper.py (HopperTask costs). Parity-pinned at 1e-6."""
from __future__ import annotations

import jax.numpy as jnp

from ._base import soft_ramp


class HopperJaxCosts:
    """Built from a live HopperTask; constants become jit-time constants."""

    def __init__(self, task):
        self._pos_adr = task._pos_adr
        self._vel_adr = task._vel_adr
        self._zax_adr = task._zax_adr
        self.height_weight = task.height_weight
        self.orientation_weight = task.orientation_weight
        self.velocity_weight = task.velocity_weight
        self.control_weight = task.control_weight
        self.target_velocity = task.target_velocity
        self.target_height = task.target_height
        self.orientation_floor = task.orientation_floor
        self.atom_soft_floor = task.atom_soft_floor
        self.atom_tail_tau = task.atom_tail_tau

    # ---- shared helpers (mirror tasks/hopper.py) ----

    def _ramp(self, r):
        if self.atom_soft_floor <= 0.0:            # static python branch: jit-safe
            return jnp.clip(r, 0.0, 1.0)
        return soft_ramp(r, f_min=self.atom_soft_floor, tau=self.atom_tail_tau)

    def _height(self, sd):
        return sd[..., self._pos_adr + 2]

    def _vel_x(self, sd):
        return sd[..., self._vel_adr]

    def _zaxis_z(self, sd):
        return sd[..., self._zax_adr + 2]

    # ---- normal cost ----

    def terminal_cost_terms(self, qpos, qvel, sensordata):
        height = self._height(sensordata)
        vel = self._vel_x(sensordata)
        zax = self._zaxis_z(sensordata)
        height_cost = self.height_weight * (height - self.target_height) ** 2
        orientation_cost = self.orientation_weight * (1.0 - zax) ** 2
        velocity_cost = self.velocity_weight * (vel - self.target_velocity) ** 2
        zero = jnp.zeros_like(height_cost)
        return jnp.stack([height_cost, orientation_cost, velocity_cost, zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u):
        terms = self.terminal_cost_terms(qpos, qvel, sensordata)
        control_cost = self.control_weight * jnp.sum(u ** 2, axis=-1)
        return terms.at[..., -1].set(control_cost)

    # ---- FPL atoms ----

    def _height_fulfillment(self, sd):
        h_full, h_floor = 1.15, 0.85
        return self._ramp((self._height(sd) - h_floor) / (h_full - h_floor))

    def _orientation_fulfillment(self, sd):
        z_full, z_floor = 0.95, self.orientation_floor
        return self._ramp((self._zaxis_z(sd) - z_floor) / (z_full - z_floor))

    def _velocity_fulfillment(self, sd):
        return self._ramp(self._vel_x(sd) / self.target_velocity)

    def _control_fulfillment(self, u):
        return self._ramp(1.0 - 0.25 * jnp.mean(u ** 2, axis=-1))

    def running_cost_terms_f(self, qpos, qvel, sensordata, u):
        return jnp.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
                self._control_fulfillment(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata):
        # 3 state atoms only; control is undefined at the terminal step (n_term < n_run).
        return jnp.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
            ],
            axis=-1,
        )
