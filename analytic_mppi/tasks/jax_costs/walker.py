"""jnp mirror of tasks/walker.py (WalkerTask costs). Parity-pinned at 1e-6."""
from __future__ import annotations

import jax.numpy as jnp


class WalkerJaxCosts:
    def __init__(self, task):
        self._pos_adr = task._pos_adr
        self._vel_adr = task._vel_adr
        self._zax_adr = task._zax_adr
        self.target_velocity = task.target_velocity
        self.target_height = task.target_height
        self.height_weight = task.height_weight
        self.orientation_weight = task.orientation_weight
        self.velocity_weight = task.velocity_weight
        self.control_weight = task.control_weight

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
        orient_cost = self.orientation_weight * (zax - 1.0) ** 2
        velocity_cost = self.velocity_weight * (vel - self.target_velocity) ** 2
        zero = jnp.zeros_like(height_cost)
        return jnp.stack([height_cost, orient_cost, velocity_cost, zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u):
        terms = self.terminal_cost_terms(qpos, qvel, sensordata)
        control_cost = self.control_weight * jnp.sum(u ** 2, axis=-1)
        return terms.at[..., -1].set(control_cost)

    # ---- FPL atoms ----

    def _height_fulfillment(self, sd):
        h_full, h_floor = 1.1, 0.7
        return jnp.clip((self._height(sd) - h_floor) / (h_full - h_floor), 0.0, 1.0)

    def _orientation_fulfillment(self, sd):
        z_full, z_floor = 0.95, 0.6
        return jnp.clip((self._zaxis_z(sd) - z_floor) / (z_full - z_floor), 0.0, 1.0)

    def _velocity_fulfillment(self, sd):
        return jnp.clip(self._vel_x(sd) / self.target_velocity, 0.0, 1.0)

    def _control_fulfillment(self, u):
        return jnp.clip(1.0 - jnp.mean(u ** 2, axis=-1), 0.0, 1.0)

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
        # 3 state atoms only (n_term < n_run), matching WalkerTask.
        return jnp.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
            ],
            axis=-1,
        )
