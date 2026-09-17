"""jnp mirror of tasks/pendulum.py (PendulumTask costs).

Added for the S13 handoff-demo path (scripts/record_demo.py) so ILQRMPC can
plan on pendulum through the MJX backend. Not part of the published locomotion
comparison; the parity test in tests/test_jax_costs.py covers it at 1e-6.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np


class PendulumJaxCosts:
    """Built from a live PendulumTask; constants become jit-time constants."""

    def __init__(self, task):
        self.target_angle = float(task.target_angle)
        # For control-fulfillment saturation: |u| / max(|u_min|, |u_max|).
        u_min = np.asarray(task.u_min, dtype=np.float32)
        u_max = np.asarray(task.u_max, dtype=np.float32)
        self._umax = jnp.asarray(np.maximum(np.abs(u_min), np.abs(u_max)))

    # ---- normal cost (mirror tasks/pendulum.py) ----

    def _distance_to_target(self, qpos):
        theta = qpos[..., 0] - self.target_angle
        return (jnp.cos(theta) - 1.0) ** 2 + jnp.sin(theta) ** 2

    def _theta_dot_cost(self, qvel):
        return 0.01 * qvel[..., 0] ** 2

    def _control_cost(self, u):
        return 0.001 * jnp.sum(u ** 2, axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u):
        return jnp.stack(
            [
                self._distance_to_target(qpos),
                self._theta_dot_cost(qvel),
                self._control_cost(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms(self, qpos, qvel, sensordata):
        d = self._distance_to_target(qpos)
        td = self._theta_dot_cost(qvel)
        zero = jnp.zeros_like(d)
        return jnp.stack([d, td, zero], axis=-1)

    # ---- FPL cost ----

    def _target_fulfillment(self, qpos):
        theta = qpos[..., 0] - self.target_angle
        return (1.0 + jnp.cos(theta)) * 0.5

    def _control_fulfillment(self, u):
        # Matches PendulumTask._control_fulfillment: floor 0.9, worst-actuator.
        frac = jnp.max(jnp.abs(u) / self._umax, axis=-1)
        u_floor = 0.9
        return jnp.clip((1.0 - frac) / (1.0 - u_floor), 0.0, 1.0)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u):
        return jnp.stack(
            [self._target_fulfillment(qpos), self._control_fulfillment(u)],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata):
        # Only the target term has a terminal contribution (mirror pendulum.py).
        return self._target_fulfillment(qpos)[..., None]
