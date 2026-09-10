"""jnp ports of the shared cost helpers (tasks/base.py) + the JaxTaskCosts protocol.

Every formula here mirrors its numpy original EXACTLY (same eps, same clips, same
normalization); tests/test_jax_costs.py pins them together at 1e-6. If you change a
formula in tasks/base.py, change it here in the same commit or that test fails.

Dtype note: nothing here forces float64 -- under jax's default f32 these run in f32
(the production GradientMPC path); the parity tests enable jax_enable_x64 so the
comparison against numpy is made at f64.
"""
from __future__ import annotations

from typing import Protocol

import jax.numpy as jnp


def power_mean(terms, p: float, eps: float = 1e-8, weights=None):
    """Generalized (power) mean along the LAST axis. Mirrors tasks/base.py:26-50."""
    t = jnp.clip(terms, eps, None)
    if weights is None:
        n = t.shape[-1]
        if p == 0:
            return jnp.exp(jnp.mean(jnp.log(t), axis=-1))
        return (jnp.sum(t ** p, axis=-1) / n) ** (1.0 / p)
    w = jnp.asarray(weights)
    w = w / w.sum(axis=-1, keepdims=True)
    if p == 0:
        return jnp.exp(jnp.sum(w * jnp.log(t), axis=-1))
    return (jnp.sum(w * t ** p, axis=-1)) ** (1.0 / p)


def soft_ramp(r, f_min: float = 0.05, tau: float = 0.5):
    """Strictly-monotone fulfillment ramp. Mirrors tasks/base.py:53-81."""
    r = jnp.asarray(r)
    inside = f_min + (1.0 - f_min) * jnp.clip(r, 0.0, 1.0)
    below = f_min * jnp.exp(jnp.minimum(r, 0.0) / tau)
    return jnp.where(r < 0.0, below, inside)


class JaxTaskCosts(Protocol):
    """The four Task cost surfaces, in jnp. Same signatures/shapes as tasks/base.Task.

    Implementations are built FROM a live Task instance (weights, sensor addresses,
    targets are read off it as plain python/numpy constants, so they become jit-time
    constants) and must keep term columns index-aligned with the task's
    `cost_term_names` / `cost_term_names_f`.
    """

    def running_cost_terms(self, qpos, qvel, sensordata, u): ...
    def terminal_cost_terms(self, qpos, qvel, sensordata): ...
    def running_cost_terms_f(self, qpos, qvel, sensordata, u): ...
    def terminal_cost_terms_f(self, qpos, qvel, sensordata): ...
