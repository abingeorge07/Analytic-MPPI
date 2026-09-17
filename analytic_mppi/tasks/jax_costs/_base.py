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


ATOM_FLOOR_LEGACY = 1e-8
ATOM_FLOOR_RECOMMENDED = 1e-3


def floor_atoms(terms, floor: float = ATOM_FLOOR_LEGACY):
    """Clamp fulfillment atoms into [floor, 1]. Mirrors tasks/base.py `floor_atoms`.

    Note for the gradient arm: `jnp.clip` has zero derivative in the clamped region, so
    a floored atom contributes no gradient. That is the intended behaviour — it is
    exactly the divergence being removed — but it means a rollout with every atom pinned
    is a flat spot, not a steep one. See REPO_STATE WO-3.4.
    """
    return jnp.clip(terms, floor, 1.0)


def discount_weights(T: int, gamma: float, terminal_value: bool):
    """Time-aggregation weights. Mirrors tasks/base.py `discount_weights`.

    T/gamma/terminal_value are all jit-time CONSTANTS here (horizon and objective spec
    are fixed for a given controller), so this builds a concrete array -- no tracing.
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    # float ** int-array promotes to the default float dtype, so this is f32 in the
    # production path and f64 under enable_x64 in the parity tests -- as intended.
    d = gamma ** jnp.arange(T)
    if not terminal_value:
        norm = (1.0 - gamma) / (1.0 - gamma ** T) if T > 1 else 1.0
        return d * norm
    w = (1.0 - gamma) * d
    return w.at[-1].set(gamma ** (T - 1))


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
