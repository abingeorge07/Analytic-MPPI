"""jnp mirrors of the three CORE scoring aggregations from sampling_base.py.

Used by gradient MPC, which must differentiate the objective: these run inside its
jitted `value_and_grad`. Each function is a formula-for-formula port of the numpy
scorer it names (parity-pinned by tests/test_jax_costs.py at 1e-6):

  score_normal  <- SamplingController._score_normal      (sampling_base.py:263-273)
  score_fpl     <- SamplingController._score_fpl          (sampling_base.py:357-449),
                   restricted to the flat power-mean collapse (fpl_conj_indices stays
                   CPU-only by scope), with `select_fpl_terms` mirroring
                   _select_fpl_terms (:275-286).

Conventions preserved exactly:
  * smaller is better; score_fpl returns -log(reward), so gradient DESCENT on these
    scores minimizes the identical objective MPPI ranks rollouts by.
  * terminal fulfillment terms are a PREFIX of the running terms (n_term <= n_run);
    trailing running terms span only H steps.
  * all reductions use negative axes, so any leading batch shape works ((H, n) for a
    single rollout inside gradient MPC, (K, H, n) in parity tests).
"""
from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp

from analytic_mppi.tasks.jax_costs._base import (ATOM_FLOOR_LEGACY, discount_weights,
                                                 floor_atoms, power_mean)


def score_normal(running_terms, terminal_terms, *, dt: float):
    """Discretized time-integral of running cost + one-shot terminal cost.

    running_terms (..., H, n), terminal_terms (..., n) -> (...,)
    """
    running = running_terms.sum(axis=(-2, -1)) * dt
    terminal = terminal_terms.sum(axis=-1)
    return running + terminal


def select_fpl_terms(running_f, terminal_f, term_indices: Sequence[int]):
    """Restrict fulfillment atoms to `term_indices`, preserving the prefix convention:
    selected indices that HAVE a terminal value (< n_term) come first."""
    n_term = terminal_f.shape[-1]
    with_term = [i for i in term_indices if i < n_term]
    without_term = [i for i in term_indices if i >= n_term]
    running_sel = running_f[..., with_term + without_term]
    terminal_sel = terminal_f[..., with_term]
    return running_sel, terminal_sel


def score_fpl(running_f, terminal_f, *, mode: str, p: float, gamma: float,
              time_p: Optional[float] = None, time_discount: bool = False,
              weights: Optional[Sequence[float]] = None,
              terminal_value: bool = False):
    """FPL cost = -log(reward). running_f (..., H, n_run), terminal_f (..., n_term).

    mode="fpl_cost": per-step power-mean over objectives, then discounted mean over
    time (time_p None) or soft-min-over-time power-mean at time_p (optionally
    discount-weighted). mode="fpl_discounted": per-term normalized discount-sums over
    each term's lifetime, then one power-mean over the per-term scalars.
    """
    H, n_run = running_f.shape[-2], running_f.shape[-1]
    n_term = terminal_f.shape[-1]
    if n_term > n_run:
        raise ValueError(f"n_term={n_term} > n_run={n_run}; expected n_term <= n_run")

    H1 = H + 1
    # Weights carry the normalization (mirrors sampling_base._score_fpl); `terminal_value`
    # swaps the renormalized discounted mean for an explicit post-horizon tail (WO-3.3).
    discounts_full = discount_weights(H1, gamma, terminal_value)  # (H+1,)
    norm_full = 1.0
    discounts_run = discount_weights(H, gamma, terminal_value)    # (H,)
    norm_run = 1.0

    w = None if weights is None else jnp.asarray(weights)

    if mode == "fpl_cost":
        w_term = None if w is None else w[:n_term]
        per_step_run = power_mean(running_f, p, weights=w)              # (..., H)
        if n_term > 0:
            per_step_term = power_mean(terminal_f, p, weights=w_term)   # (...,)
            per_step = jnp.concatenate(
                [per_step_run, per_step_term[..., None]], axis=-1)      # (..., H+1)
            disc, norm = discounts_full, norm_full
        else:
            per_step = per_step_run
            disc, norm = discounts_run, norm_run
        if time_p is None:
            reward = (per_step * disc).sum(axis=-1) * norm
        else:
            time_w = disc if time_discount else None
            reward = power_mean(per_step, time_p, weights=time_w)
    elif mode == "fpl_discounted":
        chunks = []
        if n_term > 0:
            extended = jnp.concatenate(
                [running_f[..., :n_term], terminal_f[..., None, :]], axis=-2
            )                                                           # (..., H+1, n_term)
            chunks.append((extended * discounts_full[:, None]).sum(axis=-2) * norm_full)
        if n_run > n_term:
            chunks.append(
                (running_f[..., n_term:] * discounts_run[:, None]).sum(axis=-2) * norm_run)
        per_term_sums = jnp.concatenate(chunks, axis=-1)                # (..., n_run)
        reward = power_mean(per_term_sums, p, weights=w)
    else:
        raise ValueError(f"unknown mode {mode!r} (fpl_cost | fpl_discounted)")

    return -jnp.log(reward)
