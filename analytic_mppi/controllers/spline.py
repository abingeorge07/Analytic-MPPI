"""Knot-spline interpolation for sampling-based MPC.

Algorithms parameterize a control trajectory by a few "knots" placed at
times `tk` (uniformly spaced over `[0, plan_horizon]`) and the executed
control sequence is obtained by interpolating the knots to the simulator
step grid `t_eval` (H samples on `[0, plan_horizon)`).

Interpolation is purely numpy and broadcasts over arbitrary leading
batch dims, e.g. (K, num_knots, nu) -> (K, H, nu).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def make_knot_times(plan_horizon: float, num_knots: int) -> np.ndarray:
    """Uniformly spaced knot times on [0, plan_horizon].

    With num_knots == 1 we return [0.0]. With >= 2 knots, includes endpoints.
    """
    if num_knots < 1:
        raise ValueError(f"num_knots must be >= 1, got {num_knots}")
    if num_knots == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, float(plan_horizon), num_knots, dtype=np.float64)


def make_eval_times(plan_horizon: float, dt: float) -> Tuple[np.ndarray, int]:
    """Per-step evaluation times for a rollout of dt timestep.

    Returns (t_eval, H) where t_eval has H = round(plan_horizon / dt) points
    placed at t_eval[i] = i * dt, so the first action lands at t=0.
    """
    H = max(1, int(round(float(plan_horizon) / float(dt))))
    return np.arange(H, dtype=np.float64) * dt, H


def interp_zero(knots: np.ndarray, tk: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Zero-order-hold interpolation along the second-to-last axis.

    knots:   (..., num_knots, nu)
    tk:      (num_knots,)
    t_eval:  (H,)
    returns: (..., H, nu)

    Each eval time picks the most recent knot at-or-before it (clamped to
    knot 0 if t_eval[i] < tk[0], which shouldn't happen with default usage).
    """
    idx = np.searchsorted(tk, t_eval, side="right") - 1
    idx = np.clip(idx, 0, tk.shape[0] - 1)
    return knots[..., idx, :]


def interp_linear(knots: np.ndarray, tk: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Clamped linear interpolation along the second-to-last axis.

    knots:   (..., num_knots, nu)
    tk:      (num_knots,) — monotone increasing
    t_eval:  (H,)
    returns: (..., H, nu)
    """
    K = tk.shape[0]
    if K == 1:
        return interp_zero(knots, tk, t_eval)
    right = np.clip(np.searchsorted(tk, t_eval, side="left"), 1, K - 1)
    left = right - 1
    tl, tr = tk[left], tk[right]
    denom = np.where(tr > tl, tr - tl, 1.0)
    w = np.clip((t_eval - tl) / denom, 0.0, 1.0)  # (H,)
    # broadcast w to (..., H, 1) so we can mix knot vectors per-eval-time
    kl = knots[..., left, :]
    kr = knots[..., right, :]
    return kl + (kr - kl) * w[..., :, None]

def interp_cubic(knots: np.ndarray, tk: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Clamped cubic-Hermite (Catmull-Rom) interpolation along the second-to-last axis.

    knots:   (..., num_knots, nu)
    tk:      (num_knots,) — monotone increasing
    t_eval:  (H,)
    returns: (..., H, nu)

    Tangents at each knot are finite differences of neighbors (centered for
    interior knots, one-sided at the endpoints) — no tridiagonal solve,
    unlike a natural cubic spline. Output is C^1 and each evaluation
    depends on the 4 surrounding knots only, so perturbing one knot only
    changes a local stretch of the curve — the right property for
    knot-spline MPC.

    Degenerates to interp_linear when num_knots == 2 and to interp_zero
    when num_knots == 1.
    """
    K = tk.shape[0]
    if K == 1:
        return interp_zero(knots, tk, t_eval)
    if K == 2:
        return interp_linear(knots, tk, t_eval)

    # per-knot tangents m: (..., K, nu)
    m_first = (knots[..., 1:2, :] - knots[..., 0:1, :]) / (tk[1] - tk[0])
    m_last  = (knots[..., -1:, :] - knots[..., -2:-1, :]) / (tk[-1] - tk[-2])
    m_mid   = (knots[..., 2:, :] - knots[..., :-2, :]) / (tk[2:] - tk[:-2])[:, None]
    m = np.concatenate([m_first, m_mid, m_last], axis=-2)

    right = np.clip(np.searchsorted(tk, t_eval, side="left"), 1, K - 1)
    left = right - 1
    tl, tr = tk[left], tk[right]
    h = np.where(tr > tl, tr - tl, 1.0)
    u = np.clip((t_eval - tl) / h, 0.0, 1.0)              # (H,)
    u2, u3 = u * u, u * u * u
    h00 =  2 * u3 - 3 * u2 + 1
    h10 =      u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 =      u3 -     u2

    yl = knots[..., left,  :]
    yr = knots[..., right, :]
    ml = m[...,    left,  :]
    mr = m[...,    right, :]
    hb = h[:, None]
    return (h00[:, None] * yl
            + h10[:, None] * (hb * ml)
            + h01[:, None] * yr
            + h11[:, None] * (hb * mr))


def shift_plan(mean: np.ndarray, tk: np.ndarray, dt: float, spline_type: str,
               shift_accum: float) -> Tuple[np.ndarray, float]:
    """Receding-horizon warm start: advance a knot plan `mean` (num_knots, nu) by `dt`.

    Extracted verbatim from SamplingController._shift_mean so gradient MPC warm-starts
    with the IDENTICAL semantics as the sampling stack (same plans are interchangeable).
    Returns (new_mean, new_shift_accum).

    Zero-order-hold is piecewise constant, so resampling it at tk+dt is a no-op whenever
    dt < knot spacing -- that would silently disable the warm-start time-advance for the
    default coarse-knot ZOH config. Instead accumulate elapsed time in `shift_accum` and
    roll the plan one knot toward t=0 each time a full knot spacing has elapsed
    (repeating the terminal knot). Continuous splines (linear/cubic) represent a sub-knot
    shift exactly, so they resample at tk + dt and ignore the accumulator.
    """
    num_knots = mean.shape[0]
    if num_knots == 1:
        return mean, shift_accum

    if spline_type == "zero":
        spacing = float(tk[1] - tk[0])
        shift_accum += dt
        # +eps so an exact multiple of `spacing` reached via float accumulation
        # (e.g. 10 * 0.02 = 0.19999...) fires on the intended step, not one late.
        n_roll = int((shift_accum + 1e-9) // spacing)
        if n_roll > 0:
            shift_accum -= n_roll * spacing
            n_roll = min(n_roll, num_knots - 1)
            tail = np.repeat(mean[-1:], n_roll, axis=0)
            mean = np.concatenate([mean[n_roll:], tail], axis=0)
        return mean, shift_accum

    return interpolate(mean[None, ...], tk, tk + dt, spline_type)[0], shift_accum


SPLINE_INTERPOLATORS = {
    "zero": interp_zero,
    "linear": interp_linear,
    "cubic": interp_cubic
}


def interpolate(knots: np.ndarray, tk: np.ndarray, t_eval: np.ndarray, spline_type: str = "zero") -> np.ndarray:
    if spline_type not in SPLINE_INTERPOLATORS:
        raise ValueError(
            f"Unknown spline_type {spline_type!r}; expected one of "
            f"{sorted(SPLINE_INTERPOLATORS)}"
        )
    return SPLINE_INTERPOLATORS[spline_type](knots, tk, t_eval)
