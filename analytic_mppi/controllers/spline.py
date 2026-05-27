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


SPLINE_INTERPOLATORS = {
    "zero": interp_zero,
    "linear": interp_linear,
}


def interpolate(knots: np.ndarray, tk: np.ndarray, t_eval: np.ndarray, spline_type: str = "zero") -> np.ndarray:
    if spline_type not in SPLINE_INTERPOLATORS:
        raise ValueError(
            f"Unknown spline_type {spline_type!r}; expected one of "
            f"{sorted(SPLINE_INTERPOLATORS)}"
        )
    return SPLINE_INTERPOLATORS[spline_type](knots, tk, t_eval)
