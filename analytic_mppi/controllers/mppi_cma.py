"""MPPI-CMA — block-diagonal Covariance Matrix Adaptation (CPU port of mis/algs/mppi_cma.py).

State: mean (K_knots, nu) + covariance (K_knots, nu, nu).
Update:
  mean ← sum(weights * knots) where weights = softmax(-cost/temp)
  cov  ← (1-alpha) * cov + alpha * sum(weights * (delta delta^T))
         with eigenvalues clamped to minimum_noise_level**2.
"""
from __future__ import annotations

import numpy as np

from .sampling_base import SamplingController, Trajectory


def _clamp_eigenvalues(cov: np.ndarray, min_eig: float) -> np.ndarray:
    """Per-knot symmetric eigenvalue clamp. cov: (K_knots, nu, nu) -> same shape.

    Uses np.linalg.eigh (symmetric eigendecomp); reconstructs after clipping
    eigenvalues from below. Vectorized across the leading knot axis.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, min_eig)
    # cov = V * diag(lambda) * V^T, batched
    return np.einsum("...ij,...j,...kj->...ik", eigvecs, eigvals, eigvecs)


class MppiCma(SamplingController):
    def __init__(
        self,
        task,
        backend,
        *,
        num_samples: int,
        initial_noise_level: float,
        temperature: float,
        minimum_noise_level: float | None = None,
        covariance_adaptation_rate: float = 0.1,
        num_knots: int = 4,
        plan_horizon: float = 1.0,
        spline_type: str = "zero",
        iterations: int = 1,
        seed: int = 0,
        use_fpl_cost: bool = False,
        use_fpl_discounted: bool = False,
        fpl_p: float = 0.1,
        fpl_gamma: float = 0.99,
    ):
        super().__init__(
            task=task, backend=backend, num_samples=num_samples,
            num_knots=num_knots, plan_horizon=plan_horizon, spline_type=spline_type,
            iterations=iterations, seed=seed,
            use_fpl_cost=use_fpl_cost, use_fpl_discounted=use_fpl_discounted,
            fpl_p=fpl_p, fpl_gamma=fpl_gamma,
        )
        self.initial_noise_level = float(initial_noise_level)
        self.minimum_noise_level = (
            float(minimum_noise_level)
            if minimum_noise_level is not None
            else self.initial_noise_level
        )
        self.alpha = float(covariance_adaptation_rate)
        self.temperature = float(temperature)
        # Per-knot covariance, initially sigma_0^2 * I.
        eye = np.eye(self.nu, dtype=np.float64)
        self.cov = np.broadcast_to(
            (self.initial_noise_level ** 2) * eye,
            (self.num_knots, self.nu, self.nu),
        ).copy()

    def reset(self):
        super().reset()
        eye = np.eye(self.nu, dtype=np.float64)
        self.cov[:] = (self.initial_noise_level ** 2) * eye

    def sample_knots(self) -> np.ndarray:
        # Per-knot multivariate-normal samples. rng.multivariate_normal isn't
        # vectorized over a leading axis, so loop over the num_knots dimension
        # (which is small — typically 4–8).
        K = self.num_samples
        out = np.empty((K, self.num_knots, self.nu), dtype=np.float64)
        zero = np.zeros(self.nu)
        for h in range(self.num_knots):
            out[:, h, :] = self.rng.multivariate_normal(zero, self.cov[h], size=K)
        out += self.mean[None, ...]
        return out

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        scores = traj.scores
        if self.use_fpl_cost or self.use_fpl_discounted:
            # FPL path: pick best sample (no covariance update — match mis MPPI behavior)
            best = int(np.argmin(scores))
            return traj.knots[best].copy()

        # softmax weights (lower cost = higher weight)
        z = -(scores - scores.min()) / self.temperature
        w = np.exp(z)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            best = int(np.argmin(scores))
            return traj.knots[best].copy()
        w = w / s

        # delta := knots - mean_prev    shape (K, num_knots, nu)
        delta = traj.knots - self.mean
        # weighted outer product per knot: einsum("k,khi,khj->hij")
        new_sample_cov = np.einsum("k,khi,khj->hij", w, delta, delta)
        new_cov = (1.0 - self.alpha) * self.cov + self.alpha * new_sample_cov
        new_cov = _clamp_eigenvalues(new_cov, self.minimum_noise_level ** 2)
        self.cov = new_cov

        return np.einsum("k,khj->hj", w, traj.knots)
