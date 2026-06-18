"""DIAL — Diffusion-Inspired Annealing for Legged MPC (CPU port of mis/algs/dial.py).

Sampling noise per knot h and iteration i:
    sigma[i, h] = sigma_0 * exp(-i/(beta_opt_iter*N) - (H-1-h)/(beta_horizon*H))
Update: same softmax-weighted average as MPPI (FPL uses argmin best-sample).
"""
from __future__ import annotations

import numpy as np

from .sampling_base import SamplingController, Trajectory


class DIAL(SamplingController):
    def __init__(
        self,
        task,
        backend,
        *,
        num_samples: int,
        noise_level: float,
        temperature: float,
        beta_opt_iter: float,
        beta_horizon: float,
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
        if beta_opt_iter <= 0:
            raise ValueError("beta_opt_iter must be positive")
        if beta_horizon <= 0:
            raise ValueError("beta_horizon must be positive")
        super().__init__(
            task=task, backend=backend, num_samples=num_samples,
            num_knots=num_knots, plan_horizon=plan_horizon, spline_type=spline_type,
            iterations=iterations, seed=seed,
            use_fpl_cost=use_fpl_cost, use_fpl_discounted=use_fpl_discounted,
            fpl_p=fpl_p, fpl_gamma=fpl_gamma,
        )
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)
        self.beta_opt_iter = float(beta_opt_iter)
        self.beta_horizon = float(beta_horizon)
        self._opt_iter = 0

    def reset(self):
        super().reset()
        self._opt_iter = 0

    def sample_knots(self) -> np.ndarray:
        h = np.arange(self.num_knots, dtype=np.float64)
        sigma = self.noise_level * np.exp(
            -self._opt_iter / (self.beta_opt_iter * max(1, self.iterations))
            - (self.num_knots - 1 - h) / (self.beta_horizon * self.num_knots)
        )  # (num_knots,)
        noise = self.rng.normal(size=(self.num_samples, self.num_knots, self.nu))
        knots = self.mean[None, ...] + sigma[None, :, None] * noise
        self._opt_iter = (self._opt_iter + 1) % max(1, self.iterations)
        return knots

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        # FPL: best-sample via argmax on positive reward.
        # Normal: softmax-weighted average via cost-convention scores.
        if self.use_fpl_cost or self.use_fpl_discounted:
            best = int(traj.reward.argmax())
            return traj.knots[best].copy()
        scores = traj.scores
        z = -(scores - scores.min()) / self.temperature
        w = np.exp(z)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            best = int(np.argmin(scores))
            return traj.knots[best].copy()
        w = w / s
        return (w[:, None, None] * traj.knots).sum(axis=0)
