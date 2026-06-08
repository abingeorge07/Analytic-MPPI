"""Predictive Sampling (CPU port of mis/algs/predictive_sampling.py).

Greedy best-sample selection — no softmax. The current mean is included as one
of the samples so the algorithm can never regress.
"""
from __future__ import annotations

import numpy as np

from .sampling_base import SamplingController, Trajectory


class PredictiveSampling(SamplingController):
    def __init__(
        self,
        task,
        backend,
        *,
        num_samples: int,
        noise_level: float,
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
        self.noise_level = float(noise_level)

    def sample_knots(self) -> np.ndarray:
        noise = self.rng.normal(
            scale=self.noise_level,
            size=(self.num_samples, self.num_knots, self.nu),
        )
        knots = self.mean[None, ...] + noise
        # Always include the current mean as the first sample.
        knots[0] = self.mean
        return knots

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        # FPL: argmax on positive reward. Normal: argmin on cost-convention scores.
        if self.use_fpl_cost or self.use_fpl_discounted:
            best = int(traj.reward.argmax())
        else:
            best = int(np.argmin(traj.scores))
        return traj.knots[best].copy()
