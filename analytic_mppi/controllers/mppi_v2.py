"""MPPI (knot-spline, CPU) — port of mis/algs/mppi.py minus the JAX bits.

Update rule:
  * normal cost     -> softmax-weighted average of knots
  * FPL (either mode) -> best-sample (lowest score == highest reward) selection
"""
from __future__ import annotations

import numpy as np

from .sampling_base import SamplingController, Trajectory


class MPPIv2(SamplingController):
    def __init__(
        self,
        task,
        backend,
        *,
        num_samples: int,
        noise_level: float,
        temperature: float,
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
            task=task,
            backend=backend,
            num_samples=num_samples,
            num_knots=num_knots,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            iterations=iterations,
            seed=seed,
            use_fpl_cost=use_fpl_cost,
            use_fpl_discounted=use_fpl_discounted,
            fpl_p=fpl_p,
            fpl_gamma=fpl_gamma,
        )
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)

    def sample_knots(self) -> np.ndarray:
        noise = self.rng.normal(
            scale=self.noise_level,
            size=(self.num_samples, self.num_knots, self.nu),
        )
        return self.mean[None, ...] + noise

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        scores = traj.scores
        # FPL modes use best-sample. Normal mode uses softmax-weighted average.
        if self.use_fpl_cost or self.use_fpl_discounted:
            best = int(np.argmin(scores))
            return traj.knots[best].copy()

        # numerically stable softmax of (-scores / temperature)
        z = -(scores - scores.min()) / self.temperature
        w = np.exp(z)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            best = int(np.argmin(scores))
            return traj.knots[best].copy()
        w = w / s
        return (w[:, None, None] * traj.knots).sum(axis=0)
