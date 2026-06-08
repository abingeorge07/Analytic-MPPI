"""Cross-Entropy Method (CPU port of mis/algs/cem.py).

State: per-knot per-actuator diagonal sigma plus the mean.
Update: fit a Gaussian to the top-`num_elites` rollouts; clamp sigma to sigma_min.
"""
from __future__ import annotations

import numpy as np

from .sampling_base import SamplingController, Trajectory


class CEM(SamplingController):
    def __init__(
        self,
        task,
        backend,
        *,
        num_samples: int,
        num_elites: int,
        sigma_start: float,
        sigma_min: float,
        explore_fraction: float = 0.0,
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
        if not 0.0 <= explore_fraction <= 1.0:
            raise ValueError(f"explore_fraction must be in [0,1], got {explore_fraction}")
        super().__init__(
            task=task, backend=backend, num_samples=num_samples,
            num_knots=num_knots, plan_horizon=plan_horizon, spline_type=spline_type,
            iterations=iterations, seed=seed,
            use_fpl_cost=use_fpl_cost, use_fpl_discounted=use_fpl_discounted,
            fpl_p=fpl_p, fpl_gamma=fpl_gamma,
        )
        self.num_elites = int(num_elites)
        self.sigma_start = float(sigma_start)
        self.sigma_min = float(sigma_min)
        self.num_explore = int(round(num_samples * explore_fraction))
        # Per-knot per-actuator sigma; starts at sigma_start everywhere.
        self.cov = np.full_like(self.mean, self.sigma_start)

    def reset(self):
        super().reset()
        self.cov[:] = self.sigma_start

    def sample_knots(self) -> np.ndarray:
        K_main = self.num_samples - self.num_explore
        shape_main = (K_main, self.num_knots, self.nu)
        shape_explore = (self.num_explore, self.num_knots, self.nu)

        main = self.mean + self.cov * self.rng.normal(size=shape_main) if K_main > 0 else np.empty(shape_main)
        explore = (
            self.mean + self.sigma_start * self.rng.normal(size=shape_explore)
            if self.num_explore > 0
            else np.empty(shape_explore)
        )
        return np.concatenate([main, explore], axis=0)

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        # FPL: pick the highest-reward samples. Normal: pick the lowest-cost samples.
        if self.use_fpl_cost or self.use_fpl_discounted:
            elites = np.argsort(-traj.reward)[: self.num_elites]   # descending by reward
        else:
            elites = np.argsort(traj.scores)[: self.num_elites]    # ascending by cost
        elite_knots = traj.knots[elites]
        new_mean = elite_knots.mean(axis=0)
        new_cov = np.maximum(elite_knots.std(axis=0), self.sigma_min)
        self.cov = new_cov
        return new_mean
