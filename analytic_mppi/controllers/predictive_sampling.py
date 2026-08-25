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
        **kwargs,
    ):
        # Forward the whole spline / FPL keyword surface to SamplingController rather than
        # re-declaring a subset here: hand-enumerating it is what made this controller
        # unusable with the linear arm (`fpl_weights`) and the temporal weakest-link
        # (`fpl_time_p`). Same passthrough pattern as FplColoredMPPI / FplTemperedMPPI.
        # There is deliberately no `temperature`: selection is argmax by construction.
        super().__init__(task=task, backend=backend, num_samples=num_samples, **kwargs)
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
        # Greedy selection puts all weight on one rollout, so ESS is 1 by construction.
        # Recorded explicitly so this sampler appears in the ESS diagnostics alongside
        # the softmax controllers instead of logging NaN.
        self.last_ess = 1.0
        return traj.knots[best].copy()
