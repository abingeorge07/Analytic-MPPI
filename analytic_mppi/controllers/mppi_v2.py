"""MPPI (knot-spline, CPU) — port of mis/algs/mppi.py minus the JAX bits.

Update rule (identical for normal and FPL — this is what makes the FPL-vs-linear
comparison fair; only the *scoring* upstream differs):
  * softmax-weighted average of knots, w_k ∝ exp(-(S_k - min_k S_k)/temperature).
  * In FPL mode S_k = -log(u_k) (see sampling_base._score_fpl), so this becomes
    w_k ∝ (u_k / max_k u_k)^(1/temperature) — a true MPPI update over the FPL
    composite, per FPL_MPPI_HANDOFF §3. (Greedy best-sample selection lives in
    PredictiveSampling; use that for the argmax ablation.)

FPL weighting modes (`fpl_weighting`, FPL scoring only):
  * "softmax"      (default) — the -log(u)+softmax path above.
  * "proportional" — the simplest possible reward weighting: w_k = u_k / Σ_j u_j
    directly on the raw FPL composite u_k ∈ (0,1]. No temperature, no -log, no
    min-shift. Selectivity then comes ENTIRELY from the spread of u (i.e. fpl_p),
    not from temperature — clustered u's give near-uniform (mushy) weights, a very
    negative fpl_p gives u's with the dynamic range needed to actually discriminate.
  * "exp"          — vanilla-MPPI exponential form on the reward: w_k ∝ exp(r_k/λ)
    (max-shifted for stability), i.e. exp weighting of the FPL composite in [0,1].

`fpl_term_indices` (FPL scoring only) restricts the reward to a subset of the task's
fulfillment atoms — e.g. [2] on g1_standup runs the normal fpl_cost/fpl_discounted
pipeline over ONLY nominal_fulfillment (a single-term FPL reward).
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
        use_fpl_layered: bool = False,
        fpl_p: float = 0.1,
        fpl_gamma: float = 0.99,
        fpl_time_p: float | None = None,
        fpl_group_p: float | None = None,
        fpl_term_indices: "list[int] | None" = None,
        fpl_weights: "list[float] | None" = None,
        use_hybrid: bool = False,
        floor_weight: float = 1.0,
        fpl_weighting: str = "softmax",
    ):
        if fpl_weighting not in ("softmax", "proportional", "exp"):
            raise ValueError(
                f"fpl_weighting must be 'softmax', 'proportional' or 'exp', got {fpl_weighting!r}"
            )
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
            use_fpl_layered=use_fpl_layered,
            fpl_term_indices=fpl_term_indices,
            fpl_p=fpl_p,
            fpl_gamma=fpl_gamma,
            fpl_time_p=fpl_time_p,
            fpl_group_p=fpl_group_p,
            fpl_weights=fpl_weights,
            use_hybrid=use_hybrid,
            floor_weight=floor_weight,
        )
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)
        self.fpl_weighting = fpl_weighting
        # Effective sample size of the last softmax weighting: (Σw)²/Σw² ∈ [1, K].
        # Collapse toward 1 == the update is riding a single rollout (FPL_MPPI_HANDOFF §5).
        self.last_ess: float | None = None

    def sample_knots(self) -> np.ndarray:
        noise = self.rng.normal(
            scale=self.noise_level,
            size=(self.num_samples, self.num_knots, self.nu),
        )
        return self.mean[None, ...] + noise

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        # Proportional (linear) FPL weighting: w_k = u_k / Σ_j u_j on the raw FPL
        # reward u_k ∈ (0,1]. No temperature / -log / min-shift — the bounded reward
        # is already a valid unnormalized weight. Only applies when _score_fpl ran
        # (traj.reward populated); non-FPL modes fall through to the softmax below.
        if self.fpl_weighting == "proportional" and traj.reward is not None:
            u = traj.reward
            s = u.sum()
            if s <= 0 or not np.isfinite(s):
                self.last_ess = 1.0
                return traj.knots[int(np.argmax(u))].copy()
            w = u / s
            self.last_ess = float(1.0 / np.sum(w ** 2))
            return (w[:, None, None] * traj.knots).sum(axis=0)

        # Vanilla-MPPI exponential weighting applied directly to the FPL reward:
        #   w_k ∝ exp(r_k / λ), stabilized by subtracting max_k r_k.
        # With use_fpl_normal (reward = -normal cost) this is exp(-cost/λ) == vanilla.
        if self.fpl_weighting == "exp" and traj.reward is not None:
            r = traj.reward
            z = (r - r.max()) / self.temperature
            w = np.exp(z)
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                self.last_ess = 1.0
                return traj.knots[int(np.argmax(r))].copy()
            w = w / s
            self.last_ess = float(1.0 / np.sum(w ** 2))
            return (w[:, None, None] * traj.knots).sum(axis=0)

        # Single softmax-weighted average for both normal and FPL. `scores` is in
        # cost convention (smaller is better) in either mode — for FPL it is
        # -log(u_k), so this is a true MPPI update over the FPL composite.
        scores = traj.scores
        z = -(scores - scores.min()) / self.temperature
        w = np.exp(z)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            self.last_ess = 1.0
            best = int(np.argmin(scores))
            return traj.knots[best].copy()
        w = w / s
        # ESS = (Σw)²/Σw² = 1/Σw² for normalized w. Diagnoses weight collapse.
        self.last_ess = float(1.0 / np.sum(w ** 2))
        return (w[:, None, None] * traj.knots).sum(axis=0)
