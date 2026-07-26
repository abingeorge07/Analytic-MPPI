"""FPL-informed adaptive-covariance MPPI — "gain information from the rollouts, then
sample better," using the two things FPL gives a sampler that a scalar cost cannot.

The MPPI update (softmax-weighted mean) is inherited UNCHANGED from MPPIv2 — this
controller only changes HOW the next batch is SAMPLED, by adapting a per-knot-dimension
diagonal covariance from the information in the current rollout population.

Two FPL-specific signals drive the adaptation
---------------------------------------------
1. ABSOLUTE SCALE  ->  explore/exploit magnitude.
   FPL rewards live in a fixed [0,1] frame, so "is any rollout actually good yet?" is a
   well-posed question. When the best composite fulfillment is low (every plan is bad) the
   sampler INFLATES the global step (explore); once a high-fulfillment plan appears it
   COLLAPSES toward it (exploit). A scalar cost has no fixed reference for "good," so it
   cannot calibrate this — it can only rank. (This is the FplGmmSampler thesis, applied to
   a Gaussian search distribution.)

2. PER-OBJECTIVE VECTOR  ->  WHERE to explore (information-directed).
   The binding objective is j* = argmin_j s_j (least-satisfied, on the absolute scale). We
   measure, per control knot-dimension d, how strongly perturbing d moved the binding
   objective's fulfillment across the sampled cloud (a cheap per-dim correlation — a
   one-step sensitivity estimate). We then widen the sampling variance along the dims that
   most affect the binding objective and keep the rest tight: samples are spent where they
   are most INFORMATIVE about satisfying the currently-unmet objective, instead of
   isotropically. This needs the per-objective fulfillment vector, so it is intrinsic to FPL.

Fairness / ablations (`steer_mode`, `explore_mode`)
---------------------------------------------------
The covariance-adaptation MACHINERY is not FPL-specific; only these two signals are. So the
same controller runs as a NON-FPL baseline for an apples-to-apples read:
  * steer_mode:   "binding" (FPL, sensitivity of the worst objective)   [default]
                  "scalar"  (sensitivity of the composite reward — works on ANY cost)
                  "none"    (isotropic; no directional steering)
  * explore_mode: "absolute" (FPL, global scale from the absolute best fulfillment) [default]
                  "gap"      (relative best-vs-mean spread — no absolute frame; non-FPL ok)
                  "fixed"    (constant global scale — vanilla MPPI covariance)
A strictly-fair comparison uses steer_mode="scalar"/explore_mode="gap" for the linear
baseline vs "binding"/"absolute" for FPL, with everything else identical.

Requires an FPL cost mode exposing the per-objective vector (`use_fpl_discounted` or
`use_fpl_layered`) whenever steer_mode="binding" — `traj.reward_terms` must be populated.
"""
from __future__ import annotations

import numpy as np

from .mppi_v2 import MPPIv2
from .sampling_base import Trajectory


class FplAdaptiveMPPI(MPPIv2):
    def __init__(
        self,
        task,
        backend,
        *,
        sigma_floor: float = 0.05,
        sigma_ceil: float = 1.2,
        explore_scale_lo: float = 0.15,   # global std when fully committed (exploit)
        explore_scale_hi: float = 0.6,    # global std when nothing is good yet (explore)
        steer_gain: float = 2.0,          # max anisotropic variance boost (1 + steer_gain)
        steer_ema: float = 0.5,           # EMA smoothing of the per-dim steer profile
        commit_w_best: float = 0.7,       # blend of best vs gap in the commitment scalar
        gap_ref: float = 0.3,             # gap normalizer for explore_mode="gap"
        steer_mode: str = "binding",      # "binding" | "scalar" | "none"
        explore_mode: str = "absolute",   # "absolute" | "gap" | "fixed"
        **kwargs,
    ):
        # noise_level is inherited but unused for sampling here (we use per-dim sigma).
        kwargs.setdefault("noise_level", explore_scale_hi)
        super().__init__(task, backend, **kwargs)
        if steer_mode not in ("binding", "scalar", "none"):
            raise ValueError(f"steer_mode must be binding|scalar|none, got {steer_mode!r}")
        if explore_mode not in ("absolute", "gap", "fixed"):
            raise ValueError(f"explore_mode must be absolute|gap|fixed, got {explore_mode!r}")
        if steer_mode == "binding" and not (self.use_fpl_discounted or self.use_fpl_layered):
            raise ValueError(
                "steer_mode='binding' needs a per-objective FPL vector: use cost_mode "
                "'fpl_discounted' or 'fpl_layered' (so traj.reward_terms is populated)."
            )
        self.sigma_floor = float(sigma_floor)
        self.sigma_ceil = float(sigma_ceil)
        self.explore_scale_lo = float(explore_scale_lo)
        self.explore_scale_hi = float(explore_scale_hi)
        self.steer_gain = float(steer_gain)
        self.steer_ema = float(steer_ema)
        self.commit_w_best = float(commit_w_best)
        self.gap_ref = float(gap_ref)
        self.steer_mode = steer_mode
        self.explore_mode = explore_mode

        # Per-knot-dimension diagonal std, adapted each iteration. Start isotropic.
        self.sigma = np.full((self.num_knots, self.nu), explore_scale_hi, dtype=np.float64)
        # Persisted anisotropic steer profile in [0,1] per knot-dim (EMA-smoothed).
        self._steer = np.zeros((self.num_knots, self.nu), dtype=np.float64)
        # Diagnostics.
        self.last_commit: float | None = None
        self.last_binding_obj: int | None = None
        self.last_sigma_mean: float | None = None

    # ---- sampling: anisotropic per-knot-dim Gaussian ----

    def sample_knots(self) -> np.ndarray:
        z = self.rng.standard_normal((self.num_samples, self.num_knots, self.nu))
        return self.mean[None, ...] + self.sigma[None, ...] * z

    # ---- warm-start: shift sigma forward like the mean ----

    def act(self, state):
        u0 = super().act(state)
        self._shift_sigma(self.backend.dt)
        return u0

    def _shift_sigma(self, dt):
        if self.num_knots <= 1 or self.spline_type != "zero":
            return
        spacing = float(self.tk[1] - self.tk[0])
        # Reuse the base class's accumulated-shift bookkeeping by mirroring its rule: roll
        # whole knots once a knot spacing has elapsed. We track our own accumulator so we
        # don't perturb the base _shift_accum (already advanced in super().act()).
        acc = getattr(self, "_sigma_shift_accum", 0.0) + dt
        n_roll = int((acc + 1e-9) // spacing)
        self._sigma_shift_accum = acc - n_roll * spacing
        if n_roll > 0:
            n_roll = min(n_roll, self.num_knots - 1)
            tail = np.repeat(self.sigma[-1:], n_roll, axis=0)
            self.sigma = np.concatenate([self.sigma[n_roll:], tail], axis=0)
            steer_tail = np.repeat(self._steer[-1:], n_roll, axis=0)
            self._steer = np.concatenate([self._steer[n_roll:], steer_tail], axis=0)

    # ---- covariance adaptation from the rollout information ----

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        mean_old = self.mean                      # samples were drawn around this
        new_mean = super().update_mean(traj)      # inherited softmax mean + last_ess
        self._adapt_sigma(traj, mean_old)
        return new_mean

    def _commitment(self, traj: Trajectory) -> float:
        """Scalar in [0,1]: how much to exploit (1) vs explore (0)."""
        if self.explore_mode == "fixed":
            return 1.0  # constant scale handled by caller (uses explore_scale_lo)
        u = traj.reward
        if u is None:
            # Non-FPL fallback: rank-only, so use normalized cost spread as a pseudo-gap.
            s = traj.scores
            rng = float(s.max() - s.min())
            denom = abs(float(s.mean())) + 1e-6
            return float(np.clip(rng / denom, 0.0, 1.0))
        f_best, f_mean = float(u.max()), float(u.mean())
        if self.explore_mode == "gap":
            # Relative: big spread => a clear winner exists => commit. No absolute frame.
            return float(np.clip((f_best - f_mean) / self.gap_ref, 0.0, 1.0))
        # absolute: the FPL frame. High best fulfillment => genuinely good => commit.
        gap = np.clip((f_best - f_mean) / self.gap_ref, 0.0, 1.0)
        return float(np.clip(self.commit_w_best * f_best
                             + (1.0 - self.commit_w_best) * gap, 0.0, 1.0))

    def _steer_profile(self, traj: Trajectory, mean_old: np.ndarray) -> np.ndarray:
        """Per-knot-dim value in [0,1]: how much perturbing that dim moved the target
        objective (binding objective for FPL, composite reward for the scalar mirror)."""
        if self.steer_mode == "none":
            return np.zeros((self.num_knots, self.nu))
        E = traj.knots - mean_old[None, ...]                     # (K, kn, nu)
        if self.steer_mode == "binding" and traj.reward_terms is not None:
            V = traj.reward_terms                                # (K, J)
            s = V.mean(axis=0)
            j = int(np.argmin(s))
            self.last_binding_obj = j
            target = V[:, j]
        else:
            target = traj.reward if traj.reward is not None else -traj.scores
        # Per-dim Pearson correlation |corr(E[:,d], target)|: a cheap one-step sensitivity.
        t = target - target.mean()
        t_norm = np.linalg.norm(t) + 1e-12
        Ec = E - E.mean(axis=0, keepdims=True)                   # center per dim
        num = np.abs(np.einsum("kij,k->ij", Ec, t))              # (kn, nu)
        den = np.linalg.norm(Ec, axis=0) * t_norm + 1e-12        # (kn, nu)
        corr = num / den                                         # in [0,1]
        m = corr.max()
        return corr / m if m > 1e-9 else corr

    def _adapt_sigma(self, traj: Trajectory, mean_old: np.ndarray) -> None:
        commit = self._commitment(traj)
        self.last_commit = commit
        # Global explore/exploit scale: low when committed, high when nothing is good.
        if self.explore_mode == "fixed":
            base = self.explore_scale_lo
        else:
            base = self.explore_scale_lo + (self.explore_scale_hi - self.explore_scale_lo) * (1.0 - commit)
        # Anisotropic steering toward the informative dims (EMA-smoothed across iterations).
        prof = self._steer_profile(traj, mean_old)
        self._steer = (1.0 - self.steer_ema) * self._steer + self.steer_ema * prof
        steer_factor = 1.0 + self.steer_gain * self._steer      # (kn, nu) in [1, 1+gain]
        sigma = base * steer_factor
        self.sigma = np.clip(sigma, self.sigma_floor, self.sigma_ceil)
        self.last_sigma_mean = float(self.sigma.mean())

    def reset(self):
        super().reset()
        self.sigma = np.full((self.num_knots, self.nu), self.explore_scale_hi, dtype=np.float64)
        self._steer = np.zeros((self.num_knots, self.nu), dtype=np.float64)
        self._sigma_shift_accum = 0.0


__all__ = ["FplAdaptiveMPPI"]
