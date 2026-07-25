"""Experimental sampling-MPC controllers developed in `analysis.ipynb`.

These were originally defined inline in the analysis notebook (and wired in via
`make_ctrl` monkey-patches). They live here now so any notebook can simply
`import` them — with `%autoreload 2` you can keep editing this file and re-run a
cell without restarting the kernel, and every environment notebook sees the same
latest code.

Algorithms (all subclass `SamplingController`):
  UniformGD    - per-step Uniform(-w, +w) noise around the best rollout + curvature GD
  GaussianGD   - per-step N(0, w^2) noise around the best rollout + curvature GD
  RankCMA      - per-step multivariate Gaussian with within-step rank-weighted CMA refit
  FplValueCMA  - RankCMA + an FPL value-softmax bonus on the elite cohort (FPL mode only)

All four use the `(num_knots = H, tk = t_eval, spline_type="zero")` trick so the
base-class `interpolate()` becomes the identity and `sample_knots()` emits
per-step controls directly.
"""
from __future__ import annotations

import numpy as np

from .sampling_base import SamplingController, Trajectory
from .mppi_cma import _clamp_eigenvalues
from .spline import interpolate


def smoothness_curv_grad(u: np.ndarray) -> np.ndarray:
    """Gradient of the curvature roughness penalty S = Σ ||u_{i+1} - 2u_i + u_{i-1}||².

    Returns 2·L²u where L is the 1-D discrete Laplacian along time. Same shape as
    u (..., H, nu). Endpoints get one-sided contributions only.
    """
    d2 = u[..., 2:, :] - 2.0 * u[..., 1:-1, :] + u[..., :-2, :]   # (..., H-2, nu)
    g = np.zeros_like(u)
    g[..., :-2,  :] += 2.0 * d2     # contribution to u_{i-1}
    g[..., 1:-1, :] -= 4.0 * d2     # contribution to u_i
    g[..., 2:,   :] += 2.0 * d2     # contribution to u_{i+1}
    return g


# ---------------------------------------------------------------------------
#  Per-step, best-of-K, curvature-smoothed samplers (UniformGD / GaussianGD)
# ---------------------------------------------------------------------------

class _PerStepCurvatureGD(SamplingController):
    """Shared machinery for the knot-free samplers that draw per-step noise around
    the previous-best rollout, smooth it with a few curvature-GD steps, and keep
    the single highest-reward sample. Subclasses implement only `_sample_noise`.
    """

    def __init__(self, task, backend, *, noise_level=0.3, n_gd_iter=5, gd_lr=0.05, **kwargs):
        kwargs.pop("spline_type", None)
        super().__init__(task, backend, spline_type="zero", **kwargs)
        # per-step parameterization: one "knot" per horizon step, identity spline
        self.num_knots = self.H
        self.tk = self.t_eval.copy()
        self.mean = np.zeros((self.num_knots, self.nu), dtype=np.float64)
        self.noise_level = float(noise_level)
        self.n_gd_iter = int(n_gd_iter)
        self.gd_lr = float(gd_lr)

    def _sample_noise(self, shape) -> np.ndarray:
        raise NotImplementedError

    def sample_knots(self) -> np.ndarray:
        u_min = np.asarray(self.task.u_min, dtype=np.float64)
        u_max = np.asarray(self.task.u_max, dtype=np.float64)
        u = self.mean[None, ...] + self._sample_noise((self.num_samples, self.H, self.nu))
        np.clip(u, u_min, u_max, out=u)
        for _ in range(self.n_gd_iter):
            u -= self.gd_lr * smoothness_curv_grad(u)
            np.clip(u, u_min, u_max, out=u)
        return u                                   # (K, H, nu) == (K, num_knots, nu)

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        # Best-of-K: argmax reward (FPL) or argmin cost-convention scores (normal).
        if traj.reward is not None:
            best = int(traj.reward.argmax())
        else:
            best = int(traj.scores.argmin())
        return traj.controls[best].copy()          # (H, nu) == (num_knots, nu)


class UniformGD(_PerStepCurvatureGD):
    """Per-step Uniform(-noise_level, +noise_level) noise around the previous-best
    rollout, refined by curvature GD, best-of-K update."""

    def _sample_noise(self, shape) -> np.ndarray:
        return self.rng.uniform(-self.noise_level, self.noise_level, size=shape)


class GaussianGD(_PerStepCurvatureGD):
    """Per-step N(0, noise_level^2) noise around the previous-best rollout,
    refined by curvature GD, best-of-K update."""

    def _sample_noise(self, shape) -> np.ndarray:
        return self.rng.normal(scale=self.noise_level, size=shape)


# ---------------------------------------------------------------------------
#  RankCMA — per-step multivariate Gaussian with rank-weighted CMA refit
# ---------------------------------------------------------------------------

class RankCMA(SamplingController):
    """Per-step multivariate-Gaussian sampler with within-step rank-weighted CMA refit.

    Each MPC step (for `iterations` internal rounds):
      a. Sample (K, H, nu) controls: per step h, draw K samples from
         N(self.mean[h], self.cov[h]); clip to bounds; curvature-GD smooth.
      b. Roll out and score (FPL or normal).
      c. Rank the K samples 1 (best) .. K (worst).
      d. CMA log-rank weights: positive on the top-`num_elites`, an active-CMA
         negative tail (scaled by `alpha_neg`) on the rest.
      e. mean <- Σ w_pos · sample;  cov <- EMA blend of the rank-weighted outer
         products (only when `use_cov_update`), eigenvalue-floored at sigma_floor².

    The per-step cov is warm-shifted across MPC steps (cov[h] <- cov[h+1], last
    slot reset to sigma_init²·I), mirroring the base-class mean warm-start, so
    elite directions accumulate across the closed-loop run.
    """

    def __init__(self, task, backend, *, sigma_init=0.5, sigma_floor=0.1,
                 num_elites=None, alpha_neg=1.0,
                 n_gd_iter=5, gd_lr=0.05, use_cov_update=False,
                 cov_ema_alpha=0.1, **kwargs):
        kwargs.pop("spline_type", None)
        super().__init__(task, backend, spline_type="zero", **kwargs)
        self.num_knots = self.H
        self.tk = self.t_eval.copy()
        self.mean = np.zeros((self.num_knots, self.nu), dtype=np.float64)
        self.sigma_init = float(sigma_init)
        self.sigma_floor = float(sigma_floor)
        self.num_elites = (int(num_elites) if num_elites is not None
                           else max(1, self.num_samples // 2))
        # alpha_neg = 0 disables the active-CMA negative tail; 1 makes the negative
        # contributions sum to -1 in magnitude.
        self.alpha_neg = float(alpha_neg)
        # When False, cov stays isotropic at sigma_init²·I and only the mean update
        # propagates elite info. When True, the rank-mu (+ active tail) cov fit is
        # EMA-blended into self.cov and eigenvalue-floored.
        self.use_cov_update = bool(use_cov_update)
        self.cov_ema_alpha = float(cov_ema_alpha)
        # Curvature-GD smoothing applied to each sampled rollout before rollout.
        self.n_gd_iter = int(n_gd_iter)
        self.gd_lr = float(gd_lr)
        eye = np.eye(self.nu, dtype=np.float64)
        self.cov = np.broadcast_to(
            (self.sigma_init ** 2) * eye,
            (self.num_knots, self.nu, self.nu),
        ).copy()

    def act(self, state):
        # We intentionally do NOT reset cov each MPC step: cov learned from this
        # step's elites carries into the next (after the dt warm-shift below).
        u0 = super().act(state)
        self._shift_cov(self.backend.dt)
        return u0

    def _shift_cov(self, dt):
        """Warm-shift self.cov forward by one MPC step (cov[h] <- cov[h+1]; last
        slot reset to sigma_init²·I), matching the per-step mean warm-start."""
        if self.num_knots <= 1:
            return
        new_cov = np.empty_like(self.cov)
        new_cov[:-1] = self.cov[1:]
        eye = np.eye(self.nu, dtype=np.float64)
        new_cov[-1] = (self.sigma_init ** 2) * eye
        self.cov = new_cov

    def sample_knots(self):
        K, H = self.num_samples, self.num_knots
        out = np.empty((K, H, self.nu), dtype=np.float64)
        zero = np.zeros(self.nu)
        for h in range(H):
            out[:, h, :] = self.rng.multivariate_normal(zero, self.cov[h], size=K)
        out += self.mean[None, ...]
        u_min = np.asarray(self.task.u_min, dtype=np.float64)
        u_max = np.asarray(self.task.u_max, dtype=np.float64)
        np.clip(out, u_min, u_max, out=out)
        # Curvature-GD smoothing so the cov is fit from smoothness-respecting rollouts.
        for _ in range(self.n_gd_iter):
            out -= self.gd_lr * smoothness_curv_grad(out)
            np.clip(out, u_min, u_max, out=out)
        return out

    def update_mean(self, traj: Trajectory):
        # Rank samples 1 (best) .. K (worst).
        if traj.reward is not None:
            order = np.argsort(-traj.reward)        # FPL: higher reward = better
        else:
            order = np.argsort(traj.scores)         # normal: lower cost = better

        K, mu = self.num_samples, self.num_elites
        alpha = self.alpha_neg

        # Positive weights for ranks 1..mu: log(mu + 0.5) - log(i), strictly > 0.
        pos_ranks = np.arange(1, mu + 1, dtype=np.float64)
        w_pos = np.log(mu + 0.5) - np.log(pos_ranks)
        w_pos /= w_pos.sum()                         # sum(w_pos) = 1

        # Negative (active-CMA) tail for ranks mu+1..K; sum(w_neg) = -alpha_neg.
        n_neg = K - mu
        if n_neg > 0 and alpha > 0.0:
            j = np.arange(1, n_neg + 1, dtype=np.float64)
            w_neg_mag = np.log(n_neg + 0.5) - np.log(n_neg + 1 - j)
            w_neg = -alpha * w_neg_mag / w_neg_mag.sum()
        else:
            w_neg = np.zeros(max(n_neg, 0), dtype=np.float64)

        # Scatter into K-length vectors in sample order. Mean uses positives only;
        # cov uses positives + negatives.
        w_mean = np.zeros(K, dtype=np.float64)
        w_cov = np.zeros(K, dtype=np.float64)
        w_mean[order[:mu]] = w_pos
        w_cov[order[:mu]] = w_pos
        if n_neg > 0:
            w_cov[order[mu:]] = w_neg

        new_mean = np.einsum("k,khj->hj", w_mean, traj.knots)

        if self.use_cov_update:
            delta = traj.knots - new_mean[None, ...]
            new_sample_cov = np.einsum("k,khi,khj->hij", w_cov, delta, delta)
            blended = (1.0 - self.cov_ema_alpha) * self.cov + self.cov_ema_alpha * new_sample_cov
            self.cov = _clamp_eigenvalues(blended, self.sigma_floor ** 2)
        return new_mean


# ---------------------------------------------------------------------------
#  FplValueCMA — RankCMA log-rank prior × FPL value-softmax bonus (FPL only)
# ---------------------------------------------------------------------------

class FplValueCMA(RankCMA):
    """RankCMA's asymmetric-commitment log-rank weighting combined multiplicatively
    with an FPL-value-softmax bonus on the top-`num_elites` cohort:

        w_k ∝ [log(mu + 0.5) - log(rank(k))] · exp(beta · r_k)     (r_k in [0, 1])

    The rank prior breaks the symmetry of the iid sample cloud (so the mean keeps
    making directional progress even when rewards are clustered); the value bonus
    uses the absolute FPL scale so one clearly-better elite can dominate. Requires
    FPL mode — normal cost has no fixed reward scale for the value term.

      beta = 0        -> pure RankCMA log-rank weights
      beta -> inf     -> one-hot on argmax_k r_k (greedy on absolute value)
    """

    def __init__(self, *args, beta=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.use_fpl_cost or self.use_fpl_discounted or self.use_fpl_layered):
            raise ValueError(
                "FplValueCMA requires FPL mode (use_fpl_cost / use_fpl_discounted / "
                "use_fpl_layered). Normal-cost mode has no fixed reward scale, so "
                "absolute-value weighting is undefined there -- use RankCMA / MppiCma."
            )
        self.beta = float(beta)

    def update_mean(self, traj: Trajectory):
        if traj.reward is None:
            raise RuntimeError("FplValueCMA: traj.reward is None; FPL scoring did not populate it.")
        r = traj.reward                                       # (K,) in [0, 1]
        K, mu = self.num_samples, self.num_elites

        # Stage 1: top-mu truncation by FPL reward.
        order = np.argsort(-r)                                # best -> worst
        elite_idx = order[:mu]
        r_elites = r[elite_idx]                               # (mu,)

        # Stage 2a: log-rank prior (asymmetric commitment).
        pos_ranks = np.arange(1, mu + 1, dtype=np.float64)
        rank_raw = np.log(mu + 0.5) - np.log(pos_ranks)       # (mu,), strictly > 0
        log_rank = np.log(rank_raw)

        # Stage 2b: FPL value bonus on the absolute [0, 1] scale.
        log_val = self.beta * r_elites

        # Stage 2c: combine in log-space and softmax.
        log_w = log_rank + log_val
        log_w = log_w - log_w.max()                           # numerical stability
        w_elites = np.exp(log_w)
        s = w_elites.sum()
        if s <= 0.0 or not np.isfinite(s):
            best = int(np.argmax(r))
            return traj.knots[best].copy()
        w_elites = w_elites / s                               # sums to 1

        w = np.zeros(K, dtype=np.float64)
        w[elite_idx] = w_elites
        new_mean = np.einsum("k,khj->hj", w, traj.knots)

        if self.use_cov_update:
            delta = traj.knots - new_mean[None, ...]          # (K, H, nu)
            new_sample_cov = np.einsum("k,khi,khj->hij", w, delta, delta)
            blended = (1.0 - self.cov_ema_alpha) * self.cov + self.cov_ema_alpha * new_sample_cov
            self.cov = _clamp_eigenvalues(blended, self.sigma_floor ** 2)
        return new_mean


# ---------------------------------------------------------------------------
#  FplGmmSampler — FPL-weighted multi-mode Gaussian / evolutionary resampler
# ---------------------------------------------------------------------------

class FplGmmSampler(SamplingController):
    """FPL-driven sampler that treats the population of rollouts as a Gaussian
    mixture over the knot-spline control: each rollout k is a mode
    N(center_k, sigma_k^2 I) with mixture weight pi_k = softmax(beta * f_k)
    (weight PROPORTIONAL to FPL) and a per-mode std sigma_k that DECREASES with
    FPL (good mode -> tight/exploit, bad mode -> wide/explore). A single
    commitment scalar c in [0,1], read off the ABSOLUTE [0,1] FPL scale, drives
    BOTH the selection sharpness beta and the global exploration scale, so the
    sampler is soft/wide when every rollout is bad (explore) and sharp/collapsed
    once a high-FPL rollout appears (exploit).

    Two ways to turn the mixture weights into the next N samples (``allocation``):
      'sus'     -> Stochastic Universal Sampling integer offspring (evolutionary
                   resampler; minimum-variance, deterministic).
      'mixture' -> honest multinomial categorical draw from the mixture.
    With ``width_mode='afper'`` (default) the two are equal in distribution, so
    'sus' is just the low-variance realization of the same GMM.

    Sampling happens in KNOT space (default ``num_knots`` coarse knots), so
    smoothness comes from the spline interpolation, not from post-hoc smoothing
    — this keeps the search low-dimensional and lets the plan use energy-pumping
    controls (e.g. pendulum swing-up) that a curvature penalty would suppress.
    The population is carried across MPC steps and warm-shifted with the same
    receding-horizon rule the base class applies to the mean; n_elite best
    rollouts are copied verbatim so the best plan is never lost.

    Requires FPL mode (use_fpl_cost or use_fpl_discounted): the absolute [0,1]
    reward scale is what drives the explore<->exploit adaptation, exactly like
    FplValueCMA. Prototyped in verification/fpl_based_sampling.ipynb.
    """

    def __init__(self, task, backend, *,
                 allocation="mixture", width_mode="afper",
                 sigma_min=0.1, sigma_max=1.0, kappa=2.0, eps_floor=0.05,
                 tau_hi=1.0, tau_lo=0.05, w_abs=0.7, g_ref=0.5,
                 n_elite=1, **kwargs):
        super().__init__(task, backend, **kwargs)
        if not (self.use_fpl_cost or self.use_fpl_discounted):
            raise ValueError(
                "FplGmmSampler requires FPL mode (use_fpl_cost or use_fpl_discounted); "
                "its explore<->exploit adaptation is driven by the absolute [0,1] reward "
                "scale. For normal cost use CEM / RankCMA / MppiCma instead."
            )
        if allocation not in ("sus", "mixture"):
            raise ValueError(f"allocation must be 'sus' or 'mixture', got {allocation!r}")
        if width_mode not in ("afper", "convex"):
            raise ValueError(f"width_mode must be 'afper' or 'convex', got {width_mode!r}")
        self.allocation = allocation
        self.width_mode = width_mode
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.kappa = float(kappa)
        self.eps_floor = float(eps_floor)
        self.tau_hi = float(tau_hi)
        self.tau_lo = float(tau_lo)
        self.w_abs = float(w_abs)
        self.g_ref = float(g_ref)
        self.n_elite = int(n_elite)
        self.population = None       # (K, num_knots, nu) — carried across act() steps
        self._pop_shift_accum = 0.0

    # ---- lifecycle ----

    def reset(self):
        super().reset()
        self.population = None
        self._pop_shift_accum = 0.0

    def act(self, state):
        u0 = super().act(state)
        self._shift_population(self.backend.dt)   # warm-start the population like the mean
        return u0

    def _shift_population(self, dt):
        """Advance the population one MPC step in time, mirroring `_shift_mean`:
        zero-order-hold rolls whole knots once a knot spacing has elapsed;
        continuous splines resample at knot times advanced by dt."""
        if self.population is None or self.num_knots <= 1:
            return
        if self.spline_type == "zero":
            spacing = float(self.tk[1] - self.tk[0])
            self._pop_shift_accum += dt
            n_roll = int((self._pop_shift_accum + 1e-9) // spacing)
            if n_roll > 0:
                self._pop_shift_accum -= n_roll * spacing
                n_roll = min(n_roll, self.num_knots - 1)
                tail = np.repeat(self.population[:, -1:, :], n_roll, axis=1)
                self.population = np.concatenate([self.population[:, n_roll:, :], tail], axis=1)
            return
        self.population = interpolate(self.population, self.tk, self.tk + dt, self.spline_type)

    # ---- FPL frontend (mirrors verification/fpl_based_sampling.ipynb) ----

    def _frontend(self, f):
        """Absolute-scale FPL frontend -> (commitment c, inverse-temp beta,
        mixture weights pi, per-mode std sigma). f is (K,) in [0,1], larger better."""
        f_best, f_mean = float(f.max()), float(f.mean())
        gap = f_best - f_mean
        c = float(np.clip(self.w_abs * f_best
                          + (1.0 - self.w_abs) * np.clip(gap / self.g_ref, 0.0, 1.0), 0.0, 1.0))
        beta = 1.0 / (self.tau_hi * (self.tau_lo / self.tau_hi) ** c)
        z = beta * f
        z = z - z.max()
        w = np.exp(z)
        s = w.sum()
        pi = w / s if (s > 0 and np.isfinite(s)) else np.full(f.shape[0], 1.0 / f.shape[0])
        sg = self.sigma_min + (self.sigma_max - self.sigma_min) * (1.0 - c) ** self.kappa
        if self.width_mode == "afper":
            sig = sg * (self.eps_floor + (1.0 - self.eps_floor) * (1.0 - f))
        else:  # convex
            sig = (1.0 - c) * (self.sigma_min
                               + (self.sigma_max - self.sigma_min) * (1.0 - f) ** self.kappa)
        sig = np.maximum(sig, self.eps_floor * self.sigma_min)     # sigma floor
        return c, beta, pi, sig

    # ---- SamplingController hooks ----

    def sample_knots(self):
        K = self.num_samples
        if self.population is None or self.population.shape != (K, self.num_knots, self.nu):
            # First batch (or after reset): jitter the warm-started mean broadly.
            pop = self.mean[None, ...] + self.sigma_max * self.rng.standard_normal(
                (K, self.num_knots, self.nu))
            np.clip(pop, np.asarray(self.task.u_min), np.asarray(self.task.u_max), out=pop)
            self.population = pop
        return self.population

    def update_mean(self, traj: Trajectory):
        if traj.reward is None:
            raise RuntimeError("FplGmmSampler: traj.reward is None; FPL scoring did not populate it.")
        f = traj.reward                        # (K,) in [0,1], larger is better
        centers = traj.knots                   # (K, num_knots, nu) — the evaluated population
        K = self.num_samples
        M = centers.shape[0]
        c, beta, pi, sig = self._frontend(f)

        n_elite = min(self.n_elite, K - 1)
        elite = np.argsort(-f)[:n_elite]
        R = K - n_elite
        if self.allocation == "sus":
            cdf = np.cumsum(pi)
            u0 = self.rng.uniform(0.0, 1.0 / R)
            ptr = u0 + np.arange(R) / R
            counts = np.bincount(np.clip(np.searchsorted(cdf, ptr, side="right"), 0, M - 1),
                                 minlength=M)
        else:
            counts = self.rng.multinomial(R, pi)

        u_min = np.asarray(self.task.u_min, dtype=np.float64)
        u_max = np.asarray(self.task.u_max, dtype=np.float64)
        kid_list = [
            centers[i][None] + sig[i] * self.rng.standard_normal((int(counts[i]), self.num_knots, self.nu))
            for i in range(M) if counts[i] > 0
        ]
        if kid_list:
            kids = np.clip(np.concatenate(kid_list, axis=0), u_min, u_max)
            self.population = np.concatenate([centers[elite], kids], axis=0)
        else:
            self.population = centers.copy()
        # Commanded plan = the best-reward rollout; act() applies its knot 0.
        return centers[int(f.argmax())].copy()


# name -> class, mirroring controllers.SAMPLING_CONTROLLERS. The `_cov` behaviour
# (RankCMA / FplValueCMA with use_cov_update=True) is a kwarg, not a separate class.
EXPERIMENTAL_CONTROLLERS = {
    "uniform_gd": UniformGD,
    "gaussian_gd": GaussianGD,
    "rank_cma": RankCMA,
    "fpl_value_cma": FplValueCMA,
    "fpl_gmm": FplGmmSampler,
}

__all__ = [
    "smoothness_curv_grad",
    "UniformGD", "GaussianGD", "RankCMA", "FplValueCMA", "FplGmmSampler",
    "EXPERIMENTAL_CONTROLLERS",
]
