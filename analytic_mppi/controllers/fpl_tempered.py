"""FPL-calibrated importance WEIGHTING — the second FPL-native layer (not sampling).

The sampler race showed the PROPOSAL stage doesn't benefit from FPL. This targets the other half
of MPPI-as-inference — the importance WEIGHTS / temperature — which the literature keeps trying to
improve (adaptive importance sampling, "adaptive cooling via entropy feedback", Stein-guided
reweighting). Those methods are all RELATIVE (feedback on the weight entropy / ESS / cost spread):
they have no absolute reference for "is this rollout actually good", so the temperature λ still
needs per-task tuning. FPL supplies exactly that missing reference — the reward u_k ∈ (0,1] is a
CALIBRATED fulfillment (1 = every objective satisfied) — so the weights can be set on an ABSOLUTE
scale that transfers across robots.

Everything upstream is unchanged (same Gaussian MPPI proposal, same FPL cost). Only the map from
the scored rollouts to the softmax weights changes:

  relative        (baseline)  w_k ∝ (u_k / max_j u_j)^{1/λ}     — the standard MPPI weighting; the
                              sharpness depends on the SPREAD of u, which differs per task, so λ
                              must be retuned per robot. (Weighting on raw absolute regret 1-u is
                              degenerate here — FPL composites cluster in a narrow band, which is
                              exactly why the framework spreads them with -log; so the only
                              non-relative levers that carry information are the two below.)
  adaptive_ess                pick λ each step (bisection) so the weight ESS hits a target
                              FRACTION of K. With fpl_calibrated=True the target fraction is
                              MODULATED by the absolute best fulfillment u_best: commit (low ESS)
                              when a genuinely good plan exists, hedge (high ESS) when nothing is
                              good yet — the calibration only a bounded reward makes well-posed.
  feasibility_gate            zero-weight any rollout whose composite fulfillment is below an
                              ABSOLUTE floor τ (drop the unsafe ones outright), then temper the
                              survivors. A scalar cost has no transferable τ; FPL does.

The claim to test (verification/fpl_weighting_study.py): with the absolute-scale weightings a
SINGLE global setting transfers across robots and matches the best per-robot-tuned relative λ —
removing the temperature knob — and the feasibility gate improves survival under mismatch.
"""
from __future__ import annotations

import numpy as np

from .mppi_v2 import MPPIv2
from .sampling_base import Trajectory


def _weights_ess(g: np.ndarray, lam: float) -> np.ndarray:
    """Softmax weights on regret g>=0 at temperature lam (min-shifted); returns normalized w."""
    z = -(g - g.min()) / max(lam, 1e-9)
    w = np.exp(z)
    s = w.sum()
    return w / s if s > 0 and np.isfinite(s) else np.full_like(g, 1.0 / g.size)


def _ess(w: np.ndarray) -> float:
    return float(1.0 / np.sum(w ** 2))


def _solve_lambda_for_ess(g: np.ndarray, target_ess: float,
                          lo: float = 1e-4, hi: float = 100.0, iters: int = 40) -> float:
    """Bisection for the λ whose weight ESS equals target_ess. ESS(λ) is monotone increasing
    (λ→0 ⇒ argmax ⇒ ESS→#best; λ→∞ ⇒ uniform ⇒ ESS→K), so a clean bracket-and-bisect works."""
    if _ess(_weights_ess(g, lo)) >= target_ess:
        return lo
    if _ess(_weights_ess(g, hi)) <= target_ess:
        return hi
    for _ in range(iters):
        mid = np.sqrt(lo * hi)              # geometric bisection (λ spans orders of magnitude)
        if _ess(_weights_ess(g, mid)) < target_ess:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


class FplTemperedMPPI(MPPIv2):
    def __init__(
        self,
        task,
        backend,
        *,
        weight_mode: str = "adaptive_ess",       # relative | adaptive_ess | feasibility_gate
        ess_frac_lo: float = 0.05,              # target ESS fraction when u_best is high (commit)
        ess_frac_hi: float = 0.5,               # target ESS fraction when u_best is low (hedge)
        fpl_calibrated: bool = True,            # modulate target ESS by absolute u_best
        feas_floor: float = 0.0,                # absolute fulfillment floor τ (feasibility_gate)
        **kwargs,
    ):
        super().__init__(task, backend, **kwargs)
        if weight_mode not in ("relative", "adaptive_ess", "feasibility_gate"):
            raise ValueError(f"bad weight_mode {weight_mode!r}")
        self.weight_mode = weight_mode
        self.ess_frac_lo = float(ess_frac_lo)
        self.ess_frac_hi = float(ess_frac_hi)
        self.fpl_calibrated = bool(fpl_calibrated)
        self.feas_floor = float(feas_floor)
        self.last_lambda: float | None = None

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        u = traj.reward
        if u is None or self.weight_mode == "relative":
            # Fall back to the inherited MPPI softmax (the relative baseline).
            return super().update_mean(traj)

        K = u.shape[0]
        u = np.clip(u, 1e-8, 1.0)
        u_best = float(u.max())
        # Spread scale: -log(u) is what the framework already uses to expand the narrow FPL
        # composite band into a usable dynamic range; regret on it (g = log u_best - log u_k)
        # is what both non-relative modes temper / gate on.
        g = np.log(u_best) - np.log(u)                   # >=0, log-fulfillment regret to the best

        if self.weight_mode == "adaptive_ess":
            # Target ESS fraction; if FPL-calibrated, interpolate by the ABSOLUTE best fulfillment
            # (u_best≈1 ⇒ commit to the low fraction; u_best low ⇒ hedge to the high fraction).
            if self.fpl_calibrated:
                frac = self.ess_frac_lo + (self.ess_frac_hi - self.ess_frac_lo) * (1.0 - u_best)
            else:
                frac = 0.5 * (self.ess_frac_lo + self.ess_frac_hi)
            lam = _solve_lambda_for_ess(g, max(2.0, frac * K))
            w = _weights_ess(g, lam)
            self.last_lambda = lam

        else:  # feasibility_gate: drop rollouts below the absolute floor τ, temper the survivors
            keep = u >= self.feas_floor
            if not keep.any():
                keep = u >= np.quantile(u, 0.9)          # nothing feasible: keep the top decile
            lam = self.temperature
            w = np.zeros(K)
            w[keep] = _weights_ess(g[keep], lam)
            w = w / w.sum()

        self.last_ess = _ess(w)
        return np.einsum("k,khj->hj", w, traj.knots)


__all__ = ["FplTemperedMPPI"]
