"""Multi-objective gradient composition MPPI (FPL-only).

Ordinary FPL-MPPI collapses the per-objective fulfillment vector into ONE scalar
(`power_mean(reward_terms, fpl_p)`) and then weights whole samples by that composite.
When objectives genuinely compete (g1 standup: get tall vs. stay upright vs. hold
posture), no single sample is good at everything, so the composite weighting goes mushy
— effective sample size collapses and the update averages toward nothing.

This controller instead keeps the per-objective values `V = traj.reward_terms` (K, J)
and forms a SEPARATE evolution-strategies (score-function) gradient per objective, then
COMPOSES the directions with weights that favour the currently worst-satisfied
objective.

Per iteration, with V[:, j] the [0,1] fulfillment of objective j across the K samples:

  1. per-objective sample weights   W[:, j]  (each column ≥0, sums to 1)
  2. per-objective weighted-mean    M[j] = Σ_k W[k,j] · knots_k           (a convex combo)
     — equivalently the ES gradient  g_j = M[j] − mean.
  3. satisfaction                    s_j = mean_k V[k, j]                   (current value)
  4. composition weights             α_j  (≥0, sums to 1) from s
  5. update                          new_mean = Σ_j α_j · M[j]

Because Σ_j α_j = 1 and every M[j] is a convex combination of the sampled knots,
`new_mean` is itself a convex combination of the knots — it stays inside the sample hull,
so it inherits MPPI's stability with no step size and no clipping. (Displacement /
null-space "constrained descent" forms would leave the hull and need a step size; those
are a deliberate future extension, not part of this MVP.)

Why the default `alpha_mode="power_p"` is principled
----------------------------------------------------
For the scalar reward R = power_mean(s, p), ∂R/∂s_j ∝ s_j^{p-1}. Weighting the
per-objective gradients by α_j ∝ s_j^{p-1} therefore makes  Σ_j α_j g_j  a sampled
estimate of ∇_μ R — i.e. this controller is doing natural-gradient ascent on the EXACT
scalar FPL objective the rest of the framework already optimizes, only decomposed per
objective so the direction stays sharp under competition. For p < 1 (incl. the default
fpl_p = 0.1) the exponent p-1 < 0, so s_j^{p-1} is decreasing in s_j — the worst-satisfied
objective gets the most pull, exactly the desired "improve the worst objective" behaviour.
Its one hazard is that s_j^{p-1} → ∞ as s_j → 0; `s_floor` caps that. Note the residual
edge case: an objective that is hopeless (≈0 for every sample) gets a large α but a
non-informative M[j] ≈ the population mean, so it pulls the step toward "stay near the
current plan" rather than anywhere useful — visible, and intended, behaviour to inspect
in experiments.

Per-objective weighting must be SHARP (default `obj_weighting="softmax"`)
----------------------------------------------------------------------
Each objective's sample weights use w_kj ∝ v_kj^(1/temperature) — the per-objective
analogue of MPPI's composite w_k ∝ reward_k^(1/temperature), reusing the same
`temperature`. This matters: `obj_weighting="proportional"` (w_kj = v_kj/Σ_k v_kj) is
near-uniform whenever an objective is well-satisfied by every sample (all v≈const → no
gradient), which on an unstable system (g1 balance) fails to correct and topples. The
sharp softmax weighting actively stabilises and, on g1 standup, holds the worst group's
fulfillment and the orientation floor well ABOVE the scalar-composite baseline at a fixed
budget. `proportional`/`exp` are kept for ablations.

Requires an FPL cost mode that exposes a per-objective vector: `use_fpl_discounted`
(atoms) or `use_fpl_layered` (groups). `use_fpl_cost` collapses objectives per-step
before any vector exists; `normal`/`hybrid` have no [0,1] atom vector — all rejected.
"""
from __future__ import annotations

import numpy as np

from .mppi_v2 import MPPIv2
from .sampling_base import Trajectory


class ComposedGradientMPPI(MPPIv2):
    def __init__(
        self,
        task,
        backend,
        *,
        compose: str = "worst_first",       # "worst_first" | "uniform"
        alpha_mode: str = "power_p",        # "power_p" | "softmax"  (worst_first only)
        obj_weighting: str = "softmax",     # "softmax" | "proportional" | "exp"
        compose_temp: float = 0.1,          # β for alpha_mode="softmax"
        s_floor: float = 1e-2,              # floor on s_j (guards power_p blow-up)
        s_stat: str = "mean",               # "mean" | "best" satisfaction estimate
        obj_lambda: float = 0.1,            # λ for obj_weighting="exp"
        **kwargs,
    ):
        # Forwards noise_level / temperature / use_fpl_* / fpl_p / ... to MPPIv2 + base.
        # We inherit MPPIv2.sample_knots (shared Gaussian cloud): every objective is scored
        # on the SAME samples, so the per-objective W[:, j] are just different reweightings
        # of one population (a multiple-importance setup) — this is what yields a clean
        # direction per objective even when no sample is jointly good.
        super().__init__(task, backend, **kwargs)

        if not (self.use_fpl_discounted or self.use_fpl_layered):
            raise ValueError(
                "ComposedGradientMPPI needs a per-objective FPL vector: set cost_mode to "
                "'fpl_discounted' (atoms) or 'fpl_layered' (groups). 'fpl_cost' collapses "
                "objectives per-step; 'normal'/'hybrid' have no [0,1] atom vector."
            )
        if compose not in ("worst_first", "uniform"):
            raise ValueError(f"compose must be 'worst_first' or 'uniform', got {compose!r}")
        if alpha_mode not in ("power_p", "softmax"):
            raise ValueError(f"alpha_mode must be 'power_p' or 'softmax', got {alpha_mode!r}")
        if obj_weighting not in ("softmax", "proportional", "exp"):
            raise ValueError(
                f"obj_weighting must be 'softmax', 'proportional' or 'exp', "
                f"got {obj_weighting!r}"
            )
        if s_stat not in ("mean", "best"):
            raise ValueError(f"s_stat must be 'mean' or 'best', got {s_stat!r}")

        self.compose = compose
        self.alpha_mode = alpha_mode
        self.obj_weighting = obj_weighting
        self.compose_temp = float(compose_temp)
        self.s_floor = float(s_floor)
        self.s_stat = s_stat
        self.obj_lambda = float(obj_lambda)

        # Per-objective diagnostics (length-J vectors), refreshed each update_mean.
        # last_ess (effective combined ESS) is inherited from MPPIv2.
        self.last_alpha: np.ndarray | None = None
        self.last_obj_ess: np.ndarray | None = None
        self.last_obj_satisfaction: np.ndarray | None = None

    # ---- weighting helpers ----

    def _per_objective_weights(self, V: np.ndarray) -> np.ndarray:
        """V (K, J) in [0,1] -> W (K, J), each column ≥0 summing to 1."""
        K = V.shape[0]
        if self.obj_weighting == "softmax":
            # w_kj ∝ v_kj^(1/temperature): the per-objective analogue of MPPI's composite
            # weighting w_k ∝ reward_k^(1/temperature) (same `temperature` knob, so the
            # baseline comparison stays apples-to-apples). This is sharp enough to actively
            # drive an unstable system; proportional weighting on well-satisfied [0,1]
            # values is near-uniform (mushy) and fails to stabilise — see obj_weighting docs.
            logv = np.log(np.clip(V, 1e-12, None))
            z = (logv - logv.max(axis=0, keepdims=True)) / self.temperature
            W = np.exp(z)
            return W / W.sum(axis=0, keepdims=True)
        if self.obj_weighting == "proportional":
            col_sums = V.sum(axis=0, keepdims=True)                 # (1, J)
            safe = col_sums > 0
            # All-zero column (objective unsatisfied by every sample, e.g. hopper velocity
            # at standstill) -> uniform 1/K, a neutral no-information direction rather than
            # an argmax spike. NOTE: proportional is mushy on well-satisfied objectives
            # (all v≈const -> W≈1/K -> no gradient); prefer "softmax". Kept for ablations.
            return np.where(safe, V / np.where(safe, col_sums, 1.0), 1.0 / K)
        # exp: w ∝ exp(v / λ), max-shifted per objective for numerical stability.
        z = (V - V.max(axis=0, keepdims=True)) / self.obj_lambda
        W = np.exp(z)
        return W / W.sum(axis=0, keepdims=True)

    def _composition_weights(self, s: np.ndarray) -> np.ndarray:
        """s (J,) in [0,1] -> α (J,) ≥0 summing to 1."""
        J = s.shape[0]
        if self.compose == "uniform":
            return np.full(J, 1.0 / J)
        # compose == "worst_first"
        if self.alpha_mode == "power_p":
            # α_j ∝ clip(s_j, s_floor, 1)^(fpl_p - 1). For fpl_p < 1 this weights the
            # worst-satisfied objective most and equals ∂/∂s_j power_mean(s, fpl_p) up to
            # a common factor -> natural-gradient weights on the scalar FPL reward.
            raw = np.clip(s, self.s_floor, 1.0) ** (self.fpl_p - 1.0)
            return raw / raw.sum()
        # softmax: α = softmax(-s / β), min-shifted so the worst objective gets weight 1.
        w = np.exp(-(s - s.min()) / self.compose_temp)
        return w / w.sum()

    # ---- update ----

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        V = traj.reward_terms
        if V is None:
            raise RuntimeError(
                "ComposedGradientMPPI.update_mean requires traj.reward_terms, which is "
                "populated only by the fpl_discounted / fpl_layered scorers."
            )
        W = self._per_objective_weights(V)                          # (K, J)
        # Per-objective weighted-mean knots: M[j] = Σ_k W[k,j] · knots_k.
        M = np.einsum("kj,khn->jhn", W, traj.knots)                 # (J, num_knots, nu)
        s = V.mean(axis=0) if self.s_stat == "mean" else V.max(axis=0)   # (J,)
        alpha = self._composition_weights(s)                        # (J,)
        new_mean = np.einsum("j,jhn->hn", alpha, M)                 # (num_knots, nu)

        # Diagnostics. w_eff is the effective per-sample weight of the composed update
        # (new_mean = Σ_k w_eff[k] knots_k); it sums to 1 by construction, so last_ess is
        # directly comparable to MPPIv2's. Per-objective ESS shows how peaked each
        # objective's own reweighting is.
        self.last_obj_satisfaction = s
        self.last_alpha = alpha
        self.last_obj_ess = 1.0 / np.sum(W ** 2, axis=0)            # (J,)
        w_eff = W @ alpha                                           # (K,)
        self.last_ess = float(1.0 / np.sum(w_eff ** 2))
        return new_mean


__all__ = ["ComposedGradientMPPI"]
