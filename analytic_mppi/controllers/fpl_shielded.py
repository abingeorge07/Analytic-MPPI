"""FPL Shielded MPPI — lexicographic safety-first control the scalar cost can't express.

A scalar weighted cost J = Σ wᵢ cᵢ can NEVER encode strict priority: for any finite weights there
is always enough performance reward to justify a little less safety, so under pressure (aggressive
command, model mismatch) the optimizer trades safety away and the robot falls. FPL's per-objective
BOUNDED [0,1] fulfillment lets us do what the weighted sum cannot — a hard lexicographic split:

  1. FEASIBILITY (hard):   a rollout is admissible iff every SAFETY atom stays above an absolute
                           floor τ for the WHOLE horizon (min over time & safety atoms ≥ τ), and,
                           optionally, the TERMINAL safety fulfillment ≥ τ_T (the plan must END in
                           a safe state — a recursive-feasibility / "never commit to an
                           irrecoverable trajectory" condition). τ is a transferable ABSOLUTE
                           threshold only because fulfillment is bounded; a scalar cost has none.
  2. PERFORMANCE (soft):   among the admissible rollouts ONLY, do the ordinary MPPI softmax on the
                           PERFORMANCE atom (go as fast / rotate as far as possible).
  3. FALLBACK:             if nothing is admissible (e.g. a big disturbance), follow the SAFEST
                           rollout set (softmax on the worst-safety margin) — degrade gracefully
                           toward recovery instead of chasing performance off a cliff.

This decouples "how hard you try" from "how safe you are": you can crank the performance objective
arbitrarily and the shield still vetoes any unsafe rollout, so the (speed, survival) frontier
should move UP-and-RIGHT of the soft min-fulfillment cost, especially under mismatch. Needs an FPL
cost mode that exposes the per-atom running fulfillment (`fpl_cost` / `fpl_discounted` both
populate `traj.running_terms_f`). Safety vs performance atom indices are task-specific
(hopper/walker: safety [0,1]=height,orient, perf [2]=velocity; cube: safety [0,1]=hold,height,
perf [2]=alignment).
"""
from __future__ import annotations

import numpy as np

from .mppi_v2 import MPPIv2
from .sampling_base import Trajectory


class FplShieldedMPPI(MPPIv2):
    def __init__(
        self,
        task,
        backend,
        *,
        safety_indices,                       # list[int] into the running fulfillment atoms
        perf_indices,                         # list[int] (the progress atom(s))
        safety_floor: float = 0.3,            # absolute τ: min over-time safety fulfillment
        terminal_safety_floor: float | None = None,  # optional recursive-feasibility floor τ_T
        perf_mode: str = "balanced",          # among-feasible objective:
                                              #   "performance" = maximize the perf atom (greedy,
                                              #     boundary-seeking — the naive lexicographic form)
                                              #   "balanced"    = keep the soft min-fulfillment
                                              #     composite (traj.reward) among the safe set (hard
                                              #     safety, but no boundary-seeking)
        **kwargs,
    ):
        super().__init__(task, backend, **kwargs)
        if not (self.use_fpl_cost or self.use_fpl_discounted):
            raise ValueError("FplShieldedMPPI needs running per-atom fulfillment: use cost_mode "
                             "'fpl_cost' or 'fpl_discounted' (populates traj.running_terms_f).")
        if perf_mode not in ("performance", "balanced"):
            raise ValueError(f"perf_mode must be 'performance' or 'balanced', got {perf_mode!r}")
        self.perf_mode = perf_mode
        self.safety_indices = list(safety_indices)
        self.perf_indices = list(perf_indices)
        self.safety_floor = float(safety_floor)
        self.terminal_safety_floor = (None if terminal_safety_floor is None
                                      else float(terminal_safety_floor))
        self.last_feasible_frac: float | None = None

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        rf = traj.running_terms_f                       # (K, H, J)
        if rf is None:
            return super().update_mean(traj)
        K = rf.shape[0]

        # worst safety fulfillment across time AND safety atoms (the binding safety moment)
        safety_min = rf[..., self.safety_indices].min(axis=(1, 2))        # (K,)
        feasible = safety_min >= self.safety_floor
        # optional terminal safety constraint (recursive feasibility)
        if self.terminal_safety_floor is not None and traj.terminal_terms_f is not None:
            tf = traj.terminal_terms_f                                    # (K, n_term)
            sidx = [i for i in self.safety_indices if i < tf.shape[-1]]
            if sidx:
                tsafe = tf[..., sidx].min(axis=-1)                        # (K,)
                feasible = feasible & (tsafe >= self.terminal_safety_floor)

        # among-feasible objective: greedy performance (boundary-seeking) OR the soft composite
        # (balanced — hard safety without chasing the feasibility edge).
        if self.perf_mode == "balanced" and traj.reward is not None:
            obj = np.log(np.clip(traj.reward, 1e-8, 1.0))               # soft min-fulfillment (log)
        else:
            obj = rf[..., self.perf_indices].mean(axis=(1, 2))          # perf atom time-mean
        self.last_feasible_frac = float(feasible.mean())

        temp = self.temperature
        if feasible.any():
            # softmax on the objective among the admissible rollouts ONLY.
            z = np.full(K, -np.inf)
            of = obj[feasible]
            z[feasible] = (of - of.max()) / temp
            w = np.exp(z)
        else:
            # nothing admissible: follow the SAFEST rollouts (softmax on the safety margin).
            w = np.exp((safety_min - safety_min.max()) / temp)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            best = int(np.argmax(perf if feasible.any() else safety_min))
            self.last_ess = 1.0
            return traj.knots[best].copy()
        w = w / s
        self.last_ess = float(1.0 / np.sum(w ** 2))
        return np.einsum("k,khj->hj", w, traj.knots)


__all__ = ["FplShieldedMPPI"]
