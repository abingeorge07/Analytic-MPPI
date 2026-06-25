"""Sampling-based MPC controller base class (CPU, knot-spline parameterization).

All sampling MPC algorithms ported from the JAX `mis/algs` tree (MPPI, MPPI-CMA,
CEM, DIAL, Predictive Sampling) share this skeleton:

    on each act(state):
      for _ in range(iterations):
        knots    = self.sample_knots()              # (K, num_knots, nu)
        knots    = clip(knots, u_min, u_max)
        controls = interp(knots, tk, t_eval)        # (K, H, nu)
        states, sensordata = backend.rollout(state, controls)
        scores = self.score_rollouts(states, sensordata, controls)
        self.mean = self.update_mean(knots, scores, ...)
      action = self.mean[0]                         # spline value at t=0
      self._shift_mean(dt)                          # warm-start
      return action

`scores` is in cost convention: SMALLER IS BETTER for both normal and FPL modes.
FPL aggregations negate the per-rollout reward so subclasses don't need to
branch on FPL inside their update_mean.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from analytic_mppi.tasks.base import Task, power_mean
from .spline import interpolate, make_eval_times, make_knot_times


@dataclass
class Trajectory:
    """Bundle of arrays carried through one act() iteration."""
    knots: np.ndarray            # (K, num_knots, nu)
    controls: np.ndarray         # (K, H, nu)
    states: np.ndarray           # (K, H, nstate)
    sensordata: np.ndarray       # (K, H, nsensordata)
    qpos: np.ndarray             # (K, H, nq)
    qvel: np.ndarray             # (K, H, nv)
    running_terms: Optional[np.ndarray] = None    # (K, H, n_terms) — normal terms
    terminal_terms: Optional[np.ndarray] = None   # (K, n_terms)
    running_terms_f: Optional[np.ndarray] = None  # (K, H, n_terms_f) — fulfillment in [0,1]
    terminal_terms_f: Optional[np.ndarray] = None # (K, n_terms_f)
    scores: Optional[np.ndarray] = None           # (K,) — smaller is better (cost convention)
    reward: Optional[np.ndarray] = None           # (K,) — populated by _score_fpl only;
                                                  # positive in [0, 1], LARGER is better.


class SamplingController:
    """Base class shared by every sampling-MPC algorithm in this framework."""

    def __init__(
        self,
        task: Task,
        backend,
        *,
        num_samples: int,
        num_knots: int = 4,
        plan_horizon: float = 1.0,
        spline_type: str = "zero",
        iterations: int = 1,
        seed: int = 0,
        # FPL settings
        use_fpl_cost: bool = False,
        use_fpl_discounted: bool = False,
        fpl_p: float = 0.1,
        fpl_gamma: float = 0.99,
    ):
        if use_fpl_cost and use_fpl_discounted:
            raise ValueError("use_fpl_cost and use_fpl_discounted are mutually exclusive")
        # γ must be in [0, 1); γ=1 makes the finite-horizon normaliser (1-γ)/(1-γ^H) = 0/0.
        assert 0.0 <= fpl_gamma < 1.0, f"fpl_gamma must be in [0, 1), got {fpl_gamma}"
        self.task = task
        self.backend = backend
        self.num_samples = int(num_samples)
        self.num_knots = int(num_knots)
        self.plan_horizon = float(plan_horizon)
        self.spline_type = spline_type
        self.iterations = int(iterations)
        self.use_fpl_cost = bool(use_fpl_cost)
        self.use_fpl_discounted = bool(use_fpl_discounted)
        self.fpl_p = float(fpl_p)
        self.fpl_gamma = float(fpl_gamma)
        self.nu = int(task.nu)

        self.tk = make_knot_times(self.plan_horizon, self.num_knots)
        self.t_eval, self.H = make_eval_times(self.plan_horizon, float(backend.dt))

        self.rng = np.random.default_rng(int(seed))
        self.mean = np.zeros((self.num_knots, self.nu), dtype=np.float64)
        self.last_trajectory: Optional[Trajectory] = None
        # Accumulated, not-yet-applied warm-start shift time (seconds). Only the
        # zero-order-hold path uses it — see _shift_mean.
        self._shift_accum = 0.0

    # ---- subclass extension points ----

    def reset(self):
        self.mean[:] = 0.0
        self._shift_accum = 0.0

    def sample_knots(self) -> np.ndarray:
        """Return (K, num_knots, nu) candidate knot sequences.

        Default: Gaussian perturbation of `self.mean` with `self.noise_level`.
        Subclasses with different sampling distributions (CMA, CEM, ...) override.
        """
        raise NotImplementedError

    def update_mean(self, traj: Trajectory) -> np.ndarray:
        """Return the new (num_knots, nu) mean given the scored trajectory."""
        raise NotImplementedError

    # ---- main MPC step ----

    def act(self, state: np.ndarray) -> np.ndarray:
        """Run one MPC iteration; return the action to apply at the current step."""
        for _ in range(self.iterations):
            traj = self._rollout_and_score(state)
            self.mean = self.update_mean(traj)
            self.last_trajectory = traj

        # action at t=0 (knot 0 is always at t=0)
        u0 = self.mean[0].copy()
        # warm-start shift: re-interp mean onto knot-times advanced by dt
        self._shift_mean(self.backend.dt)
        return u0

    # ---- internals ----

    def _rollout_and_score(self, state: np.ndarray) -> Trajectory:
        K, H = self.num_samples, self.H
        knots = self.sample_knots()                                          # (K, num_knots, nu)
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        controls = interpolate(knots, self.tk, self.t_eval, self.spline_type)  # (K, H, nu)

        initial = np.broadcast_to(state, (K, self.backend.nstate)).copy()
        states, sensordata = self.backend.rollout(initial, controls)         # (K, H, *)
        qpos = self.task.qpos_of(states)
        qvel = self.task.qvel_of(states)

        traj = Trajectory(
            knots=knots, controls=controls, states=states, sensordata=sensordata,
            qpos=qpos, qvel=qvel,
        )

        if self.use_fpl_cost or self.use_fpl_discounted:
            traj.running_terms_f = self.task.running_cost_terms_f(qpos, qvel, sensordata, controls)
            traj.terminal_terms_f = self.task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
            traj.scores = self._score_fpl(traj)
        else:
            traj.running_terms = self.task.running_cost_terms(qpos, qvel, sensordata, controls)
            traj.terminal_terms = self.task.terminal_cost_terms(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
            traj.scores = self._score_normal(traj)

        return traj

    # ---- scoring (returns (K,) costs; SMALLER IS BETTER) ----

    def _score_normal(self, traj: Trajectory) -> np.ndarray:
        # Running cost terms are rates (cost-per-unit-time evaluated at each
        # step), so the total accumulated cost over the horizon is the time
        # integral  ∫ ℓ dt  ≈  Σ ℓ_i · dt. Multiplying by dt matches hydrax's
        # per-step scaling in alg_base.py:312-313 and keeps the running vs.
        # terminal balance consistent across different timestep sizes.
        # Terminal cost is a one-shot penalty (not a rate) and stays unscaled.
        dt = float(self.backend.dt)
        running  = traj.running_terms.sum(axis=(1, 2)) * dt   # (K,)
        terminal = traj.terminal_terms.sum(axis=-1)           # (K,)
        return running + terminal

    def _score_fpl(self, traj: Trajectory) -> np.ndarray:
        # Convention: `terminal_terms_f` may return FEWER terms than
        # `running_terms_f`. The i-th terminal term corresponds to the i-th
        # running term; trailing running terms (i >= n_term) have no terminal
        # contribution and span only H steps. This avoids constant-1
        # placeholders for terms like control-fulfillment that are undefined
        # at the terminal step. When n_term == n_run (e.g., walker, hopper)
        # this collapses to the original behaviour (one (H+1)-step series).
        gamma = self.fpl_gamma
        running  = traj.running_terms_f    # (K, H, n_run)
        terminal = traj.terminal_terms_f   # (K, n_term)  with n_term <= n_run
        H, n_run = running.shape[1], running.shape[-1]
        n_term = terminal.shape[-1]
        if n_term > n_run:
            raise ValueError(
                f"terminal_cost_terms_f returned {n_term} terms but "
                f"running_cost_terms_f returned {n_run}; expected n_term <= n_run"
            )

        H1 = H + 1
        discounts_full = gamma ** np.arange(H1, dtype=np.float64)   # (H+1,)
        norm_full = (1.0 - gamma) / (1.0 - gamma ** H1)
        discounts_run = gamma ** np.arange(H,  dtype=np.float64)    # (H,)
        norm_run = (1.0 - gamma) / (1.0 - gamma ** H) if H > 1 else 1.0

        if self.use_fpl_cost:
            # Per-step scalar via power-mean, then discount over time.
            # Terminal step's power-mean only sees terms that actually have a
            # terminal value (the first n_term), so a "missing" term doesn't
            # silently contribute 1.0.
            per_step_run = power_mean(running, self.fpl_p)                # (K, H)
            if n_term > 0:
                per_step_term = power_mean(terminal, self.fpl_p)          # (K,)
                per_step = np.concatenate([per_step_run, per_step_term[:, None]], axis=1)
                reward = (per_step * discounts_full[None, :]).sum(axis=1) * norm_full
            else:
                reward = (per_step_run * discounts_run[None, :]).sum(axis=1) * norm_run
        else:  # use_fpl_discounted
            # Each term gets its own discount-sum across its lifetime,
            # normalized to [0,1] by its own finite-horizon factor, then
            # power-mean over the per-term scalars.
            chunks = []
            if n_term > 0:
                # Terms 0..n_term-1 span H+1 steps (running + terminal).
                extended = np.concatenate(
                    [running[..., :n_term], terminal[:, None, :]], axis=1
                )                                                         # (K, H+1, n_term)
                chunks.append(
                    (extended * discounts_full[:, None]).sum(axis=1) * norm_full
                )
            if n_run > n_term:
                # Terms n_term..n_run-1 span only H steps.
                chunks.append(
                    (running[..., n_term:] * discounts_run[:, None]).sum(axis=1) * norm_run
                )
            per_term_sums = np.concatenate(chunks, axis=-1)               # (K, n_run)
            reward = power_mean(per_term_sums, self.fpl_p)                # (K,)

        # Expose the raw reward (positive, in [0,1], LARGER is better) so subclasses
        # that prefer reward semantics can argmax it directly. The returned scores
        # keep the cost convention (smaller is better) for backwards compatibility
        # with existing update_mean implementations.
        traj.reward = reward
        return -reward

    # ---- warm-start: shift mean forward in time by dt ----

    def _shift_mean(self, dt: float) -> None:
        if self.num_knots == 1:
            return

        # Zero-order-hold is piecewise constant, so resampling it at tk+dt is a
        # no-op whenever dt < knot spacing: every shifted knot time still lands
        # in its own segment. That silently disables the warm-start time-advance
        # for the default (coarse-knot ZOH) config. Instead, accumulate elapsed
        # time and roll the plan one knot toward t=0 each time a full knot
        # spacing has elapsed (repeating the terminal knot) — a genuine
        # receding-horizon advance at the representation's resolution.
        if self.spline_type == "zero":
            spacing = float(self.tk[1] - self.tk[0])
            self._shift_accum += dt
            # +eps so an exact multiple of `spacing` reached via float accumulation
            # (e.g. 10 * 0.02 = 0.19999…) fires on the intended step, not one late.
            n_roll = int((self._shift_accum + 1e-9) // spacing)
            if n_roll > 0:
                self._shift_accum -= n_roll * spacing
                n_roll = min(n_roll, self.num_knots - 1)
                tail = np.repeat(self.mean[-1:], n_roll, axis=0)
                self.mean = np.concatenate([self.mean[n_roll:], tail], axis=0)
            return

        # Continuous splines (linear/cubic) can represent a sub-knot shift
        # exactly: resample the current plan at knot times advanced by dt.
        shifted = interpolate(self.mean[None, ...], self.tk, self.tk + dt, self.spline_type)[0]
        self.mean = shifted
