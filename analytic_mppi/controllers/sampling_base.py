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
    scores: Optional[np.ndarray] = None           # (K,) — smaller is better


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

    # ---- subclass extension points ----

    def reset(self):
        self.mean[:] = 0.0

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
        # sum running cost over (H, n_terms) plus terminal sum over (n_terms,)
        running = traj.running_terms.sum(axis=(1, 2))         # (K,)
        terminal = traj.terminal_terms.sum(axis=-1)           # (K,)
        return running + terminal

    def _score_fpl(self, traj: Trajectory) -> np.ndarray:
        gamma = self.fpl_gamma
        H = traj.running_terms_f.shape[1]
        discounts = gamma ** np.arange(H, dtype=np.float64)    # (H,)

        if self.use_fpl_cost:
            # per-step running fulfillment scalar (power-mean over terms) then discount over time.
            per_step = power_mean(traj.running_terms_f, self.fpl_p)   # (K, H)
            reward = (per_step * discounts[None, :]).sum(axis=1) * (1.0 - gamma)
        else:  # use_fpl_discounted
            # discount each term independently over time, then power-mean over terms.
            per_term_sums = (traj.running_terms_f * discounts[:, None]).sum(axis=1) * (1.0 - gamma)  # (K, n_terms)
            reward = power_mean(per_term_sums, self.fpl_p)        # (K,)

        # cost convention: lower is better → negate the reward
        return -reward

    # ---- warm-start: shift mean forward in time by dt ----

    def _shift_mean(self, dt: float) -> None:
        if self.num_knots == 1:
            return
        shifted = interpolate(self.mean[None, ...], self.tk, self.tk + dt, self.spline_type)[0]
        self.mean = shifted
