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
from typing import Optional, Sequence

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
    reward_terms: Optional[np.ndarray] = None     # (K, J) — per-objective [0,1] values,
                                                  # LARGER is better. Populated ONLY by the
                                                  # discounted / layered FPL scorers (the
                                                  # modes where a per-objective vector exists
                                                  # before the power_mean collapse); None
                                                  # otherwise. Invariant when set:
                                                  # reward == power_mean(reward_terms, fpl_p).


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
        use_fpl_layered: bool = False,
        fpl_p: float = 0.1,
        fpl_gamma: float = 0.99,
        fpl_time_p: Optional[float] = None,
        # Per-objective weights for the objective-axis power-mean collapse (broadcast to
        # the atom/group axis, auto-normalized). None -> uniform. Together with fpl_p this
        # spans baseline<->FPL: (fpl_p=1, fpl_weights=w) is a LINEAR scalarization Σ w_i f_i
        # (vary w -> the whole "linear-weight family"); (fpl_p<0, uniform) is FPL. Used by
        # the discounted collapse and the layered OUTER collapse (across groups).
        fpl_weights: Optional[Sequence[float]] = None,
        # Inner power-mean exponent for layered FPL (across atoms WITHIN a group, e.g.
        # per-joint). None -> reuse fpl_p (same conjunctiveness inner and outer).
        fpl_group_p: Optional[float] = None,
        # Restrict the FPL reward to a SUBSET of the task's fulfillment atoms (indices
        # into `task.cost_term_names_f`). None -> use all atoms. e.g. [2] on g1_standup
        # scores ONLY nominal_fulfillment through the normal fpl_cost/fpl_discounted
        # pipeline (single-term reward — power_mean of one atom is that atom).
        fpl_term_indices: Optional[Sequence[int]] = None,
        # Split the objective-axis collapse into a CONJUNCTION over constraint atoms and a
        # separate composition with the progress atoms. `fpl_conj_indices` lists the atoms
        # that belong inside the p<0 conjunction (the "must hold" floors); every other atom
        # is combined with the conjunction's scalar at the OUTER level using `fpl_outer_p`
        # (default 0 -> geometric mean). None keeps the original single flat power-mean
        # over all atoms.
        #
        # Why this exists: a power-mean with p<0 is "improve the least-satisfied objective".
        # That is a safety operator only if the argmin is a SAFETY atom. Put a progress atom
        # (forward speed) inside the same conjunction and, whenever speed is the argmin, a
        # sharper p means "go faster, ignore posture" — so tightening the conjunction can
        # reduce survival. Progress objectives have no floor to hold, so they do not belong
        # inside the conjunction; keeping them outside makes "sharper p" a strictly
        # safety-directed operation.
        fpl_conj_indices: Optional[Sequence[int]] = None,
        fpl_outer_p: float = 0.0,
        fpl_outer_weights: Optional[Sequence[float]] = None,
        # Hybrid: unbounded quadratic penalties for "target" objectives (strong signal
        # to reach/maximize) + FPL log-barrier -log(fulfillment) for "floor" objectives
        # (hard "never fail" safety). Which terms are floors is set by the task's
        # `floor_term_indices`. floor_weight scales the barrier.
        use_hybrid: bool = False,
        floor_weight: float = 1.0,
        # When fpl_time_p is set (soft-min/weakest-moment over time), also weight that
        # power-mean by the gamma discount (weights=disc) instead of ignoring gamma
        # entirely. False (default) reproduces every existing fpl_time_p result exactly
        # -- this only changes behavior for callers that opt in.
        fpl_time_discount: bool = False,
    ):
        if sum([bool(use_fpl_cost), bool(use_fpl_discounted), bool(use_fpl_layered),
                bool(use_hybrid)]) > 1:
            raise ValueError("use_fpl_cost / use_fpl_discounted / use_fpl_layered / "
                             "use_hybrid are mutually exclusive")
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
        self.use_fpl_layered = bool(use_fpl_layered)
        self.fpl_term_indices = None if fpl_term_indices is None else list(fpl_term_indices)
        self.fpl_p = float(fpl_p)
        self.fpl_gamma = float(fpl_gamma)
        # Time aggregation for fpl_cost mode. None -> discounted arithmetic mean over
        # time (original: a rollout's value is its time-AVERAGE, which launders away a
        # single catastrophic step — e.g. a go-fast-then-faceplant rollout still scores
        # high). A float q (use q<=0) -> power-mean over time = "soft-min over time", so
        # a rollout's value is dominated by its WORST moment. This is what makes the
        # min-fulfillment floor hold across the whole trajectory, not just per-step.
        self.fpl_time_p = None if fpl_time_p is None else float(fpl_time_p)
        self.fpl_weights = (None if fpl_weights is None
                            else np.asarray(fpl_weights, dtype=np.float64))
        self.fpl_group_p = None if fpl_group_p is None else float(fpl_group_p)
        self.fpl_conj_indices = (None if fpl_conj_indices is None
                                 else list(int(i) for i in fpl_conj_indices))
        self.fpl_outer_p = float(fpl_outer_p)
        self.fpl_outer_weights = (None if fpl_outer_weights is None
                                  else np.asarray(fpl_outer_weights, dtype=np.float64))
        self.use_hybrid = bool(use_hybrid)
        self.floor_weight = float(floor_weight)
        self.fpl_time_discount = bool(fpl_time_discount)
        self.nu = int(task.nu)

        self.tk = make_knot_times(self.plan_horizon, self.num_knots)
        self.t_eval, self.H = make_eval_times(self.plan_horizon, float(backend.dt))

        self.rng = np.random.default_rng(int(seed))
        # Warm-start the plan at the task's nominal control if it provides one. Position
        # actuators (g1, quadruped) need this: the standing pose requires ctrl = u_ref, so a
        # zeros-initialized mean would command a collapsed pose for the first steps and topple
        # before MPPI recovers. Tasks with torque/relative actuators leave it None -> zeros.
        nominal = getattr(task, "nominal_control", None)
        self._mean_init = (np.zeros((self.num_knots, self.nu), dtype=np.float64)
                           if nominal is None else
                           np.broadcast_to(np.asarray(nominal, dtype=np.float64),
                                           (self.num_knots, self.nu)).copy())
        self.mean = self._mean_init.copy()
        self.last_trajectory: Optional[Trajectory] = None
        # Accumulated, not-yet-applied warm-start shift time (seconds). Only the
        # zero-order-hold path uses it — see _shift_mean.
        self._shift_accum = 0.0

    # ---- subclass extension points ----

    def reset(self):
        self.mean[:] = self._mean_init
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

        if self.use_hybrid:
            # Needs BOTH representations: quadratic penalties for target terms,
            # fulfillments for the log-barrier floor terms.
            traj.running_terms = self.task.running_cost_terms(qpos, qvel, sensordata, controls)
            traj.terminal_terms = self.task.terminal_cost_terms(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
            traj.running_terms_f = self.task.running_cost_terms_f(qpos, qvel, sensordata, controls)
            traj.terminal_terms_f = self.task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
            traj.scores = self._score_hybrid(traj)
        elif self.use_fpl_cost or self.use_fpl_discounted:
            rf = self.task.running_cost_terms_f(qpos, qvel, sensordata, controls)
            tf = self.task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
            if self.fpl_term_indices is not None:
                rf, tf = self._select_fpl_terms(rf, tf)
            traj.running_terms_f = rf
            traj.terminal_terms_f = tf
            traj.scores = self._score_fpl(traj)
        elif self.use_fpl_layered:
            # Grouped fulfillment atoms (e.g. per-joint) + task.fpl_groups partition.
            traj.running_terms_f = self.task.running_cost_terms_f_grouped(qpos, qvel, sensordata, controls)
            traj.terminal_terms_f = self.task.terminal_cost_terms_f_grouped(
                qpos[:, -1], qvel[:, -1], sensordata[:, -1]
            )
            traj.scores = self._score_fpl_layered(traj)
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

    def _select_fpl_terms(self, running_f: np.ndarray, terminal_f: np.ndarray):
        # Restrict the fulfillment atoms to self.fpl_term_indices before scoring.
        # _score_fpl assumes terminal terms are a PREFIX of the running terms (the
        # i-th terminal ↔ i-th running; trailing running terms have no terminal), so
        # order the selected indices that HAVE a terminal value (< n_term) first.
        n_term = terminal_f.shape[-1]
        sel = self.fpl_term_indices
        with_term = [i for i in sel if i < n_term]
        without_term = [i for i in sel if i >= n_term]
        running_sel = running_f[..., with_term + without_term]
        terminal_sel = terminal_f[..., with_term]
        return running_sel, terminal_sel

    def _collapse_objectives(self, f: np.ndarray,
                             weights: "np.ndarray | None") -> np.ndarray:
        """Collapse the atom axis of `f` (*lead, n) to a scalar composite in (0,1].

        Default (`fpl_conj_indices is None`): one flat power-mean at `fpl_p` over all
        atoms — the original behaviour. Otherwise a two-level composition:

            conj  = power_mean(f[conj_indices], fpl_p)        # the "must hold" floors
            value = power_mean([conj, *f[others]], fpl_outer_p)

        Indices at or beyond `n` are dropped, so this works unchanged on the terminal
        slice (which carries a PREFIX of the running atoms — hopper's control atom has no
        terminal value). If the terminal slice happens to contain no non-conjunction atom,
        the outer level degenerates to the conjunction itself, which is the right limit.
        """
        n = f.shape[-1]
        idx = self.fpl_conj_indices
        if idx is None:
            return power_mean(f, self.fpl_p, weights=weights)
        conj = [i for i in idx if 0 <= i < n]
        rest = [i for i in range(n) if i not in conj]
        if not conj:
            return power_mean(f, self.fpl_p, weights=weights)
        w_conj = None if weights is None else np.asarray(weights)[conj]
        inner = power_mean(f[..., conj], self.fpl_p, weights=w_conj)   # (*lead,)
        if not rest:
            return inner
        outer = np.concatenate([inner[..., None], f[..., rest]], axis=-1)
        ow = self.fpl_outer_weights
        if ow is None:
            # Default: the conjunction carries the mass of the atoms it absorbed. With this
            # weighting the split is an exact no-op when fpl_outer_p == fpl_p — e.g. at
            # p = -1, 1/(|C|/n · 1/M + Σ_rest 1/n · 1/f_i) with M = |C|/Σ_C 1/f_j collapses
            # back to n/Σ_all 1/f_i. So `fpl_conj_indices` changes only WHERE a sharper
            # fpl_p acts, not the operating point, and it stays exact on the terminal slice
            # where some atoms are absent.
            ow = np.array([float(len(conj))] + [1.0] * len(rest))
        else:
            # Caller supplies [w_conjunction, w_atom0, w_atom1, ...] in the task's atom
            # order; select the entries that survived into `rest`.
            ow = np.concatenate([np.asarray(ow)[:1], np.asarray(ow)[1:][rest]])
        return power_mean(outer, self.fpl_outer_p, weights=ow)

    def _score_hybrid(self, traj: Trajectory) -> np.ndarray:
        # cost = Σ_targets (∫ quadratic penalty dt)  +  floor_weight · Σ_floors (∫ -log(f) dt)
        # Target terms keep the unbounded quadratic's strong signal (reach/maximize);
        # floor terms get an FPL log-barrier that blows up as the fulfillment → 0
        # (a hard "never fail" safety floor). Which term indices are floors is set by
        # the task; normal terms and FPL atoms are assumed index-aligned (true for g1,
        # hopper). Smaller-is-better, so the standard softmax update consumes it directly.
        dt = float(self.backend.dt)
        n = traj.running_terms.shape[-1]
        floors = [i for i in self.task.floor_term_indices if 0 <= i < n]
        targets = [i for i in range(n) if i not in floors]

        target_cost = (traj.running_terms[..., targets].sum(axis=(1, 2)) * dt
                       + traj.terminal_terms[..., targets].sum(axis=-1))          # (K,)

        if not floors:
            return target_cost
        barrier = -np.log(np.clip(traj.running_terms_f[..., floors], 1e-8, 1.0))  # (K,H,|floors|)
        barrier_cost = barrier.sum(axis=(1, 2)) * dt
        n_term_f = traj.terminal_terms_f.shape[-1]
        term_floors = [i for i in floors if i < n_term_f]
        if term_floors:
            bt = -np.log(np.clip(traj.terminal_terms_f[..., term_floors], 1e-8, 1.0))
            barrier_cost = barrier_cost + bt.sum(axis=-1)
        return target_cost + self.floor_weight * barrier_cost

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
            # Per-step scalar via power-mean (conjunction over objectives), then
            # aggregate over time. Terminal step's power-mean only sees terms that
            # actually have a terminal value (the first n_term), so a "missing" term
            # doesn't silently contribute 1.0. Weights (if any) are on the OBJECTIVE
            # axis; the terminal slice uses the matching prefix.
            w = self.fpl_weights
            w_term = None if w is None else w[:n_term]
            per_step_run = self._collapse_objectives(running, w)            # (K, H)
            if n_term > 0:
                per_step_term = self._collapse_objectives(terminal, w_term)  # (K,)
                per_step = np.concatenate([per_step_run, per_step_term[:, None]], axis=1)
                disc, norm = discounts_full, norm_full
            else:
                per_step = per_step_run
                disc, norm = discounts_run, norm_run
            if self.fpl_time_p is None:
                # Original: discounted arithmetic mean over time.
                reward = (per_step * disc[None, :]).sum(axis=1) * norm
            else:
                # Soft-min over time: value = (power-mean q<=0 of) the per-step
                # composites, so one catastrophic step tanks the whole rollout.
                # fpl_time_discount additionally weights this power-mean by the gamma
                # discount (disc), so a near-term failure still dominates more than an
                # identical failure late in the plan -- weakest-moment AND discounted,
                # rather than the two being mutually exclusive.
                time_w = disc if self.fpl_time_discount else None
                reward = power_mean(per_step, self.fpl_time_p, weights=time_w)  # (K,)
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
            # Expose the per-objective [0,1] vector (pre-collapse) so multi-objective
            # controllers can form a separate gradient per objective. reward below is
            # exactly power_mean(reward_terms, fpl_p).
            traj.reward_terms = per_term_sums
            reward = power_mean(per_term_sums, self.fpl_p, weights=self.fpl_weights)  # (K,)

        # Expose the raw reward (positive, in [0,1], LARGER is better) so subclasses
        # that prefer reward semantics can argmax it directly. The returned scores
        # keep the cost convention (smaller is better) for backwards compatibility
        # with existing update_mean implementations.
        #
        # Cost is S_k = -log(u_k), NOT -u_k (FPL_MPPI_HANDOFF §3 step 4). Feeding
        # this into the standard softmax exp(-(S_k - min S_k)/λ) yields
        #   w_k ∝ (u_k / max_k u_k)^(1/λ)
        # i.e. the handoff's w_k ∝ u_k^(1/λ) with the mandatory min-shift baked in.
        # Using -log (rather than -u directly) is what gives the weights spread:
        # near-catastrophic rollouts (u_k → 0) get a large cost instead of a raw
        # value cramped into [0, 1]. `reward > 0` always (power_mean clips terms at
        # eps=1e-8 and the discounted sum is over positives), so -log is finite.
        traj.reward = reward
        return -np.log(reward)

    def _score_fpl_layered(self, traj: Trajectory) -> np.ndarray:
        # Two-level composition (FPL_MPPI_HANDOFF §3, grouped): discount-sum each atom
        # over time into an FQ-value in [0,1], inner-power-mean the atoms WITHIN each
        # group (task.fpl_groups), then outer-power-mean the group scalars. Inner uses
        # fpl_group_p (falls back to fpl_p); outer uses fpl_p. Cost = -log(reward), same
        # convention as _score_fpl. Grouped atoms are all state-based, so terminal aligns
        # 1:1 with running (every atom spans the full H+1 series).
        gamma = self.fpl_gamma
        running  = traj.running_terms_f    # (K, H, n)
        terminal = traj.terminal_terms_f   # (K, n)
        H, n = running.shape[1], running.shape[-1]
        if terminal.shape[-1] != n:
            raise ValueError(
                f"layered FPL expects grouped terminal atoms aligned 1:1 with running "
                f"atoms; got {terminal.shape[-1]} terminal vs {n} running"
            )
        groups = self.task.fpl_groups
        if not groups:
            raise ValueError(
                f"task {type(self.task).__name__} defines no fpl_groups; "
                f"layered FPL unavailable"
            )
        H1 = H + 1
        discounts_full = gamma ** np.arange(H1, dtype=np.float64)   # (H+1,)
        norm_full = (1.0 - gamma) / (1.0 - gamma ** H1)
        # Per-atom discounted [0,1] value over the extended H+1 series (FQ-value analog).
        extended = np.concatenate([running, terminal[:, None, :]], axis=1)   # (K, H+1, n)
        per_term = (extended * discounts_full[:, None]).sum(axis=1) * norm_full  # (K, n)
        # Inner power-mean within each group, then outer power-mean across the groups.
        inner_p = self.fpl_p if self.fpl_group_p is None else self.fpl_group_p
        group_scores = np.stack(
            [power_mean(per_term[:, idx], inner_p) for idx in groups], axis=-1
        )                                                                    # (K, n_groups)
        # Per-group [0,1] vector (pre-collapse) for multi-objective composition; reward
        # below is exactly power_mean(reward_terms, fpl_p) over the groups.
        traj.reward_terms = group_scores
        reward = power_mean(group_scores, self.fpl_p, weights=self.fpl_weights)  # (K,)
        traj.reward = reward
        return -np.log(reward)

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
