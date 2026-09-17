"""First-order gradient MPC: projected Adam on spline knots, through MJX autodiff.

The non-sampling arm of the controller choice: where MPPI evaluates K perturbed knot
plans and reweights, GradientMPC keeps ONE plan and descends the exact gradient of the
same objective, obtained by `jax.grad` through the differentiable MJX rollout
(dynamics/mjx_backend.rollout_jax) and the jnp cost/scoring mirrors (tasks/jax_costs,
controllers/jax_scoring). Everything the two arms can share IS shared:

  * spline semantics -- the knot->controls map is linear for all three spline types, so
    it is precomputed as a basis matrix W by running the numpy `interpolate` on one-hot
    knots: identical interpolation by construction, and in-jit it is one einsum.
  * warm start -- the same `spline.shift_plan` the sampling stack uses, so a gradient
    plan and an MPPI plan are interchangeable mid-experiment.
  * objective -- jax_scoring.score_normal / score_fpl are parity-pinned mirrors of the
    numpy scorers; gradient descent minimizes exactly what MPPI ranks rollouts by
    (score = -log(reward) in FPL modes).

Only the CORE cost modes exist here (normal / fpl_cost / fpl_discounted); layered,
hybrid and the conjunction-split collapse are CPU-only by scope (config.resolve
enforces this before anything is built).

Bounds: projected Adam -- knots are clipped to [u_min, u_max] after every step, the
same u-space clip the sampling stack applies to its sampled knots. (A tanh
reparameterization was considered and rejected: it would need an atanh conversion of
warm-start plans and distorts per-dimension step sizes for asymmetric ctrl ranges.)

iLQG seam: subclass and override `_build_plan_fn` (returns the jitted
(t0, qpos0, qvel0, knots) -> (knots, losses) planner) -- nothing else in this class
assumes the planner is first-order. Register the subclass under its own dispatch pair
(e.g. ("gradient", "ilqg")). Cheap first-order variants (momentum, plain GD) only need
`_adam_step` swapped.

This module stays importable without jax (jax is imported in __init__): the controller
registry -- and the config tests introspecting it -- must work without the [mjx] extra.
"""
from __future__ import annotations

import time as _time
from typing import Optional, Sequence

import numpy as np

from analytic_mppi.tasks.base import ATOM_FLOOR_LEGACY, Task
from analytic_mppi.tasks.jax_costs import make_jax_costs
from .spline import interpolate, make_eval_times, make_knot_times, shift_plan


class GradientMPC:
    """Single-plan gradient-descent MPC on an MJX backend. `act(state) -> u`, numpy."""

    def __init__(
        self,
        task: Task,
        backend,
        *,
        num_samples: int = 1,
        num_knots: int = 4,
        plan_horizon: float = 1.0,
        spline_type: str = "zero",
        iterations: int = 20,          # Adam steps per act() (run.iterations semantics)
        seed: int = 0,                 # accepted for protocol symmetry; deterministic
        learning_rate: float = 0.05,
        warmup: bool = True,           # compile in the ctor (on a dummy state)
        # core cost modes -- the same flags eval._fpl_kwargs emits:
        use_fpl_cost: bool = False,
        use_fpl_discounted: bool = False,
        fpl_p: float = 0.1,
        fpl_gamma: float = 0.99,
        fpl_time_p: Optional[float] = None,
        fpl_time_discount: bool = False,
        fpl_weights: Optional[Sequence[float]] = None,
        fpl_term_indices: Optional[Sequence[int]] = None,
        # Lower clamp on the fulfillment atoms (WO-3.4). Mirrors sampling_base's kwarg of
        # the same name so the sampling and gradient arms score an identical objective.
        fpl_atom_floor: float = ATOM_FLOOR_LEGACY,
        fpl_terminal_value: bool = False,
    ):
        import jax
        import jax.numpy as jnp
        from . import jax_scoring

        if not hasattr(backend, "rollout_jax"):
            raise TypeError(
                f"GradientMPC differentiates through the rollout and needs an MJX "
                f"backend (rollout_jax); got {type(backend).__name__}. "
                f"Set run.backend='mjx'."
            )
        if int(num_samples) != 1:
            raise ValueError(
                f"GradientMPC optimizes a single plan; num_samples must be 1 "
                f"(got {num_samples}). Multi-start descent is future work."
            )
        if use_fpl_cost and use_fpl_discounted:
            raise ValueError("use_fpl_cost / use_fpl_discounted are mutually exclusive")
        assert 0.0 <= fpl_gamma < 1.0, f"fpl_gamma must be in [0, 1), got {fpl_gamma}"

        self._jax = jax
        self.task = task
        self.backend = backend
        self.num_samples = 1
        self.num_knots = int(num_knots)
        self.plan_horizon = float(plan_horizon)
        self.spline_type = spline_type
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.nu = int(task.nu)
        self.mode = ("fpl_cost" if use_fpl_cost
                     else "fpl_discounted" if use_fpl_discounted
                     else "normal")
        self.fpl_p = float(fpl_p)
        self.fpl_gamma = float(fpl_gamma)
        self.fpl_time_p = None if fpl_time_p is None else float(fpl_time_p)
        self.fpl_time_discount = bool(fpl_time_discount)
        self.fpl_weights = None if fpl_weights is None else list(fpl_weights)
        self.fpl_term_indices = (None if fpl_term_indices is None
                                 else [int(i) for i in fpl_term_indices])
        if not (0.0 < float(fpl_atom_floor) <= 1.0):
            raise ValueError(
                f"fpl_atom_floor must be in (0, 1], got {fpl_atom_floor}")
        self.fpl_atom_floor = float(fpl_atom_floor)
        self.fpl_terminal_value = bool(fpl_terminal_value)

        self.tk = make_knot_times(self.plan_horizon, self.num_knots)
        self.t_eval, self.H = make_eval_times(self.plan_horizon, float(backend.dt))

        # Spline as a linear map: W[h, k] = value at t_eval[h] of the spline whose knot
        # vector is the k-th one-hot. Built with the SAME numpy interpolator the sampling
        # stack uses, so the two arms share interpolation semantics by construction.
        eye = np.eye(self.num_knots, dtype=np.float64)[:, :, None]      # (k, num_knots, 1)
        self._W = interpolate(eye, self.tk, self.t_eval, spline_type)[:, :, 0].T  # (H, k)

        # Warm-start at the task's nominal control if it provides one (mirrors
        # SamplingController: position-actuated robots topple from a zeros plan).
        nominal = getattr(task, "nominal_control", None)
        self._mean_init = (np.zeros((self.num_knots, self.nu), dtype=np.float64)
                           if nominal is None else
                           np.broadcast_to(np.asarray(nominal, dtype=np.float64),
                                           (self.num_knots, self.nu)).copy())
        self.mean = self._mean_init.copy()
        self._shift_accum = 0.0

        # jit-time constants for the loss
        self._costs = make_jax_costs(task)
        self._scoring = jax_scoring
        self._jnp = jnp
        self._u_min = jnp.asarray(task.u_min)
        self._u_max = jnp.asarray(task.u_max)
        self._W_j = jnp.asarray(self._W)

        self._plan = jax.jit(self._build_plan_fn())

        # run.py / eval diagnostics surface (mirrors the sampling stack's attributes)
        self.last_trajectory = None
        self.last_ess = float("nan")
        self.last_loss_curve: Optional[np.ndarray] = None   # (iterations,) per act()
        self.warmup_s: Optional[float] = None
        if warmup:
            tic = _time.perf_counter()
            dummy = np.zeros(backend.nstate, dtype=np.float64)
            k, losses = self._plan(dummy[0], dummy[backend.qpos_slice],
                                   dummy[backend.qvel_slice], jnp.asarray(self.mean))
            jax.block_until_ready((k, losses))
            self.warmup_s = _time.perf_counter() - tic

    # ---- objective (jnp; runs inside jit) ----

    def _loss(self, knots, t0, qpos0, qvel0):
        jnp = self._jnp
        controls = jnp.einsum("hk,ku->hu", self._W_j, knots)             # (H, nu)
        _, qpos_seq, qvel_seq, sd_seq = self.backend.rollout_jax(t0, qpos0, qvel0, controls)
        if self.mode == "normal":
            rt = self._costs.running_cost_terms(qpos_seq, qvel_seq, sd_seq, controls)
            tt = self._costs.terminal_cost_terms(qpos_seq[-1], qvel_seq[-1], sd_seq[-1])
            return self._scoring.score_normal(rt, tt, dt=float(self.backend.dt))
        floor = self._scoring.floor_atoms
        rf = floor(self._costs.running_cost_terms_f(qpos_seq, qvel_seq, sd_seq, controls),
                   self.fpl_atom_floor)
        tf = floor(self._costs.terminal_cost_terms_f(qpos_seq[-1], qvel_seq[-1], sd_seq[-1]),
                   self.fpl_atom_floor)
        if self.fpl_term_indices is not None:
            rf, tf = self._scoring.select_fpl_terms(rf, tf, self.fpl_term_indices)
        return self._scoring.score_fpl(
            rf, tf, mode=self.mode, p=self.fpl_p, gamma=self.fpl_gamma,
            time_p=self.fpl_time_p, time_discount=self.fpl_time_discount,
            weights=self.fpl_weights, terminal_value=self.fpl_terminal_value)

    # ---- planner (the iLQG seam: override to replace the whole inner optimizer) ----

    def _build_plan_fn(self):
        jax, jnp = self._jax, self._jnp
        lr = self.learning_rate
        b1, b2, eps = 0.9, 0.999, 1e-8

        def plan_fn(t0, qpos0, qvel0, knots0):
            def body(carry, _):
                knots, m, v, t = carry
                val, g = jax.value_and_grad(self._loss)(knots, t0, qpos0, qvel0)
                t = t + 1.0
                m = b1 * m + (1.0 - b1) * g
                v = b2 * v + (1.0 - b2) * g * g
                mhat = m / (1.0 - b1 ** t)
                vhat = v / (1.0 - b2 ** t)
                knots = knots - lr * mhat / (jnp.sqrt(vhat) + eps)
                knots = jnp.clip(knots, self._u_min, self._u_max)
                return (knots, m, v, t), val
            zeros = jnp.zeros_like(knots0)
            init = (knots0, zeros, zeros, jnp.asarray(0.0))
            (knots, _, _, _), losses = jax.lax.scan(body, init, None,
                                                    length=self.iterations)
            return knots, losses

        return plan_fn

    # ---- protocol surface ----

    def reset(self):
        self.mean[:] = self._mean_init
        self._shift_accum = 0.0
        self.last_loss_curve = None

    def act(self, state: np.ndarray) -> np.ndarray:
        """One MPC step: descend `iterations` Adam steps from the warm start, execute
        the plan's t=0 action, shift the plan by dt. numpy in / numpy out."""
        state = np.asarray(state, dtype=np.float64)
        b = self.backend
        knots, losses = self._plan(state[0], state[b.qpos_slice], state[b.qvel_slice],
                                   self._jnp.asarray(self.mean))
        self.mean = np.asarray(self._jax.device_get(knots), dtype=np.float64)
        self.last_loss_curve = np.asarray(self._jax.device_get(losses), dtype=np.float64)

        u0 = self.mean[0].copy()               # knot 0 sits at t=0 for every spline type
        self.mean, self._shift_accum = shift_plan(
            self.mean, self.tk, float(b.dt), self.spline_type, self._shift_accum)
        return u0
