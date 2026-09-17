"""S10 / WO-1 — iLQR on the augmented (physics, accumulator) state, through MJX.

The λ-free optimizer arm: where MPPI reweights sampled rollouts through a softmax at
temperature λ, and GradientMPC runs first-order Adam, this module solves the SAME
objective by iterative LQR — no temperature, no softmax, no `-log(u)` bridge. That is
the point of Phase 4's 2x2: if FPL survives in this column, the λ/p confound is
eliminated structurally rather than by sweeping around it.

The objective enters through `controllers/accumulator.py` (S9): every mode is
(additive stage) + (terminal fold) + (readout), on the augmented state

    x_{t+1} = F(x_t, u_t)                    physics (mjx.step)
    z_{t+1} = z_t + phi_t(x_t, u_t)          accumulator stage (+ terminal fold at H-1)
    J       = readout(z_H)                   the ONLY nonlinearity in z

Where the curvature comes from (this is the load-bearing design decision)
-------------------------------------------------------------------------
Linearizing the augmented dynamics and quadratic-izing only the readout would DROP all
stage-cost curvature: in `normal` mode the readout is linear in z, V_xx would be
identically zero, and the "iLQR" would degenerate to steepest descent. The backward
pass here is therefore Gauss-Newton **through the physics only**:

  * physics block:  first-order (A_t, B_t) via `vmap(jax.jacfwd(step))` — jacfwd, never
    jacrev: reverse mode cannot differentiate the solver's `lax.while_loop`, and it only
    surfaces on models big enough to reach that path (see dynamics/mjx_manifold.py).
  * cost channel:   the stage increment phi_t = c(y) with y = (x_{t+1}, sensordata, u_t)
    is expanded to SECOND order in y (pure cost-function Hessians, cheap, no solver
    derivatives), then chained through the first-order step Jacobians:
    phi_xx ~= Y_x^T c_yy Y_x. Dropping Y_xx is the Gauss-Newton step; the cost curvature
    itself is exact. On a quadratic cost this reproduces exact LQR (GATE G8).
  * readout:        exact `jax.grad`/`jax.hessian` at the nominal z_H (dim n_acc, tiny).
    Note for p<0 FPL readouts (`-(1/q) log z`, q<0) this Hessian is CONCAVE — the
    Levenberg λ is expected to do real work there, which is why its trace is
    instrumented from the first commit.

Sensordata is handled honestly: the scorer's convention (from `mujoco.rollout`) pairs
step t's post-step state with the sensordata computed DURING that step, so the stage at
t is a clean function of (x_t, u_t) through the step map, and the terminal atoms — which
see sd from step H-1 — fold into the LAST stage rather than the readout
(`AugmentedObjective.fold_terminal` is additive in z for every mode, which is what makes
this exact).

Bounds are box-constrained iLQR: the feedforward k_t solves the Levenberg-regularized
`Q_uu` sub-problem by projected Newton (fixed clamp-and-solve iterations), and the
feedback rows of clamped dims are zeroed — NOT clip-after-update (the clipping confound,
docs/mppi_math.md).

Parameterization (settled, WO-1.2): iLQR searches per-step control space; the result is
projected to the knot basis with theta = W^+ u* and the projection residual
`‖(I - W W^+) u*‖/‖u*‖` plus J_pre = J(u*) and J_post = J(W W^+ u*) are recorded —
J_post is what executes and is the headline number (GATE G10).

Caveats inherited from Phase 1: G5 is mitigated, not passed — multi-contact Jacobians
carry a ~1.6e-3 residual, fine for search directions validated by a line search on the
true cost (this module), NOT fine for deploying the feedback gains K_t on a robot. Gain
synthesis for deployment (Phase 7 / S19) must run one final backward pass on
`mjd_transitionFD` Jacobians (plumbing in cost_gd.py).

This module stays importable without jax (lazy imports, matching gradient_mpc.py); the
jax entry points route through dynamics/mjx_manifold, which pins the
`--xla_gpu_enable_triton_gemm=false` guard before jax initializes.
"""
from __future__ import annotations

import time as _time
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np

from .gradient_mpc import GradientMPC
from .spline import shift_plan


def make_ilqr_solver(
    model,
    *,
    u_min,
    u_max,
    n_acc: int,
    stage_fn,
    fold_fn,
    readout_fn,
    iterations: int,
    alphas: Optional[Sequence[float]] = None,
    pn_iterations: int = 6,
    accept_ratio: float = 1e-4,
    lam_up: float = 10.0,
    lam_dn: float = 0.5,
    lam_min: float = 1e-6,
    lam_max: float = 1e8,
):
    """Build a jit-able iLQR solver over one MuJoCo model.

    `model` is a CPU MjModel configured for linearization — warm-starting disabled via
    `mjx_manifold.linearization_model` (G5) and a non-CG solver (both refused otherwise,
    invariant 11.5). The forward passes run on the same model: with warm-starting off the
    forward solution is unaffected and the line-search cost is evaluated on exactly the
    dynamics the backward pass linearizes.

    Objective callables (all single-trajectory, jnp in/out):
      stage_fn(t, qpos, qvel, sensordata, u) -> (n_acc,)   stage increment dz
      fold_fn(qpos, qvel, sensordata)        -> (n_acc,)   terminal fold increment
      readout_fn(z_final (n_acc,))           -> scalar     the score (smaller = better)

    Returns a namespace with:
      solve(t0, qpos0, qvel0, u_init (H,nu), lam0) -> (u_star, J, stats)
      open_loop_cost(t0, qpos0, qvel0, useq)       -> J
    Both are pure and jit-compatible; H is taken from the control shape at trace time.
    """
    from analytic_mppi.dynamics import mjx_manifold as manifold  # noqa: F401  (XLA guard)
    import jax
    import jax.numpy as jnp
    from mujoco import mjx
    from mujoco.mjx._src.types import DisableBit, SolverType

    if int(model.opt.solver) == int(SolverType.CG):
        raise ValueError(
            "MJX autodiff of the CG solver's constraint path is unusable (relFro 1.4e+02 "
            "on 16 active constraints, GATE G5). Set solver='Newton' on the model. See "
            "docs/jacobian_audit.md.")
    if not (int(model.opt.disableflags) & int(DisableBit.WARMSTART)):
        raise ValueError(
            "solver warm-starting is enabled; MJX's contact Jacobian is then wrong by "
            "~4 orders of magnitude (G5). Build the model with "
            "`mjx_manifold.linearization_model(model)` first.")

    mjx_model = mjx.put_model(model)
    data0 = mjx.make_data(mjx_model)
    nv, nu, nsd = int(model.nv), int(model.nu), int(model.nsensordata)
    nx = 2 * nv                       # physics tangent dim
    naug = nx + int(n_acc)            # augmented dim
    u_min_j = jnp.asarray(u_min)
    u_max_j = jnp.asarray(u_max)
    alphas_j = jnp.asarray(alphas if alphas is not None else 0.5 ** np.arange(8))

    def _step(d, u):
        return mjx.step(mjx_model, d.replace(ctrl=u))

    def _mkdata(t0, qpos, qvel):
        return data0.replace(qpos=qpos, qvel=qvel, time=t0)

    # ---- open-loop rollout: nominal trajectory, per-step z prefix, final z, cost ----

    def _rollout(t0, qpos0, qvel0, useq):
        H = useq.shape[0]
        ts = jnp.arange(H)

        def body(carry, per):
            d, z = carry
            t, u = per
            d2 = _step(d, u)
            z2 = z + stage_fn(t, d2.qpos, d2.qvel, d2.sensordata, u)
            return (d2, z2), (d2.qpos, d2.qvel, d2.sensordata, z)

        init = (_mkdata(t0, qpos0, qvel0), jnp.zeros(n_acc))
        (_, zH), (qs, vs, sds, z_pre) = jax.lax.scan(body, init, (ts, useq))
        zf = zH + fold_fn(qs[-1], vs[-1], sds[-1])
        return readout_fn(zf), qs, vs, sds, z_pre, zf

    def open_loop_cost(t0, qpos0, qvel0, useq):
        return _rollout(t0, qpos0, qvel0, useq)[0]

    # ---- closed-loop forward pass (one alpha; vmapped over the schedule) ----

    def _forward(t0, qpos0, qvel0, u_nom, q_pre, v_pre, z_pre, k, K, alpha):
        H = u_nom.shape[0]
        ts = jnp.arange(H)

        def body(carry, per):
            d, z = carry
            t, un, qn, vn, zn, k_t, K_t = per
            dq = manifold.differentiate_pos(model, d.qpos, qn)
            dxa = jnp.concatenate([dq, d.qvel - vn, z - zn])
            u = jnp.clip(un + alpha * k_t + K_t @ dxa, u_min_j, u_max_j)
            d2 = _step(d, u)
            z2 = z + stage_fn(t, d2.qpos, d2.qvel, d2.sensordata, u)
            return (d2, z2), (u, d2.qpos, d2.qvel, d2.sensordata, z)

        init = (_mkdata(t0, qpos0, qvel0), jnp.zeros(n_acc))
        (_, zH), (us, qs, vs, sds, z_pre2) = jax.lax.scan(
            body, init, (ts, u_nom, q_pre, v_pre, z_pre, k, K))
        zf = zH + fold_fn(qs[-1], vs[-1], sds[-1])
        return readout_fn(zf), us, qs, vs, sds, z_pre2, zf

    _forward_sweep = jax.vmap(_forward, in_axes=(None,) * 9 + (0,))

    # ---- per-step linearization + second-order cost expansion ----

    def _lin_one(qn, vn, un, qn2, vn2, sdn2, t, is_last):
        """(A, B, phi_x, phi_u, Pxx, Puu, Pux) of one augmented step.

        (qn, vn, un): pre-step nominal; (qn2, vn2, sdn2): post-step nominal state and the
        sensordata computed during the step (the scorer pairing). `is_last` folds the
        terminal atoms into this stage (only ever 1.0 at t = H-1); everything the fold
        touches is clip-protected, so evaluating it at every t is safe under vmap.
        """
        def step_res(dq, dv, du):
            d = _mkdata(0.0, manifold.integrate_pos(model, qn, dq), vn + dv)
            d2 = _step(d, un + du)
            dx = jnp.concatenate(
                [manifold.differentiate_pos(model, d2.qpos, qn2), d2.qvel - vn2])
            return dx, d2.sensordata

        zq = jnp.zeros(nv)
        zu = jnp.zeros(nu)
        # `jax.jacfwd` returns a pytree matching the OUTPUT structure at the outer
        # level and the argnums tuple at the inner level -- i.e. jac[output_leaf][arg],
        # NOT the other way around. Getting this backwards is a silent bug on models
        # with nsensordata > 0 (shapes still concat) and crashes with a shape-(0, nv)
        # slot on the pendulum where nsd == 0.
        (dxdq, dxdv, dxdu), (dsdq, dsdv, dsdu) = jax.jacfwd(
            step_res, argnums=(0, 1, 2))(zq, zq, zu)
        A = jnp.concatenate([dxdq, dxdv], axis=1)                # (nx, nx)
        B = dxdu                                                 # (nx, nu)
        Sx = jnp.concatenate([dsdq, dsdv], axis=1)               # (nsd, nx)
        Su = dsdu                                                # (nsd, nu)

        def stage_y(y):
            dxq, dxv, dsd, du = jnp.split(y, (nv, nx, nx + nsd))
            qpos = manifold.integrate_pos(model, qn2, dxq)
            qvel = vn2 + dxv
            sd = sdn2 + dsd
            u = un + du
            inc = stage_fn(t, qpos, qvel, sd, u)
            return inc + is_last * fold_fn(qpos, qvel, sd)

        y0 = jnp.zeros(nx + nsd + nu)
        c_y = jax.jacrev(stage_y)(y0)                            # (n_acc, ny)
        c_yy = jax.jacfwd(jax.jacrev(stage_y))(y0)               # (n_acc, ny, ny)

        Tx = jnp.concatenate([A, Sx, jnp.zeros((nu, nx))], axis=0)   # (ny, nx)
        Tu = jnp.concatenate([B, Su, jnp.eye(nu)], axis=0)           # (ny, nu)
        phi_x = c_y @ Tx                                             # (n_acc, nx)
        phi_u = c_y @ Tu                                             # (n_acc, nu)
        Pxx = jnp.einsum("np,jnm,mq->jpq", Tx, c_yy, Tx)
        Puu = jnp.einsum("np,jnm,mq->jpq", Tu, c_yy, Tu)
        Pux = jnp.einsum("np,jnm,mq->jpq", Tu, c_yy, Tx)
        return A, B, phi_x, phi_u, Pxx, Puu, Pux

    _lin_all = jax.vmap(_lin_one)

    # ---- box-constrained feedforward: projected Newton on the Q_uu sub-problem ----

    def _box_qp(Quu, Qu, lo, hi):
        tol = 1e-6 * jnp.maximum(hi - lo, 1e-3)

        def clamp_mask(x, g):
            return ((x <= lo + tol) & (g > 0)) | ((x >= hi - tol) & (g < 0))

        x0 = jnp.clip(-jnp.linalg.solve(Quu, Qu), lo, hi)

        def it(x, _):
            g = Qu + Quu @ x
            c = clamp_mask(x, g).astype(x.dtype)
            f = 1.0 - c
            M = Quu * jnp.outer(f, f) + jnp.diag(c)
            rhs = f * (-(Qu + Quu @ (x * c))) + c * x
            return jnp.clip(jnp.linalg.solve(M, rhs), lo, hi), None

        x, _ = jax.lax.scan(it, x0, None, length=pn_iterations)
        return x, clamp_mask(x, Qu + Quu @ x)

    # ---- backward pass ----

    def _backward(lin, zf, lam, useq):
        A, B, phi_x, phi_u, Pxx, Puu, Pux = lin
        r_g = jax.grad(readout_fn)(zf)                       # (n_acc,)
        r_H = jax.hessian(readout_fn)(zf)                    # (n_acc, n_acc)
        Vx0 = jnp.concatenate([jnp.zeros(nx), r_g])
        Vxx0 = jnp.zeros((naug, naug)).at[nx:, nx:].set(r_H)
        I_acc = jnp.eye(n_acc)
        I_u = jnp.eye(nu)

        def bstep(carry, per):
            Vx, Vxx, dV1, dV2, ok = carry
            A_t, B_t, phx, phu, Pxx_t, Puu_t, Pux_t, u_t = per
            Atil = (jnp.zeros((naug, naug))
                    .at[:nx, :nx].set(A_t)
                    .at[nx:, :nx].set(phx)
                    .at[nx:, nx:].set(I_acc))
            Btil = jnp.concatenate([B_t, phu], axis=0)       # (naug, nu)
            Vz = Vx[nx:]

            Qx = Atil.T @ Vx
            Qu = Btil.T @ Vx
            VxxA = Vxx @ Atil
            Qxx = (Atil.T @ VxxA).at[:nx, :nx].add(jnp.einsum("j,jpq->pq", Vz, Pxx_t))
            Quu = Btil.T @ Vxx @ Btil + jnp.einsum("j,jpq->pq", Vz, Puu_t)
            Qux = (Btil.T @ VxxA).at[:, :nx].add(jnp.einsum("j,jpq->pq", Vz, Pux_t))
            Quu = 0.5 * (Quu + Quu.T)

            s = jnp.linalg.svd(Quu, compute_uv=False)
            cond = s[0] / jnp.maximum(s[-1], jnp.finfo(s.dtype).tiny)
            Quu_reg = Quu + lam * I_u
            ok_t = jnp.all(jnp.isfinite(jnp.linalg.cholesky(Quu_reg)))

            k_t, clamped = _box_qp(Quu_reg, Qu, u_min_j - u_t, u_max_j - u_t)
            f = 1.0 - clamped.astype(Quu.dtype)
            M = Quu_reg * jnp.outer(f, f) + jnp.diag(1.0 - f)
            K_t = jnp.linalg.solve(M, -(f[:, None] * Qux))   # clamped rows -> 0

            dV1 = dV1 + k_t @ Qu
            dV2 = dV2 + 0.5 * k_t @ (Quu_reg @ k_t)
            KtQ = K_t.T @ Quu_reg
            Vx_n = Qx + KtQ @ k_t + K_t.T @ Qu + Qux.T @ k_t
            Vxx_n = Qxx + KtQ @ K_t + K_t.T @ Qux + Qux.T @ K_t
            Vxx_n = 0.5 * (Vxx_n + Vxx_n.T)
            return ((Vx_n, Vxx_n, dV1, dV2, ok & ok_t),
                    (k_t, K_t, cond, jnp.linalg.norm(Qu)))

        init = (Vx0, Vxx0, jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(True))
        (Vx, Vxx, dV1, dV2, ok), (k, K, conds, qu_norms) = jax.lax.scan(
            bstep, init, (A, B, phi_x, phi_u, Pxx, Puu, Pux, useq), reverse=True)
        return k, K, dV1, dV2, ok, conds.max(), qu_norms.max()

    # ---- the iLQR loop ----

    def solve(t0, qpos0, qvel0, u_init, lam0):
        H = u_init.shape[0]
        ts = jnp.arange(H)
        is_last = (ts == H - 1).astype(u_init.dtype)
        u0 = jnp.clip(u_init, u_min_j, u_max_j)
        J0, qs, vs, sds, z_pre, zf = _rollout(t0, qpos0, qvel0, u0)

        def pre_states(qs, vs):
            return (jnp.concatenate([qpos0[None], qs[:-1]], axis=0),
                    jnp.concatenate([qvel0[None], vs[:-1]], axis=0))

        q_pre0, v_pre0 = pre_states(qs, vs)
        lin0 = _lin_all(q_pre0, v_pre0, u0, qs, vs, sds, ts, is_last)

        def iteration(carry, _):
            useq, qs, vs, sds, z_pre, zf, J, lam, lin, need_lin = carry
            q_pre, v_pre = pre_states(qs, vs)
            lin = jax.lax.cond(
                need_lin,
                lambda _lin: _lin_all(q_pre, v_pre, useq, qs, vs, sds, ts, is_last),
                lambda _lin: _lin,
                lin)

            k, K, dV1, dV2, bwd_ok, cond_max, qu_max = _backward(lin, zf, lam, useq)

            Ja, ua, qa, va, sda, zpa, zfa = _forward_sweep(
                t0, qpos0, qvel0, useq, q_pre, v_pre, z_pre, k, K, alphas_j)
            red = J - Ja
            expected = -(alphas_j * dV1 + alphas_j ** 2 * dV2)
            ok_a = (jnp.isfinite(Ja) & (red > 0.0)
                    & (red > accept_ratio * jnp.maximum(expected, 0.0)))
            idx = jnp.argmax(ok_a)                       # first (largest) accepted alpha
            accept = ok_a.any() & bwd_ok

            useq2 = jnp.where(accept, ua[idx], useq)
            qs2 = jnp.where(accept, qa[idx], qs)
            vs2 = jnp.where(accept, va[idx], vs)
            sds2 = jnp.where(accept, sda[idx], sds)
            z_pre2 = jnp.where(accept, zpa[idx], z_pre)
            zf2 = jnp.where(accept, zfa[idx], zf)
            J2 = jnp.where(accept, Ja[idx], J)
            lam2 = jnp.where(accept,
                             jnp.maximum(lam * lam_dn, lam_min),
                             jnp.minimum(lam * lam_up, lam_max))
            alpha_used = jnp.where(accept, alphas_j[idx], jnp.nan)
            out = (J2, lam, alpha_used, accept, cond_max, qu_max)
            return (useq2, qs2, vs2, sds2, z_pre2, zf2, J2, lam2, lin, accept), out

        carry0 = (u0, qs, vs, sds, z_pre, zf, J0, lam0, lin0, jnp.asarray(False))
        carryF, outs = jax.lax.scan(iteration, carry0, None, length=iterations)
        useqF, _, _, _, _, _, JF, lamF, _, _ = carryF
        J_iter, lam_tr, alpha_tr, acc_tr, cond_tr, qu_tr = outs
        stats = dict(
            cost_curve=jnp.concatenate([J0[None], J_iter]),
            lambda_trace=lam_tr,
            alpha_trace=alpha_tr,
            accepted=acc_tr,
            quu_cond=cond_tr,
            qu_max=qu_tr,
            lam_final=lamF,
        )
        return useqF, JF, stats

    return SimpleNamespace(solve=solve, open_loop_cost=open_loop_cost,
                           n_acc=int(n_acc), alphas=np.asarray(alphas_j))


class ILQRMPC(GradientMPC):
    """Single-plan iLQR MPC on an MJX backend, registered at ("gradient", "ilqg").

    Uses GradientMPC's seam: everything except `_build_plan_fn` / `act` (spline basis,
    warm-start shift, ctor validation, cost mirrors) is inherited, so an iLQR plan and
    an MPPI/GradientMPC plan are interchangeable mid-experiment.

    `iterations` (run.iterations) is the number of iLQR iterations per act(). The
    Levenberg λ persists across MPC steps (warm-started MPC reuses last solve's λ);
    `reset()` restores `lam_init`.

    Default spline_type is "linear": cubic overshoots under the W^+ projection and
    needs a post-projection clip — the clipping-breaks-the-change-of-measure confound
    again (NEXT_STEPS S10).
    """

    def __init__(self, task, backend, *,
                 lam_init: float = 1.0,
                 lam_up: float = 10.0,
                 lam_dn: float = 0.5,
                 lam_min: float = 1e-6,
                 lam_max: float = 1e8,
                 n_alphas: int = 8,
                 pn_iterations: int = 6,
                 accept_ratio: float = 1e-4,
                 warmup: bool = True,
                 **kwargs):
        self.lam_init = float(lam_init)
        self.lam_up = float(lam_up)
        self.lam_dn = float(lam_dn)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)
        self.n_alphas = int(n_alphas)
        self.pn_iterations = int(pn_iterations)
        self.accept_ratio = float(accept_ratio)
        self._lam = float(lam_init)
        kwargs.setdefault("spline_type", "linear")
        super().__init__(task, backend, warmup=False, **kwargs)

        # instrumentation surface (NEXT_STEPS S10: from the first commit)
        self.last_cost_curve: Optional[np.ndarray] = None      # (iterations+1,)
        self.last_lambda_trace: Optional[np.ndarray] = None    # (iterations,)
        self.last_alpha_trace: Optional[np.ndarray] = None     # (iterations,) nan=rejected
        self.last_accepted: Optional[np.ndarray] = None        # (iterations,) bool
        self.last_quu_cond: Optional[np.ndarray] = None        # (iterations,) max over t
        self.last_qu_max: Optional[np.ndarray] = None          # (iterations,)
        self.last_accept_rate: float = float("nan")
        # NOTE the name: run_episode interprets `last_alpha` as ComposedGradientMPPI's
        # per-objective composition-weight VECTOR; a scalar there crashes its stacking.
        self.last_alpha_used: float = float("nan")             # last accepted alpha
        self.last_J_pre: float = float("nan")                  # J(u*)  (diagnostic)
        self.last_J_post: float = float("nan")                 # J(W W^+ u*) (headline)
        self.last_proj_resid: float = float("nan")             # ||(I-WW^+)u*|| / ||u*||

        if warmup:
            tic = _time.perf_counter()
            dummy = np.zeros(backend.nstate, dtype=np.float64)
            out = self._plan(dummy[0], dummy[backend.qpos_slice],
                             dummy[backend.qvel_slice],
                             self._jnp.asarray(self.mean),
                             self._jnp.asarray(self._lam))
            self._jax.block_until_ready(out)
            self.warmup_s = _time.perf_counter() - tic

    # ---- planner ----

    def _build_plan_fn(self):
        jax, jnp = self._jax, self._jnp
        from analytic_mppi.dynamics import mjx_manifold as manifold
        from .accumulator import build_augmented_objective, jax_ops

        task = self.task
        backend = self.backend
        model = manifold.linearization_model(backend.model)
        costs = self._costs
        mode = self.mode

        q_s = jax.ShapeDtypeStruct((task.nq,), jnp.float32)
        v_s = jax.ShapeDtypeStruct((task.nv,), jnp.float32)
        s_s = jax.ShapeDtypeStruct((task.nsensordata,), jnp.float32)
        u_s = jax.ShapeDtypeStruct((task.nu,), jnp.float32)
        if mode == "normal":
            n_run = jax.eval_shape(costs.running_cost_terms, q_s, v_s, s_s, u_s).shape[-1]
            n_term = jax.eval_shape(costs.terminal_cost_terms, q_s, v_s, s_s).shape[-1]
            run_perm = term_keep = None
        else:
            n_run = jax.eval_shape(costs.running_cost_terms_f, q_s, v_s, s_s, u_s).shape[-1]
            n_term = jax.eval_shape(costs.terminal_cost_terms_f, q_s, v_s, s_s).shape[-1]
            if self.fpl_term_indices is not None:
                with_term = [i for i in self.fpl_term_indices if i < n_term]
                without = [i for i in self.fpl_term_indices if i >= n_term]
                run_perm = with_term + without
                term_keep = with_term
                n_run, n_term = len(run_perm), len(term_keep)
            else:
                run_perm = term_keep = None

        obj = build_augmented_objective(
            ops=jax_ops(), mode=mode, H=self.H, n_run=n_run, n_term=n_term,
            p=self.fpl_p, gamma=self.fpl_gamma, time_p=self.fpl_time_p,
            time_discount=self.fpl_time_discount,
            terminal_value=self.fpl_terminal_value,
            weights=self.fpl_weights, dt=float(backend.dt))

        floor = self._scoring.floor_atoms
        atom_floor = self.fpl_atom_floor
        n_acc = obj.n_acc

        if mode == "normal":
            def stage_fn(t, qpos, qvel, sd, u):
                return obj.stage(t, terms=costs.running_cost_terms(qpos, qvel, sd, u))

            def fold_fn(qpos, qvel, sd):
                return obj.fold_terminal(
                    jnp.zeros(n_acc), terms=costs.terminal_cost_terms(qpos, qvel, sd))
        else:
            def stage_fn(t, qpos, qvel, sd, u):
                rf = floor(costs.running_cost_terms_f(qpos, qvel, sd, u), atom_floor)
                if run_perm is not None:
                    rf = rf[..., run_perm]
                return obj.stage(t, terms_f=rf)

            def fold_fn(qpos, qvel, sd):
                tf = floor(costs.terminal_cost_terms_f(qpos, qvel, sd), atom_floor)
                if term_keep is not None:
                    tf = tf[..., term_keep]
                return obj.fold_terminal(jnp.zeros(n_acc), terms_f=tf)

        solver = make_ilqr_solver(
            model, u_min=task.u_min, u_max=task.u_max, n_acc=n_acc,
            stage_fn=stage_fn, fold_fn=fold_fn, readout_fn=obj.readout,
            iterations=self.iterations,
            alphas=0.5 ** np.arange(self.n_alphas),
            pn_iterations=self.pn_iterations, accept_ratio=self.accept_ratio,
            lam_up=self.lam_up, lam_dn=self.lam_dn,
            lam_min=self.lam_min, lam_max=self.lam_max)
        self._solver = solver

        W_j = self._W_j
        Wp_j = jnp.asarray(np.linalg.pinv(self._W))
        u_min_j, u_max_j = self._u_min, self._u_max

        def plan_fn(t0, qpos0, qvel0, knots0, lam0):
            u0 = jnp.clip(jnp.einsum("hk,ku->hu", W_j, knots0), u_min_j, u_max_j)
            useq, J_pre, stats = solver.solve(t0, qpos0, qvel0, u0, lam0)
            knots = jnp.einsum("kh,hu->ku", Wp_j, useq)
            u_fit = jnp.einsum("hk,ku->hu", W_j, knots)
            resid = (jnp.linalg.norm(useq - u_fit)
                     / jnp.maximum(jnp.linalg.norm(useq), 1e-12))
            knots = jnp.clip(knots, u_min_j, u_max_j)
            u_exec = jnp.einsum("hk,ku->hu", W_j, knots)
            J_post = solver.open_loop_cost(t0, qpos0, qvel0, u_exec)
            stats = dict(stats, J_pre=J_pre, J_post=J_post, proj_resid=resid)
            return knots, stats

        return plan_fn

    # ---- protocol surface ----

    def reset(self):
        super().reset()
        self._lam = self.lam_init

    def act(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        b = self.backend
        knots, stats = self._plan(state[0], state[b.qpos_slice], state[b.qvel_slice],
                                  self._jnp.asarray(self.mean),
                                  self._jnp.asarray(self._lam))
        stats = {key: np.asarray(self._jax.device_get(val))
                 for key, val in stats.items()}
        self.mean = np.asarray(self._jax.device_get(knots), dtype=np.float64)

        self._lam = float(stats["lam_final"])
        self.last_cost_curve = stats["cost_curve"].astype(np.float64)
        self.last_loss_curve = self.last_cost_curve[1:].copy()
        self.last_lambda_trace = stats["lambda_trace"].astype(np.float64)
        self.last_alpha_trace = stats["alpha_trace"].astype(np.float64)
        self.last_accepted = stats["accepted"].astype(bool)
        self.last_quu_cond = stats["quu_cond"].astype(np.float64)
        self.last_qu_max = stats["qu_max"].astype(np.float64)
        self.last_accept_rate = float(self.last_accepted.mean())
        acc = self.last_alpha_trace[self.last_accepted]
        self.last_alpha_used = float(acc[-1]) if acc.size else float("nan")
        self.last_J_pre = float(stats["J_pre"])
        self.last_J_post = float(stats["J_post"])
        self.last_proj_resid = float(stats["proj_resid"])

        u0 = self.mean[0].copy()
        self.mean, self._shift_accum = shift_plan(
            self.mean, self.tk, float(b.dt), self.spline_type, self._shift_accum)
        return u0


class WarmStartedILQR:
    """S11 — the MPPI -> iLQR architecture: a sampler explores, iLQR polishes its mean.

    Mirrors cost_gd.py's composition-over-inheritance: takes ANY SamplingController
    instance plus an ILQRMPC and runs, per MPC step,

      1. the sampler's full inner update loop (its `iterations` x rollout/score/update),
         WITHOUT its trailing warm-start shift,
      2. iLQR from the sampler's post-update mean as the nominal,
      3. executes the iLQR plan's first action; both plans are then shifted identically
         (the iLQR shift is copied back to the sampler, so next step it explores around
         the POLISHED plan rather than drifting from what actually executed).

    The two controllers must share the knot grid (num_knots, plan_horizon, spline_type)
    — the mean is transferred in knot space. This is what makes contact tasks feasible
    for iLQR at all: the sampler supplies a nominal on the right side of contact-mode
    boundaries, and iLQR's line search only has to refine it.
    """

    def __init__(self, sampler, ilqr: ILQRMPC):
        if not hasattr(sampler, "_rollout_and_score"):
            raise TypeError(f"expected a SamplingController, got {type(sampler).__name__}")
        if not isinstance(ilqr, ILQRMPC):
            raise TypeError(f"expected an ILQRMPC, got {type(ilqr).__name__}")
        if sampler.num_knots != ilqr.num_knots or sampler.spline_type != ilqr.spline_type \
                or not np.allclose(sampler.tk, ilqr.tk):
            raise ValueError(
                f"sampler and iLQR must share the knot grid to transfer the mean plan: "
                f"knots {sampler.num_knots} vs {ilqr.num_knots}, spline "
                f"{sampler.spline_type!r} vs {ilqr.spline_type!r}, horizon "
                f"{sampler.plan_horizon} vs {ilqr.plan_horizon}")
        if sampler.nu != ilqr.nu:
            raise ValueError("sampler and iLQR are built on different tasks")
        self.sampler = sampler
        self.ilqr = ilqr
        self.nu = ilqr.nu
        self.num_samples = sampler.num_samples
        self.num_knots = ilqr.num_knots
        self.plan_horizon = ilqr.plan_horizon
        self.spline_type = ilqr.spline_type

    def reset(self):
        self.sampler.reset()
        self.ilqr.reset()

    # run.py / eval diagnostics passthrough
    @property
    def last_trajectory(self):
        return self.sampler.last_trajectory

    @property
    def last_ess(self):
        return getattr(self.sampler, "last_ess", float("nan"))

    @property
    def mean(self):
        return self.ilqr.mean

    def act(self, state: np.ndarray) -> np.ndarray:
        s = self.sampler
        for _ in range(s.iterations):
            traj = s._rollout_and_score(state)
            s.mean = s.update_mean(traj)
            s.last_trajectory = traj
        self.ilqr.mean = s.mean.copy()             # nominal = sampler's mean plan
        u0 = self.ilqr.act(state)                  # solve + shift (ilqr.mean now shifted)
        s.mean = self.ilqr.mean.copy()             # explore around the polished plan
        s._shift_accum = self.ilqr._shift_accum
        return u0
