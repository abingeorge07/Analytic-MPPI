"""Cost-gradient refinement of sampling-MPC rollouts (BPTT through MuJoCo FD).

Ported from `analysis.ipynb` Part 1d. `wrap_controller_with_gd_refine(ctrl, ...)`
adds a `_costgd`-style refinement to any `SamplingController` *instance*: each MPC
step it samples + rolls out as usual, then takes a few gradient-descent steps on
the total trajectory cost of the top-`num_refine` rollouts. The gradient is
obtained by reverse-mode BPTT through per-step Jacobians from
`mujoco.mjd_transitionFD`, with the cost-aggregation gradient computed
analytically (closed-form for `normal` / `fpl_cost` / `fpl_discounted`) and the
inner cost-term Jacobians via per-step finite differences.

This is *instance-level* wrapping (it replaces `ctrl._rollout_and_score` on the
one object), not a global monkey-patch — pass a freshly built controller in.

Free-joint models (`nq != nv`, e.g. cube / g1_standup) are supported: the cost-
and sensor-term position gradients are differentiated in *tangent* space (width
`nv`) via `mujoco.mj_integratePos`, so they line up with `mjd_transitionFD`'s
velocity-space `A` / `B` Jacobians. For `nq == nv` (pendulum / walker / hopper)
this reduces exactly to plain componentwise qpos finite differences.

Note:
  * `gd_iterations == 0` short-circuits the refinement and reproduces the
    unwrapped baseline bit-for-bit (the seed-stable equivalence check).
  * Free-joint refinement is slower (per-state `mj_integratePos` / `mj_forward`
    for the tangent FD); pass `use_sensor_jac_from_FD=True` to take the sensor
    Jacobian from `mjd_transitionFD`'s `C` matrix and skip the separate pass.
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import mujoco

from .sampling_base import Trajectory
from .spline import interpolate


def _integrate_pos(model, qpos_flat, tan_flat):
    """Apply tangent-space deltas to a batch of qpos via the configuration-manifold
    exponential map (`mujoco.mj_integratePos`, dt=1).

    qpos_flat (N, nq), tan_flat (N, nv)  ->  (N, nq). For hinge/slide joints this is
    plain addition; for quaternion (free/ball) joints it is the proper quaternion
    integration — the *same* map `mjd_transitionFD` uses internally, so a tangent FD
    built on it is consistent with the velocity-space A/B Jacobians by construction.
    """
    N, nq = qpos_flat.shape
    out = np.empty((N, nq), dtype=np.float64)
    qp = np.empty(nq, dtype=np.float64)
    for n in range(N):
        qp[:] = qpos_flat[n]
        mujoco.mj_integratePos(model, qp, np.ascontiguousarray(tan_flat[n], dtype=np.float64), 1.0)
        out[n] = qp
    return out


# ---------------------------------------------------------------------------
#  Per-step Jacobians + sensor Jacobian
# ---------------------------------------------------------------------------

def rollout_with_jacobians(backend, initial, controls, fd_eps=1e-6, flg_centered=1,
                           use_sensor_jac_from_FD=False):
    """Stepped rollout collecting per-step Jacobians via mujoco.mjd_transitionFD.

    Returns states (M,H,nstate), sensordata (M,H,ns_sensor),
    A_seq (M,H,ns_v,ns_v), B_seq (M,H,ns_v,nu), and optionally
    C_seq (M,H,ns_sensor,ns_v), D_seq (M,H,ns_sensor,nu); else those are None.
    ns_v = 2*nv + na (mjd_transitionFD's velocity-space layout).
    """
    model = backend.model
    M, H, nu = controls.shape
    nstate = backend.nstate
    nv = int(model.nv)
    na = int(model.na)
    ns_v = 2 * nv + na
    ns_sensor = backend.nsensordata

    states = np.empty((M, H, nstate), dtype=np.float64)
    sensordata = np.empty((M, H, ns_sensor), dtype=np.float64)
    A_seq = np.empty((M, H, ns_v, ns_v), dtype=np.float64)
    B_seq = np.empty((M, H, ns_v, nu), dtype=np.float64)
    C_seq = np.empty((M, H, ns_sensor, ns_v), dtype=np.float64) if use_sensor_jac_from_FD else None
    D_seq = np.empty((M, H, ns_sensor, nu), dtype=np.float64) if use_sensor_jac_from_FD else None

    A_buf = np.empty((ns_v, ns_v), dtype=np.float64)
    B_buf = np.empty((ns_v, nu), dtype=np.float64)
    C_buf = np.empty((ns_sensor, ns_v), dtype=np.float64) if use_sensor_jac_from_FD else None
    D_buf = np.empty((ns_sensor, nu), dtype=np.float64) if use_sensor_jac_from_FD else None

    state_spec_int = int(backend.state_spec)
    data = mujoco.MjData(model)

    for m in range(M):
        mujoco.mj_setState(model, data, initial[m], state_spec_int)
        mujoco.mj_forward(model, data)
        for t in range(H):
            data.ctrl[:] = controls[m, t]
            mujoco.mjd_transitionFD(model, data, fd_eps, flg_centered, A_buf, B_buf, C_buf, D_buf)
            A_seq[m, t] = A_buf
            B_seq[m, t] = B_buf
            if use_sensor_jac_from_FD:
                C_seq[m, t] = C_buf
                D_seq[m, t] = D_buf
            mujoco.mj_step(model, data)
            mujoco.mj_getState(model, data, states[m, t], state_spec_int)
            sensordata[m, t] = data.sensordata

    return states, sensordata, A_seq, B_seq, C_seq, D_seq


def sensor_jac_via_forward(backend, states_full, fd_eps=1e-6):
    """Per-step sensor Jacobian d(sensor)/d(position, qvel) via mj_forward + FD.

    Returns (dsens_dpos, dsens_dqvel) each of width `nv` so both align with
    `mjd_transitionFD`'s velocity-space layout. The position block is differentiated
    in **tangent space** via `mj_integratePos` — for `nq == nv` (no quaternion joints)
    this is identical to plain componentwise qpos FD; for free/ball joints it is the
    correct manifold derivative. Costs ~M*H*2*nv mj_forward calls.
    """
    model = backend.model
    M, H, nstate = states_full.shape
    nq = int(model.nq)
    nv = int(model.nv)
    ns = backend.nsensordata
    state_spec_int = int(backend.state_spec)
    qpos_start = backend.qpos_slice.start
    qvel_start = backend.qvel_slice.start

    dsens_dpos = np.empty((M, H, ns, nv), dtype=np.float64)
    dsens_dqvel = np.empty((M, H, ns, nv), dtype=np.float64)
    quaternion_joints = (nq != nv)

    data = mujoco.MjData(model)
    qp = np.empty(nq, dtype=np.float64)
    tan = np.zeros(nv, dtype=np.float64)

    for m in range(M):
        for t in range(H):
            state_ref = states_full[m, t]
            qpos_ref = state_ref[qpos_start:qpos_start + nq]
            for j in range(nv):
                # tangent-space perturbation of the position via the manifold exp map
                if quaternion_joints:
                    tan[:] = 0.0; tan[j] = fd_eps
                    sp = state_ref.copy()
                    qp[:] = qpos_ref; mujoco.mj_integratePos(model, qp, tan, 1.0)
                    sp[qpos_start:qpos_start + nq] = qp
                    sm = state_ref.copy()
                    qp[:] = qpos_ref; mujoco.mj_integratePos(model, qp, tan, -1.0)
                    sm[qpos_start:qpos_start + nq] = qp
                else:
                    sp = state_ref.copy(); sp[qpos_start + j] += fd_eps
                    sm = state_ref.copy(); sm[qpos_start + j] -= fd_eps
                mujoco.mj_setState(model, data, sp, state_spec_int); mujoco.mj_forward(model, data)
                splus = data.sensordata.copy()
                mujoco.mj_setState(model, data, sm, state_spec_int); mujoco.mj_forward(model, data)
                sminus = data.sensordata.copy()
                dsens_dpos[m, t, :, j] = (splus - sminus) / (2 * fd_eps)
            for j in range(nv):
                sp = state_ref.copy(); sp[qvel_start + j] += fd_eps
                sm = state_ref.copy(); sm[qvel_start + j] -= fd_eps
                mujoco.mj_setState(model, data, sp, state_spec_int); mujoco.mj_forward(model, data)
                splus = data.sensordata.copy()
                mujoco.mj_setState(model, data, sm, state_spec_int); mujoco.mj_forward(model, data)
                sminus = data.sensordata.copy()
                dsens_dqvel[m, t, :, j] = (splus - sminus) / (2 * fd_eps)
    return dsens_dpos, dsens_dqvel


# ---------------------------------------------------------------------------
#  Cost aggregation gradient (closed-form per mode) + inner FD on cost terms
# ---------------------------------------------------------------------------

def _power_mean_and_grad(terms, p, eps=1e-8):
    """Power mean over the last axis and its gradient. Gradient is zero where the
    original term was below eps (clipping creates a flat region)."""
    t = np.clip(terms, eps, None)
    n = t.shape[-1]
    if p == 0:
        pm = np.exp(np.mean(np.log(t), axis=-1))
        d_pm_d_t = pm[..., None] / (n * t)
    else:
        avg = np.sum(t ** p, axis=-1) / n
        pm = avg ** (1.0 / p)
        d_pm_d_t = (1.0 / n) * (pm[..., None] ** (1.0 - p)) * (t ** (p - 1.0))
    mask = (terms >= eps).astype(np.float64)
    return pm, d_pm_d_t * mask


def _score_grad_normal(rt, tt, dt):
    """J = rt.sum * dt + tt.sum. Returns (J (M,), dJ_drt (M,H,n_run), dJ_dtt (M,n_term))."""
    J = rt.sum(axis=(1, 2)) * dt + tt.sum(axis=-1)
    return J, np.full_like(rt, dt), np.ones_like(tt)


def _score_grad_fpl_cost(rt_f, tt_f, p, gamma):
    """Mirrors sampling_base._score_fpl (use_fpl_cost) plus analytic gradient. J = -reward."""
    M, H, n_run = rt_f.shape
    n_term = tt_f.shape[-1]
    H1 = H + 1
    discounts_full = gamma ** np.arange(H1, dtype=np.float64)
    norm_full = (1.0 - gamma) / (1.0 - gamma ** H1)
    discounts_run = gamma ** np.arange(H, dtype=np.float64)
    norm_run = (1.0 - gamma) / (1.0 - gamma ** H) if H > 1 else 1.0

    per_step_run, d_per_step_run_d_rt = _power_mean_and_grad(rt_f, p)

    if n_term > 0:
        per_step_term, d_per_step_term_d_tt = _power_mean_and_grad(tt_f, p)
        per_step = np.concatenate([per_step_run, per_step_term[:, None]], axis=1)
        reward = (per_step * discounts_full[None, :]).sum(axis=1) * norm_full
        d_reward_d_rt = (discounts_full[:H] * norm_full)[None, :, None] * d_per_step_run_d_rt
        d_reward_d_tt = (discounts_full[H] * norm_full) * d_per_step_term_d_tt
    else:
        reward = (per_step_run * discounts_run[None, :]).sum(axis=1) * norm_run
        d_reward_d_rt = (discounts_run * norm_run)[None, :, None] * d_per_step_run_d_rt
        d_reward_d_tt = np.zeros((M, 0), dtype=np.float64)
    return -reward, -d_reward_d_rt, -d_reward_d_tt


def _score_grad_fpl_discounted(rt_f, tt_f, p, gamma):
    """Mirrors sampling_base._score_fpl (use_fpl_discounted) plus analytic gradient."""
    M, H, n_run = rt_f.shape
    n_term = tt_f.shape[-1]
    H1 = H + 1
    discounts_full = gamma ** np.arange(H1, dtype=np.float64)
    norm_full = (1.0 - gamma) / (1.0 - gamma ** H1)
    discounts_run = gamma ** np.arange(H, dtype=np.float64)
    norm_run = (1.0 - gamma) / (1.0 - gamma ** H) if H > 1 else 1.0

    chunks = []
    if n_term > 0:
        per_term_sum_a = (
            (rt_f[..., :n_term] * discounts_full[:H, None]).sum(axis=1)
            + tt_f * discounts_full[H]
        ) * norm_full
        chunks.append(per_term_sum_a)
    if n_run > n_term:
        per_term_sum_b = (rt_f[..., n_term:] * discounts_run[:, None]).sum(axis=1) * norm_run
        chunks.append(per_term_sum_b)
    per_term_sums = np.concatenate(chunks, axis=-1)
    reward, d_reward_d_per_term = _power_mean_and_grad(per_term_sums, p)

    d_reward_d_rt = np.zeros((M, H, n_run), dtype=np.float64)
    d_reward_d_tt = np.zeros((M, n_term), dtype=np.float64)
    if n_term > 0:
        d_reward_d_rt[..., :n_term] = (
            d_reward_d_per_term[:, None, :n_term] * (discounts_full[:H] * norm_full)[None, :, None]
        )
        d_reward_d_tt = d_reward_d_per_term[:, :n_term] * (discounts_full[H] * norm_full)
    if n_run > n_term:
        d_reward_d_rt[..., n_term:] = (
            d_reward_d_per_term[:, None, n_term:] * (discounts_run * norm_run)[None, :, None]
        )
    return -reward, -d_reward_d_rt, -d_reward_d_tt


def cost_and_grad(qpos, qvel, sensordata, controls, task, cost_mode, dt, fpl_p, fpl_gamma,
                  fd_eps=1e-6):
    """Total cost J and its gradients w.r.t. qpos / qvel / sensordata / controls
    (per-step + terminal). Per-step FD on the cost terms, analytic chain through
    the aggregation. Returns
    (J, dJ_dqpos, dJ_dqvel, dJ_dsens, dJ_du, dJ_dqpos_T, dJ_dqvel_T, dJ_dsens_T).
    """
    M, H = qpos.shape[:2]

    if cost_mode == "normal":
        rt = task.running_cost_terms(qpos, qvel, sensordata, controls)
        tt = task.terminal_cost_terms(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
        get_rt = task.running_cost_terms
        get_tt = task.terminal_cost_terms
        J, dJ_drt, dJ_dtt = _score_grad_normal(rt, tt, dt)
    elif cost_mode == "fpl_cost":
        rt = task.running_cost_terms_f(qpos, qvel, sensordata, controls)
        tt = task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
        get_rt = task.running_cost_terms_f
        get_tt = task.terminal_cost_terms_f
        J, dJ_drt, dJ_dtt = _score_grad_fpl_cost(rt, tt, fpl_p, fpl_gamma)
    elif cost_mode == "fpl_discounted":
        rt = task.running_cost_terms_f(qpos, qvel, sensordata, controls)
        tt = task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sensordata[:, -1])
        get_rt = task.running_cost_terms_f
        get_tt = task.terminal_cost_terms_f
        J, dJ_drt, dJ_dtt = _score_grad_fpl_discounted(rt, tt, fpl_p, fpl_gamma)
    else:
        raise ValueError(f"Unknown cost_mode: {cost_mode}")

    # ---- Per-step FD on running terms ----
    def _fd_running(var_arr, var_name):
        D = var_arr.shape[-1]
        n_rt = rt.shape[-1]
        jac = np.empty((M, H, D, n_rt), dtype=np.float64)
        for j in range(D):
            vp = var_arr.copy(); vp[..., j] += fd_eps
            vm = var_arr.copy(); vm[..., j] -= fd_eps
            if var_name == 'qpos':
                rp, rm = get_rt(vp, qvel, sensordata, controls), get_rt(vm, qvel, sensordata, controls)
            elif var_name == 'qvel':
                rp, rm = get_rt(qpos, vp, sensordata, controls), get_rt(qpos, vm, sensordata, controls)
            elif var_name == 'sensordata':
                rp, rm = get_rt(qpos, qvel, vp, controls), get_rt(qpos, qvel, vm, controls)
            elif var_name == 'controls':
                rp, rm = get_rt(qpos, qvel, sensordata, vp), get_rt(qpos, qvel, sensordata, vm)
            jac[..., j, :] = (rp - rm) / (2 * fd_eps)
        return jac

    # For free-joint models (nq != nv), differentiate the cost w.r.t. qpos in
    # TANGENT space (width nv) via mj_integratePos, so it aligns with the
    # velocity-space A/B Jacobians. For nq == nv this is identical to the plain
    # componentwise qpos FD, so we keep that (faster, vectorized) path.
    nv = qvel.shape[-1]
    nq = qpos.shape[-1]
    if nq != nv:
        model = task.mj_model
        n_rt = rt.shape[-1]
        qpos_flat = qpos.reshape(M * H, nq)
        drt_dq = np.empty((M, H, nv, n_rt), dtype=np.float64)
        for i in range(nv):
            tan = np.zeros((M * H, nv), dtype=np.float64); tan[:, i] = fd_eps
            qp_p = _integrate_pos(model, qpos_flat, tan).reshape(M, H, nq)
            qp_m = _integrate_pos(model, qpos_flat, -tan).reshape(M, H, nq)
            rp = get_rt(qp_p, qvel, sensordata, controls)
            rm = get_rt(qp_m, qvel, sensordata, controls)
            drt_dq[..., i, :] = (rp - rm) / (2 * fd_eps)
    else:
        drt_dq = _fd_running(qpos, 'qpos')
    drt_dv = _fd_running(qvel, 'qvel')
    drt_ds = _fd_running(sensordata, 'sensordata')
    drt_du = _fd_running(controls, 'controls')

    dJ_dqpos = np.einsum("mhn,mhdn->mhd", dJ_drt, drt_dq)
    dJ_dqvel = np.einsum("mhn,mhdn->mhd", dJ_drt, drt_dv)
    dJ_dsens = np.einsum("mhn,mhdn->mhd", dJ_drt, drt_ds)
    dJ_du = np.einsum("mhn,mhdn->mhd", dJ_drt, drt_du)

    # ---- Per-step FD on terminal terms (last step only) ----
    def _fd_terminal(var_arr_T, var_name):
        D = var_arr_T.shape[-1]
        n_tt = tt.shape[-1]
        jac = np.empty((M, D, n_tt), dtype=np.float64)
        q_T, v_T, s_T = qpos[:, -1], qvel[:, -1], sensordata[:, -1]
        for j in range(D):
            vp = var_arr_T.copy(); vp[..., j] += fd_eps
            vm = var_arr_T.copy(); vm[..., j] -= fd_eps
            if var_name == 'qpos':
                tp, tm = get_tt(vp, v_T, s_T), get_tt(vm, v_T, s_T)
            elif var_name == 'qvel':
                tp, tm = get_tt(q_T, vp, s_T), get_tt(q_T, vm, s_T)
            elif var_name == 'sensordata':
                tp, tm = get_tt(q_T, v_T, vp), get_tt(q_T, v_T, vm)
            jac[..., j, :] = (tp - tm) / (2 * fd_eps)
        return jac

    if nq != nv:
        model = task.mj_model
        n_tt = tt.shape[-1]
        q_T = qpos[:, -1]                                  # (M, nq)
        v_T, s_T = qvel[:, -1], sensordata[:, -1]
        dtt_dq_T = np.empty((M, nv, n_tt), dtype=np.float64)
        for i in range(nv):
            tan = np.zeros((M, nv), dtype=np.float64); tan[:, i] = fd_eps
            tp = get_tt(_integrate_pos(model, q_T, tan), v_T, s_T)
            tm = get_tt(_integrate_pos(model, q_T, -tan), v_T, s_T)
            dtt_dq_T[..., i, :] = (tp - tm) / (2 * fd_eps)
    else:
        dtt_dq_T = _fd_terminal(qpos[:, -1], 'qpos')
    dtt_dv_T = _fd_terminal(qvel[:, -1], 'qvel')
    dtt_ds_T = _fd_terminal(sensordata[:, -1], 'sensordata')

    dJ_dqpos_T = np.einsum("mn,mdn->md", dJ_dtt, dtt_dq_T)
    dJ_dqvel_T = np.einsum("mn,mdn->md", dJ_dtt, dtt_dv_T)
    dJ_dsens_T = np.einsum("mn,mdn->md", dJ_dtt, dtt_ds_T)

    return J, dJ_dqpos, dJ_dqvel, dJ_dsens, dJ_du, dJ_dqpos_T, dJ_dqvel_T, dJ_dsens_T


def bptt_grad_du(A_seq, B_seq, dJ_dstate, dJ_du):
    """Reverse-mode adjoint backprop. A_seq[t]=ds_{t+1}/ds_t, B_seq[t]=ds_{t+1}/du_t.
    Returns grad (M, H, nu)."""
    M, H, ns_v, _ = A_seq.shape
    nu = B_seq.shape[-1]
    grad = np.empty((M, H, nu), dtype=np.float64)
    adj = dJ_dstate[:, -1].copy()
    grad[:, -1] = dJ_du[:, -1] + np.einsum("mji,mj->mi", B_seq[:, -1], adj)
    for t in range(H - 2, -1, -1):
        adj = dJ_dstate[:, t] + np.einsum("mji,mj->mi", A_seq[:, t + 1], adj)
        grad[:, t] = dJ_du[:, t] + np.einsum("mji,mj->mi", B_seq[:, t], adj)
    return grad


# ---------------------------------------------------------------------------
#  Top-mu refinement orchestrator + instance wrapper
# ---------------------------------------------------------------------------

def _knot_basis_matrix(tk, t_eval, spline_type):
    """Phi (H, num_knots): controls = Phi @ knots for unit knots."""
    num_knots = tk.shape[0]
    H = t_eval.shape[0]
    Phi = np.zeros((H, num_knots), dtype=np.float64)
    for k in range(num_knots):
        onehot = np.zeros((1, num_knots, 1), dtype=np.float64)
        onehot[0, k, 0] = 1.0
        Phi[:, k] = interpolate(onehot, tk, t_eval, spline_type)[0, :, 0]
    return Phi


def gd_refine_topmu(backend, task, initial, controls, pre_scores, pre_states, pre_sensordata,
                    cost_mode, dt, fpl_p, fpl_gamma,
                    gd_iterations, gd_lr, num_refine, u_min, u_max,
                    fd_eps=1e-6, flg_centered=1, use_sensor_jac_from_FD=False):
    """Refine the top-`num_refine` rollouts (lowest pre_scores) via BPTT-cost-GD.

    Returns refined_controls (K,H,nu), refined_states (K,H,ns),
    refined_sensor (K,H,nsensordata), elite_idx (num_refine,). Non-elites untouched.
    """
    K, H, nu = controls.shape
    nv = int(backend.model.nv); na = int(backend.model.na)
    # Free-joint models (nq != nv) are handled: the cost/sensor position gradients
    # are taken in tangent space (width nv) so they align with mjd_transitionFD's
    # velocity-space A/B Jacobians. See cost_and_grad / sensor_jac_via_forward.
    ns_v = 2 * nv + na

    M = min(num_refine, K)
    elite_idx = np.argpartition(pre_scores, M - 1)[:M] if M < K else np.arange(K)
    elite_idx = np.sort(elite_idx)

    initial_e = initial[elite_idx].copy()
    controls_e = controls[elite_idx].copy()

    u_min = np.asarray(u_min, dtype=np.float64)
    u_max = np.asarray(u_max, dtype=np.float64)

    for _gd_iter in range(gd_iterations):
        states_e, sensordata_e, A_seq, B_seq, C_seq, D_seq = rollout_with_jacobians(
            backend, initial_e, controls_e,
            fd_eps=fd_eps, flg_centered=flg_centered,
            use_sensor_jac_from_FD=use_sensor_jac_from_FD,
        )
        qpos_e = task.qpos_of(states_e)
        qvel_e = task.qvel_of(states_e)

        (J, dJ_dq, dJ_dv, dJ_ds, dJ_du,
         dJ_dq_T, dJ_dv_T, dJ_ds_T) = cost_and_grad(
            qpos_e, qvel_e, sensordata_e, controls_e,
            task, cost_mode, dt, fpl_p, fpl_gamma, fd_eps=fd_eps,
        )

        dJ_dq = dJ_dq.copy(); dJ_dv = dJ_dv.copy(); dJ_ds = dJ_ds.copy()
        dJ_dq[:, -1] += dJ_dq_T
        dJ_dv[:, -1] += dJ_dv_T
        dJ_ds[:, -1] += dJ_ds_T

        if np.any(dJ_ds):
            if use_sensor_jac_from_FD and C_seq is not None:
                dsens_dq = C_seq[..., :nv]
                dsens_dv = C_seq[..., nv:2 * nv]
            else:
                dsens_dq, dsens_dv = sensor_jac_via_forward(backend, states_e, fd_eps=fd_eps)
            dJ_dq = dJ_dq + np.einsum("mhs,mhsj->mhj", dJ_ds, dsens_dq)
            dJ_dv = dJ_dv + np.einsum("mhs,mhsj->mhj", dJ_ds, dsens_dv)

        dJ_dstate = np.zeros((M, H, ns_v), dtype=np.float64)
        dJ_dstate[..., :nv] = dJ_dq
        dJ_dstate[..., nv:2 * nv] = dJ_dv

        grad = bptt_grad_du(A_seq, B_seq, dJ_dstate, dJ_du)
        controls_e = np.clip(controls_e - gd_lr * grad, u_min, u_max)

    final_states_e, final_sensor_e = backend.rollout(initial_e, controls_e)

    refined_controls = controls.copy()
    refined_states = pre_states.copy()
    refined_sensor = pre_sensordata.copy()
    refined_controls[elite_idx] = controls_e
    refined_states[elite_idx] = final_states_e
    refined_sensor[elite_idx] = final_sensor_e

    return refined_controls, refined_states, refined_sensor, elite_idx


def wrap_controller_with_gd_refine(ctrl, gd_iterations, gd_lr, num_refine=None,
                                   fd_eps=1e-6, flg_centered=1,
                                   use_sensor_jac_from_FD=False, refit_knots=True):
    """Add cost-GD refinement of the top-mu rollouts to a controller *instance* by
    replacing its `_rollout_and_score`. Returns the same (mutated) controller.

    `gd_iterations == 0` reproduces the unwrapped baseline bit-for-bit.
    `num_refine` defaults to the controller's `num_elites` (or min(8, num_samples)).
    """
    if num_refine is None:
        num_refine_eff = (ctrl.num_elites if (hasattr(ctrl, "num_elites") and ctrl.num_elites is not None)
                          else min(8, ctrl.num_samples))
    else:
        num_refine_eff = int(num_refine)
    num_refine_eff = max(1, min(num_refine_eff, ctrl.num_samples))

    if ctrl.use_fpl_cost:
        cost_mode = "fpl_cost"
    elif ctrl.use_fpl_discounted:
        cost_mode = "fpl_discounted"
    else:
        cost_mode = "normal"

    Phi = None
    if refit_knots and ctrl.num_knots != ctrl.H:
        Phi = _knot_basis_matrix(ctrl.tk, ctrl.t_eval, ctrl.spline_type)

    def wrapped_rollout_and_score(self, state):
        K, H = self.num_samples, self.H

        knots = self.sample_knots()
        knots = np.clip(knots, self.task.u_min, self.task.u_max)
        controls = interpolate(knots, self.tk, self.t_eval, self.spline_type)

        initial = np.broadcast_to(state, (K, self.backend.nstate)).copy()
        pre_states, pre_sensordata = self.backend.rollout(initial, controls)

        if gd_iterations > 0:
            qpos_pre = self.task.qpos_of(pre_states)
            qvel_pre = self.task.qvel_of(pre_states)
            traj_pre = Trajectory(
                knots=knots, controls=controls, states=pre_states, sensordata=pre_sensordata,
                qpos=qpos_pre, qvel=qvel_pre,
            )
            if self.use_fpl_cost or self.use_fpl_discounted:
                traj_pre.running_terms_f = self.task.running_cost_terms_f(qpos_pre, qvel_pre, pre_sensordata, controls)
                traj_pre.terminal_terms_f = self.task.terminal_cost_terms_f(qpos_pre[:, -1], qvel_pre[:, -1], pre_sensordata[:, -1])
                pre_scores = self._score_fpl(traj_pre)
            else:
                traj_pre.running_terms = self.task.running_cost_terms(qpos_pre, qvel_pre, pre_sensordata, controls)
                traj_pre.terminal_terms = self.task.terminal_cost_terms(qpos_pre[:, -1], qvel_pre[:, -1], pre_sensordata[:, -1])
                pre_scores = self._score_normal(traj_pre)

            refined_controls, refined_states, refined_sensor, elite_idx = gd_refine_topmu(
                self.backend, self.task, initial, controls,
                pre_scores, pre_states, pre_sensordata,
                cost_mode, self.backend.dt, self.fpl_p, self.fpl_gamma,
                gd_iterations, gd_lr, num_refine_eff,
                self.task.u_min, self.task.u_max,
                fd_eps=fd_eps, flg_centered=flg_centered,
                use_sensor_jac_from_FD=use_sensor_jac_from_FD,
            )
            controls = refined_controls
            states = refined_states
            sensordata = refined_sensor

            if refit_knots and elite_idx.size > 0:
                if self.num_knots == self.H:
                    knots = knots.copy()
                    knots[elite_idx] = controls[elite_idx]
                elif Phi is not None:
                    knots = knots.copy()
                    M_e = elite_idx.shape[0]
                    nu = controls.shape[-1]
                    c_e = controls[elite_idx]
                    rhs = c_e.transpose(1, 0, 2).reshape(self.H, M_e * nu)
                    sol, *_ = np.linalg.lstsq(Phi, rhs, rcond=None)
                    knots[elite_idx] = sol.reshape(self.num_knots, M_e, nu).transpose(1, 0, 2)
        else:
            states = pre_states
            sensordata = pre_sensordata

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

    ctrl._rollout_and_score = MethodType(wrapped_rollout_and_score, ctrl)
    ctrl._costgd_wrapped = True
    ctrl._costgd_num_refine = num_refine_eff
    ctrl._costgd_iterations = gd_iterations
    return ctrl


__all__ = [
    "rollout_with_jacobians", "sensor_jac_via_forward",
    "cost_and_grad", "bptt_grad_du", "gd_refine_topmu",
    "wrap_controller_with_gd_refine",
]
