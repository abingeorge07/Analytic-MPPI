"""Configuration-manifold exp/log maps for MJX, differentiable AT THE BASE POINT.

Why this module exists
----------------------
`mjd_transitionFD` reports `A`/`B` in MuJoCo's **velocity-space** layout: the state is
`[qpos_tangent (nv), qvel (nv), act (na)]`, not `qpos (nq)`. For any model with a free or
ball joint `nq != nv`, so comparing an autodiff Jacobian of `mjx.step` against it requires
pushing the perturbation on via the manifold exp map and reading the difference off via the
log map -- the same maps `mjd_transitionFD` uses internally (`mj_integratePos` /
`mj_differentiatePos`).

MJX already ships those maps (`math.quat_integrate`, `math.quat_sub`). **They cannot be
differentiated at the point a Jacobian needs them.** Both route through
`normalize_with_norm(v)`, which is `v / (n + 1e-6*(n == 0))` with `n = |v|`:

  * `quat_integrate(q, v, dt)` is evaluated at `v = 0` (zero perturbation),
  * `quat_sub(qa, qb)` is evaluated at `qa == qb` (zero residual),

and at `|v| = 0` that guarded division has **zero derivative**, not the true one. The
symptom is silent and specific: the rotational block of `A` comes back as exactly `0.0`
where it should be the identity, while every translational and hinge block is correct. A
`relFro` error of 0.5 on a free body, and 1e-11 on a pendulum.

The fix is to use maps that agree with the true ones to **first order at the base point**,
which is all a Jacobian is:

    exp:  q (x) exp(v) ~ normalize(q (x) (1, v/2))       d/dv at v=0 is exact
    log:  log(qb^-1 (x) qa) ~ 2 * vec(qb^-1 (x) qa)      d/dqa at qa=qb is exact

For a unit quaternion `(w, u)`, `log = 2*atan2(|u|, w) * u/|u| -> 2u` as `u -> 0`, so the
value and the first derivative both match; only second-order terms differ, and a Jacobian
does not see them. Away from the base point these are approximations -- **use them for
linearization, not for integrating a trajectory.** Use `mj_integratePos` / MJX's own maps
for that.

Not imported by `dynamics/__init__.py`: it imports jax at module scope, and the CPU path
must never pay for the `[mjx]` extra (invariant 11.4).
"""
from __future__ import annotations

import os
from typing import List, Tuple

import mujoco

# Same Triton-gemm guard as MJXBackend.__init__ (mjx_backend.py:47-55), and for the same
# reason: XLA's gemm-fusion autotuner hard-aborts (Non-OK-status, core dump) compiling
# differentiated mjx solver graphs on this stack (jax 0.6.2 / RTX 4070). This module is
# the OTHER entry point that initializes jax for derivative work, and whichever entry
# point initializes jax first decides the flag for the whole process -- so both must set
# it. Must run before the `import jax` below on first import.
_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_gpu_enable_triton_gemm=false").strip()

import jax.numpy as jnp
from mujoco.mjx._src import math as mjxmath

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)


def joint_layout(model) -> List[Tuple[int, int, int]]:
    """[(joint type, qpos address, dof address)] for every joint, in model order."""
    return [(int(model.jnt_type[j]), int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j]))
            for j in range(int(model.njnt))]


def quat_exp_local(q, v):
    """`q (x) exp(v)`, first-order-exact in `v` and differentiable at `v = 0`.

    Mirrors `mujoco.mj_integratePos` on a quaternion joint with dt=1, but replaces
    `axis_angle_to_quat(normalize(v), |v|)` -- whose derivative at `v=0` is zero because
    of the guarded normalize -- with the small-angle form `(1, v/2)`.
    """
    dq = jnp.concatenate([jnp.ones((1,), dtype=v.dtype), 0.5 * v])
    return mjxmath.normalize(mjxmath.quat_mul(q, dq))


def quat_log_local(qa, qb):
    """`log(qb^-1 (x) qa)` as a 3-vector, first-order-exact and differentiable at `qa = qb`.

    Mirrors `mujoco.mj_differentiatePos` on a quaternion joint, without `quat_sub`'s
    singular `quat_to_axis_angle` at the identity rotation.
    """
    rel = mjxmath.quat_mul(mjxmath.quat_inv(qb), qa)
    return 2.0 * rel[1:]


def integrate_pos(model, qpos, dq):
    """`qpos (nq,)` displaced by tangent `dq (nv,)`. Mirrors `mj_integratePos(..., dt=1)`.

    Differentiable at `dq = 0`; see the module docstring.
    """
    out = qpos
    for jtype, qadr, vadr in joint_layout(model):
        if jtype == _FREE:
            out = out.at[qadr:qadr + 3].add(dq[vadr:vadr + 3])
            out = out.at[qadr + 3:qadr + 7].set(
                quat_exp_local(qpos[qadr + 3:qadr + 7], dq[vadr + 3:vadr + 6]))
        elif jtype == _BALL:
            out = out.at[qadr:qadr + 4].set(
                quat_exp_local(qpos[qadr:qadr + 4], dq[vadr:vadr + 3]))
        else:                                   # hinge / slide: one qpos, one dof
            out = out.at[qadr].add(dq[vadr])
    return out


def differentiate_pos(model, qa, qb):
    """Tangent-space `qa - qb`, width `nv`. Mirrors `mj_differentiatePos(..., dt=1)`.

    Differentiable at `qa = qb`; see the module docstring.
    """
    out = jnp.zeros((int(model.nv),), dtype=qa.dtype)
    for jtype, qadr, vadr in joint_layout(model):
        if jtype == _FREE:
            out = out.at[vadr:vadr + 3].set(qa[qadr:qadr + 3] - qb[qadr:qadr + 3])
            out = out.at[vadr + 3:vadr + 6].set(
                quat_log_local(qa[qadr + 3:qadr + 7], qb[qadr + 3:qadr + 7]))
        elif jtype == _BALL:
            out = out.at[vadr:vadr + 3].set(
                quat_log_local(qa[qadr:qadr + 4], qb[qadr:qadr + 4]))
        else:
            out = out.at[vadr].set(qa[qadr] - qb[qadr])
    return out


def linearization_model(model):
    """A copy of `model` configured so `mjx.step` can actually be differentiated.

    **Disables solver warm-starting**, which is the single largest source of error in MJX
    contact Jacobians (GATE G5, docs/jacobian_audit.md). `solver.py` seeds the solver from
    `d.qacc_warmstart`, which is an INPUT and therefore carries zero tangent; the solver's
    per-iteration updates are then gated by the zero-derivative `improved` comparison, so
    the tangent never recovers from that tangent-free seed. Taking the `d.qacc_smooth`
    branch instead seeds it with a differentiable quantity.

    Measured on a box resting on a plane (16 active constraints), vs `mjd_transitionFD`:
    relative Frobenius error **48.05 -> 1.55e-03**, and `‖A_ad‖` moves from 447 onto the
    true 9.14.

    Note `jax.lax.stop_gradient` on `qacc_warmstart` does NOT help -- it is already
    constant, which IS the bug. The intervention must be to stop USING it.

    Only for linearization. Production rollouts should keep warm-starting (it is a real
    speedup and the forward solution is unaffected), and the closed loop is CPU-stepped
    regardless (invariant 11.2).
    """
    import copy

    import mujoco
    from mujoco.mjx._src.types import DisableBit

    out = copy.copy(model)
    out.opt.disableflags = int(model.opt.disableflags) | int(DisableBit.WARMSTART)
    return out


def transition_jacobians(model, mjx_model, qpos0, qvel0, ctrl0, step_fn=None,
                         allow_cg=False, allow_warmstart=False):
    """`(A, B)` of one `mjx.step`, in `mjd_transitionFD`'s velocity-space layout.

    A is `(2nv, 2nv)`, B is `(2nv, nu)`; `na == 0` is assumed (as everywhere else in this
    repo -- see MJXBackend's state-layout assert). Returns jax arrays.

    This is the shape WO-1's iLQR needs per timestep: one `vmap` of this across the
    horizon gives every `(A_t, B_t)` in a single batched call.

    **`mjx_model` must come from `linearization_model(model)`** when constraints can be
    active, or the result is wrong by orders of magnitude -- see that function and G5.

    Accuracy, with warm-starting disabled and the Newton solver (see G5 for the full
    table): ~1e-9 contact-free, ~1e-7 for a single scalar constraint, ~1.6e-3 for
    16-constraint multi-contact. That last figure does NOT meet G4's 1e-6 bar; it is a
    usable search direction, not an exact Jacobian, and it must not be used to synthesize
    deployed feedback gains.
    """
    import jax
    import mujoco
    import mujoco.mjx as mjx
    from mujoco.mjx._src.types import DisableBit, SolverType

    # The CG solver's differentiated constraint path is unusable even with warm-starting
    # off: 3.4 relFro on a 4-constraint contact and 1.4e+02 on 16 -- WORSE than leaving
    # warm-start on. Newton is the only configuration measured to be usable. Refusing
    # rather than silently returning a garbage Jacobian (invariant 11.5).
    if int(model.opt.solver) == int(SolverType.CG) and not allow_cg:
        raise ValueError(
            "MJX autodiff of the CG solver's constraint path is unusable (measured "
            "relFro 1.4e+02 on 16 active constraints, vs 1.6e-03 for Newton). Set "
            "solver='Newton' on the model, or pass allow_cg=True if you are deliberately "
            "characterising the failure. See docs/jacobian_audit.md, GATE G5.")
    if not (int(model.opt.disableflags) & int(DisableBit.WARMSTART)) and not allow_warmstart:
        raise ValueError(
            "solver warm-starting is enabled; MJX's contact Jacobian is then wrong by "
            "~4 orders of magnitude (G5). Build the model with "
            "`mjx_manifold.linearization_model(model)` first.")

    step = step_fn if step_fn is not None else (lambda d: mjx.step(mjx_model, d))
    nv, nu = int(model.nv), int(model.nu)
    q0 = jnp.asarray(qpos0)
    v0 = jnp.asarray(qvel0)
    u0 = jnp.asarray(ctrl0)

    def _advance(dq, dv, du):
        d = mjx.make_data(mjx_model).replace(
            qpos=integrate_pos(model, q0, dq), qvel=v0 + dv, ctrl=u0 + du)
        d = step(d)
        return d.qpos, d.qvel

    zq, zv, zu = jnp.zeros(nv), jnp.zeros(nv), jnp.zeros(nu)
    qpos_nom, qvel_nom = _advance(zq, zv, zu)

    def _residual(dq, dv, du):
        qp, vp = _advance(dq, dv, du)
        return jnp.concatenate([differentiate_pos(model, qp, qpos_nom), vp - qvel_nom])

    # FORWARD mode, deliberately. `jax.jacobian` is jacrev, and reverse-mode cannot
    # differentiate the `lax.while_loop` MJX's constraint solver uses -- it raises
    # "Reverse-mode differentiation does not work for lax.while_loop". Longer kinematic
    # chains hit that path where a 1-2 dof toy does not, so jacrev fails only on the
    # bigger models. jacfwd has no such restriction and is the better fit regardless:
    # this map is R^(2nv+nu) -> R^(2nv), so forward mode costs ~the same as reverse and
    # composes with `vmap` across the horizon for WO-1's per-timestep (A_t, B_t).
    jac = jax.jacfwd(_residual, argnums=(0, 1, 2))(zq, zv, zu)
    A = jnp.concatenate([jac[0], jac[1]], axis=1)
    return A, jac[2]
