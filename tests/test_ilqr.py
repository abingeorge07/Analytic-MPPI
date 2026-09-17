"""S10 / GATE G8 — iLQR is correct on problems with known answers.

The gate proper: on a pendulum and a cart-pole with quadratic costs, the iLQR core must
converge to the analytic finite-horizon LQR solution with cost gap <= 1e-4. Both the
reference and the iLQR cost are evaluated through the SAME true-dynamics rollout+cost
path, so the gap measures the optimizer, not engine parity. On fail: it is almost always
backward-pass sign convention or the λ schedule — debug here, on these linear problems,
never on hopper (NEXT_STEPS G8).

There is no cart-pole task in the repo on purpose: these are inline diagnostic MJCF
systems following test_jacobians.py's pattern, driven through `make_ilqr_solver`
directly with quadratic stage/terminal callables. The accumulator-backed controller
path (ILQRMPC) is exercised on hopper at the bottom of the file, including the
propagation check the handoff institutionalises: the FPL objective must provably REACH
the iLQR cost (the terminal_value no-op class of bug).

f64 discipline (S5): everything jax runs inside `jax.experimental.enable_x64()`,
scoped to fixtures/tests — production stays f32. Heavy solves are module-scoped
fixtures returning numpy, so each system compiles once.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
mjx = pytest.importorskip("mujoco.mjx")

import mujoco  # noqa: E402

from analytic_mppi.controllers.ilqr import ILQRMPC, make_ilqr_solver  # noqa: E402
from analytic_mppi.dynamics.mjx_manifold import (linearization_model,  # noqa: E402
                                                 transition_jacobians)

GATE_G8_TOL = 1e-4          # cost gap vs the analytic LQR solution. Do NOT relax.

# ------------------------------- diagnostic systems -------------------------------

PENDULUM = """
<mujoco>
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="j1" gear="1"/></actuator>
</mujoco>
"""

# Cart (slide) + hanging pole (hinge), only the cart actuated: nu < nv exercises the
# underactuated coupling through B. Regulation about the STABLE (pole-down) equilibrium
# keeps the linear regime honest at these horizons.
CARTPOLE = """
<mujoco>
  <option timestep="0.01" integrator="Euler"/>
  <worldbody>
    <body pos="0 0 1">
      <joint name="cart" type="slide" axis="1 0 0" damping="0.1"/>
      <geom type="box" size="0.1 0.05 0.05" mass="1"/>
      <body pos="0 0 0">
        <joint name="pole" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="cart" gear="1"/></actuator>
</mujoco>
"""


# ------------------------------- helpers -------------------------------

def _quad_solver(xml, Q, R, Qf, u_lim, iterations):
    """iLQR core over `xml` with J = sum dt*(x_{t+1}'Qx_{t+1} + u'Ru) + x_H'Qf x_H.

    The cost sits on the POST-step state, matching the scorer convention the real
    objective uses (running term t pairs with the state after step t).
    """
    import jax.numpy as jnp

    model = linearization_model(mujoco.MjModel.from_xml_string(xml))
    dt = float(model.opt.timestep)
    nu = int(model.nu)
    Qj, Rj, Qfj = jnp.asarray(Q), jnp.asarray(R), jnp.asarray(Qf)

    def stage_fn(t, qpos, qvel, sd, u):
        x = jnp.concatenate([qpos, qvel])
        return (dt * (x @ Qj @ x + u @ Rj @ u))[None]

    def fold_fn(qpos, qvel, sd):
        x = jnp.concatenate([qpos, qvel])
        return (x @ Qfj @ x)[None]

    solver = make_ilqr_solver(
        model, u_min=-u_lim * np.ones(nu), u_max=u_lim * np.ones(nu), n_acc=1,
        stage_fn=stage_fn, fold_fn=fold_fn, readout_fn=lambda z: z[0],
        iterations=iterations)
    return model, solver, dt


def _lqr_reference(A, B, Q, R, Qf, dt, H, x0):
    """Finite-horizon discrete Riccati for the SAME post-step-state cost convention.

    V_t(x) = min_u [(Ax+Bu)' (Q dt) (Ax+Bu) + u' (R dt) u + V_{t+1}(Ax+Bu)].
    Returns the open-loop control sequence of the LQR policy rolled on the LINEAR model
    and the predicted optimal cost x0' P0 x0.
    """
    P = Qf.copy()
    Ks = [None] * H
    for t in range(H - 1, -1, -1):
        G = Q * dt + P
        Huu = R * dt + B.T @ G @ B
        Ks[t] = -np.linalg.solve(Huu, B.T @ G @ A)
        P = A.T @ G @ A + A.T @ G @ B @ Ks[t]
        P = 0.5 * (P + P.T)
    xs, us = [x0], []
    for t in range(H):
        us.append(Ks[t] @ xs[-1])
        xs.append(A @ xs[-1] + B @ us[-1])
    return np.asarray(us), float(x0 @ P @ x0)


def _g8_case(xml, Q, R, Qf, x0q, x0v, H, iterations=30, lam0=1e-3):
    import jax.numpy as jnp

    with jax.experimental.enable_x64():
        model, solver, dt = _quad_solver(xml, Q, R, Qf, u_lim=50.0,
                                         iterations=iterations)
        nq, nv, nu = int(model.nq), int(model.nv), int(model.nu)
        A, B = transition_jacobians(model, mjx.put_model(model), np.zeros(nq),
                                    np.zeros(nv), np.zeros(nu))
        A, B = np.asarray(A), np.asarray(B)
        x0 = np.concatenate([x0q, x0v])
        u_lqr, J_lin_pred = _lqr_reference(A, B, Q, R, Qf, dt, H, x0)

        q0, v0 = jnp.asarray(x0q), jnp.asarray(x0v)
        solve = jax.jit(solver.solve)
        cost = jax.jit(solver.open_loop_cost)
        u_star, J_ilqr, stats = solve(0.0, q0, v0, jnp.zeros((H, nu)),
                                      jnp.asarray(lam0))
        u_star2, _, _ = solve(0.0, q0, v0, jnp.zeros((H, nu)), jnp.asarray(lam0))
        out = dict(
            J_ilqr=float(J_ilqr),
            J_check=float(cost(0.0, q0, v0, u_star)),
            J_lqr_true=float(cost(0.0, q0, v0, jnp.asarray(u_lqr))),
            J_lin_pred=J_lin_pred,
            u_star=np.asarray(u_star), u_star2=np.asarray(u_star2), u_lqr=u_lqr,
            stats={k: np.asarray(v) for k, v in stats.items()},
        )
    return out


@pytest.fixture(scope="module")
def pendulum_g8():
    Q = np.diag([5.0, 0.5])
    R = np.array([[0.5]])
    Qf = np.diag([5.0, 0.5])
    return _g8_case(PENDULUM, Q, R, Qf, np.array([0.06]), np.array([0.3]), H=50)


@pytest.fixture(scope="module")
def cartpole_g8():
    Q = np.diag([2.0, 5.0, 0.2, 0.2])
    R = np.array([[0.2]])
    Qf = np.diag([2.0, 5.0, 0.5, 0.5])
    return _g8_case(CARTPOLE, Q, R, Qf, np.array([0.05, 0.08]), np.zeros(2), H=60)


# ------------------------------- GATE G8 -------------------------------

@pytest.mark.parametrize("case", ["pendulum_g8", "cartpole_g8"])
def test_g8_converges_to_analytic_lqr(case, request):
    """GATE G8: cost gap vs the analytic LQR solution <= 1e-4, both systems.

    Both costs are evaluated on the TRUE dynamics through the same rollout+cost path;
    iLQR optimizes the true system, so it must never be materially worse than the
    linearization-optimal policy in this near-linear regime.
    """
    r = request.getfixturevalue(case)
    gap = abs(r["J_ilqr"] - r["J_lqr_true"])
    assert gap <= GATE_G8_TOL, (
        f"{case}: |J_ilqr - J_lqr| = {gap:.3e} > {GATE_G8_TOL:.0e} "
        f"(J_ilqr={r['J_ilqr']:.6e}, J_lqr={r['J_lqr_true']:.6e}). Checklist: "
        f"backward-pass sign convention first, λ schedule second (NEXT_STEPS G8).")
    assert r["J_ilqr"] <= r["J_lqr_true"] + 1e-5, "iLQR worse than LQR on true dynamics"
    # Loose sanity: the Riccati-predicted optimum on the LINEAR model is in the same
    # place (nonlinearity + engine parity live in this slack; the gate above does not).
    assert abs(r["J_lin_pred"] - r["J_ilqr"]) <= 0.1 * abs(r["J_ilqr"]) + 1e-6


@pytest.mark.parametrize("case", ["pendulum_g8", "cartpole_g8"])
def test_solve_cost_is_the_rollout_cost(case, request):
    """Propagation check: the J the solver reports IS the open-loop cost of the u* it
    returns (no stale carry, no off-by-one in the fold)."""
    r = request.getfixturevalue(case)
    assert abs(r["J_ilqr"] - r["J_check"]) <= 1e-9 * max(1.0, abs(r["J_ilqr"]))


@pytest.mark.parametrize("case", ["pendulum_g8", "cartpole_g8"])
def test_cost_curve_monotone_nonincreasing(case, request):
    """Accepted iterations reduce cost; rejected ones leave it unchanged. The curve can
    therefore never rise."""
    cc = request.getfixturevalue(case)["stats"]["cost_curve"]
    assert np.all(np.isfinite(cc))
    assert np.all(np.diff(cc) <= 1e-12), f"cost rose: {cc}"


def test_backward_reaches_stationarity(pendulum_g8):
    """Near-LQR problem, 30 iterations: max_t ||Q_u|| must collapse (the KKT residual
    of the unconstrained problem)."""
    qu = pendulum_g8["stats"]["qu_max"]
    assert qu[-1] <= 1e-6 * max(1.0, qu[0]), f"Qu did not collapse: {qu[0]} -> {qu[-1]}"


def test_solver_is_deterministic(pendulum_g8):
    np.testing.assert_array_equal(pendulum_g8["u_star"], pendulum_g8["u_star2"])


def test_instrumentation_shapes_and_ranges(pendulum_g8):
    s = pendulum_g8["stats"]
    n = s["lambda_trace"].shape[0]
    assert s["cost_curve"].shape == (n + 1,)
    for key in ("alpha_trace", "accepted", "quu_cond", "qu_max"):
        assert s[key].shape == (n,), key
    assert np.all((s["lambda_trace"] >= 1e-6) & (s["lambda_trace"] <= 1e8))
    assert set(np.unique(s["accepted"])) <= {False, True}
    assert np.all(s["quu_cond"] >= 1.0 - 1e-9)
    acc = s["accepted"].astype(bool)
    assert np.all(np.isfinite(s["alpha_trace"][acc]))
    assert np.all(np.isnan(s["alpha_trace"][~acc]))
    assert acc.any(), "no iteration was ever accepted on a near-LQR problem"


# ------------------------------- box constraints -------------------------------

@pytest.fixture(scope="module")
def pendulum_box():
    """Torque limit far below what the unconstrained solution wants (gravity torque
    ~1.2 Nm at 0.5 rad vs a 0.3 Nm limit): saturation is guaranteed."""
    import jax.numpy as jnp

    Q = np.diag([5.0, 0.5])
    R = np.array([[0.05]])
    Qf = np.diag([5.0, 0.5])
    with jax.experimental.enable_x64():
        model, solver, dt = _quad_solver(PENDULUM, Q, R, Qf, u_lim=0.3, iterations=20)
        u_star, J, stats = jax.jit(solver.solve)(
            0.0, jnp.asarray([0.5]), jnp.asarray([0.0]),
            jnp.zeros((40, 1)), jnp.asarray(1e-3))
        out = dict(u_star=np.asarray(u_star), J=float(J),
                   stats={k: np.asarray(v) for k, v in stats.items()})
    return out


def test_box_constrained_solution_respects_and_uses_bounds(pendulum_box):
    u = pendulum_box["u_star"]
    assert np.isfinite(pendulum_box["J"])
    assert np.all(np.abs(u) <= 0.3 + 1e-9), "projected-Newton violated the box"
    assert np.max(np.abs(u)) >= 0.3 * (1 - 1e-6), "solution never saturates -- the box "\
        "test is vacuous (did the torque limit or initial state change?)"
    cc = pendulum_box["stats"]["cost_curve"]
    assert np.all(np.diff(cc) <= 1e-12)


# ------------------------------- refusals (invariant 11.5) -------------------------------

def _noop3(*a):
    raise AssertionError("should never be called")


def test_refuses_warmstart_enabled_model():
    model = mujoco.MjModel.from_xml_string(PENDULUM)     # warm-start still on
    with pytest.raises(ValueError, match="warm-start"):
        make_ilqr_solver(model, u_min=-np.ones(1), u_max=np.ones(1), n_acc=1,
                         stage_fn=_noop3, fold_fn=_noop3, readout_fn=_noop3,
                         iterations=1)


def test_refuses_cg_solver():
    xml = PENDULUM.replace('<option timestep="0.01"',
                           '<option solver="CG" timestep="0.01"')
    model = linearization_model(mujoco.MjModel.from_xml_string(xml))
    with pytest.raises(ValueError, match="CG"):
        make_ilqr_solver(model, u_min=-np.ones(1), u_max=np.ones(1), n_acc=1,
                         stage_fn=_noop3, fold_fn=_noop3, readout_fn=_noop3,
                         iterations=1)


# ------------------------------- ILQRMPC on hopper -------------------------------
# The controller path: accumulator objective, spline projection, instrumentation.
# One module-scoped controller (compile is the expensive part, as in test_gradient_mpc).

HOPPER_FLAGS = dict(use_fpl_cost=True, fpl_p=0.1, fpl_time_p=-2.0, fpl_atom_floor=1e-3)


@pytest.fixture(scope="module")
def hopper_setup():
    from analytic_mppi.dynamics import MujocoBackend
    from analytic_mppi.dynamics.mjx_backend import MJXBackend
    from analytic_mppi.eval import init_hopper_stand
    from analytic_mppi.tasks import make_task

    task = make_task("hopper")
    backend = MJXBackend(task.model_path)
    ctrl = ILQRMPC(task, backend, num_knots=3, plan_horizon=0.2, spline_type="linear",
                   iterations=3, warmup=False, **HOPPER_FLAGS)
    cpu = MujocoBackend(task.model_path, nthread=1)
    init_hopper_stand(cpu)             # settled standing state; see test_gradient_mpc
    return task, backend, ctrl, cpu.get_state()


def test_hopper_act_runs_and_instruments(hopper_setup):
    task, backend, ctrl, state = hopper_setup
    ctrl.reset()
    u = ctrl.act(state)
    assert u.shape == (task.nu,)
    assert np.isfinite(u).all()
    assert (u >= task.u_min - 1e-9).all() and (u <= task.u_max + 1e-9).all()
    assert ctrl.last_cost_curve.shape == (ctrl.iterations + 1,)
    assert ctrl.last_loss_curve.shape == (ctrl.iterations,)
    assert ctrl.last_lambda_trace.shape == (ctrl.iterations,)
    assert np.isfinite(ctrl.last_cost_curve).all()
    assert 0.0 <= ctrl.last_accept_rate <= 1.0
    assert np.isfinite(ctrl.last_J_pre) and np.isfinite(ctrl.last_J_post)
    assert np.isfinite(ctrl.last_proj_resid) and ctrl.last_proj_resid >= 0.0
    assert np.all(np.diff(ctrl.last_cost_curve) <= 1e-6), "cost curve rose within act()"


def test_hopper_objective_reaches_the_ilqr_cost(hopper_setup):
    """The propagation habit (NEXT_STEPS S10 acceptance): the FPL objective the config
    declares must provably reach the iLQR cost. cost_curve[0] is the solver's score of
    the initial plan; GradientMPC._loss (an INDEPENDENT path: rollout_jax + jax_scoring,
    parity-pinned to the numpy scorer) must agree on the same plan. Slack covers
    f32 + warm-start-on/off forward differences; wiring bugs (wrong mode, floor,
    weights, time_p) are O(0.1-1) and cannot hide in it."""
    task, backend, ctrl, state = hopper_setup
    ctrl.reset()
    mean0 = ctrl.mean.copy()
    ctrl.act(state)
    got = float(ctrl.last_cost_curve[0])
    ref = float(ctrl._loss(ctrl._jnp.asarray(mean0), state[0],
                           state[backend.qpos_slice], state[backend.qvel_slice]))
    assert abs(got - ref) <= 1e-2 * max(1.0, abs(ref)), (got, ref)


def test_hopper_deterministic(hopper_setup):
    task, backend, ctrl, state = hopper_setup
    ctrl.reset()
    u1 = ctrl.act(state)
    ctrl.reset()
    u2 = ctrl.act(state)
    np.testing.assert_array_equal(u1, u2)


def test_hopper_lambda_persists_across_acts_and_resets(hopper_setup):
    task, backend, ctrl, state = hopper_setup
    ctrl.reset()
    assert ctrl._lam == ctrl.lam_init
    ctrl.act(state)
    assert np.isfinite(ctrl._lam) and ctrl._lam > 0.0
    ctrl.reset()
    assert ctrl._lam == ctrl.lam_init


def test_requires_mjx_backend():
    from analytic_mppi.dynamics import MujocoBackend
    from analytic_mppi.tasks import make_task
    task = make_task("hopper")
    cpu = MujocoBackend(task.model_path, nthread=1)
    with pytest.raises(TypeError, match="rollout_jax"):
        ILQRMPC(task, cpu, warmup=False)


def test_warm_started_ilqr_runs_and_stays_aligned(hopper_setup):
    """S11: the sampler's mean seeds iLQR; afterwards both plans are the same,
    identically shifted, so the sampler explores around what actually executed."""
    from analytic_mppi.controllers import MPPIv2
    from analytic_mppi.controllers.ilqr import WarmStartedILQR
    from analytic_mppi.dynamics import MujocoBackend

    task, backend, ctrl, state = hopper_setup
    cpu = MujocoBackend(task.model_path, nthread=1)
    sampler = MPPIv2(task, cpu, num_samples=8, noise_level=0.3, temperature=0.2,
                     num_knots=ctrl.num_knots, plan_horizon=ctrl.plan_horizon,
                     spline_type=ctrl.spline_type, seed=0, **HOPPER_FLAGS)
    w = WarmStartedILQR(sampler, ctrl)
    w.reset()
    u = w.act(state)
    assert u.shape == (task.nu,) and np.isfinite(u).all()
    np.testing.assert_array_equal(sampler.mean, ctrl.mean)
    assert sampler._shift_accum == ctrl._shift_accum


def test_warm_started_ilqr_rejects_mismatched_grid(hopper_setup):
    from analytic_mppi.controllers import MPPIv2
    from analytic_mppi.controllers.ilqr import WarmStartedILQR
    from analytic_mppi.dynamics import MujocoBackend

    task, backend, ctrl, state = hopper_setup
    cpu = MujocoBackend(task.model_path, nthread=1)
    bad = MPPIv2(task, cpu, num_samples=8, noise_level=0.3, temperature=0.2,
                 num_knots=ctrl.num_knots + 1, plan_horizon=ctrl.plan_horizon,
                 spline_type=ctrl.spline_type, seed=0, **HOPPER_FLAGS)
    with pytest.raises(ValueError, match="knot grid"):
        WarmStartedILQR(bad, ctrl)
