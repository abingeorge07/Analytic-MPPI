"""GradientMPC unit tests (skipped wholesale without the [mjx] extra).

Runtime note: compiling plan_fn costs ~40 s, so ALL descent tests share one
module-scoped controller (hopper, small horizon, few iterations).
"""
import numpy as np
import pytest

pytest.importorskip("mujoco.mjx")

from analytic_mppi.controllers import GradientMPC
from analytic_mppi.controllers.spline import shift_plan
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.dynamics.mjx_backend import MJXBackend
from analytic_mppi.eval import init_hopper_stand
from analytic_mppi.tasks import make_task

HOPPER_XML = "analytic_mppi/envs/hopper/scene.xml"


@pytest.fixture(scope="module")
def setup():
    task = make_task("hopper")
    backend = MJXBackend(task.model_path)
    ctrl = GradientMPC(task, backend, num_knots=3, plan_horizon=0.3,
                       spline_type="linear", iterations=8, learning_rate=0.01,
                       use_fpl_cost=True, fpl_p=-1.0, warmup=False)
    cpu = MujocoBackend(task.model_path, nthread=1)
    # The settled standing state episodes actually start from. NOT the raw default
    # state: there the hopper hangs with vx=0, the hard-clipped velocity atom sits deep
    # in its zero-gradient region, the p<0 reward pins at eps, and every f32 gradient
    # flushes to exactly 0 -- the degenerate landscape soft_ramp exists to fix, and a
    # useless operating point for testing descent.
    init_hopper_stand(cpu)
    state = cpu.get_state()
    return task, backend, ctrl, state


def test_requires_mjx_backend():
    task = make_task("hopper")
    cpu = MujocoBackend(task.model_path, nthread=1)
    with pytest.raises(TypeError, match="rollout_jax"):
        GradientMPC(task, cpu, warmup=False)


def test_rejects_multiple_samples(setup):
    task, backend, *_ = setup
    with pytest.raises(ValueError, match="num_samples"):
        GradientMPC(task, backend, num_samples=8, warmup=False)


def test_act_is_finite_bounded_and_shifts_plan(setup):
    task, backend, ctrl, state = setup
    ctrl.reset()
    mean_before = ctrl.mean.copy()
    u = ctrl.act(state)
    assert u.shape == (task.nu,)
    assert np.isfinite(u).all()
    assert (u >= task.u_min - 1e-9).all() and (u <= task.u_max + 1e-9).all()
    assert (ctrl.mean >= task.u_min - 1e-9).all() and (ctrl.mean <= task.u_max + 1e-9).all()
    assert not np.array_equal(ctrl.mean, mean_before), "descent did not move the plan"
    assert ctrl.last_loss_curve.shape == (ctrl.iterations,)
    assert np.isfinite(ctrl.last_loss_curve).all()


def test_descent_reduces_loss_within_act(setup):
    """From a warm start (2nd act), the loss over Adam steps must end below where it
    began -- the 'gradients are descent-useful, not garbage' gate."""
    task, backend, ctrl, state = setup
    ctrl.reset()
    ctrl.act(state)
    ctrl.act(state)
    lc = ctrl.last_loss_curve
    assert lc[-1] < lc[0], f"loss rose over the act: {lc[0]:.4f} -> {lc[-1]:.4f}"


def test_deterministic(setup):
    task, backend, ctrl, state = setup
    ctrl.reset()
    u1 = ctrl.act(state)
    ctrl.reset()
    u2 = ctrl.act(state)
    np.testing.assert_array_equal(u1, u2)


def test_normal_mode_builds_and_descends(setup):
    task, backend, _, state = setup
    ctrl = GradientMPC(task, backend, num_knots=3, plan_horizon=0.3,
                       spline_type="linear", iterations=8, learning_rate=0.01,
                       warmup=False)  # mode: normal
    ctrl.act(state)
    ctrl.act(state)
    lc = ctrl.last_loss_curve
    assert np.isfinite(lc).all() and lc[-1] < lc[0]


def test_warm_start_uses_shared_shift_plan(setup):
    """GradientMPC's post-act shift must equal spline.shift_plan on the same inputs."""
    task, backend, ctrl, state = setup
    ctrl.reset()
    ctrl.act(state)
    # replay: capture the pre-shift mean by re-running the planner deterministically
    ctrl.reset()
    knots, _ = ctrl._plan(state[0], state[backend.qpos_slice], state[backend.qvel_slice],
                          ctrl._jnp.asarray(ctrl.mean))
    pre_shift = np.asarray(knots, dtype=np.float64)
    want, _ = shift_plan(pre_shift.copy(), ctrl.tk, backend.dt, ctrl.spline_type, 0.0)
    ctrl.act(state)
    np.testing.assert_allclose(ctrl.mean, want, atol=1e-12)


def test_spline_basis_matches_interpolate(setup):
    """controls = W @ knots must equal the numpy interpolate() on random knots."""
    from analytic_mppi.controllers.spline import interpolate
    task, backend, ctrl, _ = setup
    rng = np.random.default_rng(0)
    knots = rng.uniform(task.u_min, task.u_max, (ctrl.num_knots, task.nu))
    want = interpolate(knots[None], ctrl.tk, ctrl.t_eval, ctrl.spline_type)[0]
    got = ctrl._W @ knots
    np.testing.assert_allclose(got, want, atol=1e-12)
