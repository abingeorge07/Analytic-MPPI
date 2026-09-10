"""Parity: jnp cost/scoring mirrors vs the numpy originals, at 1e-6 in f64.

These tests are the ONLY thing pinning tasks/jax_costs/* and controllers/jax_scoring.py
to the numpy implementations that define every published result. A formula change on
either side without the twin change fails here.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")


@pytest.fixture(autouse=True)
def _f64_parity_mode():
    """f64 for the numpy comparison, scoped per-test: mutating the global x64 flag
    would leak into other test modules (it measurably shifts MJX rollouts)."""
    with jax.experimental.enable_x64():
        yield

from analytic_mppi.controllers import MPPIv2
from analytic_mppi.controllers.sampling_base import Trajectory
from analytic_mppi.controllers import jax_scoring
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.tasks import make_task
from analytic_mppi.tasks.jax_costs import JAX_COST_TASKS, make_jax_costs

K, H = 5, 7
ATOL = 1e-6


def _random_inputs(task, rng):
    """Random (K, H, ·) batches. sensordata uniform in [-1, 2] covers every ramp's
    full/floor band; qpos near qstand for g1 so joint atoms aren't all clipped."""
    qpos = rng.uniform(-1.0, 1.0, (K, H, task.nq))
    if hasattr(task, "qstand"):
        qpos = task.qstand + 0.3 * qpos
    qvel = rng.uniform(-2.0, 2.0, (K, H, task.nv))
    sd = rng.uniform(-1.0, 2.0, (K, H, task.nsensordata))
    u = rng.uniform(task.u_min, task.u_max, (K, H, task.nu))
    return qpos, qvel, sd, u


@pytest.mark.parametrize("name", sorted(JAX_COST_TASKS))
def test_cost_terms_match_numpy(name):
    task = make_task(name)
    jc = make_jax_costs(task)
    rng = np.random.default_rng(0)
    qpos, qvel, sd, u = _random_inputs(task, rng)

    for np_fn, j_fn, args, names in [
        (task.running_cost_terms, jc.running_cost_terms, (qpos, qvel, sd, u),
         task.cost_term_names),
        (task.terminal_cost_terms, jc.terminal_cost_terms, (qpos[:, -1], qvel[:, -1], sd[:, -1]),
         task.cost_term_names),
        (task.running_cost_terms_f, jc.running_cost_terms_f, (qpos, qvel, sd, u),
         task.cost_term_names_f),
        (task.terminal_cost_terms_f, jc.terminal_cost_terms_f, (qpos[:, -1], qvel[:, -1], sd[:, -1]),
         None),  # terminal may be a prefix (hopper/walker drop the control atom)
    ]:
        want = np_fn(*args)
        got = np.asarray(j_fn(*[jax.numpy.asarray(a) for a in args]))
        assert got.shape == want.shape
        if names is not None:
            assert got.shape[-1] == len(names)
        np.testing.assert_allclose(got, want, atol=ATOL)


def test_soft_floor_hopper_variant_matches():
    """Hopper's opt-in soft_ramp path (atom_soft_floor > 0) must also match."""
    task = make_task("hopper", atom_soft_floor=0.05, atom_tail_tau=0.4)
    jc = make_jax_costs(task)
    rng = np.random.default_rng(1)
    qpos, qvel, sd, u = _random_inputs(task, rng)
    want = task.running_cost_terms_f(qpos, qvel, sd, u)
    got = np.asarray(jc.running_cost_terms_f(qpos, qvel, sd, u))
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_task_kwargs_flow_through():
    task = make_task("hopper", target_velocity=3.5, height_weight=2.0)
    jc = make_jax_costs(task)
    rng = np.random.default_rng(2)
    qpos, qvel, sd, u = _random_inputs(task, rng)
    np.testing.assert_allclose(
        np.asarray(jc.running_cost_terms(qpos, qvel, sd, u)),
        task.running_cost_terms(qpos, qvel, sd, u), atol=ATOL)


# --- scoring parity ---------------------------------------------------------------


@pytest.fixture(scope="module")
def hopper_ctrl_factory():
    """Builds a real MPPIv2 (its _score_* are the reference implementation)."""
    task = make_task("hopper")
    backend = MujocoBackend(task.model_path, nthread=1)

    def build(**flags):
        return task, backend, MPPIv2(task, backend, num_samples=K, num_knots=2,
                                     plan_horizon=H * backend.dt, noise_level=0.3,
                                     temperature=0.2, **flags)
    return build


def _fake_traj(rng, n_run=4, n_term=3):
    running_f = rng.uniform(0.0, 1.0, (K, H, n_run))
    terminal_f = rng.uniform(0.0, 1.0, (K, n_term))
    return running_f, terminal_f


def test_score_normal_matches(hopper_ctrl_factory):
    task, backend, ctrl = hopper_ctrl_factory()
    rng = np.random.default_rng(3)
    running = rng.uniform(0.0, 5.0, (K, H, 4))
    terminal = rng.uniform(0.0, 5.0, (K, 4))
    traj = Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None, running_terms=running, terminal_terms=terminal)
    want = ctrl._score_normal(traj)
    got = np.asarray(jax_scoring.score_normal(running, terminal, dt=backend.dt))
    np.testing.assert_allclose(got, want, atol=ATOL)


FPL_CASES = [
    # (cost_mode flags, jnp mode, extra score kwargs)
    (dict(use_fpl_cost=True, fpl_p=-1.0), "fpl_cost", {}),
    (dict(use_fpl_cost=True, fpl_p=0.1, fpl_time_p=-2.0), "fpl_cost", {}),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_time_p=-2.0, fpl_time_discount=True),
     "fpl_cost", {}),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_weights=[1.0, 1.0, 4.0, 0.5]), "fpl_cost", {}),
    (dict(use_fpl_discounted=True, fpl_p=0.1), "fpl_discounted", {}),
    (dict(use_fpl_discounted=True, fpl_p=-1.0, fpl_weights=[1.0, 1.0, 4.0, 0.5]),
     "fpl_discounted", {}),
]


@pytest.mark.parametrize("flags,mode,extra", FPL_CASES)
@pytest.mark.parametrize("n_term", [3, 4])   # prefix convention and 1:1 alignment
def test_score_fpl_matches(hopper_ctrl_factory, flags, mode, extra, n_term):
    task, backend, ctrl = hopper_ctrl_factory(**flags)
    rng = np.random.default_rng(4)
    running_f, terminal_f = _fake_traj(rng, n_run=4, n_term=n_term)
    traj = Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None,
                      running_terms_f=running_f, terminal_terms_f=terminal_f)
    want = ctrl._score_fpl(traj)
    got = np.asarray(jax_scoring.score_fpl(
        running_f, terminal_f, mode=mode, p=ctrl.fpl_p, gamma=ctrl.fpl_gamma,
        time_p=ctrl.fpl_time_p, time_discount=ctrl.fpl_time_discount,
        weights=ctrl.fpl_weights, **extra))
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_select_fpl_terms_matches(hopper_ctrl_factory):
    task, backend, ctrl = hopper_ctrl_factory(use_fpl_cost=True, fpl_p=-1.0,
                                              fpl_term_indices=[3, 0, 2])
    rng = np.random.default_rng(5)
    running_f, terminal_f = _fake_traj(rng, n_run=4, n_term=3)
    want_r, want_t = ctrl._select_fpl_terms(running_f, terminal_f)
    got_r, got_t = jax_scoring.select_fpl_terms(running_f, terminal_f, [3, 0, 2])
    np.testing.assert_allclose(np.asarray(got_r), want_r, atol=ATOL)
    np.testing.assert_allclose(np.asarray(got_t), want_t, atol=ATOL)


def test_score_fpl_is_differentiable():
    """The whole point: grad of the score w.r.t. the atoms exists and is finite."""
    rng = np.random.default_rng(6)
    running_f, terminal_f = _fake_traj(rng)

    def loss(rf):
        return jax_scoring.score_fpl(rf, jax.numpy.asarray(terminal_f), mode="fpl_cost",
                                     p=-1.0, gamma=0.99, time_p=-2.0).sum()

    g = jax.grad(loss)(jax.numpy.asarray(running_f))
    assert np.isfinite(np.asarray(g)).all()
