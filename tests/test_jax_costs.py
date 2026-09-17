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


@pytest.mark.parametrize("floor", [1e-8, 1e-4, 1e-3, 1e-2])
def test_floor_atoms_matches_numpy(floor):
    """WO-3.4's floor must be bit-for-bit the same operation on both sides -- it is the
    one place an FPL-vs-linear asymmetry can enter by accident (invariant 11.1/#4)."""
    from analytic_mppi.tasks.base import floor_atoms as np_floor
    from analytic_mppi.tasks.jax_costs._base import floor_atoms as j_floor
    rng = np.random.default_rng(7)
    # deliberately spans below-floor, in-band and above-1 so every branch is exercised
    x = np.concatenate([np.zeros(16), rng.uniform(0.0, 1.0, 256), np.full(4, 1.5)])
    np.testing.assert_allclose(np.asarray(j_floor(x, floor)), np_floor(x, floor), atol=ATOL)


@pytest.mark.parametrize("p", [-1.0, 1.0])
def test_floored_score_matches_numpy(p):
    """The floor composes with scoring identically on both sides, for the FPL arm AND the
    linear arm. Run at p=1 too: that arm needs no floor numerically, so it is exactly the
    one a future refactor might 'optimize away'."""
    from analytic_mppi.tasks.base import floor_atoms as np_floor
    from analytic_mppi.tasks.jax_costs._base import floor_atoms as j_floor
    task = make_task("hopper")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = MPPIv2(task, backend, use_fpl_cost=True, fpl_p=p, fpl_gamma=0.99,
                  num_samples=K, noise_level=0.4, temperature=1.0, num_knots=4,
                  plan_horizon=0.6, spline_type="zero", seed=0, fpl_atom_floor=1e-3)
    rng = np.random.default_rng(8)
    running_f, terminal_f = _fake_traj(rng, n_run=4, n_term=3)
    running_f[:, :2, 1] = 0.0          # force some atoms into the clamped region
    terminal_f[:, 0] = 0.0

    traj = Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None,
                      running_terms_f=np_floor(running_f, 1e-3),
                      terminal_terms_f=np_floor(terminal_f, 1e-3))
    want = ctrl._score_fpl(traj)
    got = np.asarray(jax_scoring.score_fpl(
        j_floor(running_f, 1e-3), j_floor(terminal_f, 1e-3), mode="fpl_cost",
        p=p, gamma=ctrl.fpl_gamma, time_p=ctrl.fpl_time_p,
        time_discount=ctrl.fpl_time_discount, weights=ctrl.fpl_weights))
    np.testing.assert_allclose(got, want, atol=ATOL)


@pytest.mark.parametrize("tv", [False, True])
@pytest.mark.parametrize("T,gamma", [(7, 0.5), (31, 0.99), (50, 0.9)])
def test_discount_weights_match_numpy(T, gamma, tv):
    """WO-3.3's time aggregation, both sides. Same reason as the floor: a change that
    lands in one path and not the other silently forks the objective."""
    from analytic_mppi.tasks.base import discount_weights as np_w
    from analytic_mppi.tasks.jax_costs._base import discount_weights as j_w
    np.testing.assert_allclose(np.asarray(j_w(T, gamma, tv)), np_w(T, gamma, tv), atol=ATOL)


@pytest.mark.parametrize("flags,mode,extra", [
    c for c in FPL_CASES
    # An unweighted soft-min over time never consults the weights, so terminal_value is
    # inert there and the controller now refuses that combination. Skip those cases.
    if not (c[0].get("fpl_time_p") is not None and not c[0].get("fpl_time_discount"))
])
def test_score_fpl_matches_with_terminal_value(hopper_ctrl_factory, flags, mode, extra):
    """Every FPL case again with terminal_value=True, so the JAX scorer cannot drift from
    the numpy one on the new aggregation path."""
    task, backend, ctrl = hopper_ctrl_factory(fpl_terminal_value=True, **flags)
    rng = np.random.default_rng(9)
    running_f, terminal_f = _fake_traj(rng, n_run=4, n_term=3)
    traj = Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None,
                      running_terms_f=running_f, terminal_terms_f=terminal_f)
    want = ctrl._score_fpl(traj)
    got = np.asarray(jax_scoring.score_fpl(
        running_f, terminal_f, mode=mode, p=ctrl.fpl_p, gamma=ctrl.fpl_gamma,
        time_p=ctrl.fpl_time_p, time_discount=ctrl.fpl_time_discount,
        weights=ctrl.fpl_weights, terminal_value=True, **extra))
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_score_fpl_is_differentiable():
    """The whole point: grad of the score w.r.t. the atoms exists and is finite."""
    rng = np.random.default_rng(6)
    running_f, terminal_f = _fake_traj(rng)

    def loss(rf):
        return jax_scoring.score_fpl(rf, jax.numpy.asarray(terminal_f), mode="fpl_cost",
                                     p=-1.0, gamma=0.99, time_p=-2.0).sum()

    g = jax.grad(loss)(jax.numpy.asarray(running_f))
    assert np.isfinite(np.asarray(g)).all()


# --- GATE G7: additive-accumulator decomposition (S9), JAX side ---------------------
#
# The decomposition rewrites each objective as (additive stage in an accumulator z) +
# (terminal readout); these tests pin its JAX evaluation to the NUMPY reference scorer
# at 1e-6, on REAL hopper and walker atom surfaces (not synthetic batches), for every
# mode the mjx path can run: normal, fpl_cost in all three time_p branches, and
# fpl_discounted. fpl_layered and hybrid are unreachable on hopper/walker (no
# fpl_groups, no floor_term_indices) and excluded from config.JAX_COST_MODES; they are
# gated against the numpy scorer on g1_standup in tests/test_accumulator.py.

from analytic_mppi.controllers.accumulator import build_augmented_objective, jax_ops
from analytic_mppi.tasks.base import floor_atoms as _np_floor

G7_FPL_CASES = [
    (dict(use_fpl_cost=True, fpl_p=-1.0), "cost/discounted-mean"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_time_p=-2.0), "cost/softmin(published)"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_time_p=-2.0, fpl_time_discount=True),
     "cost/softmin-discounted"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_time_p=0.0), "cost/geometric-time"),
    (dict(use_fpl_cost=True, fpl_p=0.1, fpl_weights=[1.0, 1.0, 4.0, 0.5]),
     "cost/weighted"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_terminal_value=True), "cost/terminal-value"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_time_p=-2.0, fpl_time_discount=True,
          fpl_terminal_value=True), "cost/softmin-tv"),
    (dict(use_fpl_discounted=True, fpl_p=0.1), "disc/plain"),
    (dict(use_fpl_discounted=True, fpl_p=-1.0, fpl_weights=[1.0, 1.0, 4.0, 0.5]),
     "disc/weighted"),
    (dict(use_fpl_discounted=True, fpl_p=0.1, fpl_terminal_value=True),
     "disc/terminal-value"),
]


@pytest.fixture(scope="module")
def g7_env():
    """One (task, backend) per gated robot, shared across the G7 cases."""
    envs = {}
    for name in ("hopper", "walker"):
        task = make_task(name)
        envs[name] = (task, MujocoBackend(task.model_path, nthread=1))
    return envs


def _g7_real_atoms(task, rng, floor):
    """Real task fulfillment surfaces, floored exactly as sampling_base._floor does."""
    qpos, qvel, sd, u = _random_inputs(task, rng)
    rf = _np_floor(task.running_cost_terms_f(qpos, qvel, sd, u), floor)
    tf = _np_floor(task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sd[:, -1]), floor)
    return rf, tf


@pytest.mark.parametrize("task_name", ["hopper", "walker"])
def test_g7_normal_augmented_parity(g7_env, task_name):
    task, backend = g7_env[task_name]
    ctrl = MPPIv2(task, backend, num_samples=K, num_knots=2,
                  plan_horizon=H * backend.dt, noise_level=0.3, temperature=0.2)
    rng = np.random.default_rng(10)
    qpos, qvel, sd, u = _random_inputs(task, rng)
    running = task.running_cost_terms(qpos, qvel, sd, u)
    terminal = task.terminal_cost_terms(qpos[:, -1], qvel[:, -1], sd[:, -1])
    want = ctrl._score_normal(Trajectory(
        knots=None, controls=None, states=None, sensordata=None, qpos=None, qvel=None,
        running_terms=running, terminal_terms=terminal))

    obj = build_augmented_objective(
        ops=jax_ops(), mode="normal", H=H,
        n_run=running.shape[-1], n_term=terminal.shape[-1], dt=float(backend.dt))
    got = np.asarray(obj.score(running_terms=jax.numpy.asarray(running),
                               terminal_terms=jax.numpy.asarray(terminal)))
    np.testing.assert_allclose(got, want, atol=ATOL)


@pytest.mark.parametrize("task_name", ["hopper", "walker"])
@pytest.mark.parametrize("flags,label", G7_FPL_CASES, ids=[c[1] for c in G7_FPL_CASES])
def test_g7_fpl_augmented_parity(g7_env, task_name, flags, label):
    task, backend = g7_env[task_name]
    ctrl = MPPIv2(task, backend, num_samples=K, num_knots=2,
                  plan_horizon=H * backend.dt, noise_level=0.3, temperature=0.2,
                  fpl_atom_floor=1e-3, **flags)
    rng = np.random.default_rng(11)
    running_f, terminal_f = _g7_real_atoms(task, rng, ctrl.fpl_atom_floor)
    want = ctrl._score_fpl(Trajectory(
        knots=None, controls=None, states=None, sensordata=None, qpos=None, qvel=None,
        running_terms_f=running_f, terminal_terms_f=terminal_f))

    mode = "fpl_cost" if ctrl.use_fpl_cost else "fpl_discounted"
    obj = build_augmented_objective(
        ops=jax_ops(), mode=mode, H=H,
        n_run=running_f.shape[-1], n_term=terminal_f.shape[-1],
        p=ctrl.fpl_p, gamma=ctrl.fpl_gamma, time_p=ctrl.fpl_time_p,
        time_discount=ctrl.fpl_time_discount, terminal_value=ctrl.fpl_terminal_value,
        weights=ctrl.fpl_weights)
    got = np.asarray(obj.score(running_terms_f=jax.numpy.asarray(running_f),
                               terminal_terms_f=jax.numpy.asarray(terminal_f)))
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_g7_augmented_score_is_differentiable():
    """iLQR consumes stage/terminal derivatives: grad of the augmented score w.r.t. the
    atoms exists and is finite for the published objective (fpl_cost, time_p=-2)."""
    rng = np.random.default_rng(12)
    running_f, terminal_f = _fake_traj(rng)
    obj = build_augmented_objective(ops=jax_ops(), mode="fpl_cost", H=H,
                                    n_run=4, n_term=3, p=-1.0, time_p=-2.0)

    def loss(rf):
        return obj.score(running_terms_f=rf,
                         terminal_terms_f=jax.numpy.asarray(terminal_f)).sum()

    g = jax.grad(loss)(jax.numpy.asarray(running_f))
    assert np.isfinite(np.asarray(g)).all()
