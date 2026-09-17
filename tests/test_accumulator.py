"""S9 / GATE G7 (numpy half): the additive-accumulator decomposition is EXACT.

Every test here scores the same random atom batches twice -- once through the
controller's reference scorer (`_score_normal` / `_score_fpl` / `_score_fpl_layered` /
`_score_hybrid`, the implementations behind every published number) and once through
`accumulator.AugmentedObjective`'s scan-then-readout -- and pins them together at 1e-6.

Mode coverage follows the Phase 2 scoping decision:
  * `normal`, `fpl_cost` (all three time_p branches), `fpl_discounted` are gated on
    hopper-shaped AND walker-shaped atom layouts here and again in JAX in
    tests/test_jax_costs.py (the G7 gate proper).
  * `fpl_layered` and `hybrid` are reachable only on g1 (hopper/walker define neither
    fpl_groups nor floor_term_indices, and config.JAX_COST_MODES excludes both from the
    mjx path), so they are gated against the NUMPY scorer on g1_standup only.

Also here, because the handoff demands them explicitly:
  * a propagation check -- the accumulator provably CHANGES the score (the
    terminal_value no-op class of bug),
  * the block-triangularity assertion (`dx'/dz == 0`) through a REAL mjx step,
    in test_augmented_dynamics_block_triangular (jax-gated).
"""
import numpy as np
import pytest

from analytic_mppi.controllers import MPPIv2
from analytic_mppi.controllers.accumulator import build_augmented_objective, numpy_ops
from analytic_mppi.controllers.sampling_base import Trajectory
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.tasks import make_task

K, H = 5, 7
ATOL = 1e-6
OPS = numpy_ops()


# ---------------------------------------------------------------------------------
# Reference controllers. The backend is only needed for dt / ctor plumbing; all
# scoring runs on injected random atoms, so one hopper controller serves every
# non-layered case regardless of which task's atom LAYOUT the batch mimics.
# ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ctrl_factory():
    task = make_task("hopper")
    backend = MujocoBackend(task.model_path, nthread=1)

    def build(**flags):
        return backend, MPPIv2(task, backend, num_samples=K, num_knots=2,
                               plan_horizon=H * backend.dt, noise_level=0.3,
                               temperature=0.2, **flags)
    return build


@pytest.fixture(scope="module")
def g1_ctrl_factory():
    task = make_task("g1_standup")
    backend = MujocoBackend(task.model_path, nthread=1)

    def build(**flags):
        return task, backend, MPPIv2(task, backend, num_samples=K, num_knots=2,
                                     plan_horizon=H * backend.dt, noise_level=0.3,
                                     temperature=0.2, **flags)
    return build


def _traj(**arrays):
    return Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None, **arrays)


def _atoms(rng, n_run, n_term):
    running_f = rng.uniform(0.0, 1.0, (K, H, n_run))
    terminal_f = rng.uniform(0.0, 1.0, (K, n_term))
    return running_f, terminal_f


# ---------------------------------------------------------------------------------
# normal
# ---------------------------------------------------------------------------------


def test_normal_decomposition_matches(ctrl_factory):
    backend, ctrl = ctrl_factory()
    rng = np.random.default_rng(0)
    running = rng.uniform(0.0, 5.0, (K, H, 4))
    terminal = rng.uniform(0.0, 5.0, (K, 4))
    want = ctrl._score_normal(_traj(running_terms=running, terminal_terms=terminal))

    obj = build_augmented_objective(ops=OPS, mode="normal", H=H, n_run=4, n_term=4,
                                    dt=float(backend.dt))
    got = obj.score(running_terms=running, terminal_terms=terminal)
    np.testing.assert_allclose(got, want, atol=ATOL)


# ---------------------------------------------------------------------------------
# fpl_cost / fpl_discounted -- hopper-shaped (n_run=4, n_term=3) and walker-shaped
# (n_run=4, n_term=4) layouts, plus the no-terminal edge (n_term=0).
# ---------------------------------------------------------------------------------

FPL_CASES = [
    # (controller flags, id)
    (dict(use_fpl_cost=True, fpl_p=-1.0), "cost/discounted-mean"),
    (dict(use_fpl_cost=True, fpl_p=0.1, fpl_time_p=-2.0), "cost/softmin(published)"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_time_p=-2.0, fpl_time_discount=True),
     "cost/softmin-discounted"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_time_p=0.0), "cost/geometric-time"),
    (dict(use_fpl_cost=True, fpl_p=-1.0, fpl_weights=[1.0, 1.0, 4.0, 0.5]),
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


def _build_from_ctrl(ctrl, mode, n_run, n_term, collapse=None):
    return build_augmented_objective(
        ops=OPS, mode=mode, H=H, n_run=n_run, n_term=n_term,
        p=ctrl.fpl_p, gamma=ctrl.fpl_gamma, time_p=ctrl.fpl_time_p,
        time_discount=ctrl.fpl_time_discount, terminal_value=ctrl.fpl_terminal_value,
        weights=ctrl.fpl_weights, collapse=collapse)


@pytest.mark.parametrize("flags,label", FPL_CASES, ids=[c[1] for c in FPL_CASES])
@pytest.mark.parametrize("n_term", [0, 3, 4])
def test_fpl_decomposition_matches(ctrl_factory, flags, label, n_term):
    if n_term == 0 and flags.get("use_fpl_discounted"):
        # reward_terms would be empty; the discounted mode requires atoms. Skip, matching
        # the scorer (power_mean over an empty axis is undefined on both sides).
        pytest.skip("fpl_discounted needs at least one atom with a series")
    backend, ctrl = ctrl_factory(**flags)
    mode = "fpl_cost" if ctrl.use_fpl_cost else "fpl_discounted"
    rng = np.random.default_rng(1)
    running_f, terminal_f = _atoms(rng, n_run=4, n_term=n_term)
    want = ctrl._score_fpl(_traj(running_terms_f=running_f, terminal_terms_f=terminal_f))

    obj = _build_from_ctrl(ctrl, mode, n_run=4, n_term=n_term)
    got = obj.score(running_terms_f=running_f, terminal_terms_f=terminal_f)
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_fpl_cost_conjunction_split_matches(ctrl_factory):
    """The conjunction-split collapse (fpl_conj_indices) rides through unchanged: the
    decomposition only touches the TIME axis, so the atom-axis collapse is injected and
    inherits whatever composition the controller uses."""
    backend, ctrl = ctrl_factory(use_fpl_cost=True, fpl_p=-2.0, fpl_time_p=-2.0,
                                 fpl_conj_indices=[0, 2], fpl_outer_p=0.0)
    rng = np.random.default_rng(2)
    running_f, terminal_f = _atoms(rng, n_run=4, n_term=3)
    want = ctrl._score_fpl(_traj(running_terms_f=running_f, terminal_terms_f=terminal_f))

    obj = _build_from_ctrl(ctrl, "fpl_cost", n_run=4, n_term=3,
                           collapse=ctrl._collapse_objectives)
    got = obj.score(running_terms_f=running_f, terminal_terms_f=terminal_f)
    np.testing.assert_allclose(got, want, atol=ATOL)


# ---------------------------------------------------------------------------------
# fpl_layered + hybrid -- g1_standup, numpy scorer only (see module docstring).
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("group_p", [None, 0.5, 0.0])
def test_layered_decomposition_matches_g1(g1_ctrl_factory, group_p):
    task, backend, ctrl = g1_ctrl_factory(use_fpl_layered=True, fpl_p=-1.0,
                                          fpl_group_p=group_p)
    n = sum(len(g) for g in task.fpl_groups)
    rng = np.random.default_rng(3)
    running_f = rng.uniform(0.0, 1.0, (K, H, n))
    terminal_f = rng.uniform(0.0, 1.0, (K, n))          # layered: strict 1:1 alignment
    want = ctrl._score_fpl_layered(
        _traj(running_terms_f=running_f, terminal_terms_f=terminal_f))

    obj = build_augmented_objective(
        ops=OPS, mode="fpl_layered", H=H, n_run=n, n_term=n,
        p=ctrl.fpl_p, gamma=ctrl.fpl_gamma, terminal_value=ctrl.fpl_terminal_value,
        weights=ctrl.fpl_weights, groups=task.fpl_groups, group_p=ctrl.fpl_group_p)
    got = obj.score(running_terms_f=running_f, terminal_terms_f=terminal_f)
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_hybrid_decomposition_matches_g1(g1_ctrl_factory):
    task, backend, ctrl = g1_ctrl_factory(use_hybrid=True, floor_weight=1.7)
    assert task.floor_term_indices, "g1_standup must declare floors or this test is vacuous"
    n = len(task.cost_term_names)
    n_f = len(task.cost_term_names_f)
    rng = np.random.default_rng(4)
    running = rng.uniform(0.0, 5.0, (K, H, n))
    terminal = rng.uniform(0.0, 5.0, (K, n))
    running_f = rng.uniform(0.0, 1.0, (K, H, n_f))
    terminal_f = rng.uniform(0.0, 1.0, (K, n_f))
    want = ctrl._score_hybrid(_traj(running_terms=running, terminal_terms=terminal,
                                    running_terms_f=running_f, terminal_terms_f=terminal_f))

    obj = build_augmented_objective(
        ops=OPS, mode="hybrid", H=H, n_run=n_f, n_term=n_f, dt=float(backend.dt),
        floor_indices=task.floor_term_indices, floor_weight=ctrl.floor_weight)
    got = obj.score(running_terms=running, terminal_terms=terminal,
                    running_terms_f=running_f, terminal_terms_f=terminal_f)
    np.testing.assert_allclose(got, want, atol=ATOL)


def test_hybrid_without_floors_is_normal(ctrl_factory):
    """On a floorless task (hopper) hybrid degenerates to the normal quadratic sum --
    the decomposition must reproduce that limit too, not just raise or drift."""
    backend, ctrl = ctrl_factory(use_hybrid=True)
    rng = np.random.default_rng(5)
    running = rng.uniform(0.0, 5.0, (K, H, 4))
    terminal = rng.uniform(0.0, 5.0, (K, 4))
    want = ctrl._score_hybrid(_traj(running_terms=running, terminal_terms=terminal))

    obj = build_augmented_objective(ops=OPS, mode="hybrid", H=H, n_run=4, n_term=4,
                                    dt=float(backend.dt), floor_indices=[])
    got = obj.score(running_terms=running, terminal_terms=terminal)
    np.testing.assert_allclose(got, want, atol=ATOL)


# ---------------------------------------------------------------------------------
# The two structural assertions the handoff demands.
# ---------------------------------------------------------------------------------


def test_accumulator_actually_propagates():
    """No-op detector (the terminal_value lesson): the accumulated z must CHANGE the
    score. Zeroing the stage increments must move the readout, for every mode."""
    rng = np.random.default_rng(6)
    running_f, terminal_f = _atoms(rng, n_run=4, n_term=3)
    running = rng.uniform(0.5, 5.0, (K, H, 4))
    terminal = rng.uniform(0.5, 5.0, (K, 4))

    cases = [
        ("normal", dict(dt=0.01), dict(running_terms=running, terminal_terms=terminal),
         dict(terms=terminal)),
        ("fpl_cost", dict(p=-1.0, time_p=-2.0),
         dict(running_terms_f=running_f, terminal_terms_f=terminal_f),
         dict(terms_f=terminal_f)),
        ("fpl_discounted", dict(p=0.1),
         dict(running_terms_f=running_f, terminal_terms_f=terminal_f),
         dict(terms_f=terminal_f)),
    ]
    for mode, kw, arrays, term_kw in cases:
        obj = build_augmented_objective(ops=OPS, mode=mode, H=H, n_run=4, n_term=3, **kw)
        with_z = obj.score(**arrays)
        without_z = obj.terminal(obj.init((K,)), **term_kw)   # z frozen at init
        assert np.all(np.abs(with_z - without_z) > 1e-12), (
            f"{mode}: accumulator is a silent no-op -- z does not reach the score")


def test_augmented_dynamics_block_triangular():
    """The `Done when` of S9: dx'/dz == 0 through a REAL mjx step, by autodiff --
    asserted, not assumed. Also checks dz'/dz == I (z is carried, not remade) and
    dz'/dx != 0 (the stage half actually reads the state)."""
    jax = pytest.importorskip("jax")
    mjx = pytest.importorskip("mujoco.mjx")
    import jax.numpy as jnp

    from analytic_mppi.controllers.accumulator import jax_ops
    from analytic_mppi.dynamics.mjx_manifold import linearization_model

    with jax.experimental.enable_x64():
        task = make_task("pendulum")
        import mujoco
        model = linearization_model(mujoco.MjModel.from_xml_path(str(task.model_path)))
        mx = mjx.put_model(model)
        d0 = mjx.make_data(mx)

        obj = build_augmented_objective(ops=jax_ops(), mode="fpl_cost", H=H,
                                        n_run=2, n_term=2, p=-1.0, time_p=-2.0)

        def atoms(qpos, qvel):
            # Synthetic smooth fulfillments of the state; the assertion is about the
            # WIRING of the augmentation, not about any particular task's ramps.
            return jnp.stack([jax.nn.sigmoid(qpos[0]), jax.nn.sigmoid(-qvel[0])])

        def augmented_step(x, z):
            qpos, qvel = x[:mx.nq], x[mx.nq:]
            d = d0.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros((mx.nu,)))
            d = mjx.step(mx, d)
            x_next = jnp.concatenate([d.qpos, d.qvel])
            z_next = z + obj.stage(0, terms_f=atoms(qpos, qvel))
            return x_next, z_next

        x0 = jnp.array([0.3, -0.5], dtype=jnp.float64)
        z0 = obj.init(()) + 0.25

        (dx_dx, dx_dz), (dz_dx, dz_dz) = jax.jacfwd(augmented_step, argnums=(0, 1))(x0, z0)

        # The gate: physics never sees z. Exactly zero, not approximately.
        np.testing.assert_array_equal(np.asarray(dx_dz), 0.0)
        # z is carried forward untouched...
        np.testing.assert_allclose(np.asarray(dz_dz), np.eye(obj.n_acc), atol=ATOL)
        # ...and the stage half is live (guards against the no-op failure mode).
        assert np.any(np.abs(np.asarray(dz_dx)) > 1e-12)
