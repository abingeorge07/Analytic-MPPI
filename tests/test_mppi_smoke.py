import numpy as np
import pytest

from analytic_mppi.envs.unicycle import make_backend, make_cost, make_controller
from analytic_mppi.tasks import make_task, TASKS
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.controllers import (
    SAMPLING_CONTROLLERS,
    MPPIv2,
    MppiCma,
    CEM,
    DIAL,
    PredictiveSampling,
    ComposedGradientMPPI,
)
from analytic_mppi.controllers.sampling_base import Trajectory
from analytic_mppi.tasks.base import power_mean


# --------------------------- legacy unicycle MPPI ---------------------------

def test_mppi_drives_toward_goal():
    backend = make_backend(nthread=2)
    cost = make_cost(goal=(1.5, 1.0))
    ctl = make_controller(backend, horizon=20, n_samples=128, seed=0)

    state = backend.get_state()
    start_xy = state[[1, 2]].copy()
    for _ in range(30):
        u = ctl.act(state, cost, backend)
        state = backend.step(u)
    end_xy = state[[1, 2]]

    goal = np.array([1.5, 1.0])
    start_dist = float(np.linalg.norm(start_xy - goal))
    end_dist = float(np.linalg.norm(end_xy - goal))
    assert end_dist < 0.6 * start_dist, (
        f"MPPI did not reduce distance to goal enough: {start_dist:.3f} -> {end_dist:.3f}"
    )


def test_rollout_shape_matches_request():
    backend = make_backend(nthread=2)
    K, H = 8, 5
    init = np.broadcast_to(backend.get_state(), (K, backend.nstate)).copy()
    ctrls = np.zeros((K, H, backend.nu))
    states, sensordata = backend.rollout(init, ctrls)
    assert states.shape == (K, H, backend.nstate)
    assert sensordata.shape == (K, H, backend.nsensordata)


# --------------------------- Task cost-term shapes --------------------------

@pytest.mark.parametrize("name", sorted(TASKS))
def test_task_cost_term_shapes(name):
    """Each task's *_cost_terms methods accept batched arrays and return
    consistent (..., n_terms) outputs."""
    task = make_task(name)
    backend = MujocoBackend(task.model_path, nthread=2)
    K, H = 3, 4
    state = backend.get_state()
    init = np.broadcast_to(state, (K, backend.nstate)).copy()
    ctrls = np.zeros((K, H, task.nu))
    states, sd = backend.rollout(init, ctrls)
    qpos, qvel = task.qpos_of(states), task.qvel_of(states)

    rt = task.running_cost_terms(qpos, qvel, sd, ctrls)
    tt = task.terminal_cost_terms(qpos[:, -1], qvel[:, -1], sd[:, -1])
    rtf = task.running_cost_terms_f(qpos, qvel, sd, ctrls)
    ttf = task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sd[:, -1])

    assert rt.shape[:-1] == (K, H) and rt.shape[-1] >= 1
    assert tt.shape[:-1] == (K,)
    assert rtf.shape[:-1] == (K, H)
    assert ttf.shape[:-1] == (K,)
    assert np.all(np.isfinite(rt))
    assert np.all(np.isfinite(rtf))
    assert np.all((rtf >= 0.0) & (rtf <= 1.0 + 1e-9)), "FPL terms must be in [0,1]"


# ---------------------- New algorithms run on pendulum ----------------------

# cheap defaults: short horizon, low sample count
_PENDULUM_RUN = dict(
    num_samples=64, num_knots=4, plan_horizon=0.6, spline_type="zero", seed=0,
)


def _algo_kwargs(name):
    return {
        "mppi": dict(noise_level=0.5, temperature=1.0),
        "mppi_cma": dict(initial_noise_level=0.5, temperature=1.0, minimum_noise_level=0.1),
        "cem": dict(num_elites=8, sigma_start=1.0, sigma_min=0.1),
        "dial": dict(noise_level=0.5, temperature=1.0, beta_opt_iter=3.0, beta_horizon=3.0),
        "predictive_sampling": dict(noise_level=0.5),
        # FplGmm requires FPL mode (absolute [0,1] reward scale drives its adaptation).
        "fpl_gmm": dict(use_fpl_cost=True, sigma_max=1.0, sigma_min=0.1, allocation="mixture"),
        # ComposedGradientMPPI requires a per-objective FPL vector (discounted here).
        "composed_grad": dict(noise_level=0.5, temperature=1.0,
                              use_fpl_discounted=True, fpl_p=0.1, fpl_gamma=0.99),
        # FplAdaptiveMPPI default steer_mode="binding" needs a per-objective FPL vector.
        "fpl_adaptive": dict(temperature=1.0, use_fpl_discounted=True, fpl_p=0.1,
                             fpl_gamma=0.99),
    }[name]


@pytest.mark.parametrize("algo_name", sorted(SAMPLING_CONTROLLERS))
def test_pendulum_algorithm_runs(algo_name):
    """Each new algorithm produces finite actions for 30 closed-loop steps."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    cls = SAMPLING_CONTROLLERS[algo_name]
    ctrl = cls(task, backend, **_algo_kwargs(algo_name), **_PENDULUM_RUN)
    state = backend.get_state()
    for _ in range(30):
        u = ctrl.act(state)
        assert u.shape == (task.nu,)
        assert np.all(np.isfinite(u))
        state = backend.step(u)


@pytest.mark.parametrize("fpl_mode", ["fpl_cost", "fpl_discounted"])
def test_pendulum_mppi_fpl_paths_run(fpl_mode):
    """FPL paths produce finite actions and aren't accidentally noop."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    kwargs = dict(use_fpl_cost=(fpl_mode == "fpl_cost"),
                  use_fpl_discounted=(fpl_mode == "fpl_discounted"),
                  fpl_p=0.1, fpl_gamma=0.99)
    ctrl = MPPIv2(task, backend, **_algo_kwargs("mppi"), **kwargs, **_PENDULUM_RUN)
    state = backend.get_state()
    actions = []
    for _ in range(30):
        u = ctrl.act(state)
        actions.append(u.copy())
        assert np.all(np.isfinite(u))
        state = backend.step(u)
    actions = np.asarray(actions)
    # at least some action variability — i.e. the controller is doing something
    assert actions.std() > 1e-3, "FPL action sequence collapsed to a constant"


# ---------------------- FPL == true MPPI (only scalarization differs) --------

def test_power_mean_p1_equals_arithmetic_mean():
    """p=+1 makes the power-mean a plain arithmetic mean == a LINEAR scalarization
    of the atoms. This is what lets one p-knob interpolate linear <-> FPL."""
    from analytic_mppi.tasks.base import power_mean
    x = np.random.default_rng(0).uniform(0.01, 1.0, size=(5, 4))
    assert np.allclose(power_mean(x, 1.0), x.mean(axis=-1))


def test_weighted_power_mean_is_linear_family():
    """power_mean(x, p=1, weights=w) == Σ w_i x_i (normalized) — the whole LINEAR-weight
    family. Uniform weights reproduce the unweighted mean for any p (the FPL side)."""
    from analytic_mppi.tasks.base import power_mean
    x = np.random.default_rng(1).uniform(0.05, 1.0, size=(6, 4))
    w = np.array([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(power_mean(x, 1.0, weights=w), (x * w).sum(-1) / w.sum())
    for p in (1.0, 0.0, -1.0, -2.0):
        assert np.allclose(power_mean(x, p), power_mean(x, p, weights=np.ones(4)))


def test_fpl_weights_recover_single_atom_and_uniform():
    """MPPIv2 fpl_weights on the discounted collapse: a near-one-hot weight makes the
    composite track that atom's FQ-value; uniform weights == the unweighted reward."""
    from analytic_mppi.tasks.base import power_mean
    task = make_task("hopper")
    backend = MujocoBackend(task.model_path, nthread=2)
    base = MPPIv2(task, backend, use_fpl_discounted=True, fpl_p=1.0, fpl_gamma=0.99,
                  noise_level=0.4, temperature=1.0, **_PENDULUM_RUN)
    traj = base._rollout_and_score(backend.get_state())
    V = traj.reward_terms                                   # (K, n_atoms) FQ-values
    # uniform weights reproduce the unweighted p=1 composite
    assert np.allclose(traj.reward, power_mean(V, 1.0))
    # a near-one-hot weight on atom 2 (velocity) makes the composite ~ that atom
    w = np.array([1e-6, 1e-6, 1.0, 1e-6])
    assert np.allclose(power_mean(V, 1.0, weights=w), V[:, 2], atol=1e-3)


def test_fpl_score_is_neg_log_reward_and_weighted_average():
    """FPL scoring must be S_k = -log(u_k) (u in (0,1]), and MPPIv2 must consume it
    with a softmax-weighted AVERAGE — not an argmax on the single best rollout.
    Guards the two core fixes that make FPL a true MPPI."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = MPPIv2(task, backend, use_fpl_cost=True, fpl_p=-1.0, fpl_gamma=0.99,
                  noise_level=0.5, temperature=1.0, **_PENDULUM_RUN)
    state = backend.get_state()
    traj = ctrl._rollout_and_score(state)

    # reward in (0,1]; scores are exactly -log(reward).
    assert np.all(traj.reward > 0.0) and np.all(traj.reward <= 1.0 + 1e-9)
    assert np.allclose(traj.scores, -np.log(traj.reward))

    new_mean = ctrl.update_mean(traj)
    # A weighted average, NOT argmax on the best rollout.
    argmax_knot = traj.knots[int(traj.reward.argmax())]
    assert not np.allclose(new_mean, argmax_knot), \
        "FPL update collapsed to argmax; expected a softmax-weighted average"
    # ESS is logged and in [1, K].
    assert ctrl.last_ess is not None
    assert 1.0 <= ctrl.last_ess <= ctrl.num_samples + 1e-6


def test_fpl_layered_two_level_power_mean():
    """Layered FPL is a TWO-level composition: discount-sum each atom, inner power-mean
    (fpl_group_p) the atoms within each task.fpl_groups group, then outer power-mean
    (fpl_p) the group scalars. Distinct inner/outer p must both take effect."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = MPPIv2(task, backend, use_fpl_layered=True, fpl_p=-2.0, fpl_group_p=1.0,
                  fpl_gamma=0.9, noise_level=0.5, temperature=1.0, **_PENDULUM_RUN)
    # group 0 = atoms {0,1}, group 1 = atom {2}; overridden on the shared task instance.
    ctrl.task.fpl_groups = [[0, 1], [2]]

    # Atoms held CONSTANT over time => the normalized discounted sum equals the constant,
    # so each per-term FQ-value is exactly its atom value and the math is checkable.
    K, H, n = 4, ctrl.H, 3
    vals = np.random.default_rng(0).uniform(0.2, 1.0, size=(K, n))
    running = np.broadcast_to(vals[:, None, :], (K, H, n)).copy()
    traj = Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None,
                      running_terms_f=running, terminal_terms_f=vals.copy())
    scores = ctrl._score_fpl_layered(traj)

    g0 = power_mean(vals[:, [0, 1]], 1.0)                    # inner p = fpl_group_p
    g1 = power_mean(vals[:, [2]], 1.0)
    expected = power_mean(np.stack([g0, g1], axis=-1), -2.0)  # outer p = fpl_p
    assert np.allclose(traj.reward, expected)
    assert np.allclose(scores, -np.log(expected))
    assert np.all(traj.reward > 0.0) and np.all(traj.reward <= 1.0 + 1e-9)


# -------------- ComposedGradientMPPI (multi-objective gradient composition) --

def test_composed_gradient_requires_per_objective_fpl():
    """The controller needs a per-objective [0,1] vector; normal cost and fpl_cost
    (which power-means objectives per-step before any vector exists) are rejected."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    with pytest.raises(ValueError):
        ComposedGradientMPPI(task, backend, noise_level=0.5, temperature=1.0, **_PENDULUM_RUN)
    with pytest.raises(ValueError):
        ComposedGradientMPPI(task, backend, use_fpl_cost=True, fpl_p=0.1,
                             noise_level=0.5, temperature=1.0, **_PENDULUM_RUN)


def test_composed_reward_terms_invariant_discounted():
    """reward_terms is the per-objective [0,1] vector and collapses to reward via
    power_mean(., fpl_p) — the invariant the composition relies on (discounted)."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = ComposedGradientMPPI(task, backend, use_fpl_discounted=True, fpl_p=0.1,
                                fpl_gamma=0.99, noise_level=0.5, temperature=1.0,
                                **_PENDULUM_RUN)
    traj = ctrl._rollout_and_score(backend.get_state())
    assert traj.reward_terms is not None
    assert traj.reward_terms.shape[0] == ctrl.num_samples
    assert np.all(traj.reward_terms >= 0.0) and np.all(traj.reward_terms <= 1.0 + 1e-9)
    assert np.allclose(traj.reward, power_mean(traj.reward_terms, ctrl.fpl_p))


def test_composed_reward_terms_invariant_layered():
    """Same invariant for layered mode: reward_terms are the per-GROUP scalars, and
    reward == power_mean(group_scores, fpl_p). (Built synthetically like the layered
    test above since pendulum has no grouped-atom method.)"""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = ComposedGradientMPPI(task, backend, use_fpl_layered=True, fpl_p=-2.0,
                                fpl_group_p=1.0, fpl_gamma=0.9, noise_level=0.5,
                                temperature=1.0, **_PENDULUM_RUN)
    ctrl.task.fpl_groups = [[0, 1], [2]]
    K, H, n = 4, ctrl.H, 3
    vals = np.random.default_rng(1).uniform(0.2, 1.0, size=(K, n))
    running = np.broadcast_to(vals[:, None, :], (K, H, n)).copy()
    traj = Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None,
                      running_terms_f=running, terminal_terms_f=vals.copy())
    ctrl._score_fpl_layered(traj)
    assert traj.reward_terms is not None and traj.reward_terms.shape == (K, 2)
    assert np.allclose(traj.reward, power_mean(traj.reward_terms, ctrl.fpl_p))


@pytest.mark.parametrize("compose", ["worst_first", "uniform"])
def test_composed_update_stays_in_sample_hull(compose):
    """Core stability guarantee: the convex-combination update lies within the
    per-coordinate hull of the sampled knots — no step size, no clipping needed."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = ComposedGradientMPPI(task, backend, use_fpl_discounted=True, fpl_p=0.1,
                                fpl_gamma=0.99, compose=compose, noise_level=0.5,
                                temperature=1.0, **_PENDULUM_RUN)
    traj = ctrl._rollout_and_score(backend.get_state())
    new_mean = ctrl.update_mean(traj)
    lo, hi = traj.knots.min(axis=0), traj.knots.max(axis=0)
    assert np.all(new_mean >= lo - 1e-9) and np.all(new_mean <= hi + 1e-9)


def test_composed_worst_objective_gets_more_weight():
    """worst_first + power_p must place MORE composition weight on the lower-satisfied
    objective. Synthetic reward_terms: objective 0 ≈0.1, objective 1 ≈0.9."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = ComposedGradientMPPI(task, backend, use_fpl_discounted=True, fpl_p=0.1,
                                fpl_gamma=0.99, compose="worst_first",
                                alpha_mode="power_p", noise_level=0.5, temperature=1.0,
                                **_PENDULUM_RUN)
    K = ctrl.num_samples
    rng = np.random.default_rng(0)
    V = np.stack([rng.uniform(0.05, 0.15, size=K),
                  rng.uniform(0.85, 0.95, size=K)], axis=1)
    knots = rng.normal(size=(K, ctrl.num_knots, task.nu))
    traj = Trajectory(knots=knots, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None, reward_terms=V)
    ctrl.update_mean(traj)
    assert ctrl.last_alpha[0] > ctrl.last_alpha[1]


def test_composed_diagnostics_populated():
    """last_alpha (sums to 1, ≥0), per-objective ESS and combined ESS are all in range."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = ComposedGradientMPPI(task, backend, use_fpl_discounted=True, fpl_p=0.1,
                                fpl_gamma=0.99, noise_level=0.5, temperature=1.0,
                                **_PENDULUM_RUN)
    traj = ctrl._rollout_and_score(backend.get_state())
    ctrl.update_mean(traj)
    J = traj.reward_terms.shape[1]
    assert ctrl.last_alpha.shape == (J,)
    assert np.isclose(ctrl.last_alpha.sum(), 1.0) and np.all(ctrl.last_alpha >= 0.0)
    assert ctrl.last_obj_ess.shape == (J,)
    K = ctrl.num_samples
    assert np.all((ctrl.last_obj_ess >= 1.0 - 1e-6) & (ctrl.last_obj_ess <= K + 1e-6))
    assert 1.0 - 1e-6 <= ctrl.last_ess <= K + 1e-6


def test_composed_gradient_closed_loop_nonconstant():
    """30 closed-loop steps on pendulum: finite actions that actually vary."""
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = ComposedGradientMPPI(task, backend, use_fpl_discounted=True, fpl_p=0.1,
                                fpl_gamma=0.99, noise_level=0.5, temperature=1.0,
                                **_PENDULUM_RUN)
    state = backend.get_state()
    actions = []
    for _ in range(30):
        u = ctrl.act(state)
        assert np.all(np.isfinite(u))
        actions.append(u.copy())
        state = backend.step(u)
    assert np.asarray(actions).std() > 1e-3


# -------------- FplAdaptiveMPPI (FPL-informed adaptive-covariance sampler) ----

def test_fpl_adaptive_binding_requires_per_objective_fpl():
    """steer_mode='binding' needs the per-objective vector (fpl_discounted/layered);
    normal cost and fpl_cost (no vector) must be rejected."""
    from analytic_mppi.controllers import FplAdaptiveMPPI
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    with pytest.raises(ValueError):
        FplAdaptiveMPPI(task, backend, steer_mode="binding", temperature=1.0, **_PENDULUM_RUN)
    with pytest.raises(ValueError):
        FplAdaptiveMPPI(task, backend, steer_mode="binding", use_fpl_cost=True, fpl_p=0.1,
                        temperature=1.0, **_PENDULUM_RUN)


def test_fpl_adaptive_scalar_gap_runs_on_normal_cost():
    """The mirror-able ablation (steer='scalar', explore='gap') needs no FPL vector, so it
    runs on plain cost — this is the fair non-FPL baseline for A/B sampling comparisons."""
    from analytic_mppi.controllers import FplAdaptiveMPPI
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = FplAdaptiveMPPI(task, backend, steer_mode="scalar", explore_mode="gap",
                           temperature=1.0, **_PENDULUM_RUN)
    state = backend.get_state()
    for _ in range(20):
        u = ctrl.act(state)
        assert np.all(np.isfinite(u))
        state = backend.step(u)
    assert ctrl.last_commit is not None and 0.0 <= ctrl.last_commit <= 1.0


def test_fpl_adaptive_binding_closed_loop_and_diagnostics():
    """Binding-steered FPL sampler: finite, varying actions; sigma stays within [floor,ceil];
    the binding-objective index and commitment scalar are exposed."""
    from analytic_mppi.controllers import FplAdaptiveMPPI
    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = FplAdaptiveMPPI(task, backend, use_fpl_discounted=True, fpl_p=0.1, fpl_gamma=0.99,
                           steer_mode="binding", explore_mode="absolute", temperature=1.0,
                           sigma_floor=0.05, sigma_ceil=1.2, iterations=2, **_PENDULUM_RUN)
    state = backend.get_state()
    actions = []
    for _ in range(25):
        u = ctrl.act(state)
        assert np.all(np.isfinite(u))
        actions.append(u.copy())
        state = backend.step(u)
    assert np.asarray(actions).std() > 1e-3
    assert np.all(ctrl.sigma >= ctrl.sigma_floor - 1e-9)
    assert np.all(ctrl.sigma <= ctrl.sigma_ceil + 1e-9)
    assert ctrl.last_binding_obj is not None
    assert 0.0 <= ctrl.last_commit <= 1.0


# --------------------------- model-mismatch perturbation ---------------------------

def test_apply_perturbation_scales_params():
    """apply_perturbation scales the requested model arrays in place; unknown keys raise."""
    from analytic_mppi.dynamics import apply_perturbation
    task = make_task("hopper")
    b = MujocoBackend(task.model_path, nthread=2)
    m0_mass = b.model.body_mass.copy()
    m0_gain = b.model.actuator_gainprm[:, 0].copy()
    m0_fric = b.model.geom_friction[:, 0].copy()
    apply_perturbation(b.model, dict(mass_scale=1.5, gain_scale=0.5, friction_scale=0.25))
    assert np.allclose(b.model.body_mass, 1.5 * m0_mass)
    assert np.allclose(b.model.actuator_gainprm[:, 0], 0.5 * m0_gain)
    assert np.allclose(b.model.geom_friction[:, 0], 0.25 * m0_fric)
    with pytest.raises(KeyError):
        apply_perturbation(b.model, dict(bogus_scale=2.0))


def test_true_perturbation_splits_plan_and_step_models():
    """make_controller with true_perturbation: the controller plans on the NOMINAL model
    while the returned (true) backend has perturbed dynamics — so the two step differently
    under the same control."""
    from analytic_mppi.eval import make_controller
    common = dict(num_samples=8, plan_horizon=0.4, num_knots=4, spline_type="zero",
                  cost_mode="fpl_cost", noise_level=0.3, temperature=0.2, fpl_p=-1.0)
    _, plan_backend, ctrl = make_controller("hopper", "mppi", **common)          # nominal
    _, true_backend, _ = make_controller("hopper", "mppi",
                                         true_perturbation=dict(mass_scale=2.0), **common)
    # The controller's own backend (planning) is nominal, distinct from a perturbed true one.
    assert ctrl.backend is plan_backend
    assert np.allclose(ctrl.backend.model.body_mass, plan_backend.model.body_mass)
    assert np.allclose(true_backend.model.body_mass, 2.0 * plan_backend.model.body_mass)
    # Same NONZERO control from the same initial state evolves differently under the perturbed
    # model. (Zero control would fall identically — gravitational acceleration is mass-
    # independent — so drive the actuators, whose forces give mass-dependent accelerations.)
    u = np.full(true_backend.nu, 0.5)
    s0 = plan_backend.get_state()
    plan_backend.set_state(s0)
    true_backend.set_state(s0)
    for _ in range(20):
        plan_backend.step(u)
        true_backend.step(u)
    assert not np.allclose(plan_backend.get_state(), true_backend.get_state())
