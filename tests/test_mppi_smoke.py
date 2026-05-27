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
)


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
