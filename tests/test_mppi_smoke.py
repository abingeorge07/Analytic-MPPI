from pathlib import Path

import numpy as np

from analytic_mppi.controllers import MPPI
from analytic_mppi.costs import GoalReachCost
from analytic_mppi.dynamics import MujocoBackend


MODEL_PATH = Path(__file__).resolve().parent.parent / "analytic_mppi" / "dynamics" / "unicycle.xml"


def test_mppi_drives_toward_goal():
    backend = MujocoBackend(MODEL_PATH, nthread=2)
    goal = np.array([1.5, 1.0])
    cost = GoalReachCost(
        goal=goal,
        state_xy_idx=(1, 2),
        Q=1.0,
        R=[0.01, 0.01, 0.001],
        terminal_weight=20.0,
    )
    ctl = MPPI(
        horizon=20,
        n_samples=128,
        nu=backend.nu,
        sigma=np.array([0.8, 0.8, 1.2]),
        lambda_=1.0,
        u_min=np.array([-2.0, -2.0, -3.0]),
        u_max=np.array([ 2.0,  2.0,  3.0]),
        seed=0,
    )

    state = backend.get_state()
    start_xy = state[[1, 2]].copy()
    for _ in range(30):
        u = ctl.act(state, cost, backend)
        state = backend.step(u)
    end_xy = state[[1, 2]]

    start_dist = float(np.linalg.norm(start_xy - goal))
    end_dist = float(np.linalg.norm(end_xy - goal))
    assert end_dist < 0.6 * start_dist, (
        f"MPPI did not reduce distance to goal enough: {start_dist:.3f} -> {end_dist:.3f}"
    )


def test_rollout_shape_matches_request():
    backend = MujocoBackend(MODEL_PATH, nthread=2)
    K, H = 8, 5
    init = np.broadcast_to(backend.get_state(), (K, backend.nstate)).copy()
    ctrls = np.zeros((K, H, backend.nu))
    states = backend.rollout(init, ctrls)
    assert states.shape == (K, H, backend.nstate)
