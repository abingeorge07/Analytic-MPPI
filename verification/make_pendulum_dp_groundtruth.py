"""Dynamic-programming (value-iteration) ground truth for pendulum swing-up.

WHY THIS EXISTS
---------------
To debug why the FPL samplers underperform we need a *reference optimum* for the
exact objective they chase. Value iteration gives the true infinite-horizon
optimum V*(s), the optimal action-values Q*(s,u), and the optimal policy pi*(s)
on a discretized (theta, theta_dot) grid — using the REAL MuJoCo pendulum
dynamics for one-step transitions.

WHICH OBJECTIVE (important correctness gate)
--------------------------------------------
DP requires an ADDITIVE per-step reward  return = sum_t gamma^t r(s_t, u_t).

  * `fpl_cost`  -> per step it does power_mean(fulfillment_terms) THEN discounts
                   over time (analytic_mppi/controllers/sampling_base.py:_score_fpl,
                   use_fpl_cost branch). The per-step scalar
                       r(s,u) = power_mean([upright_f(theta'), control_f(u)], p)
                   IS a valid per-step reward -> DP is EXACT.  <-- default here.
  * `fpl_discounted` -> discount-sums each term over the whole horizon FIRST, then
                   power-means across terms. The cross-term power-mean is on the
                   OUTSIDE -> NOT additive over time -> DP cannot represent it
                   exactly. Debug the controller in fpl_cost mode for an
                   apples-to-apples comparison against this artifact.
  * `normal`    -> additive quadratic penalties; reward = -(sum terms)*dt. Also
                   exact. Built with --objective normal for the task-success
                   reference.

Reward convention matches the controller: the per-step reward for the transition
(s --u--> s') is evaluated at the LANDING state s' and the applied control u,
because `running_cost_terms_f` is computed on the rollout states that EXCLUDE the
initial state (sampling_base.py:_rollout_and_score).

ARTIFACT (saved to data/)
-------------------------
np.savez_compressed with: theta_grid, thetadot_grid, u_grid, V, Q, pi_idx, pi_u,
reward, next_theta, next_thetadot, the optimal closed-loop trajectory from
hang-down, and a `meta` dict (objective, gamma, fpl_p, dt, grid sizes).

USAGE
-----
    python verification/make_pendulum_dp_groundtruth.py                 # fpl_cost
    python verification/make_pendulum_dp_groundtruth.py --objective normal
    python verification/make_pendulum_dp_groundtruth.py --n-theta 241 --n-thetadot 241
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analytic_mppi.tasks import make_task, power_mean          # noqa: E402
from analytic_mppi.dynamics import MujocoBackend               # noqa: E402

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
#  Grids
# ---------------------------------------------------------------------------

def build_grids(n_theta, n_thetadot, thetadot_max, n_u, u_min, u_max):
    # theta is PERIODIC: grid spans [0, 2pi) with uniform spacing so wrap-around
    # interpolation (last cell -> first cell) is exact.
    theta_grid = np.linspace(0.0, TWO_PI, n_theta, endpoint=False)
    thetadot_grid = np.linspace(-thetadot_max, thetadot_max, n_thetadot)
    u_grid = np.linspace(u_min, u_max, n_u)
    return theta_grid, thetadot_grid, u_grid


# ---------------------------------------------------------------------------
#  One-step transition model (real MuJoCo dynamics, batched)
# ---------------------------------------------------------------------------

def build_transition(backend, task, theta_grid, thetadot_grid, u_grid,
                     objective, fpl_p):
    """For every (theta, thetadot, u) grid point, step the real simulator once.

    Returns next_theta, next_thetadot, reward — each (n_th, n_td, n_u).
    """
    n_th, n_td, n_u = len(theta_grid), len(thetadot_grid), len(u_grid)
    TH, TD, U = np.meshgrid(theta_grid, thetadot_grid, u_grid, indexing="ij")
    B = TH.size

    template = backend.get_state()
    initial = np.tile(template, (B, 1)).astype(np.float64)
    initial[:, 1] = TH.ravel()      # qpos (theta)
    initial[:, 2] = TD.ravel()      # qvel (theta_dot)
    controls = U.ravel().reshape(B, 1, task.nu)

    states_out, sd_out = backend.rollout(initial, controls)   # (B,1,nstate), (B,1,nsd)
    qpos_next = states_out[:, 0, backend.qpos_slice]          # (B, nq)
    qvel_next = states_out[:, 0, backend.qvel_slice]          # (B, nv)
    u_flat = controls[:, 0, :]                                # (B, nu)
    sd_next = sd_out[:, 0, :]                                 # (B, nsd)

    reward = _step_reward(task, qpos_next, qvel_next, sd_next, u_flat,
                          objective, fpl_p, backend.dt)        # (B,)

    next_theta = np.mod(qpos_next[:, 0], TWO_PI)
    next_thetadot = qvel_next[:, 0]
    shape = (n_th, n_td, n_u)
    return (next_theta.reshape(shape),
            next_thetadot.reshape(shape),
            reward.reshape(shape))


def _step_reward(task, qpos_next, qvel_next, sd_next, u, objective, fpl_p, dt):
    """Per-step reward (LARGER is better) matching the controller's convention."""
    if objective == "normal":
        terms = task.running_cost_terms(qpos_next, qvel_next, sd_next, u)   # (B, n_terms)
        return -terms.sum(axis=-1) * dt
    # fpl_cost / (fpl_discounted uses the same per-step surrogate for DP)
    terms_f = task.running_cost_terms_f(qpos_next, qvel_next, sd_next, u)   # (B, n_terms_f)
    return power_mean(terms_f, fpl_p)


# ---------------------------------------------------------------------------
#  Bilinear interpolation of V on the (periodic-theta, clamped-thetadot) grid
# ---------------------------------------------------------------------------

def make_interpolator(theta_grid, thetadot_grid):
    n_th, n_td = len(theta_grid), len(thetadot_grid)
    dth = TWO_PI / n_th                       # periodic spacing
    td_min, td_max = thetadot_grid[0], thetadot_grid[-1]
    dtd = (td_max - td_min) / (n_td - 1)

    def interp(V, theta_q, thetadot_q):
        theta_q = np.mod(theta_q, TWO_PI)
        fth = theta_q / dth
        i0 = np.floor(fth).astype(np.int64) % n_th
        i1 = (i0 + 1) % n_th
        wt = fth - np.floor(fth)

        td_c = np.clip(thetadot_q, td_min, td_max)
        ftd = (td_c - td_min) / dtd
        j0 = np.clip(np.floor(ftd).astype(np.int64), 0, n_td - 2)
        j1 = j0 + 1
        wd = ftd - j0

        v00 = V[i0, j0]; v10 = V[i1, j0]; v01 = V[i0, j1]; v11 = V[i1, j1]
        return ((1 - wt) * (1 - wd) * v00 + wt * (1 - wd) * v10
                + (1 - wt) * wd * v01 + wt * wd * v11)

    return interp


# ---------------------------------------------------------------------------
#  Value iteration
# ---------------------------------------------------------------------------

def value_iteration(reward, next_theta, next_thetadot, interp,
                    gamma, tol, max_iters, verbose=True):
    n_th, n_td, n_u = reward.shape
    V = np.zeros((n_th, n_td), dtype=np.float64)
    nt_flat = next_theta.reshape(-1)
    ntd_flat = next_thetadot.reshape(-1)
    t0 = time.perf_counter()
    for it in range(max_iters):
        Vnext = interp(V, nt_flat, ntd_flat).reshape(n_th, n_td, n_u)
        Q = reward + gamma * Vnext
        V_new = Q.max(axis=-1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if verbose and (it % 50 == 0 or delta < tol):
            print(f"  VI iter {it:4d}  |dV|={delta:.3e}  "
                  f"({time.perf_counter()-t0:.1f}s)")
        if delta < tol:
            break
    Vnext = interp(V, nt_flat, ntd_flat).reshape(n_th, n_td, n_u)
    Q = reward + gamma * Vnext
    pi_idx = Q.argmax(axis=-1)
    return V, Q, pi_idx


# ---------------------------------------------------------------------------
#  Optimal closed-loop trajectory (one-step lookahead with V*, real dynamics)
# ---------------------------------------------------------------------------

def optimal_trajectory(backend, task, V, interp, u_grid, gamma, fpl_p,
                       objective, steps, theta0=0.0, thetadot0=0.0):
    """Greedy w.r.t. V* using the REAL simulator (no policy-grid discretization):
    at each state evaluate Q(s,u)=r(s,u)+gamma V*(s') for all candidate u via a
    one-step rollout, execute argmax. Returns states/ctrls/theta/thetadot/reward.
    """
    template = backend.get_state()
    s = template.copy()
    s[1] = theta0
    s[2] = thetadot0
    n_u = len(u_grid)

    states = [s.copy()]
    ctrls, rewards = [], []
    for _ in range(steps):
        init = np.tile(s, (n_u, 1))
        controls = u_grid.reshape(n_u, 1, task.nu)
        states_out, sd_out = backend.rollout(init, controls)   # (n_u,1,*)
        qpos_next = states_out[:, 0, backend.qpos_slice]
        qvel_next = states_out[:, 0, backend.qvel_slice]
        sd_next = sd_out[:, 0, :]
        r = _step_reward(task, qpos_next, qvel_next, sd_next,
                         controls[:, 0, :], objective, fpl_p, backend.dt)
        Vp = interp(V, np.mod(qpos_next[:, 0], TWO_PI), qvel_next[:, 0])
        Q = r + gamma * Vp
        k = int(Q.argmax())
        u = u_grid[k:k + 1].copy()
        # commit the chosen action on the real sim
        backend.set_state(s)
        s = backend.step(u).copy()
        states.append(s.copy())
        ctrls.append(u.copy())
        rewards.append(float(r[k]))

    states = np.asarray(states)
    ctrls = np.asarray(ctrls)
    rewards = np.asarray(rewards)
    disc = gamma ** np.arange(len(rewards))
    return dict(
        states=states,
        ctrls=ctrls,
        theta=states[:, 1],
        thetadot=states[:, 2],
        reward=rewards,
        discounted_return=float((disc * rewards).sum()),
    )


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", choices=["fpl_cost", "fpl_discounted", "normal"],
                    default="fpl_cost")
    ap.add_argument("--n-theta", type=int, default=181)
    ap.add_argument("--n-thetadot", type=int, default=141)
    ap.add_argument("--thetadot-max", type=float, default=10.0)
    ap.add_argument("--n-u", type=int, default=21)
    ap.add_argument("--gamma", type=float, default=0.99, help="match controller fpl_gamma")
    ap.add_argument("--fpl-p", type=float, default=0.1, help="match controller fpl_p")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument("--traj-steps", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    task = make_task("pendulum")
    backend = MujocoBackend(task.model_path, nthread=None)
    u_min = float(task.u_min[0]); u_max = float(task.u_max[0])

    print(f"objective = {args.objective}  gamma={args.gamma}  fpl_p={args.fpl_p}  dt={backend.dt}")
    print(f"grid: theta={args.n_theta} x thetadot={args.n_thetadot} "
          f"(|thetadot|<={args.thetadot_max}) x u={args.n_u}  "
          f"ctrl=[{u_min},{u_max}]")

    theta_grid, thetadot_grid, u_grid = build_grids(
        args.n_theta, args.n_thetadot, args.thetadot_max, args.n_u, u_min, u_max)

    print("building transition model (real MuJoCo one-step rollouts)...")
    t0 = time.perf_counter()
    next_theta, next_thetadot, reward = build_transition(
        backend, task, theta_grid, thetadot_grid, u_grid, args.objective, args.fpl_p)
    print(f"  {next_theta.size} (state,action) pairs in {time.perf_counter()-t0:.1f}s "
          f"| reward range [{reward.min():.4f}, {reward.max():.4f}]")

    interp = make_interpolator(theta_grid, thetadot_grid)

    print("value iteration...")
    V, Q, pi_idx = value_iteration(
        reward, next_theta, next_thetadot, interp,
        args.gamma, args.tol, args.max_iters)
    pi_u = u_grid[pi_idx]
    print(f"  V range [{V.min():.3f}, {V.max():.3f}]")

    print("rolling optimal closed-loop trajectory from hang-down...")
    traj = optimal_trajectory(
        backend, task, V, interp, u_grid, args.gamma, args.fpl_p,
        args.objective, args.traj_steps, theta0=0.0, thetadot0=0.0)
    final_theta = traj["theta"][-1]
    upright_err = abs(((final_theta - np.pi + np.pi) % TWO_PI) - np.pi)
    print(f"  discounted return = {traj['discounted_return']:.4f} | "
          f"final |theta-pi| = {upright_err:.4f} rad")
    if traj["thetadot"].__abs__().max() > 0.98 * args.thetadot_max:
        print(f"  WARNING: |thetadot| reached {abs(traj['thetadot']).max():.2f} "
              f"near grid edge {args.thetadot_max} — consider raising --thetadot-max")

    meta = dict(
        objective=args.objective, gamma=args.gamma, fpl_p=args.fpl_p,
        dt=backend.dt, n_theta=args.n_theta, n_thetadot=args.n_thetadot,
        thetadot_max=args.thetadot_max, n_u=args.n_u,
        u_min=u_min, u_max=u_max, task="pendulum",
        reward_convention="per-step at landing state s' and applied u; larger=better",
    )

    out = Path(args.out) if args.out else (REPO / "data" / f"pendulum_dp_{args.objective}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        theta_grid=theta_grid, thetadot_grid=thetadot_grid, u_grid=u_grid,
        V=V, Q=Q, pi_idx=pi_idx, pi_u=pi_u,
        reward=reward, next_theta=next_theta, next_thetadot=next_thetadot,
        traj_states=traj["states"], traj_ctrls=traj["ctrls"],
        traj_theta=traj["theta"], traj_thetadot=traj["thetadot"],
        traj_reward=traj["reward"], traj_discounted_return=traj["discounted_return"],
        meta=np.array(meta, dtype=object),
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
