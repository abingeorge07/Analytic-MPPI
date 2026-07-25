"""Does ComposedGradientMPPI's inner loop CONVERGE at a fixed state?

Each act() already runs `iterations` inner steps of  sample -> per-objective gradients ->
compose -> update  before committing an action. Here we hold the state FIXED and run that
inner loop for many steps, evaluating the NOISE-FREE mean plan after each step, to see
whether the composed FPL reward climbs and plateaus (and whether the step size ||Δmean||
settles). This is the "a few cycles of gradient descent" question made measurable.

Note: with a fixed `noise_level` the mean converges to a NEIGHBOURHOOD of the smoothed
optimum, not a point — ||Δmean|| settles to a noise floor rather than 0. Pass --anneal to
shrink the sampling noise each step for a tighter fixed-point.

Run:  python verification/composed_gradient_convergence.py            # g1 layered
      python verification/composed_gradient_convergence.py --shove 3  # harder recovery state
      python verification/composed_gradient_convergence.py --anneal 0.93
"""
from __future__ import annotations

import argparse
import numpy as np
import mujoco

from analytic_mppi.tasks import make_task
from analytic_mppi.eval import make_controller, init_g1_stand, init_hopper_stand
from analytic_mppi.controllers.spline import interpolate
from analytic_mppi.controllers.sampling_base import Trajectory


def score_mean(ctrl, task, backend, state):
    """Deterministically score the controller's CURRENT mean plan (no sampling noise):
    roll the mean out once and run it through the exact FPL scorer the controller uses.
    Returns (reward, per_objective_satisfaction)."""
    controls = interpolate(ctrl.mean[None], ctrl.tk, ctrl.t_eval, ctrl.spline_type)  # (1,H,nu)
    init = np.broadcast_to(state, (1, backend.nstate)).copy()
    states, sd = backend.rollout(init, controls)
    qpos, qvel = task.qpos_of(states), task.qvel_of(states)
    traj = Trajectory(knots=None, controls=None, states=None, sensordata=None,
                      qpos=None, qvel=None)
    if ctrl.use_fpl_layered:
        traj.running_terms_f = task.running_cost_terms_f_grouped(qpos, qvel, sd, controls)
        traj.terminal_terms_f = task.terminal_cost_terms_f_grouped(qpos[:, -1], qvel[:, -1], sd[:, -1])
        ctrl._score_fpl_layered(traj)
    else:  # discounted
        rf = task.running_cost_terms_f(qpos, qvel, sd, controls)
        tf = task.terminal_cost_terms_f(qpos[:, -1], qvel[:, -1], sd[:, -1])
        if ctrl.fpl_term_indices is not None:
            rf, tf = ctrl._select_fpl_terms(rf, tf)
        traj.running_terms_f, traj.terminal_terms_f = rf, tf
        ctrl._score_fpl(traj)
    return float(traj.reward[0]), traj.reward_terms[0].copy()


def make_shove(vx):
    def _init(backend):
        kf = backend.model.keyframe("stand")
        backend.data.qpos[:] = kf.qpos
        backend.data.qvel[:] = 0.0
        backend.data.qvel[0] = vx
        mujoco.mj_forward(backend.model, backend.data)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="g1_standup")
    ap.add_argument("--cost-mode", default="fpl_layered", choices=["fpl_layered", "fpl_discounted"])
    ap.add_argument("--iters", type=int, default=40, help="inner GD steps at the fixed state")
    ap.add_argument("--shove", type=float, default=0.0, help="g1 only: initial torso vx")
    ap.add_argument("--num-samples", type=int, default=128)
    ap.add_argument("--noise-level", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--fpl-p", type=float, default=0.1)
    ap.add_argument("--anneal", type=float, default=1.0, help="multiply noise_level each step (<1 tightens)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    extra = dict(fpl_group_p=args.fpl_p) if args.cost_mode == "fpl_layered" else {}
    task, backend, ctrl = make_controller(
        args.task, "composed_grad", cost_mode=args.cost_mode, seed=args.seed,
        num_samples=args.num_samples, plan_horizon=0.5, num_knots=4,
        noise_level=args.noise_level, temperature=args.temperature, fpl_p=args.fpl_p,
        compose="worst_first", alpha_mode="power_p", **extra,
    )
    if args.task == "g1_standup":
        (make_shove(args.shove) if args.shove else init_g1_stand)(backend)
    elif args.task == "hopper":
        init_hopper_stand(backend)
    state = backend.get_state()

    names = task.fpl_group_names if args.cost_mode == "fpl_layered" else task.cost_term_names_f

    print(f"\nComposed-gradient inner-loop convergence @ FIXED state")
    print(f"task={args.task} cost_mode={args.cost_mode} K={args.num_samples} "
          f"noise={args.noise_level} temp={args.temperature} fpl_p={args.fpl_p} "
          f"anneal={args.anneal}" + (f" shove={args.shove}" if args.shove else ""))
    print("=" * 84)
    r0, s0 = score_mean(ctrl, task, backend, state)
    print(f"{'iter':>4} {'reward':>9} {'min_s':>8} {'||dmean||':>10} {'ess':>7}   per-objective s_j")
    print(f"{0:>4} {r0:9.4f} {s0.min():8.4f} {'':>10} {'':>7}   "
          + " ".join(f"{n[:5]}={v:.3f}" for n, v in zip(names, s0)))

    rewards = [r0]
    for it in range(1, args.iters + 1):
        prev = ctrl.mean.copy()
        traj = ctrl._rollout_and_score(state)     # sample fresh noise around current mean
        ctrl.mean = ctrl.update_mean(traj)         # ONE composed-gradient step (no warm-shift)
        ctrl.noise_level *= args.anneal
        reward, s = score_mean(ctrl, task, backend, state)
        rewards.append(reward)
        dmean = float(np.linalg.norm(ctrl.mean - prev))
        if it <= 10 or it % 2 == 0:
            print(f"{it:>4} {reward:9.4f} {s.min():8.4f} {dmean:10.5f} {ctrl.last_ess:7.1f}   "
                  + " ".join(f"{n[:5]}={v:.3f}" for n, v in zip(names, s)))

    rewards = np.asarray(rewards)
    gain = rewards[-1] - rewards[0]
    if abs(gain) > 1e-6:
        thresh = rewards[0] + 0.95 * gain
        hit = int(np.argmax(rewards >= thresh)) if gain > 0 else int(np.argmax(rewards <= thresh))
    else:
        hit = 0
    print("=" * 84)
    print(f"reward: {rewards[0]:.4f} -> {rewards[-1]:.4f}  (Δ={gain:+.4f});  "
          f"reached 95% of the gain by iter {hit};  last-5 std={rewards[-5:].std():.4f}")
    print("Converged if reward plateaus (small last-5 std) and ||dmean|| settles to a floor.")


if __name__ == "__main__":
    main()
