"""LunarLander demo: vanilla MPPI flying Gymnasium's LunarLanderContinuous down
onto the pad, driven by the env's default reward converted to a cost.

    python -m analytic_mppi.envs.lunarlander.run                 # headless, save PNG
    python -m analytic_mppi.envs.lunarlander.run --live          # gymnasium viewer (needs pygame)
    python -m analytic_mppi.envs.lunarlander.run --n-samples 128 --horizon 25

Requires the optional deps: `pip install gymnasium box2d-py` (+ `pygame` for --live).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from analytic_mppi.controllers import list_controllers
from analytic_mppi.envs.lunarlander import make_env


# analytic_mppi/envs/lunarlander/run.py -> parents[3] = repo root
REPO = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO / "runs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=250, help="closed-loop steps (max episode length)")
    p.add_argument("--variant", default="vanilla", choices=list_controllers(), help="controller variant")
    p.add_argument("--n-samples", type=int, default=68, help="MPPI rollouts per step (serial Box2D)")
    p.add_argument("--horizon", type=int, default=20, help="planning horizon (steps)") # horizon * .02 (dt) = planning horizon
    p.add_argument("--sigma", type=float, default=0.5, help="control noise std (both actuators)")
    p.add_argument("--lambda", dest="lambda_", type=float, default=.1, help="MPPI temperature")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--live", action="store_true", help="render the gymnasium viewer (needs pygame)")
    p.add_argument("--out", type=Path, default=None, help="PNG path (headless); default runs/lunarlander.png")
    return p.parse_args()


def _run_closed_loop(env, steps: int, *, render: bool):
    """Receding-horizon control. Returns (obs_hist (T+1,8), ctrl_hist (T,2),
    term_hist (T,n_terms), outcome, plan_time)."""
    be, cost, ctrl = env.backend, env.cost, env.controller
    state = be.reset(env.backend.seed)
    ctrl.reset()
    obs_hist = [be.observe().copy()]
    ctrl_hist, t_plan = [], 0.0
    for _ in range(steps):
        t0 = time.perf_counter()
        u = ctrl.act(state, cost, be)
        t_plan += time.perf_counter() - t0
        state = be.step(u)
        if render:
            be.env.render()
        obs_hist.append(be.observe().copy())
        ctrl_hist.append(np.asarray(u, dtype=np.float64))
        if be.terminated:
            break
    obs_hist = np.asarray(obs_hist)
    ctrl_hist = np.asarray(ctrl_hist)
    # realized per-step running cost terms on the executed trajectory
    term_hist = cost.running_terms(obs_hist[1:], ctrl_hist)
    return obs_hist, ctrl_hist, term_hist, be.outcome(), t_plan


def _plot(obs_hist, term_hist, cost, outcome, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2))
    # left: lander x-y trajectory in observation frame (pad at origin)
    x, y = obs_hist[:, 0], obs_hist[:, 1]
    axL.plot(x, y, "-", color="tab:blue", lw=1.5)
    axL.plot(x[0], y[0], "o", color="tab:green", label="start")
    axL.plot(x[-1], y[-1], "*", color="tab:red", ms=14, label=f"end ({outcome})")
    axL.axhline(0, color="0.6", lw=0.8); axL.scatter([0], [0], marker="P", s=120, color="k", label="pad")
    axL.set_xlabel("x (norm)"); axL.set_ylabel("y (norm)"); axL.set_title("lander trajectory")
    axL.legend(loc="best"); axL.grid(alpha=0.3)

    # right: realized per-term running cost over time
    t = np.arange(term_hist.shape[0])
    for j, name in enumerate(cost.term_names):
        axR.plot(t, term_hist[:, j], lw=1.3, label=name)
    axR.plot(t, term_hist.sum(axis=1), "k--", lw=1.5, label="total")
    axR.set_xlabel("step"); axR.set_ylabel("realized running cost"); axR.set_title("per-term cost")
    axR.legend(loc="best", fontsize=8); axR.grid(alpha=0.3)

    fig.suptitle(f"LunarLanderContinuous + MPPI  (outcome: {outcome}, realized J="
                 f"{term_hist.sum():.1f})", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved {out_path}")


def main() -> None:
    args = _parse_args()
    env = make_env(
        variant=args.variant,
        seed=args.seed,
        render_mode="human" if args.live else None,
        controller_kwargs=dict(
            horizon=args.horizon, n_samples=args.n_samples,
            sigma=(args.sigma, args.sigma), lambda_=args.lambda_,
        ),
    )
    print(f"[lunarlander] variant={args.variant} K={args.n_samples} H={args.horizon} "
          f"sigma={args.sigma} lambda={args.lambda_} steps={args.steps}")
    obs_hist, ctrl_hist, term_hist, outcome, t_plan = _run_closed_loop(
        env, args.steps, render=args.live
    )
    T = len(ctrl_hist)
    print(f"  ran {T} steps | plan time {t_plan:.2f}s ({t_plan / max(T,1) * 1e3:.0f} ms/step) "
          f"| outcome={outcome} | realized running J={term_hist.sum():.1f}")
    if not args.live:
        _plot(obs_hist, term_hist, env.cost, outcome, args.out or (DEFAULT_RUNS / "lunarlander.png"))


if __name__ == "__main__":
    main()
