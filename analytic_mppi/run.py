"""Unified CLI for the CPU sampling-based MPC framework.

Two ways in.

1. A CONFIG FILE (preferred -- docs/config_design.md). One reviewable file per run, no
   silently-dropped parameters, and a JSON provenance record of exactly what ran:

    python -m analytic_mppi.run --config configs/env/hopper.py --live
    python -m analytic_mppi.run --config configs/exp/hopper_argmax_ablation.py --index 1
    python -m analytic_mppi.run --config configs/env/hopper.py \\
        --set objective.p=1.0 --set task.kwargs.target_velocity=3.0 --print-config

2. FLAGS (legacy). Note that only the flags listed in `_ALGO_PARAMS` for the chosen
   --algo are forwarded; the rest are silently ignored. Prefer --config.

    python -m analytic_mppi.run --task <name> --algo <name> [common params] \\
                                [algo params] [--fpl|--fpl-discounted ...] \\
                                [--live | --record OUT.mp4]

Tasks  (--task):  see analytic_mppi.tasks.TASKS
Algos  (--algo):  see analytic_mppi.controllers.SAMPLING_CONTROLLERS

Examples:
    python -m analytic_mppi.run --task pendulum --algo mppi \\
        --steps 200 --num-samples 512 --num-knots 6 --plan-horizon-sec 1.0 \\
        --noise-level 0.5 --temperature 1.0
    python -m analytic_mppi.run --task walker --algo dial --live \\
        --steps 200 --num-samples 256 --num-knots 4 --plan-horizon-sec 0.5 \\
        --noise-level 0.3 --temperature 1.0 --beta-opt-iter 3 --beta-horizon 3
    python -m analytic_mppi.run --task g1_standup --algo mppi --live \\
        --steps 200 --num-samples 128 --num-knots 4 --plan-horizon-sec 0.5 \\
        --noise-level 0.3 --temperature 1.0 --fpl --fpl-p 0.1 --fpl-gamma 0.99
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from analytic_mppi.tasks import TASKS, make_task
from analytic_mppi.controllers import SAMPLING_CONTROLLERS, get_sampling_controller_class
from analytic_mppi.dynamics import MujocoBackend


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = REPO / "runs"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Either --config (preferred, see docs/config_design.md) or --task + --algo.
    p.add_argument("--config", default=None, metavar="PATH",
                   help="python config file exposing CONFIG or configs() "
                        "(e.g. configs/env/hopper.py). Supersedes --task/--algo and every "
                        "hyperparameter flag below.")
    p.add_argument("--index", type=int, default=None,
                   help="--config only: which entry of configs() to run")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="--config only: dotted-path override, repeatable "
                        "(e.g. --set objective.p=1.0 --set task.kwargs.target_velocity=3.0). "
                        "Values are parsed as JSON.")
    p.add_argument("--print-config", action="store_true",
                   help="--config only: print the fully-resolved config as JSON and exit")
    p.add_argument("--save-config", default=None, metavar="DIR",
                   help="--config only: write config.resolved.json + provenance.json to DIR")

    p.add_argument("--task", choices=sorted(TASKS), help="task name (omit when using --config)")
    p.add_argument("--algo", choices=sorted(SAMPLING_CONTROLLERS),
                   help="algorithm (omit when using --config)")

    # rollout / sampling
    p.add_argument("--steps", type=int, default=20000, help="number of MPC steps (closed-loop)")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--num-knots", type=int, default=4)
    p.add_argument("--plan-horizon-sec", type=float, default=0.6)
    p.add_argument("--spline-type", choices=["zero", "linear", "cubic"], default="zero")
    p.add_argument("--iterations", type=int, default=1, help="optimization iterations per MPC step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nthread", type=int, default=None, help="rollout threads (default: cpu_count)")

    # algo-specific (only the relevant ones for the chosen --algo are applied).
    # Defaults are conservative, pendulum-tuned starting points — override as needed.
    p.add_argument("--noise-level", type=float, default=0.5,
                   help="MPPI / DIAL / Predictive Sampling — Gaussian sample stddev (default: 0.5)")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="MPPI / DIAL / MPPI-CMA — softmax temperature lambda (default: 1.0)")
    # MPPI-CMA
    p.add_argument("--initial-noise-level", type=float, default=0.3,
                   help="MPPI-CMA initial sigma (default: 0.5)")
    p.add_argument("--minimum-noise-level", type=float, default=0.3,
                   help="MPPI-CMA covariance eigenvalue floor (default: 0.1)")
    p.add_argument("--cov-rate", type=float, default=0.1,
                   help="MPPI-CMA covariance adaptation rate alpha (default: 0.1)")
    # CEM
    p.add_argument("--sigma-start", type=float, default=1.0, help="CEM initial sigma (default: 1.0)")
    p.add_argument("--sigma-min", type=float, default=0.1, help="CEM minimum sigma (default: 0.1)")
    p.add_argument("--num-elites", type=int, default=32, help="CEM number of elite samples (default: 32)")
    p.add_argument("--explore-fraction", type=float, default=0.0,
                   help="CEM fraction of samples kept at sigma_start (default: 0.0)")
    # DIAL
    p.add_argument("--beta-opt-iter", type=float, default=3.0, help="DIAL beta_1 (default: 3.0)")
    p.add_argument("--beta-horizon", type=float, default=3.0, help="DIAL beta_2 (default: 3.0)")
    # FplGmm (requires --fpl or --fpl-discounted). sigma_max <- --sigma-start, sigma_min <- --sigma-min.
    p.add_argument("--allocation", choices=["mixture", "sus"], default="mixture",
                   help="FplGmm: 'mixture' (multinomial GMM draw) or 'sus' (evolutionary resampler)")

    # FPL toggles
    fpl_group = p.add_mutually_exclusive_group()
    fpl_group.add_argument("--fpl", action="store_true",
                           help="FPL cost: per-step running_cost_f then discount-sum over time")
    fpl_group.add_argument("--fpl-discounted", action="store_true",
                           help="FPL discounted: per-term discount-sum then power-mean over terms")
    p.add_argument("--fpl-p", type=float, default=0.1, help="FPL power-mean exponent")
    p.add_argument("--fpl-gamma", type=float, default=0.99, help="FPL temporal discount")
    p.add_argument("--fpl-time-p", type=float, default=None,
                   help="FPL cost time aggregation: default (unset) = discounted mean over "
                        "time; set q<=0 for soft-min over time (rollout value = its worst moment)")
    p.add_argument("--fpl-time-discount", action="store_true",
                   help="with --fpl-time-p set, also weight the soft-min-over-time by the "
                        "gamma discount (default off = pure soft-min, gamma ignored)")

    # output / viz
    p.add_argument("--live", action="store_true", help="open interactive MuJoCo viewer")
    p.add_argument("--record", action="store_true", help="render to mp4 at --out")
    p.add_argument("--out", default=None, help="output path for --record")
    p.add_argument("--camera", default=None, help="MJCF camera name for --record (default: free)")
    p.add_argument("--print-costs", action="store_true",
                   help="print per-step cost terms (and FPL terms if enabled) of the closed-loop state")
    p.add_argument("--print-every", type=int, default=1,
                   help="thin the --print-costs output: print every Nth step (default: 1)")
    return p


# Mapping --algo -> list of (cli_arg_name, ctor_param_name) pairs.
# Only these params are forwarded from argparse to the controller constructor.
_ALGO_PARAMS: Dict[str, list[tuple[str, str]]] = {
    "mppi": [("noise_level", "noise_level"), ("temperature", "temperature")],
    "mppi_cma": [
        ("initial_noise_level", "initial_noise_level"),
        ("temperature", "temperature"),
        ("minimum_noise_level", "minimum_noise_level"),
        ("cov_rate", "covariance_adaptation_rate"),
    ],
    "cem": [
        ("sigma_start", "sigma_start"),
        ("sigma_min", "sigma_min"),
        ("num_elites", "num_elites"),
        ("explore_fraction", "explore_fraction"),
    ],
    "dial": [
        ("noise_level", "noise_level"),
        ("temperature", "temperature"),
        ("beta_opt_iter", "beta_opt_iter"),
        ("beta_horizon", "beta_horizon"),
    ],
    "predictive_sampling": [("noise_level", "noise_level")],
    "fpl_gmm": [
        ("sigma_start", "sigma_max"),
        ("sigma_min", "sigma_min"),
        ("allocation", "allocation"),
    ],
}


def _collect_algo_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    pairs = _ALGO_PARAMS[args.algo]
    out: Dict[str, Any] = {}
    for cli_name, ctor_name in pairs:
        v = getattr(args, cli_name)
        if v is not None:
            out[ctor_name] = v
    return out


def _build(args: argparse.Namespace):
    task = make_task(args.task)
    backend = MujocoBackend(task.model_path, nthread=args.nthread)
    cls = get_sampling_controller_class(args.algo)
    kwargs = dict(
        num_samples=args.num_samples,
        num_knots=args.num_knots,
        plan_horizon=args.plan_horizon_sec,
        spline_type=args.spline_type,
        iterations=args.iterations,
        seed=args.seed,
        use_fpl_cost=bool(args.fpl),
        use_fpl_discounted=bool(args.fpl_discounted),
        fpl_p=args.fpl_p,
        fpl_gamma=args.fpl_gamma,
        **_collect_algo_kwargs(args),
    )
    # fpl_time_p (soft-min over time) is currently an MPPIv2-only kwarg; only pass it
    # when set so other controllers' constructors aren't handed an unexpected arg.
    if args.fpl_time_p is not None:
        kwargs["fpl_time_p"] = args.fpl_time_p
        kwargs["fpl_time_discount"] = bool(args.fpl_time_discount)
    ctrl = cls(task, backend, **kwargs)
    return task, backend, ctrl


def _build_from_config(args: argparse.Namespace):
    """Build (task, backend, ctrl) from --config, and patch `args` with the fields the
    runners and cost printer read (steps / task / algo / fpl flags / init fn).

    Unlike `_build`, nothing is silently dropped: `config.resolve` errors on any parameter
    the chosen controller cannot accept, and reports at-default ones in `.dropped`.
    """
    from analytic_mppi.config import (apply_overrides, load_config_file, resolve,
                                      save_resolved)
    from analytic_mppi.eval import make_controller

    cfg = apply_overrides(load_config_file(args.config, args.index), args.overrides)
    if args.print_config:
        print(cfg.to_json())
        raise SystemExit(0)
    r = resolve(cfg)

    if len(cfg.run.seeds) > 1:
        print(f"note: config lists {len(cfg.run.seeds)} seeds; run.py executes one episode. "
              f"Using seed={cfg.run.seeds[0]} (multi-seed studies go through eval.run_study).")
    seed = cfg.run.seeds[0]

    # Mirror the config onto the flags the rest of this module already reads.
    args.task = r.task_name
    args.algo = r.controller
    args.steps = cfg.run.steps
    args.fpl = cfg.objective.mode == "fpl_cost"
    args.fpl_discounted = cfg.objective.mode == "fpl_discounted"
    args.fpl_p = cfg.objective.p
    args.fpl_gamma = cfg.objective.gamma
    args.fpl_time_p = cfg.objective.time_p
    args._init_fn = r.init_fn
    args._config = cfg
    args._resolved = r
    if cfg.run.mode == "live":
        args.live = True
    elif cfg.run.mode == "record":
        args.record = True
    if args.out is None and cfg.run.out is not None:
        args.out = cfg.run.out

    task, backend, ctrl = make_controller(
        r.task_name, r.controller, seed=seed, task_kwargs=r.task_kwargs, **r.build
    )
    if args.save_config:
        print(f"wrote {save_resolved(cfg, args.save_config)}")
    return task, backend, ctrl


def _print_header(task, backend, ctrl, args):
    cfg = getattr(args, "_config", None)
    if cfg is not None:
        print(f"config = {args.config}  label={cfg.label or '<unlabelled>'}  hash={cfg.hash()}")
        print(f"         proposal={cfg.proposal.kind}  update={cfg.update.rule}  "
              f"objective={cfg.objective.mode}(p={cfg.objective.p})")
        dropped = getattr(args, "_resolved").dropped
        if dropped:
            print(f"         at-default params {type(ctrl).__name__} does not accept, "
                  f"not passed: {list(dropped)}")
    print(f"task   = {args.task}  (nq={task.nq}, nv={task.nv}, nu={task.nu}, nsd={task.nsensordata}, dt={backend.dt})")
    print(f"algo   = {args.algo} -> {type(ctrl).__name__}  K={ctrl.num_samples} knots={ctrl.num_knots} H={ctrl.H} spline={ctrl.spline_type}")
    fpl = "cost" if args.fpl else ("discounted" if args.fpl_discounted else "off")
    if args.fpl_time_p is None:
        time_agg = "discounted-mean"
    elif getattr(args, "fpl_time_discount", False):
        time_agg = f"soft-min+discount(q={args.fpl_time_p})"
    else:
        time_agg = f"soft-min(q={args.fpl_time_p})"
    print(f"FPL    = {fpl} (p={args.fpl_p}, gamma={args.fpl_gamma}, time={time_agg})")


def _set_initial_state(args: argparse.Namespace, backend):
    """Initial-state setup: the config's named init fn if there is one, else per-task."""
    init_fn = getattr(args, "_init_fn", None)
    if init_fn is not None:
        init_fn(backend)
        return backend.get_state()

    task_name = args.task
    state = backend.get_state()
    if task_name == "pendulum":
        # Hang down: theta=0, theta_dot=0
        state[1] = 0.0
        state[2] = 0.0
        backend.set_state(state)
    elif task_name == "g1_standup":
        # Use the stand keyframe as starting pose by default. Users can rerun
        # from the default home pose by editing the model or starting different
        # qpos. (Stand is a sensible "test the cost surface" starting point.)
        try:
            kf = backend.model.keyframe("stand")
            backend.data.qpos[:] = kf.qpos
            backend.data.qvel[:] = 0
            import mujoco
            mujoco.mj_forward(backend.model, backend.data)
            state = backend.get_state()
        except Exception:
            pass
    return backend.get_state()


def _print_step_costs(step: int, task, backend, ctrl, u, args) -> None:
    """Print the cost terms actually used by the controller at the current
    closed-loop state, plus the best-rollout score from the last act() call.

    The mode tag makes it explicit which scoring path is live:
      [normal]          -> uses running_cost_terms (penalties)
      [fpl_cost]        -> uses running_cost_terms_f (fulfillment in [0,1])
      [fpl_discounted]  -> same source, different aggregation

    Sensordata is read from backend.data, which is up-to-date after mj_step.
    """
    qpos = np.asarray(backend.data.qpos, dtype=np.float64)
    qvel = np.asarray(backend.data.qvel, dtype=np.float64)
    sd = np.asarray(backend.data.sensordata, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    if args.fpl or args.fpl_discounted:
        mode = "fpl_cost" if args.fpl else "fpl_discounted"
        terms = task.running_cost_terms_f(qpos, qvel, sd, u)
        names = task.cost_term_names_f or [f"f{i}" for i in range(terms.shape[-1])]
        prefix = "fpl"
        fmt = "{:.3f}"
    else:
        mode = "normal"
        terms = task.running_cost_terms(qpos, qvel, sd, u)
        names = task.cost_term_names or [f"t{i}" for i in range(terms.shape[-1])]
        prefix = "running"
        fmt = "{:+.4f}"

    parts = [f"{n}=" + fmt.format(v) for n, v in zip(names, terms.tolist())]
    line = f"step {step:4d} [{mode}] {prefix}: " + " ".join(parts)

    # Show what the controller actually scored over its last batch of rollouts.
    last = getattr(ctrl, "last_trajectory", None)
    if last is not None and last.scores is not None:
        line += f"  | best_score={last.scores.min():+.4f}  K={last.scores.shape[0]}"

    print(line)


def _run_headless(task, backend, ctrl, args) -> None:
    state = _set_initial_state(args, backend)
    states_hist = [state.copy()]
    ctrls_hist = []
    plan_times = []
    for step in range(args.steps):
        t0 = time.perf_counter()
        u = ctrl.act(state)
        plan_times.append(time.perf_counter() - t0)
        state = backend.step(u)
        states_hist.append(state.copy())
        ctrls_hist.append(u.copy())
        if args.print_costs and (step % max(1, args.print_every) == 0):
            _print_step_costs(step, task, backend, ctrl, u, args)
    states_hist = np.asarray(states_hist)
    ctrls_hist = np.asarray(ctrls_hist)
    print(f"final state[1:4] = {states_hist[-1, 1:4]}")
    print(f"planning time    = {1e3 * np.mean(plan_times):.2f} ms / step ({1.0 / np.mean(plan_times):.0f} Hz)")


def _run_live(task, backend, ctrl, args) -> None:
    import mujoco.viewer as mj_viewer
    from mujoco import rollout as mj_rollout

    state = _set_initial_state(args, backend)
    # Hide the left (settings) and right (info) UI panels — toggle back at runtime with Tab.
    viewer = mj_viewer.launch_passive(backend.model, backend.data,
                                      show_left_ui=False, show_right_ui=False)
    try:
        dt = backend.dt
        for step in range(args.steps):
            if not viewer.is_running():
                break
            t0 = time.perf_counter()
            u = ctrl.act(state)
            with viewer.lock():
                state = backend.step(u)
            if args.print_costs and (step % max(1, args.print_every) == 0):
                _print_step_costs(step, task, backend, ctrl, u, args)
            viewer.sync()
            elapsed = time.perf_counter() - t0
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            mj_rollout.shutdown_persistent_pool()
        except Exception:
            pass
        try:
            viewer.close()
        except Exception:
            pass
    os._exit(0)


def _run_record(task, backend, ctrl, args) -> None:
    import mujoco
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise RuntimeError('Recording needs imageio. Install with: pip install -e ".[viz]"') from e

    out = Path(args.out or (DEFAULT_RUNS / f"{args.task}_{args.algo}.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)

    state = _set_initial_state(args, backend)
    fps = max(1, int(round(1.0 / backend.dt)))
    frames = []
    renderer = mujoco.Renderer(backend.model, width=720, height=540)
    try:
        for _ in range(args.steps):
            u = ctrl.act(state)
            state = backend.step(u)
            cam = args.camera if args.camera is not None else -1
            renderer.update_scene(backend.data, camera=cam)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    if out.suffix.lower() in (".mp4", ".m4v"):
        imageio.mimsave(out, frames, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    else:
        imageio.mimsave(out, frames, fps=fps)
    print(f"saved {out}")


def main() -> None:
    args = build_parser().parse_args()

    if args.config:
        from analytic_mppi.config import ConfigError
        try:
            task, backend, ctrl = _build_from_config(args)
        except ConfigError as e:
            raise SystemExit(f"config error: {e}")
    else:
        if not (args.task and args.algo):
            raise SystemExit("supply --config PATH, or both --task and --algo")
        for flag in ("index", "print_config", "save_config"):
            if getattr(args, flag) not in (None, False):
                raise SystemExit(f"--{flag.replace('_', '-')} requires --config")
        if args.overrides:
            raise SystemExit("--set requires --config")
        task, backend, ctrl = _build(args)

    if args.live and args.record:
        raise SystemExit("choose one of --live or --record (cannot be combined)")

    _print_header(task, backend, ctrl, args)

    if args.live:
        _run_live(task, backend, ctrl, args)
    elif args.record:
        _run_record(task, backend, ctrl, args)
    else:
        _run_headless(task, backend, ctrl, args)


if __name__ == "__main__":
    main()
