"""Closed-loop evaluation harness for the sampling-MPC framework.

Extracted from `analysis.ipynb` (Parts 5/6) so notebooks don't have to redefine
`make_ctrl` / `run_episode` / `metrics_for` in cells or monkey-patch them. A
controller is built by passing its **class** (or registry name) straight in — no
patching, no cell-ordering dependencies.

Typical use from an environment notebook:

    from analytic_mppi.eval import Config, run_study, walker_metrics, plot_study, walker_panels
    from analytic_mppi.controllers import MPPIv2
    from analytic_mppi.controllers.experimental import RankCMA

    configs = [
        Config("mppi (normal)",        MPPIv2,  "normal",         dict(noise_level=0.3, temperature=0.05)),
        Config("rank_cma (fpl)",       RankCMA, "fpl_discounted", dict(sigma_init=0.5, beta=10.0)),
    ]
    study = run_study("walker", configs, steps=500, n_episodes=5,
                      num_samples=128, plan_horizon=1.0, num_knots=6)
    task = make_task("walker")
    plot_study(study, lambda r: walker_metrics(r, task), walker_panels(), dt=...)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from analytic_mppi.tasks import make_task
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.controllers import SAMPLING_CONTROLLERS
from analytic_mppi.controllers.experimental import EXPERIMENTAL_CONTROLLERS
from analytic_mppi.controllers.cost_gd import wrap_controller_with_gd_refine


# Combined registry so configs can reference a controller by name OR by class.
ALL_CONTROLLERS: Dict[str, type] = {**SAMPLING_CONTROLLERS, **EXPERIMENTAL_CONTROLLERS}

ControllerSpec = Union[str, type]


# ---------------------------------------------------------------------------
#  Controller construction (no monkey-patching: the class is an argument)
# ---------------------------------------------------------------------------

def _resolve_controller(controller: ControllerSpec) -> type:
    if isinstance(controller, str):
        try:
            return ALL_CONTROLLERS[controller]
        except KeyError:
            raise KeyError(
                f"unknown controller {controller!r}. known: {sorted(ALL_CONTROLLERS)}"
            ) from None
    return controller


def _fpl_kwargs(cost_mode: str, fpl_p: float, fpl_gamma: float) -> Dict[str, Any]:
    if cost_mode == "normal":
        return {}
    if cost_mode == "fpl_discounted":
        return dict(use_fpl_discounted=True, fpl_p=fpl_p, fpl_gamma=fpl_gamma)
    if cost_mode == "fpl_cost":
        return dict(use_fpl_cost=True, fpl_p=fpl_p, fpl_gamma=fpl_gamma)
    if cost_mode == "fpl_layered":
        # Two-level composition over grouped atoms (task.fpl_groups). fpl_group_p (inner
        # power-mean p) flows through **algo_kwargs (Config.kwargs / build_kwargs).
        return dict(use_fpl_layered=True, fpl_p=fpl_p, fpl_gamma=fpl_gamma)
    if cost_mode == "hybrid":
        # quadratic targets + FPL log-barrier floors (task.floor_term_indices).
        # floor_weight flows through **algo_kwargs (Config.kwargs / build_kwargs).
        return dict(use_hybrid=True)
    raise ValueError(f"unknown cost_mode {cost_mode!r} "
                     f"(normal | fpl_cost | fpl_discounted | fpl_layered | hybrid)")


def make_controller(
    task_name: str,
    controller: ControllerSpec,
    *,
    num_samples: int,
    plan_horizon: float,
    num_knots: int = 1,
    spline_type: str = "zero",
    iterations: int = 1,
    cost_mode: str = "normal",
    fpl_p: float = 0.1,
    fpl_gamma: float = 0.99,
    seed: int = 0,
    nthread: Optional[int] = None,
    cost_gd: Optional[Dict[str, Any]] = None,
    task_kwargs: Optional[Dict[str, Any]] = None,
    true_perturbation: Optional[Dict[str, float]] = None,
    **algo_kwargs: Any,
):
    """Build a fresh (task, backend, controller) triple.

    `controller` is a `SamplingController` subclass or a name in `ALL_CONTROLLERS`.
    A fresh MuJoCo backend is created per call so each episode is an independent
    simulation. Sampler-common kwargs are passed explicitly; `cost_mode` selects
    the FPL flags; anything else (`noise_level`, `temperature`, `sigma_init`, ...)
    flows through `**algo_kwargs` to the controller constructor.

    `cost_gd`, if given, is a dict of cost-GD refinement params (e.g.
    `dict(gd_iterations=3, gd_lr=0.1)`) — the built controller is wrapped with
    BPTT cost-gradient refinement of its top-mu rollouts. Requires nq == nv
    (pendulum / walker / hopper).

    `true_perturbation` (model-mismatch study): if given, the controller PLANS on the
    nominal model while the returned backend (the "true" simulator that run_episode steps)
    has its dynamics parameters scaled per this dict (see dynamics.apply_perturbation). This
    is the sim-to-real gap: the controller's model is nominal, reality has drifted, and it
    cannot retune for a drift it doesn't know about.
    """
    cls = _resolve_controller(controller)
    task = make_task(task_name, **(task_kwargs or {}))
    # The controller always plans on the NOMINAL model (its belief).
    plan_backend = MujocoBackend(task.model_path, nthread=nthread)
    common = dict(
        num_samples=num_samples,
        num_knots=num_knots,
        plan_horizon=plan_horizon,
        spline_type=spline_type,
        iterations=iterations,
        seed=seed,
        **_fpl_kwargs(cost_mode, fpl_p, fpl_gamma),
    )
    ctrl = cls(task, plan_backend, **common, **algo_kwargs)
    if cost_gd:
        wrap_controller_with_gd_refine(ctrl, **cost_gd)
    # run_episode steps the returned backend. Nominal (== plan_backend) unless a
    # perturbation makes reality differ from the controller's model.
    step_backend = (MujocoBackend(task.model_path, nthread=nthread, perturb=true_perturbation)
                    if true_perturbation else plan_backend)
    return task, step_backend, ctrl


# ---------------------------------------------------------------------------
#  Initial-state helpers
# ---------------------------------------------------------------------------

def init_hang_down(backend) -> None:
    """Pendulum: hang down at theta=0, theta_dot=0 (mirror run.py:_set_initial_state)."""
    s = backend.get_state()
    s[1] = 0.0   # qpos
    s[2] = 0.0   # qvel
    backend.set_state(s)


def init_hopper_stand(backend, settle_steps: int = 60) -> None:
    """Hopper: drop from the model's default pose (torso at z≈2.5) and let it
    settle to a stationary stand (height ≈1.2, upright, vel≈0) under zero control.

    The stand-still-refusal experiment must start from a *settled stationary*
    pose, otherwise the opening transient (the torso falling from 2.5) confounds
    the "does the controller choose to stay still vs move" read.
    """
    backend.get_state()
    zero_u = np.zeros(backend.nu, dtype=np.float64)
    for _ in range(int(settle_steps)):
        backend.step(zero_u)


def init_g1_stand(backend) -> None:
    """G1 humanoid: start from the 'stand' keyframe at rest (mirrors
    run.py:_set_initial_state). The g1_standup task is really "keep standing /
    balance" from this pose, so it's the natural start for the comparison."""
    import mujoco
    kf = backend.model.keyframe("stand")
    backend.data.qpos[:] = kf.qpos
    backend.data.qvel[:] = 0.0
    mujoco.mj_forward(backend.model, backend.data)


# ---------------------------------------------------------------------------
#  Episodes / studies
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """One row of a comparison: a label, the controller class/name, the cost mode,
    any algo-specific kwargs, and (optionally) cost-GD refinement params."""
    label: str
    controller: ControllerSpec
    cost_mode: str = "normal"
    kwargs: Dict[str, Any] = field(default_factory=dict)
    cost_gd: Optional[Dict[str, Any]] = None   # e.g. dict(gd_iterations=3, gd_lr=0.1)
    fpl_p: Optional[float] = None              # power-mean exponent; None → make_controller default (0.1)
    task_kwargs: Optional[Dict[str, Any]] = None  # per-config Task ctor overrides (e.g.
                                               # dict(target_velocity=2.0)); merged over any
                                               # study-level task_kwargs shared across configs.


def _copy_or_none(x: Any) -> Optional[np.ndarray]:
    return None if x is None else np.asarray(x, dtype=float).copy()


def _stack_optional(seq: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    """Stack a per-step list of length-J vectors (or None) into (T, J).

    Returns None if no step produced a vector (controller doesn't expose the field);
    steps that are None become NaN rows. J is taken from the first present vector.
    """
    present = [x for x in seq if x is not None]
    if not present:
        return None
    out = np.full((len(seq), present[0].shape[0]), np.nan)
    for t, x in enumerate(seq):
        if x is not None:
            out[t] = x
    return out


def run_episode(
    task_name: str,
    controller: ControllerSpec,
    *,
    steps: int,
    seed: int = 0,
    cost_mode: str = "normal",
    init_fn: Optional[Callable[[Any], None]] = None,
    cost_gd: Optional[Dict[str, Any]] = None,
    task_kwargs: Optional[Dict[str, Any]] = None,
    **build_kwargs: Any,
) -> Dict[str, np.ndarray]:
    """Run one closed-loop episode; return history arrays.

    Always returns: states (T+1, nstate), ctrls (T, nu), sd (T, nsensordata), ess (T).
    When the controller exposes multi-objective diagnostics (ComposedGradientMPPI),
    also returns alpha / obj_ess / obj_satisfaction, each (T, J).
    """
    task, backend, ctrl = make_controller(
        task_name, controller, cost_mode=cost_mode, seed=seed, cost_gd=cost_gd,
        task_kwargs=task_kwargs, **build_kwargs
    )
    if init_fn is not None:
        init_fn(backend)
    state = backend.get_state()

    states_hist = [state.copy()]
    ctrls_hist: List[np.ndarray] = []
    sd_hist: List[np.ndarray] = []
    ess_hist: List[float] = []
    alpha_hist: List[Optional[np.ndarray]] = []
    obj_ess_hist: List[Optional[np.ndarray]] = []
    obj_sat_hist: List[Optional[np.ndarray]] = []
    for _ in range(steps):
        u = ctrl.act(state)
        state = backend.step(u)
        states_hist.append(state.copy())
        ctrls_hist.append(u.copy())
        sd_hist.append(np.asarray(backend.data.sensordata, dtype=np.float64).copy())
        # Effective sample size of this step's softmax weighting, if the
        # controller exposes it (MPPIv2). NaN otherwise (argmax/elite samplers).
        ess_hist.append(float(getattr(ctrl, "last_ess", np.nan) or np.nan))
        # Multi-objective composition diagnostics (ComposedGradientMPPI): the composition
        # weights, per-objective ESS, and per-objective satisfaction. None for every other
        # controller -> the keys are simply omitted from the result below.
        alpha_hist.append(_copy_or_none(getattr(ctrl, "last_alpha", None)))
        obj_ess_hist.append(_copy_or_none(getattr(ctrl, "last_obj_ess", None)))
        obj_sat_hist.append(_copy_or_none(getattr(ctrl, "last_obj_satisfaction", None)))
    out: Dict[str, np.ndarray] = dict(
        states=np.asarray(states_hist),   # (T+1, nstate)
        ctrls=np.asarray(ctrls_hist),     # (T,   nu)
        sd=np.asarray(sd_hist),           # (T,   nsensordata)
        ess=np.asarray(ess_hist),         # (T,)
    )
    for key, seq in (("alpha", alpha_hist), ("obj_ess", obj_ess_hist),
                     ("obj_satisfaction", obj_sat_hist)):
        stacked = _stack_optional(seq)    # (T, J) or None
        if stacked is not None:
            out[key] = stacked
    return out


def run_study(
    task_name: str,
    configs: List[Config],
    *,
    steps: int,
    n_episodes: int = 1,
    seed0: int = 0,
    init_fn: Optional[Callable[[Any], None]] = None,
    progress: bool = True,
    task_kwargs: Optional[Dict[str, Any]] = None,
    **shared_build_kwargs: Any,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run every config for `n_episodes` (seeds seed0..seed0+n_episodes-1) and
    stack the histories. Returns {label: {states, ctrls, sd}} with a leading
    episode axis on each array.

    `shared_build_kwargs` (num_samples, plan_horizon, num_knots, ...) apply to
    every config; per-config `kwargs` and `cost_mode` come from each `Config`.
    """
    study: Dict[str, Dict[str, np.ndarray]] = {}
    for cfg in configs:
        if progress:
            print(f"running {cfg.label:40s}", end="", flush=True)
        fpl_p_kw = {} if cfg.fpl_p is None else {"fpl_p": cfg.fpl_p}
        # Merge study-level task_kwargs (shared difficulty) with per-config overrides.
        cfg_task_kwargs = {**(task_kwargs or {}), **(cfg.task_kwargs or {})}
        eps = []
        for ep in range(n_episodes):
            eps.append(run_episode(
                task_name, cfg.controller, steps=steps, seed=seed0 + ep,
                cost_mode=cfg.cost_mode, init_fn=init_fn, cost_gd=cfg.cost_gd,
                task_kwargs=cfg_task_kwargs, **shared_build_kwargs, **cfg.kwargs, **fpl_p_kw,
            ))
            if progress:
                print(".", end="", flush=True)
        # Stack every history key the episodes produced (states/ctrls/sd/ess always;
        # alpha/obj_ess/obj_satisfaction only when the controller exposed them). Leading
        # axis is the episode index. Keys are consistent across episodes of one config
        # (same controller + cost mode).
        keys = list(eps[0].keys())
        study[cfg.label] = {k: np.stack([e[k] for e in eps]) for k in keys}
        if progress:
            print(" done")
    return study


# ---------------------------------------------------------------------------
#  Per-task metrics (operate on a stacked study entry: leading episode axis)
# ---------------------------------------------------------------------------

def pendulum_metrics(res: Dict[str, np.ndarray], task) -> Dict[str, np.ndarray]:
    """err = wrap-aware |theta - pi| (T+1); u_mag = mean|u| (T); cost = Σ normal terms (T)."""
    states, ctrls, sd = res["states"], res["ctrls"], res["sd"]
    qpos = task.qpos_of(states)
    qvel = task.qvel_of(states)
    theta = qpos[..., 0]
    err = np.abs(((theta - np.pi + np.pi) % (2.0 * np.pi)) - np.pi)
    u_mag = np.abs(ctrls).mean(axis=-1)
    qpos_u, qvel_u = qpos[:, :-1, :], qvel[:, :-1, :]
    cost = task.running_cost_terms(qpos_u, qvel_u, sd, ctrls).sum(axis=-1)
    return dict(err=err, u_mag=u_mag, cost=cost)


def walker_metrics(res: Dict[str, np.ndarray], task) -> Dict[str, np.ndarray]:
    """height_err, vel_err, upright (torso·z), u_mag — all per (episode, step)."""
    sd, ctrls = res["sd"], res["ctrls"]
    pos_z = sd[..., task._pos_adr + 2]
    vx = sd[..., task._vel_adr]
    zax_z = sd[..., task._zax_adr + 2]
    return dict(
        height_err=np.abs(pos_z - task.target_height),
        vel_err=np.abs(vx - task.target_velocity),
        upright=zax_z,
        u_mag=np.abs(ctrls).mean(axis=-1),
    )


# Panel specs: (metric_key, title, ylabel, on_state_grid). on_state_grid=True uses
# the (T+1) time axis (state metrics like err); False uses the (T) control axis.
def pendulum_panels() -> List[tuple]:
    return [
        ("err", "Pendulum upright error", "|θ − π|  (rad)", True),
        ("u_mag", "Control usage", "mean |u|", False),
        ("cost", "Running cost (Σ terms)", "cost", False),
    ]


def walker_panels() -> List[tuple]:
    return [
        ("height_err", "Height tracking", "|z − z*|  (m)", False),
        ("vel_err", "Forward velocity", "|vx − v*|  (m/s)", False),
        ("upright", "Torso uprightness", "torso·ẑ  (1=upright)", False),
        ("u_mag", "Control usage", "mean |u|", False),
    ]


# Hopper shares the walker's torso sensor layout (position / subtreelinvel / zaxis),
# so the same metrics + panels apply.
hopper_metrics = walker_metrics
hopper_panels = walker_panels


def cube_metrics(res: Dict[str, np.ndarray], task) -> Dict[str, np.ndarray]:
    """pos_err = ‖Δxy‖ of the cube; ori_err = ‖quat error‖ to the goal; u_mag."""
    from analytic_mppi.tasks.cube import _quat_sub
    sd, ctrls = res["sd"], res["ctrls"]
    pos = task._cube_pos_err(sd)                            # (..., 3)
    pos_err = np.sqrt(np.sum(pos[..., 0:2] ** 2, axis=-1))  # xy distance (ignore z)
    quat_err = _quat_sub(task._cube_quat(sd), task.goal_quat)
    ori_err = np.sqrt(np.sum(quat_err ** 2, axis=-1))
    return dict(pos_err=pos_err, ori_err=ori_err, u_mag=np.abs(ctrls).mean(axis=-1))


def cube_panels() -> List[tuple]:
    return [
        ("pos_err", "Cube position error (xy)", "‖Δxy‖  (m)", False),
        ("ori_err", "Cube orientation error", "‖quat err‖", False),
        ("u_mag", "Control usage", "mean |u|", False),
    ]


def g1_metrics(res: Dict[str, np.ndarray], task) -> Dict[str, np.ndarray]:
    """height_err = |torso_z - target|; upright = rotated-up·ẑ (1=upright); u_mag."""
    sd, ctrls = res["sd"], res["ctrls"]
    height_err = np.abs(task._torso_height(sd) - task.target_height)
    upright = task._torso_orientation(sd)[..., 2]          # rz of rotated (0,0,1)
    return dict(height_err=height_err, upright=upright, u_mag=np.abs(ctrls).mean(axis=-1))


def g1_panels() -> List[tuple]:
    return [
        ("height_err", "Torso height error", "|z − z*|  (m)", False),
        ("upright", "Torso uprightness", "rot(ẑ)·ẑ  (1=upright)", False),
        ("u_mag", "Control usage", "mean |u|", False),
    ]


def g1_walk_metrics(res: Dict[str, np.ndarray], task) -> Dict[str, np.ndarray]:
    """fwd_vel = world-frame torso vx; upright = rot(ẑ)·ẑ; height_err; u_mag."""
    sd, ctrls = res["sd"], res["ctrls"]
    fwd_vel = sd[..., task._linvel_adr]                    # torso linvel x (world frame)
    upright = task._torso_orientation(sd)[..., 2]
    height_err = np.abs(task._torso_height(sd) - task.target_height)
    return dict(fwd_vel=fwd_vel, upright=upright, height_err=height_err,
                u_mag=np.abs(ctrls).mean(axis=-1))


def g1_walk_panels() -> List[tuple]:
    return [
        ("fwd_vel", "Forward speed", "torso vx  (m/s)", False),
        ("upright", "Torso uprightness", "rot(ẑ)·ẑ  (1=upright)", False),
        ("height_err", "Torso height error", "|z − z*|  (m)", False),
        ("u_mag", "Control usage", "mean |u|", False),
    ]


# ---------------------------------------------------------------------------
#  Plotting / summary
# ---------------------------------------------------------------------------

def plot_study(study, metric_fn, panels, dt, *, title=None, figsize_per=(5.0, 3.6)):
    """Mean ± 1 std bands, one panel per entry in `panels`, one colored line per config."""
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]), sharex=True)
    if n == 1:
        axes = [axes]
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_key = panels[0][0]

    for i, (label, res) in enumerate(study.items()):
        m = metric_fn(res)
        color = cmap[i % len(cmap)]
        for ax, (key, _title, _ylabel, on_state_grid) in zip(axes, panels):
            arr = m[key]
            xs = np.arange(arr.shape[-1]) * dt
            mu, sd_ = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(xs, mu, color=color, lw=1.4, label=(label if key == legend_key else None))
            ax.fill_between(xs, mu - sd_, mu + sd_, color=color, alpha=0.15, lw=0)

    for ax, (_key, ptitle, ylabel, _g) in zip(axes, panels):
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(ptitle)
    axes[0].legend(fontsize=7, loc="best")
    if title:
        fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig


def summarize(study, metric_fn, keys: List[str]) -> None:
    """Print one scalar per config per metric (mean across episodes and time)."""
    header = f"{'config':40s}  " + "  ".join(f"{k:>12s}" for k in keys)
    print(header)
    for label, res in study.items():
        m = metric_fn(res)
        vals = "  ".join(f"{m[k].mean():12.4f}" for k in keys)
        print(f"{label:40s}  {vals}")


# ---------------------------------------------------------------------------
#  Optional video rendering
# ---------------------------------------------------------------------------

def render_video(
    task_name: str,
    controller: ControllerSpec,
    *,
    steps: int,
    out_path: Union[str, Path],
    seed: int = 0,
    cost_mode: str = "normal",
    init_fn: Optional[Callable[[Any], None]] = None,
    camera: Optional[str] = None,
    width: int = 480,
    height: int = 360,
    **build_kwargs: Any,
) -> Dict[str, Any]:
    """Run a closed-loop episode and write an mp4. Returns {path, states, ctrls, plan_ms}."""
    import time
    import mujoco
    import imageio.v2 as imageio

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    task, backend, ctrl = make_controller(
        task_name, controller, cost_mode=cost_mode, seed=seed, **build_kwargs
    )
    if init_fn is not None:
        init_fn(backend)
    state = backend.get_state()

    cam_id = -1
    if camera is not None:
        try:
            cid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            cam_id = cid if cid >= 0 else -1
        except Exception:
            cam_id = -1

    fps = max(1, int(round(1.0 / backend.dt)))
    frames, states_hist, ctrls_hist, sd_hist, plan_times = [], [state.copy()], [], [], []
    renderer = mujoco.Renderer(backend.model, width=width, height=height)
    try:
        for _ in range(steps):
            t0 = time.perf_counter()
            u = ctrl.act(state)
            plan_times.append(time.perf_counter() - t0)
            state = backend.step(u)
            states_hist.append(state.copy())
            ctrls_hist.append(u.copy())
            sd_hist.append(np.asarray(backend.data.sensordata, dtype=np.float64).copy())
            renderer.update_scene(backend.data, camera=cam_id)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()
    imageio.mimsave(str(out_path), frames, fps=fps, codec="libx264",
                    quality=8, macro_block_size=None)
    return dict(
        path=out_path,
        states=np.asarray(states_hist),
        ctrls=np.asarray(ctrls_hist),
        sd=np.asarray(sd_hist),
        plan_ms=1e3 * float(np.mean(plan_times)),
    )
