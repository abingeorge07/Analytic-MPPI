"""Shared experiment plumbing for the FPL-MPPI campaign.

One place that knows, per environment: the study build params (horizon / knots /
noise / temperature / temporal-weakest-link, matched to the published pareto sweeps),
the init function, the difficulty knob, and how to turn a closed-loop episode into
scalar outcome metrics. And one place that knows, per sampler, how to spell its algo
kwargs so the SAME FPL (or linear) cost runs identically across MPPI / CMA / CEM /
colored. Every experiment (sampler race, portability matrix, zero-tuning, robustness)
calls `run_trial` so the fairness protocol is enforced in exactly one file.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from analytic_mppi.eval import (run_episode, make_task, init_hopper_stand,
                                 init_barkour_stand, init_cube)
from analytic_mppi.tasks.base import ATOM_FLOOR_LEGACY


# --- WO-3.4 / NEXT_STEPS S1-S2: the fulfillment-atom floor ------------------------
# Every trial in the campaign goes through `run_trial`, so this one global is the only
# place the floor needs setting for an A/B. It is deliberately NOT a per-call argument:
# the floor must be identical for the FPL arm and the linear arm within a comparison
# (invariant 11.1), and a global makes it impossible to set for one and not the other.
# Drivers set it via `set_atom_floor`; ATOM_FLOOR_LEGACY (1e-8) reproduces every number
# produced before the floor existed.
ATOM_FLOOR: float = ATOM_FLOOR_LEGACY

# WO-3.3: same reasoning, same mechanism. False is the legacy renormalized discounted mean.
# NOTE on magnitude: at the study horizon (hopper H=30, gamma=0.99) the post-horizon tail is
# gamma^29 = 0.75 of the total discount mass, so flipping this is NOT a small correction --
# the terminal step's weight goes 0.029 -> 0.747. That is the point (the renormalization was
# assuming that 75% equals the in-horizon average) but it makes the objective largely
# "terminal fulfillment + in-horizon correction" under the cheap hold-r_H tail estimate.
TERMINAL_VALUE: bool = False


def set_atom_floor(floor: float) -> None:
    global ATOM_FLOOR
    if not (0.0 < float(floor) <= 1.0):
        raise ValueError(f"atom floor must be in (0, 1], got {floor}")
    ATOM_FLOOR = float(floor)


# The published spec is fpl_cost + fpl_time_p<0 + time_discount=False, i.e. an UNWEIGHTED
# soft-min over time. That path never consults the discount/terminal weights, so TERMINAL_VALUE
# cannot act there (the controller now refuses the inert combination outright). Turning this on
# makes the soft-min discount-weighted, which is the only way the terminal value can reach the
# published objective -- so a G3 comparison has to move this first, then the tail, separately.
TIME_DISCOUNT: bool = False


def set_terminal_value(on: bool) -> None:
    global TERMINAL_VALUE
    TERMINAL_VALUE = bool(on)


def set_time_discount(on: bool) -> None:
    global TIME_DISCOUNT
    TIME_DISCOUNT = bool(on)


# --- per-environment study spec (matched to verification/*_pareto_sweep.py) --------
# noise/temp/time_p/horizon/knots are the PUBLISHED sweep values; steps sized for a
# few seconds of closed loop; `lin_idx` is the atom the linear-weight family sweeps
# (velocity / alignment); `fall` is the min-uprightness fall line (locomotion).
ENVS: Dict[str, Dict[str, Any]] = {
    "hopper": dict(task="hopper", init=init_hopper_stand, steps=150,
                   horizon=0.6, knots=4, noise=0.3, temp=0.2, time_p=-2.0,
                   difficulty_key="target_velocity", difficulty=2.0,
                   n_atoms=4, lin_idx=2, safe_w=0.5, metric="locomotion", fall=0.6),
    "walker": dict(task="walker", init=None, steps=150,
                   horizon=0.6, knots=6, noise=0.8, temp=0.2, time_p=-2.0,
                   difficulty_key="target_velocity", difficulty=5.0,
                   n_atoms=4, lin_idx=2, safe_w=0.5, metric="locomotion", fall=0.6),
    "cube": dict(task="cube", init=init_cube, steps=400,
                 horizon=0.3, knots=4, noise=0.4, temp=0.1, time_p=-2.0,
                 difficulty_key="target_angle", difficulty=1.2,
                 n_atoms=4, lin_idx=2, safe_w=0.5, metric="cube", fall=None),
    "quadruped": dict(task="quadruped", init=init_barkour_stand, steps=1500,
                      horizon=0.3, knots=5, noise=0.5, temp=0.2, time_p=-2.0,
                      difficulty_key="target_velocity", difficulty=1.5,
                      n_atoms=6, lin_idx=2, safe_w=0.5, metric="quadruped", fall=0.5),
}


def linear_weights(env: str, wv: float) -> list[float]:
    """Linear-family weight vector: `wv` on the swept progress atom, 1 on the safety
    atoms, `safe_w` on control — matches each sweep's build_configs."""
    spec = ENVS[env]
    w = [1.0] * spec["n_atoms"]
    w[spec["lin_idx"]] = wv
    w[-1] = spec["safe_w"]
    return w


def sampler_kwargs(sampler: str, env: str, K: int) -> Dict[str, Any]:
    """Algo kwargs for one sampler at budget K, using the env's matched noise/temp.
    The FPL/linear COST kwargs are added separately by `cost_kwargs` so the cost is
    identical across samplers (fairness)."""
    spec = ENVS[env]
    noise, temp, time_p = spec["noise"], spec["temp"], spec["time_p"]
    if sampler in ("mppi", "fpl_colored", "fpl_colored_generic", "fpl_tempered", "fpl_shielded"):
        kw = dict(noise_level=noise, temperature=temp, fpl_time_p=time_p)
        if sampler == "fpl_colored":
            kw.update(color_beta=2.0, use_absolute_scale=True)
        elif sampler == "fpl_colored_generic":
            kw.update(color_beta=2.0, use_absolute_scale=False)
        # fpl_tempered: the weight_mode + its knobs come from the caller via run_trial(extra=...)
        return kw
    if sampler == "predictive_sampling":
        # Greedy argmax over the same Gaussian cloud (the mean is always sample 0), so
        # there is no temperature to match — this is the selection-rule ablation against
        # `mppi`: identical proposal, softmax replaced by argmax.
        return dict(noise_level=noise, fpl_time_p=time_p)
    if sampler == "mppi_cma":
        return dict(initial_noise_level=noise, temperature=temp,
                    covariance_adaptation_rate=0.1, fpl_time_p=time_p)
    if sampler == "cem":
        return dict(sigma_start=noise, sigma_min=max(0.05, 0.1 * noise),
                    num_elites=max(4, K // 8), explore_fraction=0.1, fpl_time_p=time_p)
    if sampler == "dial":
        # DIAL's ctor does not forward **kwargs (no fpl_time_p slot), unlike the other
        # samplers here -- only pass what it actually declares.
        return dict(noise_level=noise, temperature=temp, beta_opt_iter=3.0, beta_horizon=3.0)
    raise ValueError(f"unknown sampler {sampler!r}")


def _controller_name(sampler: str) -> str:
    return {"fpl_colored_generic": "fpl_colored"}.get(sampler, sampler)


def cost_kwargs(cost: str, env: str) -> Dict[str, Any]:
    """FPL vs linear cost, both through the fpl_cost pipeline so p=1 IS the linear
    weighted sum and only the objective-axis composition differs (handoff fairness).
    `cost` is 'fpl', 'linear' (best-safe weight), 'lin:wv' for a specific weight, or
    'normal' for the raw quadratic penalty (bypasses the fulfillment pipeline entirely
    -- see `cost_mode` in run_trial)."""
    if cost == "normal":
        return dict()
    if cost == "fpl":
        return dict(fpl_p=-1.0)
    if cost == "linear":
        return dict(fpl_p=1.0, fpl_weights=linear_weights(env, 1.0))
    if cost.startswith("lin:"):
        wv = float(cost.split(":", 1)[1])
        return dict(fpl_p=1.0, fpl_weights=linear_weights(env, wv))
    raise ValueError(f"unknown cost {cost!r}")


def _metrics(env: str, res: Dict[str, np.ndarray], task) -> Dict[str, Any]:
    spec = ENVS[env]
    sd = res["sd"]  # (T, nsensordata)
    if spec["metric"] in ("locomotion",):
        vx = sd[..., task._vel_adr]
        up = sd[..., task._zax_adr + 2]
        fell = bool(up.min() < spec["fall"])
        vx_mean = float(vx.mean())
        return dict(vx=vx_mean, survived=(0.0 if fell else 1.0),
                    prod=(0.0 if fell else vx_mean), minup=float(up.min()))
    if spec["metric"] == "quadruped":
        vx = task._torso_vel_x(sd)
        up = task._torso_up(sd)
        fell = bool(up.min() < spec["fall"])
        vx_mean = float(vx.mean())
        return dict(vx=vx_mean, survived=(0.0 if fell else 1.0),
                    prod=(0.0 if fell else vx_mean), minup=float(up.min()))
    if spec["metric"] == "cube":
        ang = task._angle_err(sd)                      # (T,) radians
        achieved = float(np.degrees(task.target_angle) - np.degrees(ang[-10:].mean()))
        dropped = bool(task._hold_xy(sd).max() > task.hold_xy_floor)
        return dict(achieved=achieved, survived=(0.0 if dropped else 1.0),
                    prod=(0.0 if dropped else achieved), drift=float(task._hold_xy(sd).max()))
    raise ValueError(spec["metric"])


def run_trial(env: str, sampler: str, cost: str, seed: int, K: int, *,
              steps: Optional[int] = None, difficulty: Optional[float] = None,
              true_perturbation: Optional[Dict[str, float]] = None,
              cost_gd: Optional[Dict[str, Any]] = None,
              extra: Optional[Dict[str, Any]] = None,
              task_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run ONE closed-loop episode and return its scalar outcome metrics.

    Enforces the fairness protocol: same horizon/knots/steps/difficulty for every
    (sampler, cost); only the sampler's proposal and the cost's composition vary."""
    spec = ENVS[env]
    steps = spec["steps"] if steps is None else steps
    diff = spec["difficulty"] if difficulty is None else difficulty
    build = dict(
        num_samples=K, plan_horizon=spec["horizon"], num_knots=spec["knots"],
        spline_type="zero",
        **sampler_kwargs(sampler, env, K),
        **cost_kwargs(cost, env),
    )
    # The atom floor is part of the OBJECTIVE, so it is set for every arm identically
    # (see ATOM_FLOOR above). `extra` may still override it for a deliberate sweep.
    build["fpl_atom_floor"] = ATOM_FLOOR
    build["fpl_terminal_value"] = TERMINAL_VALUE
    if TIME_DISCOUNT and build.get("fpl_time_p") is not None:
        build["fpl_time_discount"] = True
    if extra:
        build.update(extra)
    # `task_extra` carries Task-ctor settings that are part of the COST SPEC rather than
    # the sampler (e.g. hopper's atom_soft_floor). It goes to both the scoring task inside
    # the controller and the metrics task here, so they cannot drift apart.
    tkw = {spec["difficulty_key"]: diff, **(task_extra or {})}
    task = make_task(spec["task"], **tkw)
    # Every 'cost' arm except 'normal' goes through the fpl_cost pipeline (fpl_p selects
    # linear-vs-conjunctive within it); 'normal' is the raw quadratic penalty, which needs
    # cost_mode="normal" so eval.make_controller doesn't set use_fpl_cost.
    res = run_episode(
        spec["task"], _controller_name(sampler), steps=steps, seed=seed,
        cost_mode=("normal" if cost == "normal" else "fpl_cost"),
        init_fn=spec["init"], cost_gd=cost_gd,
        task_kwargs=tkw,
        true_perturbation=true_perturbation, **build,
    )
    m = _metrics(env, res, task)
    ess = res.get("ess")
    m["ess"] = (float(np.nanmean(ess)) if ess is not None and np.isfinite(ess).any()
                else float("nan"))
    return m
