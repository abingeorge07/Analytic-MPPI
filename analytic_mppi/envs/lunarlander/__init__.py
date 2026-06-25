"""Gymnasium LunarLander env for the MPPI testbed (Box2D, not MuJoCo).

This is the one non-MuJoCo env in the testbed. It wraps Gymnasium's
`LunarLanderContinuous-v3` through `GymBackend`, which gives the MPPI controllers
the batched-rollout + state-injection interface they expect (see
`analytic_mppi/dynamics/gym_backend.py`). The objective is the env's *default
reward, converted to a decomposed cost* (`LunarLanderCost`): land on the pad
(distance/velocity/angle/leg-contact shaping) with minimal fuel, avoiding crashes.

Observation (the rollout "state" the cost sees): [x, y, vx, vy, angle, omega,
leg1_contact, leg2_contact]. Action: continuous [main, side] in [-1, 1].

Performance note: unlike the MuJoCo envs, Box2D has no parallel rollout primitive,
so `GymBackend.rollout` steps the B samples *serially* in Python (~B*H Box2D
steps per `act`). Keep `n_samples`/`horizon` modest -- the defaults below plan in
well under a second per step.

Requires the optional deps: `pip install gymnasium box2d-py` (and `pygame` for
`--live` rendering). They are imported lazily, so the rest of the testbed does
not need them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from analytic_mppi.controllers import get_controller_class
from analytic_mppi.costs import LunarLanderCost
from analytic_mppi.dynamics.gym_backend import GymBackend


ENV_ID: str = "LunarLanderContinuous-v3"

# Observation layout (GymBackend rollout "states"), for callers that want indices.
X_IDX, Y_IDX = 0, 1
VX_IDX, VY_IDX = 2, 3
ANGLE_IDX, OMEGA_IDX = 4, 5
LEG1_IDX, LEG2_IDX = 6, 7


def make_backend(*, seed: int = 0, render_mode: str | None = None, **kw) -> GymBackend:
    """Build the Gymnasium/Box2D backend for LunarLander."""
    return GymBackend(ENV_ID, seed=seed, render_mode=render_mode, **kw)


def make_cost(
    *,
    distance_weight: float = 100.0,
    velocity_weight: float = 100.0,
    angle_weight: float = 100.0,
    leg_weight: float = 10.0,
    fuel_weight: float = 1.0,
    crash_weight: float = 100.0,
    land_weight: float = 100.0,
) -> LunarLanderCost:
    """LunarLander's default reward, converted to a decomposed cost.

    Defaults reproduce the env reward's own scales; pass overrides to retune.
    """
    return LunarLanderCost(
        distance_weight=distance_weight,
        velocity_weight=velocity_weight,
        angle_weight=angle_weight,
        leg_weight=leg_weight,
        fuel_weight=fuel_weight,
        crash_weight=crash_weight,
        land_weight=land_weight,
    )


def make_controller(
    backend: GymBackend,
    *,
    variant: str = "vanilla",
    horizon: int = 20,
    n_samples: int = 64,
    sigma: Sequence[float] = (0.5, 0.5),
    lambda_: float = 1.0,
    u_min: Sequence[float] = (-1.0, -1.0),
    u_max: Sequence[float] = (1.0, 1.0),
    seed: int = 0,
    **extra,
):
    """Build an MPPI-family controller (registry `variant`). Modest K/H defaults
    because the Box2D rollout is serial. `extra` is forwarded to the chosen class."""
    cls = get_controller_class(variant)
    return cls(
        horizon=horizon,
        n_samples=n_samples,
        nu=backend.nu,
        sigma=np.asarray(sigma, dtype=np.float64),
        lambda_=lambda_,
        u_min=np.asarray(u_min, dtype=np.float64),
        u_max=np.asarray(u_max, dtype=np.float64),
        seed=seed,
        **extra,
    )


@dataclass
class Env:
    backend: GymBackend
    cost: LunarLanderCost
    controller: object   # MPPI or any variant from analytic_mppi.controllers.CONTROLLERS


def make_env(
    *,
    variant: str = "vanilla",
    seed: int = 0,
    render_mode: str | None = None,
    backend_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
    controller_kwargs: dict | None = None,
) -> Env:
    """One-call constructor: backend + cost + controller wired with defaults.

    The lander spawns at the top with a random shove; the controller's job is to
    fly it down onto the pad, upright and slow, with both legs touching."""
    backend = make_backend(seed=seed, render_mode=render_mode, **(backend_kwargs or {}))
    cost = make_cost(**(cost_kwargs or {}))
    ctl = make_controller(backend, variant=variant, seed=seed, **(controller_kwargs or {}))
    return Env(backend=backend, cost=cost, controller=ctl)


__all__ = [
    "ENV_ID",
    "X_IDX", "Y_IDX", "VX_IDX", "VY_IDX", "ANGLE_IDX", "OMEGA_IDX", "LEG1_IDX", "LEG2_IDX",
    "Env",
    "make_backend",
    "make_cost",
    "make_controller",
    "make_env",
]
