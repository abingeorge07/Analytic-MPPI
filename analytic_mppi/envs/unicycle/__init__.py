"""Planar 'unicycle' env (3-DOF holonomic base) for the MPPI testbed.

Layout (matches the MJCF):
  qpos = (x, y, theta)        ->  FULLPHYSICS state indices [1, 2, 3]
  qvel = (vx, vy, omega)      ->  FULLPHYSICS state indices [4, 5, 6]
  nu   = 3 (vx_cmd, vy_cmd, omega_cmd, world-frame)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from analytic_mppi.controllers import get_controller_class
from analytic_mppi.costs import GoalReachCost
from analytic_mppi.dynamics import MujocoBackend


MODEL_PATH: Path = Path(__file__).resolve().parent / "model.xml"

# State-layout knowledge for this env (FULLPHYSICS: time, qpos, qvel, ...)
XY_IDX: tuple[int, int] = (1, 2)
THETA_IDX: int = 3


def make_backend(**kw) -> MujocoBackend:
    """Build the MuJoCo backend for the unicycle env. Forwards kwargs to MujocoBackend."""
    return MujocoBackend(MODEL_PATH, **kw)


def make_cost(
    goal: Sequence[float] = (2.0, 2.0),
    *,
    Q: float = 1.0,
    R: Sequence[float] = (0.01, 0.01, 0.001),
    terminal_weight: float = 20.0,
) -> GoalReachCost:
    return GoalReachCost(
        goal=goal,
        state_xy_idx=XY_IDX,
        Q=Q,
        R=list(R),
        terminal_weight=terminal_weight,
    )


def make_controller(
    backend: MujocoBackend,
    *,
    variant: str = "vanilla",
    horizon: int = 25,
    n_samples: int = 512,
    sigma: Sequence[float] = (0.8, 0.8, 1.2),
    lambda_: float = 1.0,
    u_min: Sequence[float] = (-2.0, -2.0, -3.0),
    u_max: Sequence[float] = (2.0, 2.0, 3.0),
    seed: int = 0,
    **extra,
):
    """Build an MPPI-family controller. `variant` selects from the registry
    in analytic_mppi.controllers.CONTROLLERS. `extra` is forwarded as-is to the
    chosen class, so variant-specific knobs can be passed without changing this
    signature."""
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
    backend: MujocoBackend
    cost: GoalReachCost
    controller: object   # MPPI or any variant from analytic_mppi.controllers.CONTROLLERS
    goal: np.ndarray
    xy_idx: tuple[int, int] = XY_IDX
    theta_idx: int = THETA_IDX


def make_env(
    goal: Sequence[float] = (2.0, 2.0),
    *,
    variant: str = "vanilla",
    seed: int = 0,
    backend_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
    controller_kwargs: dict | None = None,
) -> Env:
    """One-call constructor: backend + cost + controller wired with defaults."""
    backend = make_backend(**(backend_kwargs or {}))
    cost = make_cost(goal=goal, **(cost_kwargs or {}))
    ctl = make_controller(backend, variant=variant, seed=seed,
                          **(controller_kwargs or {}))
    return Env(
        backend=backend,
        cost=cost,
        controller=ctl,
        goal=np.asarray(goal, dtype=np.float64),
    )


__all__ = [
    "MODEL_PATH",
    "XY_IDX",
    "THETA_IDX",
    "Env",
    "make_backend",
    "make_cost",
    "make_controller",
    "make_env",
]
