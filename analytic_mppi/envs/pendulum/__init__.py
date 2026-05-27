"""Inverted-pendulum swing-up env for the MPPI testbed.

Layout (matches the MJCF):
  qpos = (theta,)      ->  FULLPHYSICS state index 1   (0 = hanging down)
  qvel = (theta_dot,)  ->  FULLPHYSICS state index 2
  nu   = 1            (torque on pendulum_joint, ctrlrange [-1, 1], gear=2)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from analytic_mppi.controllers import get_controller_class
from analytic_mppi.costs import InvPenCost
from analytic_mppi.dynamics import MujocoBackend


MODEL_PATH: Path = Path(__file__).resolve().parent / "model.xml"

# State-layout knowledge for this env (FULLPHYSICS: time, qpos, qvel, ...)
THETA_IDX: int = 1
THETA_DOT_IDX: int = 2


def make_backend(**kw) -> MujocoBackend:
    """Build the MuJoCo backend for the pendulum env. Forwards kwargs to MujocoBackend."""
    return MujocoBackend(MODEL_PATH, **kw)


def make_cost(
    *,
    theta_dot_weight: float = 0.01,
    control_weight: float = 0.001,
    terminal_weight: float = 10.0,
) -> InvPenCost:
    return InvPenCost(
        theta_idx=THETA_IDX,
        theta_dot_idx=THETA_DOT_IDX,
        theta_dot_weight=theta_dot_weight,
        control_weight=control_weight,
        terminal_weight=terminal_weight,
    )


def make_controller(
    backend: MujocoBackend,
    *,
    variant: str = "vanilla",
    horizon: int = 50,
    n_samples: int = 2048,
    sigma: Sequence[float] = (0.5,),
    lambda_: float = 1.0,
    u_min: Sequence[float] = (-1.0,),
    u_max: Sequence[float] = (1.0,),
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
    cost: InvPenCost
    controller: object   # MPPI or any variant from analytic_mppi.controllers.CONTROLLERS
    theta_idx: int = THETA_IDX
    theta_dot_idx: int = THETA_DOT_IDX


def make_env(
    *,
    variant: str = "vanilla",
    seed: int = 0,
    backend_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
    controller_kwargs: dict | None = None,
) -> Env:
    """One-call constructor: backend + cost + controller wired with defaults.

    The pendulum starts hanging (theta=0); the controller's job is to swing
    it up to theta=pi.
    """
    backend = make_backend(**(backend_kwargs or {}))
    cost = make_cost(**(cost_kwargs or {}))
    ctl = make_controller(backend, variant=variant, seed=seed,
                          **(controller_kwargs or {}))
    return Env(backend=backend, cost=cost, controller=ctl)


__all__ = [
    "MODEL_PATH",
    "THETA_IDX",
    "THETA_DOT_IDX",
    "Env",
    "make_backend",
    "make_cost",
    "make_controller",
    "make_env",
]
