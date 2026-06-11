"""Planar one-legged hopper env for the MPPI testbed.

Layout (matches the MJCF):
  qpos = (x, z, pitch, thigh, leg, foot)  ->  FULLPHYSICS state indices [1..6]
  qvel = (vx, vz, omega, ...)             ->  FULLPHYSICS state indices [7..12]
  nu   = 3 (thigh / leg / foot torques, ctrlrange [-1, 1], gear 200)

Task: hop forward (+x) while keeping the torso near `target_height` and upright.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from analytic_mppi.controllers import get_controller_class
from analytic_mppi.costs import HopperCost
from analytic_mppi.dynamics import MujocoBackend


MODEL_PATH: Path = Path(__file__).resolve().parent / "hopper.xml"

# State-layout knowledge for this env (FULLPHYSICS: time, qpos, qvel, ...)
X_IDX: int = 1       # rootx  (forward position, qpos[0])
HEIGHT_IDX: int = 2  # rootz  (torso height,    qpos[1])
ORIENT_IDX: int = 3  # rooty  (torso pitch,     qpos[2])
VX_IDX: int = 7      # rootx' (forward velocity, qvel[0])


def make_backend(**kw) -> MujocoBackend:
    """Build the MuJoCo backend for the hopper env. Forwards kwargs to MujocoBackend."""
    return MujocoBackend(MODEL_PATH, **kw)


def make_cost(
    *,
    target_height: float = 1.2,
    height_weight: float = 1.0,
    orient_weight: float = 1.0,
    forward_weight: float = 1.0,
    control_weight: float = 2.0,
    terminal_weight: float = 1.0,
) -> HopperCost:
    return HopperCost(
        height_idx=HEIGHT_IDX,
        orient_idx=ORIENT_IDX,
        vx_idx=VX_IDX,
        target_height=target_height,
        height_weight=height_weight,
        orient_weight=orient_weight,
        forward_weight=forward_weight,
        control_weight=control_weight,
        terminal_weight=terminal_weight,
    )


def make_controller(
    backend: MujocoBackend,
    *,
    variant: str = "vanilla",
    horizon: int = 40,
    n_samples: int = 2048,
    sigma: Sequence[float] = (0.4, 0.4, 0.4),
    lambda_: float = 1.0,
    u_min: Sequence[float] = (-1.0, -1.0, -1.0),
    u_max: Sequence[float] = (1.0, 1.0, 1.0),
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
    cost: HopperCost
    controller: object   # MPPI or any variant from analytic_mppi.controllers.CONTROLLERS
    x_idx: int = X_IDX
    height_idx: int = HEIGHT_IDX
    orient_idx: int = ORIENT_IDX
    vx_idx: int = VX_IDX


def make_env(
    *,
    variant: str = "vanilla",
    seed: int = 0,
    backend_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
    controller_kwargs: dict | None = None,
) -> Env:
    """One-call constructor: backend + cost + controller wired with defaults.

    The hopper starts standing; the controller's job is to hop forward (+x)
    while staying upright at the target height.
    """
    backend = make_backend(**(backend_kwargs or {}))
    cost = make_cost(**(cost_kwargs or {}))
    ctl = make_controller(backend, variant=variant, seed=seed,
                          **(controller_kwargs or {}))
    return Env(backend=backend, cost=cost, controller=ctl)


__all__ = [
    "MODEL_PATH",
    "X_IDX",
    "HEIGHT_IDX",
    "ORIENT_IDX",
    "VX_IDX",
    "Env",
    "make_backend",
    "make_cost",
    "make_controller",
    "make_env",
]
