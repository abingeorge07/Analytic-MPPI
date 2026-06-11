"""Unitree G1 humanoid standup env for the MPPI testbed.

Layout (FULLPHYSICS state = [time, qpos(nq), qvel(nv), ...], free base joint):
  qpos = (x, y, z, qw, qx, qy, qz, joints...) -> state indices [1..nq]
    height  = qpos[2]   -> state index 3
    quat    = qpos[3:7] -> state slice [4:8]   (base orientation, wxyz)
    joints  = qpos[7:]  -> state slice [8:1+nq]
  nu = 29 (actuated joint torques; bounds from the MJCF ctrlrange)

Task: rise from the collapsed default pose toward `target_height` while keeping
the torso upright and the joints near the `stand` keyframe reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from analytic_mppi.controllers import get_controller_class
from analytic_mppi.costs import G1StandupCost
from analytic_mppi.dynamics import MujocoBackend


MODEL_PATH: Path = Path(__file__).resolve().parent / "scene.xml"

# State-layout knowledge for this env (FULLPHYSICS: time, qpos, qvel, ...)
HEIGHT_IDX: int = 3                # qpos[2], pelvis z
QUAT_SLICE: slice = slice(4, 8)    # qpos[3:7], base orientation (wxyz)


def make_backend(**kw) -> MujocoBackend:
    """Build the MuJoCo backend for the G1 env. Forwards kwargs to MujocoBackend."""
    return MujocoBackend(MODEL_PATH, **kw)


def make_cost(
    backend: MujocoBackend,
    *,
    target_height: float = 0.79,
    orient_weight: float = 1.0,
    height_weight: float = 1.0,
    nominal_weight: float = 1.0,
    terminal_weight: float = 1.0,
) -> G1StandupCost:
    """Build the standup cost. Reads the standing reference (qpos[7:]) from the
    model's `stand` keyframe and the torso orientation sensor address; takes
    `backend` because the cost needs that reference, nq, and the sensor address
    (unlike the no-arg make_cost in the pendulum env)."""
    qstand = np.asarray(backend.model.keyframe("stand").qpos, dtype=np.float64)
    sid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_in_torso_quat")
    orientation_sensor_adr = int(backend.model.sensor_adr[sid])
    return G1StandupCost(
        qstand_joints=qstand[7:],
        height_idx=HEIGHT_IDX,
        quat_slice=QUAT_SLICE,
        joint_slice=slice(8, 1 + backend.nq),
        orientation_sensor_adr=orientation_sensor_adr,
        target_height=target_height,
        orient_weight=orient_weight,
        height_weight=height_weight,
        nominal_weight=nominal_weight,
        terminal_weight=terminal_weight,
    )


def make_controller(
    backend: MujocoBackend,
    *,
    variant: str = "vanilla",
    horizon: int = 25,
    n_samples: int = 512,
    sigma: float | Sequence[float] = 0.3,
    lambda_: float = 0.5,
    u_min: Sequence[float] | None = None,
    u_max: Sequence[float] | None = None,
    seed: int = 0,
    **extra,
):
    """Build an MPPI-family controller. `variant` selects from the registry
    in analytic_mppi.controllers.CONTROLLERS. `extra` is forwarded as-is to the
    chosen class.

    With nu=29, scalar `sigma` is broadcast across all joints, and `u_min`/`u_max`
    default to the model's actuator ctrlrange (hardcoding 29 bounds is impractical).
    """
    cls = get_controller_class(variant)
    nu = backend.nu
    sig = np.full(nu, float(sigma)) if np.isscalar(sigma) else np.asarray(sigma, dtype=np.float64)
    cr = np.asarray(backend.model.actuator_ctrlrange, dtype=np.float64)
    lo = cr[:, 0] if u_min is None else np.asarray(u_min, dtype=np.float64)
    hi = cr[:, 1] if u_max is None else np.asarray(u_max, dtype=np.float64)
    return cls(
        horizon=horizon,
        n_samples=n_samples,
        nu=nu,
        sigma=sig,
        lambda_=lambda_,
        u_min=lo,
        u_max=hi,
        seed=seed,
        **extra,
    )


@dataclass
class Env:
    backend: MujocoBackend
    cost: G1StandupCost
    controller: object   # MPPI or any variant from analytic_mppi.controllers.CONTROLLERS
    height_idx: int = HEIGHT_IDX
    quat_slice: slice = QUAT_SLICE


def make_env(
    *,
    variant: str = "vanilla",
    seed: int = 0,
    backend_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
    controller_kwargs: dict | None = None,
) -> Env:
    """One-call constructor: backend + cost + controller wired with defaults.

    The G1 starts from the collapsed default pose (pelvis z=0); the controller's
    job is to rise toward `target_height` while staying upright and near `qstand`.
    """
    backend = make_backend(**(backend_kwargs or {}))
    cost = make_cost(backend, **(cost_kwargs or {}))
    ctl = make_controller(backend, variant=variant, seed=seed,
                          **(controller_kwargs or {}))
    return Env(backend=backend, cost=cost, controller=ctl)


__all__ = [
    "MODEL_PATH",
    "HEIGHT_IDX",
    "QUAT_SLICE",
    "Env",
    "make_backend",
    "make_cost",
    "make_controller",
    "make_env",
]
