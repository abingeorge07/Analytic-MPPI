"""Unitree G1 humanoid motion-capture *walking* env for the MPPI testbed.

The G1 tracks a retargeted mocap walking clip (vendored ``walk1_subject1.npz``)
using the 23-DOF model shared with the g1 env (``envs/g1/scene_23dof.xml``).

Layout (FULLPHYSICS state = [time, qpos(nq=30), qvel(nv=29), ...], free base):
  qpos = (x, y, z, qw, qx, qy, qz, joints[23]) -> state indices [1..30]
  nu   = 23 PD position actuators (ctrl = target joint angle; bounds = ctrlrange)

The tracking cost (:class:`analytic_mppi.costs.HumanoidMocapCost`) reads the sim
time from ``state[..., 0]`` to index the reference clip, so it plugs directly
into the vanilla-MPPI cost interface. The sim is initialized to the first
reference frame so tracking is time-aligned from step 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from analytic_mppi.controllers import get_controller_class
from analytic_mppi.costs import HumanoidMocapCost
from analytic_mppi.dynamics import MujocoBackend


# Reuse the 23-DOF G1 model/assets from the g1 env (don't copy XML/meshes).
MODEL_PATH: Path = (
    Path(__file__).resolve().parent.parent / "g1" / "scene_23dof.xml"
)
# Vendored reference clip lives alongside this package.
REFERENCE_NPZ: Path = Path(__file__).resolve().parent / "walk1_subject1.npz"


def make_backend(**kw) -> MujocoBackend:
    """Build the MuJoCo backend for the G1 mocap env. Forwards kwargs on."""
    return MujocoBackend(MODEL_PATH, **kw)


def make_cost(backend: MujocoBackend, **weights) -> HumanoidMocapCost:
    """Build the mocap tracking cost: loads the vendored clip and precomputes the
    reference trajectory + torso/foot sensor targets from ``backend.model``.
    ``weights`` overrides the cost's tracking weights (configuration_weight, ...)."""
    return HumanoidMocapCost.from_model(
        backend.model, npz_path=REFERENCE_NPZ, **weights
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
    """Build an MPPI-family controller (mirrors the g1 env helper).

    With nu=23, scalar ``sigma`` is broadcast across all joints, and
    ``u_min``/``u_max`` default to the model's actuator ctrlrange.
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


def initial_state(backend: MujocoBackend, cost: HumanoidMocapCost,
                  start_frame: int = 0) -> np.ndarray:
    """FULLPHYSICS state at reference ``start_frame``.

    qpos/qvel come from that frame, and the sim clock is set to
    ``start_frame / fps`` so the time-indexed cost lines up (index 0 of the state
    buffer is the sim time). The walk1 clip stands roughly in place for its first
    ~2.5 s, so ``start_frame`` lets a demo begin mid-stride.
    """
    f = int(np.clip(start_frame, 0, cost.reference_qpos.shape[0] - 1))
    state = backend.get_state()
    state[0] = f / cost.fps
    state[backend.qpos_slice] = cost.reference_qpos[f]
    state[backend.qvel_slice] = cost.reference_qvel[f]
    return state


@dataclass
class Env:
    backend: MujocoBackend
    cost: HumanoidMocapCost
    controller: object   # MPPI or any variant from analytic_mppi.controllers.CONTROLLERS


def make_env(
    *,
    variant: str = "vanilla",
    seed: int = 0,
    start_frame: int = 0,
    backend_kwargs: dict | None = None,
    cost_kwargs: dict | None = None,
    controller_kwargs: dict | None = None,
) -> Env:
    """One-call constructor: backend + cost + controller wired with defaults, with
    the sim initialized to reference ``start_frame`` (time-aligned tracking)."""
    backend = make_backend(**(backend_kwargs or {}))
    cost = make_cost(backend, **(cost_kwargs or {}))
    ctl = make_controller(backend, variant=variant, seed=seed,
                          **(controller_kwargs or {}))
    backend.set_state(initial_state(backend, cost, start_frame))
    return Env(backend=backend, cost=cost, controller=ctl)


__all__ = [
    "MODEL_PATH",
    "REFERENCE_NPZ",
    "Env",
    "make_backend",
    "make_cost",
    "make_controller",
    "initial_state",
    "make_env",
]
