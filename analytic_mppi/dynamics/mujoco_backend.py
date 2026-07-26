from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import mujoco
from mujoco import rollout as mj_rollout


def apply_perturbation(model: "mujoco.MjModel", perturb: dict) -> None:
    """Scale dynamics parameters of `model` IN PLACE for a model-mismatch study.

    Keys (each an optional multiplicative scale, 1.0 = no change):
      mass_scale     — body_mass and body_inertia (heavier/lighter robot / payload)
      gain_scale     — actuator_gainprm[:,0] (stronger/weaker actuators)
      friction_scale — geom_friction[:,0] (grippier/slipperier ground & contacts)
      damping_scale  — dof_damping (more/less joint damping)
    Used to make a SEPARATE "true" backend whose dynamics differ from the controller's
    nominal planning model, so we can measure how gracefully FPL vs a fixed linear cost
    degrade as reality drifts from the model (you cannot retune weights for an unknown drift).
    """
    unknown = set(perturb) - {"mass_scale", "gain_scale", "friction_scale", "damping_scale"}
    if unknown:
        raise KeyError(f"unknown perturbation key(s) {sorted(unknown)}; expected a subset of "
                       f"mass_scale/gain_scale/friction_scale/damping_scale")
    if (s := perturb.get("mass_scale", 1.0)) != 1.0:
        model.body_mass[:] *= s
        model.body_inertia[:] *= s
    if (s := perturb.get("gain_scale", 1.0)) != 1.0:
        model.actuator_gainprm[:, 0] *= s
    if (s := perturb.get("friction_scale", 1.0)) != 1.0:
        model.geom_friction[:, 0] *= s
    if (s := perturb.get("damping_scale", 1.0)) != 1.0:
        model.dof_damping[:] *= s


class MujocoBackend:
    """MuJoCo-backed dynamics for MPPI.

    Two roles:
      * `rollout(initial_states, controls)` -- many open-loop rollouts in
        parallel via `mujoco.rollout.rollout` (multithreaded CPU).
      * `step(control)` + `get_state()` / `set_state()` -- single "real"
        simulator for the closed-loop outer loop.

    State vectors use `mjSTATE_FULLPHYSICS` so the same bytes can be passed
    back into `rollout` as initial states. Layout starts with [time, qpos, qvel, ...].
    """

    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

    def __init__(self, model_path: str | os.PathLike, nthread: int | None = None,
                 perturb: dict | None = None):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        # Optional dynamics-parameter perturbation (model-mismatch study). Applied before
        # MjData/thread_data are created so every rollout + step sees the perturbed model.
        if perturb:
            apply_perturbation(self.model, perturb)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.nthread = nthread if nthread is not None else max(1, os.cpu_count() or 1)
        self._thread_data = [mujoco.MjData(self.model) for _ in range(self.nthread)]

        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)
        self.nsensordata = int(self.model.nsensordata)
        self.nstate = int(mujoco.mj_stateSize(self.model, int(self.state_spec)))
        self.dt = float(self.model.opt.timestep)
        self.qpos_slice = slice(1, 1 + self.nq)
        self.qvel_slice = slice(1 + self.nq, 1 + self.nq + self.nv)

    def get_state(self) -> np.ndarray:
        state = np.empty(self.nstate, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, int(self.state_spec))
        return state

    def set_state(self, state: np.ndarray) -> None:
        s = np.ascontiguousarray(state, dtype=np.float64)
        mujoco.mj_setState(self.model, self.data, s, int(self.state_spec))
        mujoco.mj_forward(self.model, self.data)

    def step(self, control: np.ndarray) -> np.ndarray:
        self.data.ctrl[:] = np.asarray(control, dtype=np.float64)
        mujoco.mj_step(self.model, self.data)
        return self.get_state()

    def rollout(self, initial_states: np.ndarray, controls: np.ndarray):
        """Parallel batched rollouts.

        initial_states: (B, nstate)
        controls:       (B, H, nu)
        returns: (states, sensordata) each excluding the initial step
                 states:     (B, H, nstate)
                 sensordata: (B, H, nsensordata)
        """
        initial_states = np.ascontiguousarray(initial_states, dtype=np.float64)
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        state, sensordata = mj_rollout.rollout(
            self.model,
            self._thread_data,
            initial_states,
            controls,
        )
        return state, sensordata
