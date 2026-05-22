from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import mujoco
from mujoco import rollout as mj_rollout


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

    def __init__(self, model_path: str | os.PathLike, nthread: int | None = None):
        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.nthread = nthread if nthread is not None else max(1, os.cpu_count() or 1)
        self._thread_data = [mujoco.MjData(self.model) for _ in range(self.nthread)]

        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)
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

    def rollout(self, initial_states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Parallel batched rollouts.

        initial_states: (B, nstate)
        controls:       (B, H, nu)
        returns states: (B, H, nstate)   -- excludes the initial state
        """
        initial_states = np.ascontiguousarray(initial_states, dtype=np.float64)
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        state, _ = mj_rollout.rollout(
            self.model,
            self._thread_data,
            initial_states,
            controls,
        )
        return state
