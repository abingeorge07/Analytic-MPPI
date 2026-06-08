"""Planar biped walking task (CPU port of mis/tasks/walker.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from .base import Task


_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "walker" / "scene.xml"


class WalkerTask(Task):
    """A planar biped tasked with walking forward at `target_velocity`.

    Sensors in scene.xml (verified at load time):
      torso_position        (3,) - px, py, pz
      torso_subtreelinvel   (3,) - vx, vy, vz
      torso_zaxis           (3,) - body-z axis expressed in world frame
    """

    cost_term_names = ["height_cost", "orientation_cost", "velocity_cost", "control_cost"]
    cost_term_names_f = ["height_fulfillment", "orientation_fulfillment", "velocity_fulfillment", "control_fulfillment"]

    def __init__(self, *, target_velocity: float = 1.5, target_height: float = 1.2):
        super().__init__(_MODEL_PATH)
        self.target_velocity = float(target_velocity)
        self.target_height = float(target_height)
        # cache sensor address spans
        m = self.mj_model
        self._pos_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_position")])
        self._vel_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_subtreelinvel")])
        self._zax_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_zaxis")])

    def _torso_height(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._pos_adr + 2]

    def _torso_vel_x(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._vel_adr]

    def _torso_zaxis_z(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._zax_adr + 2]

    # ---- normal cost ----

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        height = self._torso_height(sensordata)
        vel = self._torso_vel_x(sensordata)
        zax = self._torso_zaxis_z(sensordata)
        height_cost = 10.0 * (height - self.target_height) ** 2
        orient_cost = .0 * (zax - 1.0) ** 2
        velocity_cost = 1.0 * (vel - self.target_velocity) ** 2
        zero = np.zeros_like(height_cost)
        return np.stack([height_cost, orient_cost, velocity_cost, zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        terms = self.terminal_cost_terms(qpos, qvel, sensordata)
        control_cost = 0.0000000001 * np.sum(u ** 2, axis=-1)
        # replace the last (zero) column with control cost
        out = terms.copy()
        out[..., -1] = control_cost
        return out

    # ---- FPL cost ----

    def _height_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # 1 at target height, decays with deviation, gentle band.
        err = self._torso_height(sensordata) - self.target_height
        return np.exp(-(err ** 2) / (0.1 ** 2))

    def _orientation_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # zax dot (0,0,1) in [-1,1] -> shift to [0,1]
        z = self._torso_zaxis_z(sensordata)
        return np.clip((z + 1.0) * 0.5, 0.0, 1.0)

    def _velocity_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        err = self._torso_vel_x(sensordata) - self.target_velocity
        return np.exp(-(err ** 2) / (0.5 ** 2))

    def _control_fulfillment(self, u: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - np.mean(u ** 2, axis=-1), 0.0, 1.0)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
                self._control_fulfillment(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        f1 = self._height_fulfillment(sensordata)
        ones = np.ones_like(f1)
        return np.stack(
            [
                f1,
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
                ones,
            ],
            axis=-1,
        )
