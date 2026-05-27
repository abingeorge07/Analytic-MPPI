"""In-hand cube reorientation task (CPU port of mis/tasks/cube.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from .base import Task


_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "cube" / "scene.xml"


def _quat_sub(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Hamiltonian quaternion 'subtraction' returning the 3-vec rotation
    error (axis * angle/2 approx). Inputs:
      q:     (..., 4) wxyz
      q_ref: (4,) wxyz
    Returns (..., 3) — the imaginary part of (q * conj(q_ref)).

    Matches mjx._src.math.quat_sub semantics: small-rotation error vector.
    """
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]  # conj of ref
    # q * conj(qref): only need the vector part
    vx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    vy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    vz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    # double-cover: if real part w1*w2 - x1*x2 - y1*y2 - z1*z2 < 0, flip sign
    real = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    sign = np.where(real < 0, -1.0, 1.0)
    return np.stack([vx, vy, vz], axis=-1) * sign[..., None]


class CubeRotationTask(Task):
    """Reorient a cube in the LEAP hand to the identity quaternion."""

    cost_term_names = ["position_cost", "orientation_cost", "grasp_cost"]
    cost_term_names_f = ["position_fulfillment", "orientation_fulfillment", "grasp_fulfillment"]

    def __init__(self, *, position_tol: float = 0.015):
        super().__init__(_MODEL_PATH)
        m = self.mj_model
        self._pos_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "cube_position")])
        self._ori_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "cube_orientation")])
        self.position_tol = float(position_tol)
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _cube_pos_err(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._pos_adr : self._pos_adr + 3]

    def _cube_quat(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._ori_adr : self._ori_adr + 4]

    # ---- normal cost ----

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        pos_err = self._cube_pos_err(sensordata)
        sq_dist = np.sum(pos_err[..., 0:2] ** 2, axis=-1)              # ignore z
        position_cost = 0.1 * sq_dist + 100.0 * np.maximum(sq_dist - self.position_tol ** 2, 0.0)

        quat_err = _quat_sub(self._cube_quat(sensordata), self.goal_quat)
        orientation_cost = np.sum(quat_err ** 2, axis=-1)

        grasp_cost = 0.001 * np.sum(u ** 2, axis=-1)
        return np.stack([position_cost, orientation_cost, grasp_cost], axis=-1)

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        pos_err = self._cube_pos_err(sensordata)
        position_cost = 100.0 * np.sum(pos_err ** 2, axis=-1)
        zero = np.zeros_like(position_cost)
        return np.stack([position_cost, zero, zero], axis=-1)

    # ---- FPL cost ----

    def _position_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        pos_err = self._cube_pos_err(sensordata)
        sq_dist = np.sum(pos_err[..., 0:2] ** 2, axis=-1)
        # 1 inside the position tolerance, decays to 0 outside the tolerance band.
        return np.exp(-sq_dist / (self.position_tol ** 2))

    def _orientation_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        quat_err = _quat_sub(self._cube_quat(sensordata), self.goal_quat)
        sq = np.sum(quat_err ** 2, axis=-1)
        return np.exp(-sq / (0.25 ** 2))

    def _grasp_fulfillment(self, u: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - np.mean(u ** 2, axis=-1), 0.0, 1.0)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._position_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._grasp_fulfillment(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        ones = np.ones_like(self._position_fulfillment(sensordata))
        return np.stack(
            [
                self._position_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                ones,
            ],
            axis=-1,
        )
