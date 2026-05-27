"""G1 humanoid standup task (CPU port of mis/tasks/humanoid_standup.py).

The mis version reads torso height via `state.site_xpos[torso_id, 2]`, which
isn't exported by `mujoco.rollout`. We exposed the same value as a `framepos`
sensor named `torso_pos` in `envs/g1/scene.xml` so it shows up in sensordata.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from .base import Task


_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "g1" / "scene.xml"


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a constant vector v by a batch of quaternions q (wxyz).

    q: (..., 4) — wxyz
    v: (3,)     — constant world-frame vector to rotate (we use upright = (0,0,1)).
    Returns: (..., 3)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # Standard formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    # Computed element-wise for batches.
    ax = y * vz - z * vy + w * vx
    ay = z * vx - x * vz + w * vy
    az = x * vy - y * vx + w * vz
    rx = vx + 2.0 * (y * az - z * ay)
    ry = vy + 2.0 * (z * ax - x * az)
    rz = vz + 2.0 * (x * ay - y * ax)
    return np.stack([rx, ry, rz], axis=-1)


class G1StandupTask(Task):
    """Standup task for the Unitree G1 humanoid (~36 DOF)."""

    cost_term_names = ["orientation_cost", "height_cost", "nominal_cost"]
    cost_term_names_f = ["orientation_fulfillment", "height_fulfillment", "nominal_fulfillment"]

    def __init__(self, *, target_height: float = 0.9):
        super().__init__(_MODEL_PATH)
        m = self.mj_model
        self._quat_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_in_torso_quat")])
        self._torso_pos_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_pos")])
        self.target_height = float(target_height)
        # standing reference qpos and joint deviation budget for the FPL nominal term
        kf = m.keyframe("stand")
        self.qstand = np.asarray(kf.qpos, dtype=np.float64).copy()
        # joints start at qpos index 7 (after free joint: 3 pos + 4 quat)
        jnt_range = np.asarray(m.jnt_range[1:], dtype=np.float64)  # (njoint-1, 2)
        jnt_low, jnt_high = jnt_range[:, 0], jnt_range[:, 1]
        qstand_j = self.qstand[7:]
        max_dev = np.maximum(qstand_j - jnt_low, jnt_high - qstand_j)
        self.joint_max_dev = np.where(max_dev > 1e-6, max_dev, 1.0)

    def _torso_orientation(self, sensordata: np.ndarray) -> np.ndarray:
        """Return the rotated upright vector (0,0,1) by the torso quaternion."""
        quat = sensordata[..., self._quat_adr : self._quat_adr + 4]
        return _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))

    def _torso_height(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._torso_pos_adr + 2]

    # ---- normal cost ----

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        orient = self._torso_orientation(sensordata)
        orientation_cost = 10.0 * np.sum(orient ** 2, axis=-1)
        height_cost = 10.0 * (self._torso_height(sensordata) - self.target_height) ** 2
        nominal_cost = 0.1 * np.sum((qpos[..., 7:] - self.qstand[7:]) ** 2, axis=-1)
        return np.stack([orientation_cost, height_cost, nominal_cost], axis=-1)

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms(qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu]))

    # ---- FPL cost ----

    def _height_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        h = self._torso_height(sensordata)
        # mis formula: |h - target| clipped to delta=0.9 -> mapped to [0,1].
        delta = 0.9
        err = -np.abs(h - self.target_height)
        err = np.clip(err, -delta, 0.0)
        return err / delta + 1.0

    def _orientation_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # Dot of rotated upright with desired (0,0,1) is just the rz component; shift to [0,1].
        orient = self._torso_orientation(sensordata)
        rz = orient[..., 2]
        return (rz + 1.0) * 0.5

    def _nominal_fulfillment(self, qpos: np.ndarray) -> np.ndarray:
        diff = np.abs(qpos[..., 7:] - self.qstand[7:])
        frac = np.clip(1.0 - diff / self.joint_max_dev, 1e-8, 1.0)
        # geometric mean cubed (matches mis humanoid_standup formula)
        log_mean = np.mean(np.log(frac), axis=-1)
        geom_mean = np.exp(log_mean)
        return geom_mean

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._orientation_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._nominal_fulfillment(qpos),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms_f(qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu]))
