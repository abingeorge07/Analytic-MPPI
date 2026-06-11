from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a constant vector v by a batch of quaternions q (wxyz).

    q: (..., 4) — wxyz
    v: (3,)     — constant world-frame vector to rotate (we use upright = (0,0,1)).
    Returns: (..., 3)

    (Local copy of tasks/g1_standup.py:_quat_rotate to keep costs free of a
    dependency on the tasks package.)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # Standard formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w * v)
    ax = y * vz - z * vy + w * vx
    ay = z * vx - x * vz + w * vy
    az = x * vy - y * vx + w * vz
    rx = vx + 2.0 * (y * az - z * ay)
    ry = vy + 2.0 * (z * ax - x * az)
    rz = vz + 2.0 * (x * ay - y * ax)
    return np.stack([rx, ry, rz], axis=-1)


@dataclass
class G1StandupCost:
    """Standup cost for the Unitree G1 humanoid (free-base, ~29 actuated joints).

    Height and the nominal pose come from the FULLPHYSICS state
    ([time, qpos, qvel, ...]); the torso orientation comes from the
    `imu_in_torso_quat` sensor (passed alongside `states` by the MPPI controller),
    which tracks the actual torso link rather than the pelvis base -- the two
    diverge once the waist joints bend. Layout for the G1 (leading free joint):
      height  = state[..., 3]          (qpos[2], pelvis z)
      joints  = state[..., 8:8+njnt]   (qpos[7:], actuated joint angles)
      quat    = sensordata[..., adr:adr+4]  (imu_in_torso_quat, torso, wxyz)
                fallback when sensordata is None: state[..., 4:8] (pelvis base quat)

    Per-step cost (per the user's formula, equal unit weights by default):
      orientation_cost = orient_weight  * sum((R(quat)@[0,0,1] - [0,0,1])^2)
                         (= 2*(1 - up_z); 0 upright, up to 4 inverted)
      height_cost      = height_weight  * (height - target_height)^2
      nominal_cost     = nominal_weight * sum((joints - qstand_joints)^2)

    `__call__` does the MPPI integration over (B, H):
      states:     (B, H, nstate)        -- rollout output, excludes initial
      controls:   (B, H, nu)            -- unused (no control term)
      sensordata: (B, H, nsensordata)   -- rollout output, excludes initial
      returns:    (B,)

      total = sum_{t=0..H-2} l_state(x_t) + terminal_weight * l_state(x_{H-1})
    """

    qstand_joints: np.ndarray
    height_idx: int = 3
    quat_slice: slice = slice(4, 8)
    joint_slice: slice = slice(8, 37)
    orientation_sensor_adr: int = -1
    target_height: float = 0.79
    orient_weight: float = 1.0
    height_weight: float = 1.0
    nominal_weight: float = 1.0
    terminal_weight: float = 1.0

    _qstand: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._qstand = np.asarray(self.qstand_joints, dtype=np.float64)

    # ---- per-step components (work on any leading shape) ----

    def _get_torso_orientation(self, state: np.ndarray,
                               sensordata: np.ndarray | None = None) -> np.ndarray:
        """Up-axis error vs world up: R(quat)@[0,0,1] - [0,0,1]. (..., 3).

        Reads the torso quaternion from the `imu_in_torso_quat` sensor when
        sensordata is available; otherwise falls back to the pelvis base quat.
        """
        if sensordata is not None and self.orientation_sensor_adr >= 0:
            adr = self.orientation_sensor_adr
            quat = sensordata[..., adr : adr + 4]
        else:
            quat = state[..., self.quat_slice]
        up = _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
        return up - np.array([0.0, 0.0, 1.0])

    def _get_torso_height(self, state: np.ndarray) -> np.ndarray:
        return state[..., self.height_idx]

    def _get_orientation_cost(self, state: np.ndarray,
                              sensordata: np.ndarray | None = None) -> np.ndarray:
        orient = self._get_torso_orientation(state, sensordata)
        return self.orient_weight * (orient ** 2).sum(axis=-1)

    def _get_height_cost(self, state: np.ndarray) -> np.ndarray:
        h = self._get_torso_height(state)
        return self.height_weight * (h - self.target_height) ** 2

    def _get_nominal_cost(self, state: np.ndarray) -> np.ndarray:
        joints = state[..., self.joint_slice]
        return self.nominal_weight * ((joints - self._qstand) ** 2).sum(axis=-1)

    def _state_cost(self, state: np.ndarray,
                    sensordata: np.ndarray | None = None) -> np.ndarray:
        return (
            self._get_orientation_cost(state, sensordata)
            + self._get_height_cost(state)
            + self._get_nominal_cost(state)
        )

    def running_cost(self, state: np.ndarray, control: np.ndarray,
                     sensordata: np.ndarray | None = None) -> np.ndarray:
        """Running cost l(x_t, u_t) -- no control term."""
        return self._state_cost(state, sensordata)

    def terminal_cost(self, state: np.ndarray,
                      sensordata: np.ndarray | None = None) -> np.ndarray:
        """Terminal cost phi(x_T)."""
        return self._state_cost(state, sensordata)

    # ---- MPPI batched integration over (B, H) ----

    def __call__(self, states: np.ndarray, controls: np.ndarray,
                 sensordata: np.ndarray | None = None) -> np.ndarray:
        sc = self._state_cost(states, sensordata)  # (B, H)
        running = sc[..., :-1].sum(axis=-1)
        terminal = self.terminal_weight * sc[..., -1]
        return running + terminal
