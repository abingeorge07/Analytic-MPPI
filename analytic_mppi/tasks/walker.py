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
        orient_cost = 10.0 * (zax - 1.0) ** 2
        velocity_cost = 1.0 * (vel - self.target_velocity) ** 2
        zero = np.zeros_like(height_cost)
        return np.stack([height_cost, orient_cost, velocity_cost, zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        terms = self.terminal_cost_terms(qpos, qvel, sensordata)
        control_cost = 0.001 * np.sum(u ** 2, axis=-1)
        # replace the last (zero) column with control cost
        out = terms.copy()
        out[..., -1] = control_cost
        return out

    # ---- FPL cost ----
    # Atom shapes mirror the improved HOPPER atoms (the walker's original Gaussian/generous
    # shapes were superseded): one-sided velocity ramp (spread, no reward for exceeding),
    # orientation that decays to 0 BEFORE horizontal (a real upright floor, not 0.5 at
    # horizontal), one-sided height that decays before the fall cliff. These are SHARED by
    # both the linear family (p=1) and FPL (p<0) — only the composition differs — so the
    # comparison stays apples-to-apples; only the objective scalarization is the variable.

    def _height_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # One-sided saturating band: full credit at/above h_full, linearly to 0 at h_floor
        # (well above the collapsed-torso height, so it decays BEFORE the fall). No penalty
        # for being taller than target — a walker's torso bobs up during a stride.
        h = self._torso_height(sensordata)
        h_full, h_floor = 1.1, 0.7
        return np.clip((h - h_floor) / (h_full - h_floor), 0.0, 1.0)

    def _orientation_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # zaxis_z = cos(tilt): 1 upright, 0 horizontal. Full credit near upright (z_full),
        # decaying to 0 by z_floor (≈ cos 53°) — BEFORE horizontal, so uprightness is a real
        # floor (the old (z+1)/2 gave 0.5 at horizontal, too generous to ever bind).
        z = self._torso_zaxis_z(sensordata)
        z_full, z_floor = 0.95, 0.6
        return np.clip((z - z_floor) / (z_full - z_floor), 0.0, 1.0)

    def _velocity_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # One-sided LINEAR ramp: proportional credit vx/target, capped at 1 at/above target,
        # no credit for backward. Gives the conjunction spread to discriminate on (the old
        # Gaussian pinned this atom near 0 for nearly every short rollout), and one-sided so
        # there's no incentive to sprint past target into a fall (FPL_MPPI_HANDOFF §6).
        vx = self._torso_vel_x(sensordata)
        return np.clip(vx / self.target_velocity, 0.0, 1.0)

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
        # Control fulfillment is undefined at the terminal step (no action applied), so return
        # only the 3 state-based atoms; _score_fpl handles n_term < n_run (matches hopper).
        # Avoids a constant-1 placeholder silently inflating the terminal composite.
        return np.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
            ],
            axis=-1,
        )
