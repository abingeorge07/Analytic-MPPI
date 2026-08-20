"""Planar biped walking task (CPU port of mis/tasks/walker.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from .base import Task


_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "hopper" / "scene.xml"


class HopperTask(Task):
    """A planar biped tasked with walking forward at `target_velocity`.

    Sensors in scene.xml (verified at load time):
      torso_position        (3,) - px, py, pz
      torso_subtreelinvel   (3,) - vx, vy, vz
      torso_zaxis           (3,) - body-z axis expressed in world frame
    """

    cost_term_names = ["height_cost", "orientation_cost", "velocity_cost", "control_cost"]
    # FPL atoms. Speed uses a LINEAR ramp (proportional credit) rather than a Gaussian
    # that pins near 0 from low speed. Control is a near-constant effort regularizer that
    # rarely binds but measurably prevents falls (see _control_fulfillment).
    cost_term_names_f = ["height_fulfillment", "orientation_fulfillment", "velocity_fulfillment", "control_fulfillment"]

    def __init__(self, *, target_velocity: float = 1.0, target_height: float = 1.0):
        super().__init__(_MODEL_PATH)
        # target_velocity default is deliberately high enough that STANDING STILL
        # fails the speed objective (measured: a settled hopper stands at height
        # ≈1.2 with zaxis_z≈1.0 and vel_x≈0). This is what makes the objectives
        # genuinely compete for the stand-still-refusal demo (FPL_MPPI_HANDOFF §7).
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

        height_cost      = 10.0 * (height - self.target_height) ** 2
        orientation_cost = 50.0 * (1.0 - zax) ** 2
        velocity_cost    =  5.0 * (vel - self.target_velocity) ** 2
        zero             = np.zeros_like(height_cost)
        return np.stack([height_cost, orientation_cost, velocity_cost, zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        terms = self.terminal_cost_terms(qpos, qvel, sensordata)
        control_cost = 0.3 * np.sum(u ** 2, axis=-1)
        # replace the last (zero) column with control cost
        out = terms.copy()
        out[..., -1] = control_cost
        return out

    # ---- FPL cost ----

    def _height_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # One-sided saturating band: full credit at/above `h_full`, decaying linearly
        # to 0 at `h_floor`. THE key hopper atom. The band sits in a NARROW window just
        # below the settled standing height (≈1.19): full credit at 1.15, cratering to 0
        # by 0.85. This is deliberate — a hopper's only support is one leg, so the failure
        # mode is not "tip over" (uprightness stays ≈1 while the torso sinks) but "fold the
        # leg and let the torso COLLAPSE straight down" to chase forward speed. The old wide
        # band (1.0→0.5) gave full credit for any h≥1.0, so staying tall cost nothing in the
        # conjunction until the torso was already below 1.0 with the leg folded to its joint
        # limit — too late to recover in one short rollout. Pulling `h_floor` up to 0.85
        # makes ANY sag below standing crater the min-fulfillment conjunction, so the sampler
        # keeps the leg loaded and springy and actually HOPS instead of collapse-dragging.
        # (Empirically: min torso height over an episode 0.21→0.56, frac-time-tall 0.40→0.83.)
        # No penalty for hopping higher — the torso bobs up during flight.
        h = self._torso_height(sensordata)
        h_full, h_floor = 1.15, 0.85
        return np.clip((h - h_floor) / (h_full - h_floor), 0.0, 1.0)

    def _orientation_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # zaxis_z = cos(torso tilt): 1 upright, 0 horizontal. Full credit near
        # upright (`z_full` ≈ cos 18°), decaying to 0 by `z_floor` (≈ cos 53°) —
        # i.e. before the torso is horizontal, so uprightness decays before the
        # tip-over cliff (FPL_MPPI_HANDOFF §6).
        z = self._torso_zaxis_z(sensordata)
        z_full, z_floor = 0.95, 0.6
        return np.clip((z - z_floor) / (z_full - z_floor), 0.0, 1.0)

    def _velocity_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # One-sided, LINEAR forward-speed tracking: proportional credit vx/target,
        # capped at 1 at/above the target. The old Gaussian form pinned this atom
        # near 0 for almost every sampled rollout (target is unreachable within one
        # short planning rollout from low speed), giving the conjunction nothing to
        # discriminate on. A linear ramp gives partial speed proportional credit —
        # more spread — while staying one-sided (no reward for exceeding the target,
        # so no incentive to sprint into a fall; FPL_MPPI_HANDOFF §6).
        vx = self._torso_vel_x(sensordata)
        return np.clip(vx / self.target_velocity, 0.0, 1.0)

    def _control_fulfillment(self, u: np.ndarray) -> np.ndarray:
        # Mild effort regularizer. Sits near-constant in ~[0.75, 1.0] — it is almost
        # never the binding conjunction term, but that near-constant "floor" still
        # discourages slamming the actuators, which measurably prevents falls. (An
        # earlier version dropped it as a "dead" atom; that INCREASED the fall rate —
        # liveness ≠ value.)
        return np.clip(1.0 - 0.25 * np.mean(u ** 2, axis=-1), 0.0, 1.0)

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
        # control fulfillment is undefined at the terminal step (no action applied),
        # so this returns only the 3 state-based atoms; _score_fpl handles n_term < n_run.
        return np.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
            ],
            axis=-1,
        )
