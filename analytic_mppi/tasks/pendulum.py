"""Pendulum swing-up task (CPU port of mis/tasks/pendulum.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Task


_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "pendulum" / "model.xml"


class PendulumTask(Task):
    """Inverted pendulum: drive theta to pi (upright)."""

    cost_term_names = ["theta_cost", "theta_dot_cost", "control_cost"]
    cost_term_names_f = ["theta_fulfillment", "control_fulfillment"]

    def __init__(self, *, target_angle: float = np.pi):
        super().__init__(_MODEL_PATH)
        # theta convention: 0 = hanging down, pi = upright. Default pi preserves the
        # classic swing-up task (and every existing pendulum test / call site).
        self.target_angle = float(target_angle)

    # ---- normal cost ----

    def _distance_to_target(self, qpos: np.ndarray) -> np.ndarray:
        # Smooth wrap-free distance to theta = target_angle.
        theta = qpos[..., 0] - self.target_angle
        return (np.cos(theta) - 1.0) ** 2 + np.sin(theta) ** 2

    def _theta_dot_cost(self, qvel: np.ndarray) -> np.ndarray:
        return 0.01 * qvel[..., 0] ** 2

    def _control_cost(self, u: np.ndarray) -> np.ndarray:
        return 0.001 * np.sum(u ** 2, axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._distance_to_target(qpos),
                self._theta_dot_cost(qvel),
                self._control_cost(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        d = self._distance_to_target(qpos)
        td = self._theta_dot_cost(qvel)
        zero = np.zeros_like(d)
        return np.stack([d, td, zero], axis=-1)

    # ---- FPL cost (per-step satisfaction in [0,1]) ----

    def _target_fulfillment(self, qpos: np.ndarray) -> np.ndarray:
        # Proximity of the swing angle to target_angle, in [0,1]: 1 at theta*, 0 opposite.
        # = cos of the angle error, shifted to [0,1]. (At target_angle=pi this reduces to
        # the old (1 - cos theta)/2 upright term.)
        theta = qpos[..., 0] - self.target_angle
        return (1.0 + np.cos(theta)) * 0.5

    def _control_fulfillment(self, u: np.ndarray) -> np.ndarray:
        # FLOOR, not target: ~1 through every control level the task legitimately uses,
        # decaying to 0 only as an actuator approaches SATURATION (|u| -> its ctrl limit).
        # A regularizer atom must read ~1 at the DESIRED behaviour — including the steady
        # torque needed to HOLD a pose against gravity (|u|~0.84 to hold 20 deg here) —
        # otherwise it is low at the exact state we want, fights the task under the FPL
        # min-conjunction, and the controller sits still. Only *pegging* the actuator
        # (slamming) is penalised. The old 1 - mean(u^2) form peaked at u=0 and so
        # directly rewarded doing nothing (see PendulumComTask docstring / the 2x2 study).
        umax = np.maximum(np.abs(self.u_min), np.abs(self.u_max))
        frac = np.max(np.abs(u) / umax, axis=-1)     # worst actuator, in [0,1]
        u_floor = 0.9                                 # plateau: |u|<=0.9*limit costs nothing
        return np.clip((1.0 - frac) / (1.0 - u_floor), 0.0, 1.0)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [self._target_fulfillment(qpos), self._control_fulfillment(u)],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        # Only the target term has a terminal contribution. Control fulfillment
        # is undefined at the terminal step (no action is applied), so we omit
        # it rather than padding with ones — a constant placeholder biases the
        # power-mean / discounted reward upward without conveying any info.
        return self._target_fulfillment(qpos)[..., None]


# Pendulum geometry (from envs/pendulum/pendulum.xml): the link pivots about the
# world y-axis at height PIVOT_Z; its inertial CoM sits COM_R below the pivot along
# the link. So the swinging-mass CoM in the world x-z plane is a point on a circle
# of radius COM_R centred at the pivot (y is identically 0 for this planar system).
PIVOT_Z = 1.5
COM_R = 0.5
# theta convention: 0 = hanging straight down, pi = balanced upright.


class PendulumComTask(PendulumTask):
    """Inverted pendulum driven by a **CoM-position reference**, not a joint-angle cost.

    This is the minimal test bed for the "derive the desired CoM, then cost deviation
    from it" idea. Given a `target_angle` theta*, we compute where the link CoM *should*
    be — `com_ref = (-COM_R sin theta*, PIVOT_Z - COM_R cos theta*)` — and penalise the
    Cartesian distance of the actual CoM to that point. For a single link the CoM is
    closed-form in theta (below); on a multi-body robot this same quantity is
    `data.subtree_com` / a MuJoCo `subtreecom` sensor and the reference comes from a
    reduced (template) model. The cost math is identical either way.

    Feasibility caveat this task makes vivid: the actuator (gear 2, |ctrl|<=1 -> |tau|<=2
    N*m) is WEAKER than the peak gravity torque m*g*l ~= 4.9 N*m, so the pendulum can only
    statically HOLD a CoM reference within ~24 deg of straight-down or straight-up. A
    kinematic CoM reference in the un-holdable band is dynamically infeasible — the demo
    shows the cost pulling toward it but the pendulum unable to settle there. That is the
    argument for a *dynamically feasible* (template-derived) reference rather than a
    hand-drawn one.
    """

    cost_term_names = ["com_cost", "theta_dot_cost", "control_cost"]
    cost_term_names_f = ["com_fulfillment", "control_fulfillment"]

    def __init__(self, *, target_angle: float = np.pi, com_weight: float = 10.0):
        super().__init__(target_angle=target_angle)   # base stores self.target_angle
        self.com_weight = float(com_weight)
        # Precompute the reference CoM (x, z) once; y is always 0.
        self.com_ref = np.array(
            [-COM_R * np.sin(self.target_angle),
             PIVOT_Z - COM_R * np.cos(self.target_angle)],
            dtype=np.float64,
        )

    # ---- CoM kinematics ----

    def _com_xz(self, qpos: np.ndarray) -> np.ndarray:
        """World-frame link CoM in the x-z plane, shape (*leading, 2)."""
        theta = qpos[..., 0]
        x = -COM_R * np.sin(theta)
        z = PIVOT_Z - COM_R * np.cos(theta)
        return np.stack([x, z], axis=-1)

    def _com_dist(self, qpos: np.ndarray) -> np.ndarray:
        """Euclidean CoM distance to the reference. Ranges [0, 2*COM_R] = [0, 1]."""
        d = self._com_xz(qpos) - self.com_ref
        return np.sqrt(np.sum(d ** 2, axis=-1))

    # ---- normal cost (Cartesian CoM deviation) ----

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self.com_weight * self._com_dist(qpos) ** 2,
                self._theta_dot_cost(qvel),
                self._control_cost(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        com = self.com_weight * self._com_dist(qpos) ** 2
        td = self._theta_dot_cost(qvel)
        return np.stack([com, td, np.zeros_like(com)], axis=-1)

    # ---- FPL cost (per-step CoM-proximity fulfillment in [0,1]) ----

    def _com_fulfillment(self, qpos: np.ndarray) -> np.ndarray:
        # CoM distance already lives in [0, 1] (diameter = 2*COM_R). Full credit at the
        # reference, decaying linearly to 0 at the opposite side of the swing circle.
        return np.clip(1.0 - self._com_dist(qpos), 0.0, 1.0)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [self._com_fulfillment(qpos), self._control_fulfillment(u)],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        return self._com_fulfillment(qpos)[..., None]
