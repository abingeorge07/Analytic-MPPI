"""G1 humanoid WALK task — the competing-objective flagship for FPL vs linear.

Standup (g1_standup) is a COOPERATING-objective balance task (all atoms want "stay put
and upright"), so FPL ≈ linear there. Walking makes objectives genuinely COMPETE: to track
a forward speed the torso must break the static-balance posture (shift weight, step, pitch),
which fights uprightness/height. This is the humanoid analogue of the hopper Pareto result.

Built as a subclass of G1StandupTask that reuses ALL its sensor plumbing and the
orientation / height / per-joint-posture / control / wobble atoms UNCHANGED, and swaps the
"stillness" atom (penalize torso linear velocity) for a "forward" atom (reward forward speed
toward `target_velocity`). It intentionally does NOT call G1StandupTask.__init__ (which
hard-codes the standup group names / offsets) — it wires the walk-specific groups + priority
offsets itself, on top of Task.__init__ + the shared standup field setup.

Atom order (flat, `cost_term_names_f`):
    [orientation, height, nominal, control, forward, wobble]
Priority offsets (lower = higher priority; paper Eq.7): orientation/height/forward = 0
(the competing trio fought for first), wobble/control = 0.1, nominal = 0.5 (walking must be
free to leave the stand pose).
"""
from __future__ import annotations

import numpy as np
import mujoco

from .base import Task
from .g1_standup import G1StandupTask, _MODEL_PATH


class G1WalkTask(G1StandupTask):
    cost_term_names = ["orientation_cost", "height_cost", "nominal_cost", "control_cost",
                       "forward_cost", "wobble_cost"]
    cost_term_names_f = ["orientation_fulfillment", "height_fulfillment", "nominal_fulfillment",
                         "control_fulfillment", "forward_fulfillment", "wobble_fulfillment"]
    # Hybrid: orientation + height are the anti-topple/anti-collapse floors; the rest
    # (incl. forward speed) are targets.
    floor_term_indices = [0, 1]

    DEFAULT_FPL_OFFSETS = {
        "orientation_fulfillment": 0.0,   # floor: don't tip (top priority)
        "height_fulfillment": 0.0,        # floor: don't collapse
        "forward_fulfillment": 0.0,       # fought for: WANT speed to compete with upright
        "wobble_fulfillment": 0.1,        # allow some turning during a gait
        "control_fulfillment": 0.1,
        "nominal_fulfillment": 0.5,       # low: walking must leave the stand pose
    }

    def __init__(self, *, target_velocity: float = 0.5, target_height: float = 1.0,
                 ctrl_cost_weight: float = 2.0, wobble_weight: float = 1.0,
                 vel_cost_weight: float = 5.0, ang_vel_ref: float = 3.0,
                 fpl_offsets: dict | None = None):
        # NOTE: deliberately skip G1StandupTask.__init__ (standup-specific groups/offsets).
        Task.__init__(self, _MODEL_PATH)
        # Height uses the inherited (wide symmetric) _height_fulfillment. A TIGHT one-sided
        # height band was tried to stop the crouch-collapse but it TOPPLES instead (forcing
        # height dumps pressure off orientation → torso inverts; the g1_standup finding
        # reproduces here). Orientation and height floors are in direct tension for this MPC
        # setup, so the honest g1_walk result is graceful crouch (torso never inverts), not
        # tall walking.
        m = self.mj_model
        self._quat_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_in_torso_quat")])
        self._torso_pos_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_pos")])
        self._linvel_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_in_torso_linvel")])
        self._angvel_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-angular-velocity")])
        self.target_height = float(target_height)
        self.target_velocity = float(target_velocity)

        kf = m.keyframe("stand")
        self.qstand = np.asarray(kf.qpos, dtype=np.float64).copy()
        jnt_range = np.asarray(m.jnt_range[1:], dtype=np.float64)
        jnt_low, jnt_high = jnt_range[:, 0], jnt_range[:, 1]
        qstand_j = self.qstand[7:]
        max_dev = np.maximum(qstand_j - jnt_low, jnt_high - qstand_j)
        self.joint_max_dev = np.where(max_dev > 1e-6, max_dev, 1.0)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.u_ref = np.asarray(kf.ctrl, dtype=np.float64).copy()
        self.u_half_range = 0.5 * (self.u_max - self.u_min)
        self.wobble_weight = float(wobble_weight)
        self.vel_cost_weight = float(vel_cost_weight)
        self.ang_vel_ref = float(ang_vel_ref)
        # standup reads these for the stillness atom; unused here but kept so inherited
        # helpers referencing them never KeyError.
        self.stillness_weight = 0.0
        self.lin_vel_ref = 1.0

        njoint = int(self.joint_max_dev.shape[0])
        c = 2 + njoint  # grouped index of control; forward/wobble follow
        self.fpl_group_names = ["orientation", "height", "posture", "control", "forward", "wobble"]
        self.fpl_groups = [[0], [1], list(range(2, 2 + njoint)), [c], [c + 1], [c + 2]]

        offsets = dict(self.DEFAULT_FPL_OFFSETS)
        if fpl_offsets:
            bad = [k for k in fpl_offsets if k not in self.cost_term_names_f]
            if bad:
                raise KeyError(f"unknown fpl_offsets atom(s) {bad}; "
                               f"expected a subset of {self.cost_term_names_f}")
            offsets.update(fpl_offsets)
        self.fpl_offsets = np.array([offsets[n] for n in self.cost_term_names_f], dtype=np.float64)
        self.fpl_offsets_grouped = np.concatenate([
            [offsets["orientation_fulfillment"]],
            [offsets["height_fulfillment"]],
            np.full(njoint, offsets["nominal_fulfillment"]),
            [offsets["control_fulfillment"]],
            [offsets["forward_fulfillment"]],
            [offsets["wobble_fulfillment"]],
        ]).astype(np.float64)

    # ---- forward-speed atom (replaces stillness) ----

    def _forward_speed(self, sensordata: np.ndarray) -> np.ndarray:
        """World-frame forward (x) component of torso linear velocity."""
        return self._torso_linvel(sensordata)[..., 0]

    def _forward_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # One-sided LINEAR ramp: proportional credit vx/target, capped at 1 at/above the
        # target. No reward for exceeding (no incentive to sprint into a fall) and no credit
        # for going backwards. Mirrors the hopper velocity atom (FPL_MPPI_HANDOFF §6).
        vx = self._forward_speed(sensordata)
        return np.clip(vx / self.target_velocity, 0.0, 1.0)

    # ---- FPL cost (flat) ----

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        atoms = np.stack(
            [
                self._orientation_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._nominal_fulfillment(qpos),
                self._control_fulfillment(u),
                self._forward_fulfillment(sensordata),
                self._wobble_fulfillment(sensordata),
            ],
            axis=-1,
        )
        return self._apply_offset(atoms, self.fpl_offsets)

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms_f(qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu]))

    # ---- FPL cost (grouped / layered) ----

    def running_cost_terms_f_grouped(self, qpos, qvel, sensordata, u) -> np.ndarray:
        orient = self._orientation_fulfillment(sensordata)[..., None]
        height = self._height_fulfillment(sensordata)[..., None]
        joints = self._joint_fulfillments(qpos)
        control = self._control_fulfillment(u)[..., None]
        forward = self._forward_fulfillment(sensordata)[..., None]
        wobble = self._wobble_fulfillment(sensordata)[..., None]
        atoms = np.concatenate([orient, height, joints, control, forward, wobble], axis=-1)
        return self._apply_offset(atoms, self.fpl_offsets_grouped)

    def terminal_cost_terms_f_grouped(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms_f_grouped(
            qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu])
        )

    # ---- normal cost (quadratic; for the 'normal' baseline + linear-family p=1) ----

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        rz = self._torso_orientation(sensordata)[..., 2]
        orientation_cost = 10.0 * (1.0 - rz)
        height_cost = 10.0 * (self._torso_height(sensordata) - self.target_height) ** 2
        nominal_cost = 0.1 * np.sum((qpos[..., 7:] - self.qstand[7:]) ** 2, axis=-1)
        control_cost = self.ctrl_cost_weight * self._control_effort(u)
        # forward tracking: penalize the shortfall below target (one-sided, matching the
        # FPL atom — no penalty for going faster than target).
        vx = self._forward_speed(sensordata)
        forward_cost = self.vel_cost_weight * np.clip(self.target_velocity - vx, 0.0, None) ** 2
        wobble_cost = self.wobble_weight * self._wobble_effort(sensordata)
        return np.stack([orientation_cost, height_cost, nominal_cost, control_cost,
                         forward_cost, wobble_cost], axis=-1)

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms(qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu]))


class G1ReachTask(G1WalkTask):
    """Gentle, ACHIEVABLE competing-objective humanoid task: lean the torso FORWARD to a
    target x-offset and HOLD, feet planted — no stepping/gait required, so it stays within
    static-balance MPC reach (unlike g1_walk, which needs a gait and collapses).

    The competition is the same structure as the hopper win but static: leaning the COM
    forward reduces the stability margin, so a linear cost with high lean-weight over-commits
    and topples forward, while low lean-weight is timid; the FPL floor should let it lean to
    the safe limit and hold. Everything else (orientation/height/posture/control/wobble atoms,
    groups, offsets) is inherited from G1WalkTask; ONLY the "forward" atom changes from
    velocity-tracking to POSITION-tracking (reach & hold), which removes the incentive to keep
    accelerating (the thing that made g1_walk step and fall).
    """

    def __init__(self, *, target_reach: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        # target_reach: forward torso-x offset (m) to lean to. The stand keyframe places the
        # torso at x≈0, so absolute torso x IS the forward displacement. Kept modest so the
        # COM can reach it WITHOUT stepping (a bigger offset would force a step → collapse).
        self.target_reach = float(target_reach)

    def _forward_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # Position-based (reach & hold): proportional credit x/target_reach, capped at 1 at
        # the target, no credit for going backward. Rewards a held forward lean, not speed.
        x = sensordata[..., self._torso_pos_adr]      # torso x = forward displacement
        return np.clip(x / self.target_reach, 0.0, 1.0)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        # Same as walk but the forward term penalizes the forward-position shortfall.
        rz = self._torso_orientation(sensordata)[..., 2]
        orientation_cost = 10.0 * (1.0 - rz)
        height_cost = 10.0 * (self._torso_height(sensordata) - self.target_height) ** 2
        nominal_cost = 0.1 * np.sum((qpos[..., 7:] - self.qstand[7:]) ** 2, axis=-1)
        control_cost = self.ctrl_cost_weight * self._control_effort(u)
        x = sensordata[..., self._torso_pos_adr]
        forward_cost = self.vel_cost_weight * np.clip(self.target_reach - x, 0.0, None) ** 2
        wobble_cost = self.wobble_weight * self._wobble_effort(sensordata)
        return np.stack([orientation_cost, height_cost, nominal_cost, control_cost,
                         forward_cost, wobble_cost], axis=-1)
