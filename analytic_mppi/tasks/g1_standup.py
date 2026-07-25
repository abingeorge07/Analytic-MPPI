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

    cost_term_names = ["orientation_cost", "height_cost", "nominal_cost", "control_cost",
                       "stillness_cost", "wobble_cost"]
    cost_term_names_f = ["orientation_fulfillment", "height_fulfillment", "nominal_fulfillment",
                         "control_fulfillment", "stillness_fulfillment", "wobble_fulfillment"]
    # Hybrid mode: orientation is the safety FLOOR (log-barrier — never tip); height and
    # nominal are TARGETS (unbounded quadratic — stand tall / good posture). This gives
    # the strong height signal FPL lacks while keeping a hard anti-topple floor.
    floor_term_indices = [0]

    # FPL priority offsets (paper Eq. 7, [phi]_delta): a per-atom delta that RAISES the
    # baseline fulfillment of an atom, which in the weakest-link power-mean DEPRIORITIZES
    # it (it looks more satisfied, so it stops binding) until lower-offset atoms catch up.
    # Rule: lower offset = higher priority. Aligned with `cost_term_names_f`. Priority tiers:
    #   1. orientation, height, wobble -> 0.0  (don't tip/spin, stand tall — fought for first)
    #   2. control, stillness          -> 0.1  (move as little as possible once upright/tall)
    #   3. nominal (joint pose)         -> 0.5  (matched last, once the rest are reasonably met)
    DEFAULT_FPL_OFFSETS = {
        "orientation_fulfillment": 0.0,
        "height_fulfillment": 0.0,
        "wobble_fulfillment": 0.0,
        "control_fulfillment": 0.1,
        "stillness_fulfillment": 0.1,
        "nominal_fulfillment": 0.5,
    }

    def __init__(self, *, target_height: float = 1.0, ctrl_cost_weight: float = 2.0,
                 stillness_weight: float = 1.0, wobble_weight: float = 1.0,
                 lin_vel_ref: float = 1.0, ang_vel_ref: float = 3.0,
                 fpl_offsets: dict | None = None):
        super().__init__(_MODEL_PATH)
        m = self.mj_model
        self._quat_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_in_torso_quat")])
        self._torso_pos_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_pos")])
        # Torso velocity sensors (Tier A movement terms): linear velocity -> "stillness"
        # (don't drift/lurch), angular velocity -> "wobble" (don't spin/tip-oscillate).
        self._linvel_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_in_torso_linvel")])
        self._angvel_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu-torso-angular-velocity")])
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
        # Control-effort term: these are POSITION actuators (the control u is a target
        # joint angle), so "minimize the controls" means keep u near the nominal REST
        # control u_ref (the 'stand' keyframe ctrl) — NOT near zero, which would fight the
        # standing pose. We measure effort as the mean squared deviation from u_ref,
        # normalized per-actuator by its half control-range so it's dimensionless and the
        # FPL fulfillment maps cleanly into [0,1]. Raise ctrl_cost_weight to move less.
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        self.u_ref = np.asarray(kf.ctrl, dtype=np.float64).copy()
        self.u_half_range = 0.5 * (self.u_max - self.u_min)
        # Movement (velocity) terms: effort = ||v||^2 / ref^2, so effort ~1 at the reference
        # speed. Weights scale the normal-cost penalty; the FPL atom is exp(-effort) in [0,1].
        self.stillness_weight = float(stillness_weight)
        self.wobble_weight = float(wobble_weight)
        self.lin_vel_ref = float(lin_vel_ref)
        self.ang_vel_ref = float(ang_vel_ref)
        # Layered FPL grouping: orientation (idx 0), height (idx 1), then one fulfillment
        # atom per joint (idx 2..) forming the "posture" group. The layered scorer
        # inner-power-means the posture group, then outer-power-means the three groups.
        njoint = int(self.joint_max_dev.shape[0])
        c = 2 + njoint  # grouped index of the control atom; stillness/wobble follow it
        self.fpl_group_names = ["orientation", "height", "posture", "control", "stillness", "wobble"]
        self.fpl_groups = [[0], [1], list(range(2, 2 + njoint)), [c], [c + 1], [c + 2]]

        # Resolve the per-atom offset deltas. `fpl_offsets` (name -> delta) overrides the
        # DEFAULT_FPL_OFFSETS tiers; unknown names raise so typos don't silently no-op.
        offsets = dict(self.DEFAULT_FPL_OFFSETS)
        if fpl_offsets:
            bad = [k for k in fpl_offsets if k not in self.cost_term_names_f]
            if bad:
                raise KeyError(f"unknown fpl_offsets atom(s) {bad}; "
                               f"expected a subset of {self.cost_term_names_f}")
            offsets.update(fpl_offsets)
        # Flat offset vector aligned with cost_term_names_f (applied in running_cost_terms_f).
        self.fpl_offsets = np.array([offsets[n] for n in self.cost_term_names_f], dtype=np.float64)
        # Grouped offset vector: orientation, height, then the nominal delta broadcast over
        # every per-joint posture atom, then control — aligned with running_cost_terms_f_grouped.
        self.fpl_offsets_grouped = np.concatenate([
            [offsets["orientation_fulfillment"]],
            [offsets["height_fulfillment"]],
            np.full(njoint, offsets["nominal_fulfillment"]),
            [offsets["control_fulfillment"]],
            [offsets["stillness_fulfillment"]],
            [offsets["wobble_fulfillment"]],
        ]).astype(np.float64)

    @staticmethod
    def _apply_offset(f: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """FPL priority offset, paper Eq. 7:  u([phi]_d) = (u(phi) + max(d,0)) / (1 + d).
        For d >= 0 this maps [0,1] -> [d/(1+d), 1] (raises the floor, keeps the top at 1).
        `delta` broadcasts over the last (atom) axis. Clipped to [0,1] to preserve the FPL
        invariant for any (e.g. negative) delta the caller might pass."""
        return np.clip((f + np.maximum(delta, 0.0)) / (1.0 + delta), 0.0, 1.0)

    def _torso_orientation(self, sensordata: np.ndarray) -> np.ndarray:
        """Return the rotated upright vector (0,0,1) by the torso quaternion."""
        quat = sensordata[..., self._quat_adr : self._quat_adr + 4]
        return _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))

    def _torso_height(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._torso_pos_adr + 2]

    # ---- control effort (shared by normal + FPL) ----

    def _control_effort(self, u: np.ndarray) -> np.ndarray:
        """Mean squared control deviation from the nominal rest control u_ref, normalized
        per-actuator by its half control-range. ~0 when commanding the standing pose,
        growing toward ~1 as the targets swing across their full stroke. (*lead,)."""
        u_norm = (u - self.u_ref) / self.u_half_range
        return np.mean(u_norm ** 2, axis=-1)

    def _control_fulfillment(self, u: np.ndarray) -> np.ndarray:
        # Smoothly MAP effort in [0, inf) into (0, 1] via exp(-effort): exactly 1 at the
        # rest control, decaying monotonically toward 0 as effort grows. No clipping —
        # the map is bijective over the whole effort range, so every control value gets a
        # meaningful gradient (unlike 1-effort, which saturates flat once it hits 0).
        return np.exp(-self._control_effort(u))

    # ---- movement / velocity effort (Tier A: stillness + anti-wobble) ----

    def _torso_linvel(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._linvel_adr : self._linvel_adr + 3]

    def _torso_angvel(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._angvel_adr : self._angvel_adr + 3]

    def _stillness_effort(self, sensordata: np.ndarray) -> np.ndarray:
        """Squared torso linear speed, normalized by lin_vel_ref -> ~1 at the reference
        speed. 0 when the torso is stationary; penalizes drifting/lurching. (*lead,)."""
        v = self._torso_linvel(sensordata)
        return np.sum(v ** 2, axis=-1) / (self.lin_vel_ref ** 2)

    def _wobble_effort(self, sensordata: np.ndarray) -> np.ndarray:
        """Squared torso angular speed, normalized by ang_vel_ref. 0 when the torso holds
        still; penalizes spinning / tip-oscillation. (*lead,)."""
        w = self._torso_angvel(sensordata)
        return np.sum(w ** 2, axis=-1) / (self.ang_vel_ref ** 2)

    def _stillness_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # exp(-effort): 1 when stationary, smoothly -> 0 as the torso speeds up. In (0,1].
        return np.exp(-self._stillness_effort(sensordata))

    def _wobble_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        return np.exp(-self._wobble_effort(sensordata))

    # ---- normal cost ----

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        # NOTE: the original `10 * sum(orient**2)` was a no-op — `orient` is a rotated
        # UNIT vector so its squared norm is identically 1, making orientation a constant
        # cost that never influenced vanilla MPPI. Use the tilt (1 - rz) instead, so the
        # normal cost actually penalizes leaning/toppling and mirrors the FPL orientation
        # atom (both are monotonic in rz = torso up-vector's z; 1=upright, -1=inverted).
        rz = self._torso_orientation(sensordata)[..., 2]
        orientation_cost = 10.0 * (1.0 - rz)                      # 0 upright -> 20 inverted
        height_cost = 10.0 * (self._torso_height(sensordata) - self.target_height) ** 2
        nominal_cost = 0.1 * np.sum((qpos[..., 7:] - self.qstand[7:]) ** 2, axis=-1)
        control_cost = self.ctrl_cost_weight * self._control_effort(u)
        stillness_cost = self.stillness_weight * self._stillness_effort(sensordata)
        wobble_cost = self.wobble_weight * self._wobble_effort(sensordata)
        return np.stack([orientation_cost, height_cost, nominal_cost, control_cost,
                         stillness_cost, wobble_cost], axis=-1)

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms(qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu]))

    # ---- FPL cost ----

    def _height_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # mis formula: |h - target| clipped to delta=0.9 -> mapped to [0,1].
        # NOTE (2026-07-20): tightening this band so a crouch scores ~0 (instead of
        # ~0.5) gives the atom more signal, but makes the weakest-link dump all
        # pressure on height and TOPPLE (sacrifice orientation). No single-atom shape
        # cleanly wins on this cooperating-objective balance task; see inspect_costs.py.
        h = self._torso_height(sensordata)
        delta = 0.9
        err = -np.abs(h - self.target_height)
        err = np.clip(err, -delta, 0.0)
        return err / delta + 1.0

    def _orientation_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # rz = torso up-vector's z = cos(tilt): 1 upright, 0 horizontal, -1 inverted.
        # Decay to 0 BEFORE horizontal (by ~cos 60°) so uprightness is a real floor. The
        # old (rz+1)/2 gave 0.5 at horizontal — too generous, so FPL never strongly
        # defended upright and let the torso topple on disturbances.
        rz = self._torso_orientation(sensordata)[..., 2]
        rz_full, rz_floor = 0.95, 0.5
        return np.clip((rz - rz_floor) / (rz_full - rz_floor), 0.0, 1.0)

    def _joint_fulfillments(self, qpos: np.ndarray) -> np.ndarray:
        # Per-joint fulfillment in [0,1]: 1 at the standing reference, decaying toward 0
        # at the joint's deviation budget. (*lead, njoint) — the atoms the layered
        # composition inner-power-means into a single "posture" score.
        diff = np.abs(qpos[..., 7:] - self.qstand[7:])
        return np.clip(1.0 - diff / self.joint_max_dev, 1e-8, 1.0)

    def _nominal_fulfillment(self, qpos: np.ndarray) -> np.ndarray:
        frac = self._joint_fulfillments(qpos)
        # geometric mean cubed (matches mis humanoid_standup formula)
        log_mean = np.mean(np.log(frac), axis=-1)
        geom_mean = np.exp(log_mean)
        return geom_mean

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        atoms = np.stack(
            [
                self._orientation_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._nominal_fulfillment(qpos),
                self._control_fulfillment(u),
                self._stillness_fulfillment(sensordata),
                self._wobble_fulfillment(sensordata),
            ],
            axis=-1,
        )
        # Apply the FPL priority offsets before composition (the normalized discount-sum is
        # a convex combination, so a per-step affine offset == offsetting the FQ-value).
        return self._apply_offset(atoms, self.fpl_offsets)

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms_f(qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu]))

    # ---- layered / grouped FPL ----

    def running_cost_terms_f_grouped(self, qpos, qvel, sensordata, u) -> np.ndarray:
        # Orientation and height stay single atoms; the nominal posture term is EXPANDED
        # into one fulfillment atom per joint. `fpl_groups` (set in __init__) tells the
        # layered scorer to inner-power-mean the joint atoms into a single "posture"
        # score, then outer-power-mean {orientation, height, posture}.
        orient = self._orientation_fulfillment(sensordata)[..., None]
        height = self._height_fulfillment(sensordata)[..., None]
        joints = self._joint_fulfillments(qpos)
        control = self._control_fulfillment(u)[..., None]
        stillness = self._stillness_fulfillment(sensordata)[..., None]
        wobble = self._wobble_fulfillment(sensordata)[..., None]
        return np.concatenate([orient, height, joints, control, stillness, wobble], axis=-1)

    def terminal_cost_terms_f_grouped(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.running_cost_terms_f_grouped(
            qpos, qvel, sensordata, np.zeros_like(qpos[..., :self.nu])
        )
