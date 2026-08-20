"""Quadruped (Barkour) forward-locomotion task — CPU sampling-MPC.

Mirrors the hopper/walker locomotion tasks (forward-velocity tracking + stay-upright + hold
height + control regularization) but on a 12-DoF position-actuated quadruped, so the
FPL-vs-linear comparison and the traction-robustness study extend to a whole-body legged
robot (the regime of Alvarez-Padilla et al., "Real-Time Whole-Body Control of Legged Robots
with MPPI", arXiv:2409.10469 — the same MuJoCo-parallel sampling MPC this framework implements).

Model: analytic_mppi/envs/barkour/barkour.xml (DeepMind Barkour v0, flat-ground scene with
torso_pos / torso_linvel / torso_quat sensors added). The chassis body frame is rotated:
body-X is UP and body-Z is FORWARD (world +x), so uprightness rotates the body up-axis
(1,0,0) by the torso quat and reads its world-z; forward speed is torso_linvel world-x.
Position actuators (ctrl = target joint angle), so control effort is deviation from the
standing keyframe control u_ref (as in g1_standup), not from zero.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from .base import Task


_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "barkour" / "barkour.xml"


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a constant vector v by a batch of quaternions q (wxyz). q:(...,4) v:(3,)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    ax = y * vz - z * vy + w * vx
    ay = z * vx - x * vz + w * vy
    az = x * vy - y * vx + w * vz
    rx = vx + 2.0 * (y * az - z * ay)
    ry = vy + 2.0 * (z * ax - x * az)
    rz = vz + 2.0 * (x * ay - y * ax)
    return np.stack([rx, ry, rz], axis=-1)


class QuadrupedTask(Task):
    """Barkour quadruped tasked with walking forward at `target_velocity`."""

    cost_term_names = ["height_cost", "orientation_cost", "velocity_cost", "heading_cost",
                       "posture_cost", "control_cost"]
    cost_term_names_f = ["height_fulfillment", "orientation_fulfillment",
                         "velocity_fulfillment", "heading_fulfillment", "posture_fulfillment",
                         "control_fulfillment"]

    def __init__(self, *, target_velocity: float = 0.6, target_height: float = 0.22,
                 ctrl_cost_weight: float = 1.0, heading_weight: float = 5.0,
                 lat_vel_ref: float = 0.3, lat_pos_ref: float = 0.3, heading_yaw_ref: float = 0.25,
                 posture_ref: float = 0.7, posture_weight: float = 10.0,
                 model_path=None):
        super().__init__(model_path or _MODEL_PATH)
        self.target_velocity = float(target_velocity)
        self.target_height = float(target_height)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        # Straight-line objective: penalize LATERAL velocity (world y) and lateral position
        # drift off the start line (y=0), so forward progress is along +x, not a crab/veer.
        self.heading_weight = float(heading_weight)
        self.lat_vel_ref = float(lat_vel_ref)
        self.lat_pos_ref = float(lat_pos_ref)
        self.heading_yaw_ref = float(heading_yaw_ref)
        # STABILITY term: keep the four hip-ABDUCTION joints near their standing angle so the
        # legs stay tucked under the body instead of splaying out into a low belly-sprawl (the
        # dominant "ugly gait" failure of a pure velocity+upright objective — the body can score
        # on forward speed and instantaneous uprightness while the legs paddle out sideways).
        # Abduction-only (not hips/knees) so the swing joints are free to cycle for the gait.
        self.posture_ref = float(posture_ref)
        self.posture_weight = float(posture_weight)
        m = self.mj_model
        self._pos_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_pos")])
        self._vel_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_linvel")])
        self._quat_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "torso_quat")])
        # Standing reference pose + control (position actuators hold u_ref -> stand).
        kf = m.keyframe("standing")
        self.qstand = np.asarray(kf.qpos, dtype=np.float64).copy()
        self.u_ref = np.asarray(kf.ctrl, dtype=np.float64).copy()
        self.u_half_range = 0.5 * (self.u_max - self.u_min)
        # qpos indices of the four hip-abduction joints and their standing angles.
        self._abd_qadr = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)])
                          for nm in ("abduction_front_right", "abduction_front_left",
                                     "abduction_hind_right", "abduction_hind_left")]
        self.abd_stand = self.qstand[self._abd_qadr].copy()
        # Warm-start the sampling plan at the standing control (position actuators): read by
        # SamplingController so the first commands hold the stand instead of a collapsed pose.
        self.nominal_control = self.u_ref

    # ---- torso state decode ----

    def _torso_height(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._pos_adr + 2]

    def _torso_vel_x(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._vel_adr]                       # world +x = forward

    def _torso_vel_y(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._vel_adr + 1]                   # world +y = lateral

    def _torso_pos_y(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._pos_adr + 1]                   # lateral drift (0 = line)

    def _torso_forward(self, sensordata: np.ndarray) -> np.ndarray:
        # World-frame FACING direction: the body's forward axis (body-Z for this model) rotated
        # by the torso quat. ~(1,0,0) when the robot faces straight along +x.
        quat = sensordata[..., self._quat_adr:self._quat_adr + 4]
        return _quat_rotate(quat, np.array([0.0, 0.0, 1.0]))

    def _heading_effort(self, sensordata: np.ndarray) -> np.ndarray:
        # Straight-line heading: sideways velocity + lateral drift off the y=0 start line +
        # BODY YAW. Each normalized; 0 when travelling straight along +x AND facing +x. The yaw
        # term (1 - forward_x, i.e. how far the facing direction has turned off +x) is essential:
        # without it the robot keeps its COM on the line but spins its body around (a 172° yaw
        # swing) — it crab/pirouette-walks forward. With it the robot faces where it walks.
        vy = self._torso_vel_y(sensordata)
        y = self._torso_pos_y(sensordata)
        yaw_dev = 1.0 - self._torso_forward(sensordata)[..., 0]     # 0 facing +x, ->2 facing -x
        return ((vy / self.lat_vel_ref) ** 2 + (y / self.lat_pos_ref) ** 2
                + (yaw_dev / self.heading_yaw_ref) ** 2)

    def _torso_up(self, sensordata: np.ndarray) -> np.ndarray:
        # Uprightness = world-z component of the torso's UP axis (body-x for this model),
        # i.e. cos(tilt): 1 upright, 0 on its side. Rotate body-up (1,0,0) by the torso quat.
        quat = sensordata[..., self._quat_adr:self._quat_adr + 4]
        return _quat_rotate(quat, np.array([1.0, 0.0, 0.0]))[..., 2]

    def _control_effort(self, u: np.ndarray) -> np.ndarray:
        # Mean squared deviation from the standing control, normalized per-actuator by the
        # half control-range (dimensionless). ~0 when holding the stand pose.
        u_norm = (u - self.u_ref) / self.u_half_range
        return np.mean(u_norm ** 2, axis=-1)

    def _abduction_effort(self, qpos: np.ndarray) -> np.ndarray:
        # Mean squared deviation of the four hip-abduction joints from their standing angle,
        # normalized by posture_ref (rad). ~0 with the legs tucked under the body; grows as the
        # stance splays out sideways.
        abd = qpos[..., self._abd_qadr]
        return np.mean(((abd - self.abd_stand) / self.posture_ref) ** 2, axis=-1)

    # ---- normal (quadratic) cost ----

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        h = self._torso_height(sensordata)
        up = self._torso_up(sensordata)
        vx = self._torso_vel_x(sensordata)
        height_cost = 10.0 * (h - self.target_height) ** 2
        orient_cost = 10.0 * (1.0 - up)
        # one-sided velocity shortfall (no penalty for exceeding target — matches FPL atom)
        velocity_cost = 5.0 * np.clip(self.target_velocity - vx, 0.0, None) ** 2
        heading_cost = self.heading_weight * self._heading_effort(sensordata)
        posture_cost = self.posture_weight * self._abduction_effort(qpos)
        zero = np.zeros_like(height_cost)
        return np.stack([height_cost, orient_cost, velocity_cost, heading_cost, posture_cost,
                         zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        out = self.terminal_cost_terms(qpos, qvel, sensordata)
        out = out.copy()
        out[..., -1] = self.ctrl_cost_weight * self._control_effort(u)
        return out

    # ---- FPL fulfillment atoms (all in [0,1]); shapes mirror the hopper/walker ----

    def _height_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # One-sided: full credit at/above target, decaying to 0 at h_floor (well below the
        # standing height but above a collapsed torso), so it bites before the torso hits the
        # ground. No penalty for being taller.
        h = self._torso_height(sensordata)
        h_full, h_floor = self.target_height, 0.10
        return np.clip((h - h_floor) / (h_full - h_floor), 0.0, 1.0)

    def _orientation_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # Uprightness decaying to 0 BEFORE horizontal (a real floor).
        up = self._torso_up(sensordata)
        u_full, u_floor = 0.9, 0.5
        return np.clip((up - u_floor) / (u_full - u_floor), 0.0, 1.0)

    def _velocity_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # One-sided linear ramp: proportional credit vx/target, capped at 1, no credit
        # for backward. Gives the conjunction spread without rewarding sprinting into a fall.
        vx = self._torso_vel_x(sensordata)
        return np.clip(vx / self.target_velocity, 0.0, 1.0)

    def _heading_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # exp(-effort): 1 travelling straight along +x on the line, -> 0 as it crabs/veers.
        # In the FPL conjunction this becomes a floor that keeps the gait on a straight path.
        return np.exp(-self._heading_effort(sensordata))

    def _posture_fulfillment(self, qpos: np.ndarray) -> np.ndarray:
        # STABILITY atom: exp(-abduction effort). 1 with the legs tucked under the body, decaying
        # as the stance splays. In the FPL conjunction this is a floor that forbids trading a
        # splayed belly-sprawl for forward speed — it makes the min-fulfillment gait an actual
        # legged walk. Shared identically by the linear family (fair protocol).
        return np.exp(-self._abduction_effort(qpos))

    def _control_fulfillment(self, u: np.ndarray) -> np.ndarray:
        # exp(-effort): 1 at the stand control, smoothly -> 0 as the targets swing.
        return np.exp(-self._control_effort(u))

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
                self._heading_fulfillment(sensordata),
                self._posture_fulfillment(qpos),
                self._control_fulfillment(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        # control fulfillment undefined at the terminal step -> 5 state atoms (n_term < n_run).
        return np.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
                self._heading_fulfillment(sensordata),
                self._posture_fulfillment(qpos),
            ],
            axis=-1,
        )


class QuadrupedComTask(QuadrupedTask):
    """Barkour walk + a CoM-straightness (vertical anti-bob) objective — the +CoM arm of the
    interaction-effect 2x2.

    Adds ONE atom to QuadrupedTask: the CoM should glide along a straight, constant-height
    path, so its VERTICAL velocity |vz| stays small. This is the piece of "keep the CoM
    straight" the base task does NOT already cover — the heading atom handles LATERAL drift +
    yaw, and the height atom is one-sided so vertical BOB is currently free. Shaped as a FLOOR
    (~1 through the gentle vertical motion any legged gait needs; natural gait median |vz|~0.19,
    so it clears that), dropping only for EXCESSIVE bounce -> it competes with forward SPEED
    (fast gaits pogo), the speed-vs-smoothness tradeoff where FPL's conjunction should beat a
    linear weight. The atom is inserted BEFORE control so the terminal atoms stay a prefix of
    the running atoms (control is the only running-only atom; see sampling_base._score_fpl)."""

    cost_term_names = ["height_cost", "orientation_cost", "velocity_cost", "heading_cost",
                       "posture_cost", "com_bob_cost", "control_cost"]
    cost_term_names_f = ["height_fulfillment", "orientation_fulfillment", "velocity_fulfillment",
                         "heading_fulfillment", "posture_fulfillment", "com_bob_fulfillment",
                         "control_fulfillment"]

    def __init__(self, *, vz_floor: float = 0.2, vz_sat: float = 0.7,
                 com_bob_weight: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.vz_floor = float(vz_floor)
        self.vz_sat = float(vz_sat)
        self.com_bob_weight = float(com_bob_weight)

    def _torso_vel_z(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._vel_adr + 2]                   # world +z = vertical

    def _com_bob_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # FLOOR on CoM vertical speed: ~1 through the gentle vertical motion a legged gait
        # needs (|vz| <= vz_floor), decaying to 0 as the CoM bounces harder (vz_sat ~ the
        # gait's high-bob tail). Only EXCESSIVE bounce is penalised, so it does not fight the
        # natural gait but competes with speed.
        vz = np.abs(self._torso_vel_z(sensordata))
        return np.clip((self.vz_sat - vz) / (self.vz_sat - self.vz_floor), 0.0, 1.0)

    # ---- normal (quadratic) cost: bob inserted before the control placeholder ----

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        base = super().terminal_cost_terms(qpos, qvel, sensordata)   # (...,6): [..post, zero]
        bob = (self.com_bob_weight * self._torso_vel_z(sensordata) ** 2)[..., None]
        return np.concatenate([base[..., :-1], bob, base[..., -1:]], axis=-1)   # (...,7)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        out = self.terminal_cost_terms(qpos, qvel, sensordata).copy()
        out[..., -1] = self.ctrl_cost_weight * self._control_effort(u)
        return out

    # ---- FPL atoms: bob before control (terminal stays a prefix of running) ----

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
                self._heading_fulfillment(sensordata),
                self._posture_fulfillment(qpos),
                self._com_bob_fulfillment(sensordata),
                self._control_fulfillment(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        return np.stack(
            [
                self._height_fulfillment(sensordata),
                self._orientation_fulfillment(sensordata),
                self._velocity_fulfillment(sensordata),
                self._heading_fulfillment(sensordata),
                self._posture_fulfillment(qpos),
                self._com_bob_fulfillment(sensordata),
            ],
            axis=-1,
        )


_JUMP_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "barkour" / "barkour_jump.xml"


class QuadrupedJumpTask(QuadrupedTask):
    """Hurdle jump — a genuinely COMPETING CoM objective (no safe conservative weight).

    The robot must get past an obstacle at x_obs by driving its CoM up-and-over, then land
    upright. This competes hard both ways: clear the hurdle (jump = CoM high+forward, risks
    toppling/crashing on landing) vs stay stable (don't jump = stuck at the wall, no progress).
    No single fixed weight is both — too timid never clears, too aggressive face-plants — which
    is exactly FPL's regime, unlike the (stabilizing) anti-bob atom.

    ±the CoM CLEARANCE atom is the CoM axis of the interaction 2x2. It is a spatially-gated
    jump template: "be HIGH while over the obstacle" — 1 everywhere except in the gate region
    around x_obs, where it demands CoM height >= h_clear. With `com_guidance=False` the task
    keeps only progress + stay-upright + posture + control, and the controller must discover
    the jump from forward progress against the blocking wall alone.
    """

    def __init__(self, *, x_obs: float = 0.55, h_obs: float = 0.12, x_goal: float = 1.0,
                 com_guidance: bool = True, h_clear: float = 0.38, h_lo: float = 0.24,
                 gate_half: float = 0.18, **kwargs):
        super().__init__(model_path=_JUMP_MODEL_PATH, **kwargs)
        self.x_obs = float(x_obs)
        self.h_obs = float(h_obs)
        self.x_goal = float(x_goal)
        self.com_guidance = bool(com_guidance)
        self.h_clear = float(h_clear)
        self.h_lo = float(h_lo)
        self.gate_half = float(gate_half)
        # atom order keeps every terminal-defined (state) atom BEFORE the running-only control
        # atom, so terminal stays a prefix of running (sampling_base._score_fpl). clearance is
        # inserted just before control when com_guidance is on.
        base = ["progress", "orientation", "posture"]
        clear = ["com_clearance"] if self.com_guidance else []
        self.cost_term_names = [f"{n}_cost" for n in base + clear] + ["control_cost"]
        self.cost_term_names_f = [f"{n}_fulfillment" for n in base + clear] + ["control_fulfillment"]

    # ---- state decode ----
    def _torso_pos_x(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._pos_adr]                      # world +x forward position

    # ---- atoms (all [0,1]) ----
    def _progress_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # forward position toward the goal past the obstacle; one-sided (no reward past goal).
        return np.clip(self._torso_pos_x(sensordata) / self.x_goal, 0.0, 1.0)

    def _com_clearance_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # spatially-gated jump template: 1 unless the CoM is OVER the obstacle and too LOW.
        # gate ~1 near x_obs (exp bell), height_ok ramps 0->1 as CoM rises h_lo->h_clear.
        x = self._torso_pos_x(sensordata)
        h = self._torso_height(sensordata)
        gate = np.exp(-((x - self.x_obs) / self.gate_half) ** 2)
        height_ok = np.clip((h - self.h_lo) / (self.h_clear - self.h_lo), 0.0, 1.0)
        return np.clip(1.0 - gate * (1.0 - height_ok), 0.0, 1.0)

    def _fulfillment_stack(self, qpos, sensordata, u, terminal):
        cols = [
            self._progress_fulfillment(sensordata),
            self._orientation_fulfillment(sensordata),
            self._posture_fulfillment(qpos),
        ]
        if self.com_guidance:
            cols.append(self._com_clearance_fulfillment(sensordata))
        if not terminal:
            cols.append(self._control_fulfillment(u))
        return np.stack(cols, axis=-1)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return self._fulfillment_stack(qpos, sensordata, u, terminal=False)

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        return self._fulfillment_stack(qpos, sensordata, None, terminal=True)

    # ---- normal cost = 1 - fulfillment (linear family uses fpl_cost p=1; this is for tests) ----
    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return 1.0 - self._fulfillment_stack(qpos, sensordata, u, terminal=False)

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        return 1.0 - self._fulfillment_stack(qpos, sensordata, None, terminal=True)
