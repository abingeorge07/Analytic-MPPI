"""In-hand cube reorientation task (LEAP right hand) — the dexterous-manipulation
member of the FPL-vs-linear study.

The hand is mounted palm-up and cradles a 7 cm cube on its fingers. The task is to
*reorient* the cube by a commanded yaw angle about the vertical axis (spin-in-hand)
while keeping it balanced on the fingers. This is the manipulation analogue of the
legged dynamic-competition tasks:

  * PROGRESS  (like forward velocity): rotate the cube toward the goal orientation.
  * SAFETY    (like uprightness / height): keep the cube in the hand — do not let it
    slide off the palm or fall.

The competition is dynamic: the only way to spin the cube faster is to push it harder
with the fingertips, which risks losing the cradle and dropping it. No fixed linear
weight is both fast (rotates far) and safe (never drops): a heavy orientation weight
flings the cube out of the hand, a light one never turns it. FPL's min-fulfillment
conjunction rotates as far as it can *without* letting the "hold" fulfillment collapse.

Model: analytic_mppi/envs/cube/scene.xml (LEAP hand + free-joint cube). Position
actuators (ctrl = target joint angle), so control effort is deviation from the cradle
control U_GRASP (as in the quadruped / g1), not from zero. The cube settles onto the
fingers deterministically under U_GRASP; that settled orientation q0 is the zero of the
reorientation, and the goal is q0 rotated by `target_angle` about world-z, so the
initial orientation error is exactly `target_angle` regardless of the settle tilt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from .base import Task


_MODEL_PATH = Path(__file__).resolve().parent.parent / "envs" / "cube" / "scene.xml"

# Cradle control: a cupping posture (fingers flexed, thumb opposed) that catches the
# cube and holds it balanced on the fingertips. Found by settling the free cube from
# just above the palm over a small grid of finger-flex / thumb-opposition targets and
# picking the pose with the smallest residual cube drift. Layout (16 position acts):
#   index [mcp,rot,pip,dip]  middle [mcp,rot,pip,dip]  ring [mcp,rot,pip,dip]
#   thumb [cmc,axl,mcp,ipl]
U_GRASP = np.array([0.5, 0.0, 1.0, 0.8,
                    0.5, 0.0, 1.0, 0.8,
                    0.5, 0.0, 1.0, 0.8,
                    1.0, 0.0, 0.8, 0.6], dtype=np.float64)

# Deterministic settle used both to compute the rest orientation q0 (task init) and to
# start every episode (init_cube in eval.py). Cube dropped from just above the grasp
# site, fingers driven to U_GRASP, physics run for this many steps.
_SETTLE_STEPS = 120
_CUBE_START = np.array([0.11, 0.0, 0.06], dtype=np.float64)


def _quat_sub(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Hamiltonian quaternion 'subtraction' returning the 3-vec rotation error
    (imaginary part of q * conj(q_ref); ‖·‖ ≈ sin(angle/2)). q:(...,4) wxyz, q_ref:(4,).
    Double-cover corrected so the error is the shortest rotation."""
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]  # conj of ref
    vx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    vy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    vz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    real = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    sign = np.where(real < 0, -1.0, 1.0)
    return np.stack([vx, vy, vz], axis=-1) * sign[..., None]


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b (wxyz)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


_SETTLE_CACHE: "tuple | None" = None


def settle_grasp_state(model: "mujoco.MjModel"):
    """Deterministically settle the cube into the cradle; return (fullphysics state,
    settled cube quat q0). Shared by the task (to fix q0) and init_cube (to start each
    episode from the identical grasped pose).

    The settle is deterministic and identical for every cube scene (same XML), so the
    result is MEMOIZED: it is recomputed only once per process instead of on every task
    construction and episode reset (each of which would otherwise run 120 serial physics
    steps on the contact-heavy hand). The cached grasped pose is used as the start for
    every episode — including the perturbed-friction robustness runs, so every condition
    begins from the identical grasp (the perturbation only changes what happens after)."""
    global _SETTLE_CACHE
    if _SETTLE_CACHE is not None and _SETTLE_CACHE[2] == (model.nq, model.nu):
        return _SETTLE_CACHE[0].copy(), _SETTLE_CACHE[1].copy()
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    d.qpos[:16] = U_GRASP
    d.qpos[16:19] = _CUBE_START
    d.qpos[19:23] = np.array([1.0, 0.0, 0.0, 0.0])
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    for _ in range(_SETTLE_STEPS):
        d.ctrl[:] = U_GRASP
        mujoco.mj_step(model, d)
    state = np.empty(mujoco.mj_stateSize(model, int(mujoco.mjtState.mjSTATE_FULLPHYSICS)))
    mujoco.mj_getState(model, d, state, int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
    q0 = d.qpos[19:23].copy()
    _SETTLE_CACHE = (state.copy(), q0.copy(), (model.nq, model.nu))
    return state, q0


class CubeRotationTask(Task):
    """Reorient a cube in the LEAP hand by `target_angle` about the vertical axis,
    without dropping it. Atom order: [hold, height, alignment, control] — the
    ALIGNMENT atom (rotation progress) is the one the linear-weight family sweeps,
    mirroring the velocity atom in the locomotion tasks."""

    cost_term_names = ["hold_cost", "height_cost", "alignment_cost", "control_cost"]
    cost_term_names_f = ["hold_fulfillment", "height_fulfillment",
                         "alignment_fulfillment", "control_fulfillment"]

    def __init__(self, *, target_angle: float = 1.2, target_axis: str = "x",
                 hold_xy_full: float = 0.02, hold_xy_floor: float = 0.06,
                 height_full: float = 0.02, height_floor: float = -0.04,
                 ctrl_cost_weight: float = 1.0):
        super().__init__(_MODEL_PATH)
        m = self.mj_model
        self._pos_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "cube_position")])
        self._ori_adr = int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "cube_orientation")])
        self.target_angle = float(target_angle)
        self.target_axis = str(target_axis)
        self.hold_xy_full = float(hold_xy_full)
        self.hold_xy_floor = float(hold_xy_floor)
        self.height_full = float(height_full)
        self.height_floor = float(height_floor)
        self.ctrl_cost_weight = float(ctrl_cost_weight)
        # Rest orientation the cube settles to under the cradle control; the goal is this
        # yawed by target_angle about world-z, so the initial error is exactly target_angle.
        _, self.q0 = settle_grasp_state(m)
        # Goal = the settled orientation rotated by target_angle about `target_axis` (world
        # frame). Default axis 'x' is a forward ROLL (tumbling the cube over the fingertips):
        # the drop-prone motion that creates the dynamic competition (yaw about 'z' is too
        # securely cradled to ever risk a drop — the manipulation analogue of gentle walking).
        ax = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[self.target_axis]
        s = np.sin(0.5 * self.target_angle)
        rot = np.array([np.cos(0.5 * self.target_angle), ax[0] * s, ax[1] * s, ax[2] * s],
                       dtype=np.float64)
        self.goal_quat = _quat_mul(rot, self.q0)
        # Warm-start the plan at the cradle control (position actuators), like the
        # quadruped standing keyframe: the first commands hold the grasp, not a collapse.
        self.nominal_control = U_GRASP
        self.u_ref = U_GRASP
        self.u_half_range = 0.5 * (self.u_max - self.u_min)

    # ---- cube state decode ----

    def _cube_pos_err(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._pos_adr:self._pos_adr + 3]

    def _cube_quat(self, sensordata: np.ndarray) -> np.ndarray:
        return sensordata[..., self._ori_adr:self._ori_adr + 4]

    def _hold_xy(self, sensordata: np.ndarray) -> np.ndarray:
        # Distance of the cube from the palm centre in the palm plane (xy). ~0 cradled,
        # grows as the cube slides toward the edge and off (a drop).
        pe = self._cube_pos_err(sensordata)
        return np.sqrt(np.sum(pe[..., 0:2] ** 2, axis=-1))

    def _cube_height(self, sensordata: np.ndarray) -> np.ndarray:
        # Cube height relative to the grasp site (z of the position sensor). Falls sharply
        # once the cube leaves the fingers.
        return self._cube_pos_err(sensordata)[..., 2]

    def _angle_err(self, sensordata: np.ndarray) -> np.ndarray:
        # Geodesic angle (rad) between the cube and the goal orientation.
        v = _quat_sub(self._cube_quat(sensordata), self.goal_quat)
        return 2.0 * np.arcsin(np.clip(np.linalg.norm(v, axis=-1), 0.0, 1.0))

    def _control_effort(self, u: np.ndarray) -> np.ndarray:
        u_norm = (u - self.u_ref) / self.u_half_range
        return np.mean(u_norm ** 2, axis=-1)

    # ---- normal (quadratic) cost ----

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        xy = self._hold_xy(sensordata)
        h = self._cube_height(sensordata)
        ang = self._angle_err(sensordata)
        hold_cost = 10.0 * xy ** 2 + 100.0 * np.maximum(xy - self.hold_xy_floor, 0.0) ** 2
        height_cost = 50.0 * np.clip(self.height_full - h, 0.0, None) ** 2
        alignment_cost = 5.0 * ang ** 2
        zero = np.zeros_like(hold_cost)
        return np.stack([hold_cost, height_cost, alignment_cost, zero], axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        out = self.terminal_cost_terms(qpos, qvel, sensordata).copy()
        out[..., -1] = self.ctrl_cost_weight * self._control_effort(u)
        return out

    # ---- FPL fulfillment atoms (all in [0,1]) ----

    def _hold_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # SAFETY floor: 1 while the cube is centred on the palm, decaying to 0 as it
        # slides out to hold_xy_floor (a drop). Bites before the cube leaves the fingers.
        xy = self._hold_xy(sensordata)
        return np.clip((self.hold_xy_floor - xy) / (self.hold_xy_floor - self.hold_xy_full),
                       0.0, 1.0)

    def _height_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # SAFETY floor: 1 at/above the cradle height, decaying to 0 at height_floor (the
        # cube has dropped below the fingers). One-sided — no penalty for sitting higher.
        h = self._cube_height(sensordata)
        return np.clip((h - self.height_floor) / (self.height_full - self.height_floor),
                       0.0, 1.0)

    def _alignment_fulfillment(self, sensordata: np.ndarray) -> np.ndarray:
        # PROGRESS: fraction of the commanded rotation achieved, 0 at the start (err =
        # target_angle) ramping to 1 at the goal (err = 0). One-sided linear ramp, the
        # manipulation analogue of the velocity atom — gives the conjunction its spread
        # without rewarding over-rotation past the goal.
        ang = self._angle_err(sensordata)
        return np.clip((self.target_angle - ang) / self.target_angle, 0.0, 1.0)

    def _control_fulfillment(self, u: np.ndarray) -> np.ndarray:
        return np.exp(-self._control_effort(u))

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._hold_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._alignment_fulfillment(sensordata),
                self._control_fulfillment(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        # control fulfillment undefined at the terminal step -> 3 state atoms (n_term < n_run).
        return np.stack(
            [
                self._hold_fulfillment(sensordata),
                self._height_fulfillment(sensordata),
                self._alignment_fulfillment(sensordata),
            ],
            axis=-1,
        )
