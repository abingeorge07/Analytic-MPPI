"""Motion-capture tracking cost for the Unitree G1 humanoid (CPU port).

Port of the hydrax ``HumanoidMocap`` task to this project's NumPy + CPU-MuJoCo
cost-object convention. The G1 tracks a retargeted motion-capture *walking* clip
from the LocoMuJoCo dataset (vendored as an ``.npz`` in the env directory).

The original hydrax task runs on MJX and reads body poses (``xpos``/``xquat``/
``cvel``) straight off ``mjx.Data``. Here rollouts go through ``mujoco.rollout``,
which exports only ``[time, qpos, qvel, ...]`` states plus ``sensordata`` -- so:

  * configuration (``qpos``) and generalized-velocity (``qvel``) tracking read
    from the FULLPHYSICS state (``state[..., 1:1+nq]`` / ``[1+nq:1+nq+nv]``);
  * body-position / body-orientation tracking read the torso + foot poses from
    the sensors the model already exposes (``imu_in_torso_quat``,
    ``left_foot_position``, ``right_foot_position``, ``left_foot_orientation``,
    ``right_foot_orientation``) -- the same bodies the original "traced";
  * the body-twist term is dropped (``cvel`` is not exported; its default weight
    in the original is already 0.0).

Crucially, the cost is *time-varying*: it reads the sim time from
``state[..., 0]`` (the analogue of hydrax's ``state.time``) to index the
reference clip -- exactly what the vanilla-MPPI cost interface makes available
(the cost callable receives the full ``states`` array).

Each tracking term uses the original's ``1 - exp(-||err||^2)`` shaping so it sits
in ``[0, 1]``, and the weighted sum is normalized so the per-step cost is in
``[0, 1]`` too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np


# Reference bodies tracked via the model's existing sensors (torso + both feet),
# mirroring the original task's trace_sites = [imu_in_torso, left_foot, right_foot].
_TORSO_QUAT_SENSOR = "imu_in_torso_quat"
_FOOT_POS_SENSORS = ("left_foot_position", "right_foot_position")
_FOOT_QUAT_SENSORS = ("left_foot_orientation", "right_foot_orientation")

# Default vendored reference clip (see envs/humanoid_mocap/_fetch_reference.py).
_DEFAULT_NPZ = (
    Path(__file__).resolve().parent.parent
    / "envs" / "humanoid_mocap" / "walk1_subject1.npz"
)


# ---- batched quaternion helpers (wxyz) ----

def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two batches of wxyz quaternions. (..., 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate of a batch of wxyz quaternions. (..., 4)."""
    out = q.copy()
    out[..., 1:] = -out[..., 1:]
    return out


def _quat_sub(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Orientation error as a 3-vector (log map of the relative rotation).

    Analogue of mjx ``quat_sub``: the rotation vector (axis * angle) taking
    ``q_ref`` to ``q``. Batched over any leading shape; returns (..., 3).
    """
    d = _quat_mul(_quat_conj(q_ref), q)          # relative rotation, wxyz
    w = np.clip(d[..., 0], -1.0, 1.0)
    v = d[..., 1:4]
    vnorm = np.sqrt(np.sum(v * v, axis=-1))       # (...,)
    angle = 2.0 * np.arctan2(vnorm, w)            # [0, 2pi)
    angle = (angle + np.pi) % (2.0 * np.pi) - np.pi   # wrap to [-pi, pi]
    axis = v / (vnorm[..., None] + 1e-12)
    return axis * angle[..., None]


def _track_cost(sq_err: np.ndarray) -> np.ndarray:
    """Shaping used by the original task: 1 - exp(-||err||^2), in [0, 1]."""
    return 1.0 - np.exp(-sq_err)


@dataclass
class HumanoidMocapCost:
    """Reference-tracking cost for the G1 mocap walking task.

    Build with :meth:`from_model` (loads the vendored clip and precomputes the
    reference trajectory + sensor targets). ``__call__`` integrates the per-step
    cost over the horizon like the other cost objects in this package.
    """

    # Precomputed reference (float64), indexed by frame.
    reference_qpos: np.ndarray                 # (F, nq)
    reference_qvel: np.ndarray                 # (F, nv)
    reference_torso_quat: np.ndarray           # (F, 4)
    reference_foot_pos: np.ndarray             # (F, 6)  [lfoot_xyz, rfoot_xyz]
    reference_foot_quat: np.ndarray            # (F, 2, 4) [lfoot_wxyz, rfoot_wxyz]
    fps: float
    nq: int
    nv: int

    # Sensor addresses (into sensordata).
    torso_quat_adr: int
    foot_pos_adr: tuple[int, int]
    foot_quat_adr: tuple[int, int]

    # Cost weights (normalized in __post_init__ so the per-step cost is in [0,1]).
    configuration_weight: float = 0.1
    velocity_weight: float = 0.01
    body_position_weight: float = 1.0
    body_orientation_weight: float = 0.1
    terminal_weight: float = 1.0

    _F: int = field(init=False, repr=False)

    def __post_init__(self):
        self.reference_qpos = np.ascontiguousarray(self.reference_qpos, dtype=np.float64)
        self.reference_qvel = np.ascontiguousarray(self.reference_qvel, dtype=np.float64)
        self.reference_torso_quat = np.ascontiguousarray(self.reference_torso_quat, dtype=np.float64)
        self.reference_foot_pos = np.ascontiguousarray(self.reference_foot_pos, dtype=np.float64)
        self.reference_foot_quat = np.ascontiguousarray(self.reference_foot_quat, dtype=np.float64)
        self._F = int(self.reference_qpos.shape[0])

        total = (
            self.configuration_weight
            + self.velocity_weight
            + self.body_position_weight
            + self.body_orientation_weight
        )
        if total <= 0:
            raise ValueError("Sum of tracking weights must be positive")
        self.configuration_weight /= total
        self.velocity_weight /= total
        self.body_position_weight /= total
        self.body_orientation_weight /= total

    # ---- construction / precompute ----

    @classmethod
    def from_model(
        cls,
        model: mujoco.MjModel,
        *,
        npz_path: str | Path | None = None,
        **weights,
    ) -> "HumanoidMocapCost":
        """Load the vendored reference clip and precompute reference targets.

        Runs forward kinematics per frame (once) to read the reference torso /
        foot poses from the same sensors used at rollout time, and differentiates
        consecutive frames for the reference generalized velocity (MuJoCo's qvel
        convention, matching ``state.qvel``).
        """
        npz_path = Path(npz_path) if npz_path is not None else _DEFAULT_NPZ
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Reference clip not found: {npz_path}. Regenerate it with "
                "`python -m analytic_mppi.envs.humanoid_mocap._fetch_reference` "
                "(requires the optional `mocap` extra)."
            )
        # allow_pickle: the archive also stores object/string metadata arrays;
        # we only read the numeric `qpos` / `frequency` fields.
        with np.load(npz_path, allow_pickle=True) as z:
            reference = np.asarray(z["qpos"], dtype=np.float64)     # (F, nq)
            fps = float(np.asarray(z["frequency"]).item())

        nq, nv = int(model.nq), int(model.nv)
        if reference.shape[1] != nq:
            raise ValueError(
                f"Reference qpos width {reference.shape[1]} != model nq {nq}; "
                f"the clip {npz_path.name} does not match this model."
            )

        # Sensor addresses.
        def _adr(name: str) -> int:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise KeyError(f"model has no sensor named {name!r}")
            return int(model.sensor_adr[sid])

        torso_quat_adr = _adr(_TORSO_QUAT_SENSOR)
        foot_pos_adr = tuple(_adr(n) for n in _FOOT_POS_SENSORS)
        foot_quat_adr = tuple(_adr(n) for n in _FOOT_QUAT_SENSORS)

        # Precompute reference velocities (differentiate) and sensor targets (FK).
        F = reference.shape[0]
        data = mujoco.MjData(model)
        dt = 1.0 / fps
        ref_qvel = np.zeros((F, nv))
        ref_torso_quat = np.zeros((F, 4))
        ref_foot_pos = np.zeros((F, 6))
        ref_foot_quat = np.zeros((F, 2, 4))
        for i in range(F):
            data.qpos[:] = reference[i]
            mujoco.mj_forward(model, data)
            sd = data.sensordata
            ref_torso_quat[i] = sd[torso_quat_adr : torso_quat_adr + 4]
            ref_foot_pos[i, 0:3] = sd[foot_pos_adr[0] : foot_pos_adr[0] + 3]
            ref_foot_pos[i, 3:6] = sd[foot_pos_adr[1] : foot_pos_adr[1] + 3]
            ref_foot_quat[i, 0] = sd[foot_quat_adr[0] : foot_quat_adr[0] + 4]
            ref_foot_quat[i, 1] = sd[foot_quat_adr[1] : foot_quat_adr[1] + 4]
            if i < F - 1:
                mujoco.mj_differentiatePos(
                    model, ref_qvel[i], dt, reference[i], reference[i + 1]
                )
        if F > 1:
            ref_qvel[F - 1] = ref_qvel[F - 2]

        return cls(
            reference_qpos=reference,
            reference_qvel=ref_qvel,
            reference_torso_quat=ref_torso_quat,
            reference_foot_pos=ref_foot_pos,
            reference_foot_quat=ref_foot_quat,
            fps=fps,
            nq=nq,
            nv=nv,
            torso_quat_adr=torso_quat_adr,
            foot_pos_adr=foot_pos_adr,
            foot_quat_adr=foot_quat_adr,
            **weights,
        )

    # ---- reference indexing / feedforward ----

    def _frame_index(self, t: np.ndarray) -> np.ndarray:
        """Reference frame index for sim time(s) t, clipped to the clip length."""
        i = (np.asarray(t) * self.fps).astype(np.int64)
        return np.clip(i, 0, self._F - 1)

    def reference_action(self, t: float | np.ndarray) -> np.ndarray:
        """Feedforward control (target joint angles) at time t: qpos[7:] of the
        reference frame. Useful for warm-starting the MPPI nominal, since the G1
        uses PD position actuators (ctrl = target joint angle)."""
        i = self._frame_index(t)
        return self.reference_qpos[i, 7:]

    # ---- per-step cost (works on any leading shape) ----

    def _state_cost(self, states: np.ndarray, sensordata: np.ndarray) -> np.ndarray:
        t = states[..., 0]                                  # (...,) sim time
        i = self._frame_index(t)                            # (...,) frame index

        # Configuration + generalized-velocity tracking (from the state buffer).
        q = states[..., 1 : 1 + self.nq]
        v = states[..., 1 + self.nq : 1 + self.nq + self.nv]
        q_err = q - self.reference_qpos[i]                  # (..., nq)
        v_err = v - self.reference_qvel[i]                  # (..., nv)
        q_cost = _track_cost(np.sum(q_err * q_err, axis=-1))
        v_cost = _track_cost(np.sum(v_err * v_err, axis=-1))

        # Body-position tracking (feet, from sensors).
        foot_pos = np.concatenate(
            [
                sensordata[..., self.foot_pos_adr[0] : self.foot_pos_adr[0] + 3],
                sensordata[..., self.foot_pos_adr[1] : self.foot_pos_adr[1] + 3],
            ],
            axis=-1,
        )                                                   # (..., 6)
        pos_err = foot_pos - self.reference_foot_pos[i]     # (..., 6)
        pos_cost = _track_cost(np.sum(pos_err * pos_err, axis=-1))

        # Body-orientation tracking (torso + feet, from sensors).
        torso_quat = sensordata[..., self.torso_quat_adr : self.torso_quat_adr + 4]
        lfoot_quat = sensordata[..., self.foot_quat_adr[0] : self.foot_quat_adr[0] + 4]
        rfoot_quat = sensordata[..., self.foot_quat_adr[1] : self.foot_quat_adr[1] + 4]
        ref_foot_quat = self.reference_foot_quat[i]         # (..., 2, 4)
        ori_sq = (
            np.sum(_quat_sub(torso_quat, self.reference_torso_quat[i]) ** 2, axis=-1)
            + np.sum(_quat_sub(lfoot_quat, ref_foot_quat[..., 0, :]) ** 2, axis=-1)
            + np.sum(_quat_sub(rfoot_quat, ref_foot_quat[..., 1, :]) ** 2, axis=-1)
        )
        ori_cost = _track_cost(ori_sq)

        return (
            self.configuration_weight * q_cost
            + self.velocity_weight * v_cost
            + self.body_position_weight * pos_cost
            + self.body_orientation_weight * ori_cost
        )

    def running_cost(self, state: np.ndarray, control: np.ndarray,
                     sensordata: np.ndarray) -> np.ndarray:
        """Running cost l(x_t, u_t) -- no control term."""
        return self._state_cost(state, sensordata)

    def terminal_cost(self, state: np.ndarray, sensordata: np.ndarray) -> np.ndarray:
        """Terminal cost phi(x_T)."""
        return self._state_cost(state, sensordata)

    # ---- MPPI batched integration over (B, H) ----

    def __call__(self, states: np.ndarray, controls: np.ndarray,
                 sensordata: np.ndarray) -> np.ndarray:
        sc = self._state_cost(states, sensordata)           # (B, H)
        running = sc[..., :-1].sum(axis=-1)
        terminal = self.terminal_weight * sc[..., -1]
        return running + terminal
