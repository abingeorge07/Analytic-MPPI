"""MJX (GPU) planning backend: MuJoCo MJX rollouts behind the numpy backend protocol.

One dynamics definition, two faces:

  * `rollout(initial_states, controls)` -- the SAME numpy-in/numpy-out contract as
    `MujocoBackend.rollout`, so every sampling controller runs on it unchanged.
    Internally `jit(vmap(scan(mjx.step)))`.
  * `rollout_jax(t0, qpos0, qvel0, controls)` / `step_fn(data, ctrl)` -- the jax-native
    differentiable surface gradient MPC composes into its own jitted objective. Nothing
    numpy crosses this path, so `jax.grad` flows through the whole rollout.

This backend is PLANNING-ONLY: it deliberately implements no `step`/`get_state`/
`set_state`. The closed loop is always stepped by a CPU `MujocoBackend` (see
eval.make_controller), so the physics of record -- and everything that needs a real
`MjData`: the viewer, --print-costs, video rendering -- stays exactly what it was for
every existing CPU result. An mjx-planned run differs from its CPU baseline only in
which model the controller *believes*, which is the same model-mismatch structure the
perturbation studies already use.

Precision: MJX runs in float32 on GPU (float64 on a consumer card is ~64x slower and
planning does not need it -- the sampler's noise scale dwarfs f32 error). The numpy face
returns float64 arrays to honor the protocol; the cast does not add information.

State layout: only `mjSTATE_FULLPHYSICS == [time, qpos, qvel]` models are supported
(na == 0, no actuator activation states). All current tasks satisfy this; the ctor
asserts it rather than silently mis-slicing.
"""
from __future__ import annotations

import os
import time as _time
from pathlib import Path

import numpy as np
import mujoco

from .mujoco_backend import apply_perturbation


class MJXBackend:
    """MJX-backed planning dynamics. Protocol-compatible with `MujocoBackend.rollout`."""

    state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS

    def __init__(self, model_path: str | os.PathLike, nthread: int | None = None,
                 perturb: dict | None = None):
        # XLA's Triton gemm-fusion autotuner hard-crashes (FAILED_PRECONDITION, core
        # dump) compiling the BACKWARD pass of mjx's constraint solver on this stack
        # (jax 0.6.2 / RTX 4070). Disabling triton gemm fusions avoids it; cuBLAS is
        # used instead, with no measured slowdown at our sizes. Must be set before jax
        # initializes its backend -- appending here covers every entry point, but an
        # already-initialized jax (rare: something imported+used jax first) wins.
        flags = os.environ.get("XLA_FLAGS", "")
        if "--xla_gpu_enable_triton_gemm" not in flags:
            os.environ["XLA_FLAGS"] = (flags + " --xla_gpu_enable_triton_gemm=false").strip()

        # jax is imported here, not at module top: the CPU path must never require it.
        import jax
        from mujoco import mjx

        self._jax = jax
        self._mjx = mjx

        self.model_path = Path(model_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        # Perturbation (model-mismatch studies) is applied to the CPU model BEFORE
        # put_model, so the device copy carries the scaled parameters.
        if perturb:
            apply_perturbation(self.model, perturb)
        self.mjx_model = mjx.put_model(self.model)

        # `nthread` is part of the backend ctor protocol; meaningless on GPU.
        self.nthread = nthread

        self.nq = int(self.model.nq)
        self.nv = int(self.model.nv)
        self.nu = int(self.model.nu)
        self.nsensordata = int(self.model.nsensordata)
        self.nstate = int(mujoco.mj_stateSize(self.model, int(self.state_spec)))
        self.dt = float(self.model.opt.timestep)
        self.qpos_slice = slice(1, 1 + self.nq)
        self.qvel_slice = slice(1 + self.nq, 1 + self.nq + self.nv)
        if self.nstate != 1 + self.nq + self.nv:
            raise NotImplementedError(
                f"{self.model_path.name}: FULLPHYSICS state has size {self.nstate} != "
                f"1 + nq + nv = {1 + self.nq + self.nv} (actuator activation states?). "
                f"MJXBackend only supports the [time, qpos, qvel] layout."
            )

        # Template device Data; per-rollout states are `replace`d onto it inside jit.
        self._data0 = mjx.make_data(self.mjx_model)
        self._batched_rollout = jax.jit(jax.vmap(self.rollout_jax))
        # First-call compile time per (B, H) shape, seconds. Diagnostics only.
        self.warmup_s: dict[tuple[int, int], float] = {}

    # ---- face 2: jax-native, differentiable ----

    def step_fn(self, data, ctrl):
        """One pure MJX step: write ctrl, step. `data` is `mjx.Data`, traceable."""
        data = data.replace(ctrl=ctrl)
        return self._mjx.step(self.mjx_model, data)

    def rollout_jax(self, t0, qpos0, qvel0, controls):
        """Open-loop rollout of ONE trajectory. jax arrays in, jax arrays out.

        t0 ():  qpos0 (nq,)  qvel0 (nv,)  controls (H, nu)
        -> (time (H,), qpos (H, nq), qvel (H, nv), sensordata (H, nsensordata))

        Mirrors `mujoco.rollout` recording semantics: entry i is (state AFTER step i+1,
        sensordata computed DURING that step's forward pass). Differentiable end-to-end;
        gradient MPC composes this with jnp costs inside its own `value_and_grad`.
        """
        jax = self._jax

        data = self._data0.replace(qpos=qpos0, qvel=qvel0, time=t0)

        def body(d, u):
            d = self.step_fn(d, u)
            return d, (d.time, d.qpos, d.qvel, d.sensordata)

        _, (t_seq, qpos_seq, qvel_seq, sd_seq) = jax.lax.scan(body, data, controls)
        return t_seq, qpos_seq, qvel_seq, sd_seq

    # ---- face 1: numpy protocol (sampling controllers) ----

    def rollout(self, initial_states: np.ndarray, controls: np.ndarray):
        """Batched rollouts, protocol-identical to `MujocoBackend.rollout`.

        initial_states: (B, nstate)   controls: (B, H, nu)
        returns (states (B, H, nstate), sensordata (B, H, nsensordata)), float64,
        excluding the initial step -- same convention as `mujoco.rollout`.
        """
        initial_states = np.asarray(initial_states, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        B, H = controls.shape[0], controls.shape[1]

        t0 = initial_states[:, 0]
        qpos0 = initial_states[:, self.qpos_slice]
        qvel0 = initial_states[:, self.qvel_slice]

        shape = (B, H)
        tic = _time.perf_counter() if shape not in self.warmup_s else None
        t_seq, qpos_seq, qvel_seq, sd_seq = self._batched_rollout(t0, qpos0, qvel0, controls)
        sd_seq = np.asarray(self._jax.device_get(sd_seq), dtype=np.float64)
        if tic is not None:
            # First call at this shape includes trace+compile; later calls are steady-state.
            self.warmup_s[shape] = _time.perf_counter() - tic

        states = np.empty((B, H, self.nstate), dtype=np.float64)
        # MJX accumulates data.time in f32; the schedule is deterministic, so rebuild it
        # in f64 to keep the protocol face exact (mujoco.rollout's time column is exact).
        del t_seq
        states[..., 0] = t0[:, None] + self.dt * np.arange(1, H + 1, dtype=np.float64)
        states[..., self.qpos_slice] = np.asarray(qpos_seq, dtype=np.float64)
        states[..., self.qvel_slice] = np.asarray(qvel_seq, dtype=np.float64)
        return states, sd_seq
