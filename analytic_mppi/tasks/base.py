"""Base Task class for the CPU sampling-based MPC framework.

A Task bundles:
  * a MuJoCo model (the `mj_model` attribute),
  * the per-step cost computation in TWO flavours:
      - normal (penalty)        — `running_cost_terms`, `terminal_cost_terms`
      - FPL (fulfillment [0,1]) — `running_cost_terms_f`, `terminal_cost_terms_f`,
  * actuator bounds `u_min, u_max`,
  * helpers to slice qpos / qvel from a FULLPHYSICS state buffer.

All cost methods are written in BATCHED numpy. They accept arrays with any
leading dimensions — typically `(K, H, ...)` for K rollouts × H horizon steps —
and broadcast naturally. This matches mis-style cost code (one expression for
single-state and batched-state evaluation), and avoids Python loops over (K, H).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import mujoco


def power_mean(terms: np.ndarray, p: float, eps: float = 1e-8) -> np.ndarray:
    """Generalized (power) mean along the LAST axis of `terms`.

    p == 0  -> geometric mean exp(mean(log x))
    p != 0  -> (mean(x**p))**(1/p)

    Terms are clipped from below by `eps` so log/division stay finite.
    """
    t = np.clip(terms, eps, None)
    if p == 0:
        return np.exp(np.mean(np.log(t), axis=-1))
    n = t.shape[-1]
    return (np.sum(t ** p, axis=-1) / n) ** (1.0 / p)


class Task:
    """Base class. Subclasses load a model and override the cost methods.

    Convention for cost methods: every input array has shape `(*leading, dim)`
    where `*leading` is whatever batch shape the caller used (often `(K, H)`
    or `(K,)`), and `dim` is `nq`, `nv`, `nu`, or `nsensordata`. Outputs of
    `*_terms` methods have shape `(*leading, n_terms)`.
    """

    cost_term_names: List[str] = []        # subclasses set
    cost_term_names_f: List[str] = []      # subclasses set

    def __init__(self, model_path: str | os.PathLike):
        self.model_path = Path(model_path)
        self.mj_model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.nu = int(self.mj_model.nu)
        self.nq = int(self.mj_model.nq)
        self.nv = int(self.mj_model.nv)
        self.nsensordata = int(self.mj_model.nsensordata)

        ctrl_range = np.asarray(self.mj_model.actuator_ctrlrange, dtype=np.float64)
        self.u_min = ctrl_range[:, 0].copy()
        self.u_max = ctrl_range[:, 1].copy()

        # FULLPHYSICS state layout: [time, qpos (nq), qvel (nv), ...]
        self.qpos_slice = slice(1, 1 + self.nq)
        self.qvel_slice = slice(1 + self.nq, 1 + self.nq + self.nv)

    # ---- state decoding ----

    def qpos_of(self, state: np.ndarray) -> np.ndarray:
        return state[..., self.qpos_slice]

    def qvel_of(self, state: np.ndarray) -> np.ndarray:
        return state[..., self.qvel_slice]

    # ---- normal-cost API (subclasses implement *_terms; sums are derived) ----

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        raise NotImplementedError

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        raise NotImplementedError

    def running_cost(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return self.running_cost_terms(qpos, qvel, sensordata, u).sum(axis=-1)

    def terminal_cost(self, qpos, qvel, sensordata) -> np.ndarray:
        return self.terminal_cost_terms(qpos, qvel, sensordata).sum(axis=-1)

    # ---- FPL-cost API (terms in [0,1]; subclasses implement *_terms_f) ----

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        raise NotImplementedError

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        raise NotImplementedError

    def running_cost_f(self, qpos, qvel, sensordata, u, p: float = 0.1) -> np.ndarray:
        return power_mean(self.running_cost_terms_f(qpos, qvel, sensordata, u), p)

    def terminal_cost_f(self, qpos, qvel, sensordata, p: float = 0.1) -> np.ndarray:
        return power_mean(self.terminal_cost_terms_f(qpos, qvel, sensordata), p)
