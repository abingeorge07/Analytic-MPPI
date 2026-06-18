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

    def __init__(self):
        super().__init__(_MODEL_PATH)

    # ---- normal cost ----

    def _distance_to_upright(self, qpos: np.ndarray) -> np.ndarray:
        # Smooth wrap-free distance to theta=pi.
        theta = qpos[..., 0] - np.pi
        return (np.cos(theta) - 1.0) ** 2 + np.sin(theta) ** 2

    def _theta_dot_cost(self, qvel: np.ndarray) -> np.ndarray:
        return 0.01 * qvel[..., 0] ** 2

    def _control_cost(self, u: np.ndarray) -> np.ndarray:
        return 0.001 * np.sum(u ** 2, axis=-1)

    def running_cost_terms(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [
                self._distance_to_upright(qpos),
                self._theta_dot_cost(qvel),
                self._control_cost(u),
            ],
            axis=-1,
        )

    def terminal_cost_terms(self, qpos, qvel, sensordata) -> np.ndarray:
        d = self._distance_to_upright(qpos)
        td = self._theta_dot_cost(qvel)
        zero = np.zeros_like(d)
        return np.stack([d, td, zero], axis=-1)

    # ---- FPL cost (per-step satisfaction in [0,1]) ----

    def _upright_fulfillment(self, qpos: np.ndarray) -> np.ndarray:
        # Direction (sin theta, -cos theta) hits (0, 1) at theta=pi.
        # Dot with (0,1) is -cos(theta); shift to [0,1].
        theta = qpos[..., 0]
        return (-np.cos(theta) + 1.0) * 0.5

    def _control_fulfillment(self, u: np.ndarray) -> np.ndarray:
        # 1 when u==0, decays toward 0 as |u| grows. Per-actuator average squared.
        n = u.shape[-1]
        return np.clip(1.0 - np.sum(u ** 2, axis=-1) / n, 0.0, 1.0)

    def running_cost_terms_f(self, qpos, qvel, sensordata, u) -> np.ndarray:
        return np.stack(
            [self._upright_fulfillment(qpos), self._control_fulfillment(u)],
            axis=-1,
        )

    def terminal_cost_terms_f(self, qpos, qvel, sensordata) -> np.ndarray:
        # Only the upright term has a terminal contribution. Control fulfillment
        # is undefined at the terminal step (no action is applied), so we omit
        # it rather than padding with ones — a constant placeholder biases the
        # power-mean / discounted reward upward without conveying any info.
        return self._upright_fulfillment(qpos)[..., None]
