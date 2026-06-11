from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InvPenCost:
    """Inverted-pendulum swing-up cost.

    For MuJoCo FULLPHYSICS state with nq=1, nv=1 (pendulum):
      theta     = state[..., 1]   (qpos[0],  0 = hanging down)
      theta_dot = state[..., 2]   (qvel[0])

    Per-step pieces broadcast on leading axes so they work on single states
    or on (B, H, nstate) rollouts. `__call__` does the MPPI integration:

      states:   (B, H, nstate)  -- rollout output, excludes initial
      controls: (B, H, nu)
      returns:  (B,)

      total = sum_{t=0..H-2} [ d_upright(x_t) + w_v*v_t^2 + w_u*||u_t||^2 ]
            + w_u*||u_{H-1}||^2
            + terminal_weight * [ d_upright(x_{H-1}) + w_v*v_{H-1}^2 ]
    """

    theta_idx: int = 1
    theta_dot_idx: int = 2
    theta_dot_weight: float = 0.01
    control_weight: float = 0.001
    terminal_weight: float = 1.0

    # ---- per-step components (work on any leading shape) ----

    def _distance_to_upright(self, state: np.ndarray) -> np.ndarray:
        """Smooth, wrap-free distance to theta=pi.

        theta_err = [cos(theta-pi) - 1, sin(theta-pi)];  cost = ||theta_err||^2.
        Zero at theta = pi (mod 2*pi), maximum (=4) at theta = 0.
        """
        theta = state[..., self.theta_idx]
        e = theta - np.pi
        return (np.cos(e) - 1.0) ** 2 + np.sin(e) ** 2

    def _get_theta_dot_cost(self, state: np.ndarray) -> np.ndarray:
        theta_dot = state[..., self.theta_dot_idx]
        return self.theta_dot_weight * theta_dot ** 2

    def _get_control_cost(self, control: np.ndarray) -> np.ndarray:
        return self.control_weight * (control ** 2).sum(axis=-1)

    def running_cost(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Running cost l(x_t, u_t)."""
        return (
            self._distance_to_upright(state)
            + self._get_theta_dot_cost(state)
            + self._get_control_cost(control)
        )

    def terminal_cost(self, state: np.ndarray) -> np.ndarray:
        """Terminal cost phi(x_T) -- no control term."""
        return self._distance_to_upright(state) + self._get_theta_dot_cost(state)

    # ---- MPPI batched integration over (B, H) ----

    def __call__(self, states: np.ndarray, controls: np.ndarray,
                 sensordata: np.ndarray | None = None) -> np.ndarray:
        d_up = self._distance_to_upright(states)            # (B, H)
        td = self._get_theta_dot_cost(states)               # (B, H)
        uc = self._get_control_cost(controls)               # (B, H)

        running = (
            d_up[..., :-1].sum(axis=-1)
            + td[..., :-1].sum(axis=-1)
            + uc.sum(axis=-1)
        )
        terminal = self.terminal_weight * (d_up[..., -1] + td[..., -1])
        return running + terminal
