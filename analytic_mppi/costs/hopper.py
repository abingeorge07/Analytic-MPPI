from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HopperCost:
    """Forward-hopping cost for the planar MuJoCo hopper.

    For MuJoCo FULLPHYSICS state (nq=6, nv=6) the relevant entries are:
      height  = state[..., 2]   (qpos[1], rootz)   -- torso height
      orient  = state[..., 3]   (qpos[2], rooty)   -- torso pitch (0 = upright)
      vx      = state[..., 7]   (qvel[0], rootx')  -- forward velocity

    Per-step pieces broadcast on leading axes so they work on single states
    or on (B, H, nstate) rollouts. The objective rewards forward velocity
    while keeping the torso near `target_height` and upright:

      height_cost      = height_weight  * (height - target_height)^2
      orientation_cost = orient_weight  * orient^2
      forward_cost     = -forward_weight * vx          (reward forward speed)
      control_cost     = control_weight * ||u||^2

    `__call__` does the MPPI integration:

      states:   (B, H, nstate)  -- rollout output, excludes initial
      controls: (B, H, nu)
      returns:  (B,)

      total = sum_{t=0..H-2} [ l_state(x_t) + w_u*||u_t||^2 ]
            + w_u*||u_{H-1}||^2
            + terminal_weight * l_state(x_{H-1})
    """

    height_idx: int = 2
    orient_idx: int = 3
    vx_idx: int = 7
    target_height: float = 1.2
    height_weight: float = 1.0
    orient_weight: float = 1.0
    forward_weight: float = 1.0
    control_weight: float = 2.0
    terminal_weight: float = 1.0

    # ---- per-step components (work on any leading shape) ----

    def _get_height_cost(self, state: np.ndarray) -> np.ndarray:
        height = state[..., self.height_idx]
        return self.height_weight * (height - self.target_height) ** 2

    def _get_orientation_cost(self, state: np.ndarray) -> np.ndarray:
        orient = state[..., self.orient_idx]
        return self.orient_weight * orient ** 2

    def _get_forward_cost(self, state: np.ndarray) -> np.ndarray:
        vx = state[..., self.vx_idx]
        return -self.forward_weight * vx

    def _get_control_cost(self, control: np.ndarray) -> np.ndarray:
        return self.control_weight * (control ** 2).sum(axis=-1)

    def _state_cost(self, state: np.ndarray) -> np.ndarray:
        return (
            self._get_height_cost(state)
            + self._get_orientation_cost(state)
            + self._get_forward_cost(state)
        )

    def running_cost(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Running cost l(x_t, u_t)."""
        return self._state_cost(state) + self._get_control_cost(control)

    def terminal_cost(self, state: np.ndarray) -> np.ndarray:
        """Terminal cost phi(x_T) -- no control term."""
        return self._state_cost(state)

    # ---- MPPI batched integration over (B, H) ----

    def __call__(self, states: np.ndarray, controls: np.ndarray,
                 sensordata: np.ndarray | None = None) -> np.ndarray:
        sc = self._state_cost(states)              # (B, H)
        uc = self._get_control_cost(controls)      # (B, H)

        running = sc[..., :-1].sum(axis=-1) + uc.sum(axis=-1)
        terminal = self.terminal_weight * sc[..., -1]
        return running + terminal
