from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LunarLanderCost:
    """Gymnasium LunarLander reward, converted to a decomposed MPPI cost.

    The default LunarLander reward is built from a shaping potential plus a fuel
    penalty and a terminal crash/land bonus:

      shaping = -100*sqrt(x^2 + y^2) - 100*sqrt(vx^2 + vy^2) - 100*|angle|
                + 10*leg1 + 10*leg2
      reward  = (shaping_t - shaping_{t-1}) - 0.30*m_power - 0.03*s_power
      terminal: -100 on crash (game_over or |x|>=1), +100 on landing at rest

    We re-express it as named, (mostly) positive COST terms (lower = better). The
    running cost uses the shaping *potential* per step rather than its telescoping
    delta -- a denser, better-behaved signal whose horizon sum equals the negated
    shaping change up to a constant initial offset, so it ranks trajectories the
    same way the reward does. The leg term is the constant-shifted positive form
    of the +10/leg contact bonus (offset is irrelevant to the argmin).

    Operates on the GymBackend rollout outputs:
      states     : (..., 8)  observation [x, y, vx, vy, angle, omega, leg1, leg2]
      controls   : (..., 2)  [main, side] in [-1, 1]
      sensordata : (..., 3)  per-step flags [terminated, crashed, landed]
    Every piece broadcasts over arbitrary leading axes (single state or (B, H, *)).
    """

    distance_weight: float = 100.0
    velocity_weight: float = 100.0
    angle_weight: float = 100.0
    leg_weight: float = 10.0
    fuel_weight: float = 1.0
    crash_weight: float = 100.0
    land_weight: float = 100.0

    term_names = ["distance_cost", "velocity_cost", "angle_cost", "leg_cost", "fuel_cost"]
    terminal_term_names = ["crash_cost", "land_bonus"]

    # ---- running cost pieces (work on any leading shape) ----

    def _distance_cost(self, s: np.ndarray) -> np.ndarray:
        return self.distance_weight * np.sqrt(s[..., 0] ** 2 + s[..., 1] ** 2)

    def _velocity_cost(self, s: np.ndarray) -> np.ndarray:
        return self.velocity_weight * np.sqrt(s[..., 2] ** 2 + s[..., 3] ** 2)

    def _angle_cost(self, s: np.ndarray) -> np.ndarray:
        return self.angle_weight * np.abs(s[..., 4])

    def _leg_cost(self, s: np.ndarray) -> np.ndarray:
        # +10/leg contact reward -> 10*((1-leg1)+(1-leg2)) penalty (off-pad legs).
        return self.leg_weight * ((1.0 - s[..., 6]) + (1.0 - s[..., 7]))

    def _fuel_cost(self, u: np.ndarray) -> np.ndarray:
        # LunarLander fuel: main fires when u0>0 (m_power in [0.5,1]); side fires
        # when |u1|>0.5 (s_power in [0.5,1]). Matches lunar_lander.py step().
        u0, u1 = u[..., 0], u[..., 1]
        m_power = np.where(u0 > 0.0, (np.clip(u0, 0.0, 1.0) + 1.0) * 0.5, 0.0)
        s_power = np.where(np.abs(u1) > 0.5, np.clip(np.abs(u1), 0.5, 1.0), 0.0)
        return self.fuel_weight * (0.30 * m_power + 0.03 * s_power)

    def running_terms(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Per-step running cost terms -> (..., 5)."""
        return np.stack(
            [
                self._distance_cost(states),
                self._velocity_cost(states),
                self._angle_cost(states),
                self._leg_cost(states),
                self._fuel_cost(controls),
            ],
            axis=-1,
        )

    def terminal_terms(self, sensordata: np.ndarray) -> np.ndarray:
        """Terminal crash/land terms from the rollout's final-step flags -> (..., 2).

        crash_cost = +crash_weight on a crash; land_bonus = -land_weight on a
        successful landing (negative -> lowers total cost)."""
        crashed = sensordata[..., 1]
        landed = sensordata[..., 2]
        return np.stack([self.crash_weight * crashed, -self.land_weight * landed], axis=-1)

    def running_cost(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return self.running_terms(states, controls).sum(axis=-1)

    def terminal_cost(self, sensordata: np.ndarray) -> np.ndarray:
        return self.terminal_terms(sensordata).sum(axis=-1)

    # ---- MPPI batched integration over (B, H) ----

    def __call__(self, states: np.ndarray, controls: np.ndarray,
                 sensordata: np.ndarray | None = None) -> np.ndarray:
        """Per-rollout total cost. states/controls: (B, H, *) -> (B,).

        total = sum_t running(x_t, u_t) + terminal(outcome). The terminal
        outcome is read from the last step's flags (frozen forward by the
        backend after termination)."""
        running = self.running_terms(states, controls).sum(axis=(-1, -2))   # (B,)
        if sensordata is None:
            return running
        terminal = self.terminal_terms(sensordata[..., -1, :]).sum(axis=-1)  # (B,)
        return running + terminal
