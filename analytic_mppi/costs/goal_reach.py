from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class GoalReachCost:
    """Quadratic distance-to-goal + quadratic control cost.

    states:   (B, H, nstate)   -- rollout output, excludes initial state
    controls: (B, H, nu)
    returns:  (B,)             -- per-sample total cost

    `state_xy_idx` selects which entries of the state vector are (x, y).
    For MuJoCo FULLPHYSICS state with the planar unicycle (nq=3), these are (1, 2).
    """

    goal: Sequence[float]
    state_xy_idx: Sequence[int]
    Q: float = 1.0
    R: float | Sequence[float] | None = None
    terminal_weight: float = 10.0

    goal_arr: np.ndarray = field(init=False, repr=False)
    idx: list[int] = field(init=False, repr=False)
    R_arr: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self):
        self.goal_arr = np.asarray(self.goal, dtype=np.float64)
        self.idx = list(self.state_xy_idx)
        self.R_arr = None if self.R is None else np.atleast_1d(np.asarray(self.R, dtype=np.float64))

    def __call__(self, states: np.ndarray, controls: np.ndarray,
                 sensordata: np.ndarray | None = None) -> np.ndarray:
        xy = states[..., self.idx]                            # (B, H, 2)
        sqdist = ((xy - self.goal_arr) ** 2).sum(axis=-1)     # (B, H)
        running = self.Q * sqdist[..., :-1].sum(axis=-1)      # (B,)
        if self.R_arr is not None:
            running = running + (controls ** 2 * self.R_arr).sum(axis=(-2, -1))
        terminal = self.terminal_weight * sqdist[..., -1]     # (B,)
        return running + terminal
