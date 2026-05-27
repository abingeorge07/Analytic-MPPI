from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class _Backend(Protocol):
    nstate: int
    nu: int
    def rollout(self, initial_states: np.ndarray, controls: np.ndarray): ...


CostFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class MPPI:
    """Vanilla MPPI (textbook formulation).

    Per `act(state, cost_fn, backend)` call:
      1. Sample K Gaussian noise sequences of length H, std `sigma`.
      2. Form K candidate control sequences = nominal + noise, clip to [u_min, u_max].
      3. Roll out in parallel from `state` via `backend.rollout`.
      4. Score each rollout with `cost_fn(states, controls) -> (K,)`.
      5. Re-weight: w_k = exp(-(J_k - min J) / lambda_), normalize.
      6. Update nominal = sum_k w_k * U_k.
      7. Return nominal[0]; shift nominal left, pad with zero.
    """

    horizon: int
    n_samples: int
    nu: int
    sigma: np.ndarray
    lambda_: float
    u_min: np.ndarray | None = None
    u_max: np.ndarray | None = None
    seed: int = 0
    store_samples: bool = False

    nominal: np.ndarray = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    last_samples: np.ndarray | None = field(init=False, default=None, repr=False)
    last_weights: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.sigma = np.asarray(self.sigma, dtype=np.float64).reshape(self.nu)
        if self.u_min is not None:
            self.u_min = np.asarray(self.u_min, dtype=np.float64).reshape(self.nu)
        if self.u_max is not None:
            self.u_max = np.asarray(self.u_max, dtype=np.float64).reshape(self.nu)
        self.nominal = np.zeros((self.horizon, self.nu), dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def reset(self):
        self.nominal[:] = 0.0

    def act(self, state: np.ndarray, cost_fn: CostFn, backend: _Backend) -> np.ndarray:
        K, H, nu = self.n_samples, self.horizon, self.nu
        noise = self.rng.normal(scale=self.sigma, size=(K, H, nu))
        U = self.nominal[None, :, :] + noise
        if self.u_min is not None or self.u_max is not None:
            U = np.clip(U, self.u_min, self.u_max)

        initial = np.broadcast_to(state, (K, state.shape[0])).copy()
        states, _sensordata = backend.rollout(initial, U)   # (K, H, nstate), (K, H, nsensordata)
        costs = np.asarray(cost_fn(states, U), dtype=np.float64).reshape(K)

        beta = costs.min()
        w = np.exp(-(costs - beta) / self.lambda_)
        s = w.sum()
        if s <= 0 or not np.isfinite(s):
            # Pathological case: every weight underflowed. Fall back to best sample.
            best = int(np.argmin(costs))
            self.nominal = U[best].copy()
            w_norm = np.zeros(K)
            w_norm[best] = 1.0
        else:
            w_norm = w / s
            self.nominal = (w_norm[:, None, None] * U).sum(axis=0)

        if self.store_samples:
            self.last_samples = states
            self.last_weights = w_norm

        u0 = self.nominal[0].copy()
        self.nominal = np.roll(self.nominal, -1, axis=0)
        self.nominal[-1] = 0.0
        return u0
