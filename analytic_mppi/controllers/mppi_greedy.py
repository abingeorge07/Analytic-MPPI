from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class _Backend(Protocol):
    nstate: int
    nu: int
    def rollout(self, initial_states: np.ndarray, controls: np.ndarray): ...


CostFn = Callable[..., np.ndarray]   # (states, controls, sensordata=None) -> (K,)


@dataclass
class MPPIGreedy:
    """Greedy (argmin) MPPI.

    Like vanilla MPPI but selects the single best rollout instead of a
    softmax-weighted average. Per `act(state, cost_fn, backend)` call:
      1. Sample K Gaussian noise sequences of length H, std `sigma`.
      2. Form K candidate control sequences = nominal + noise, clip to [u_min, u_max].
      3. Roll out in parallel from `state` via `backend.rollout`.
      4. Score each rollout with `cost_fn(states, controls, sensordata) -> (K,)`.
      5. Pick best = argmin(J); optionally refine it via `refine_fn` (e.g.
         gradient descent on the cost); cache that rollout as the nominal.
      6. Return nominal[0]; shift nominal left by one (the executed action),
         keep the first `keep` timesteps and zero the rest.

    `keep` controls how much of the cached best rollout warm-starts the next call:
      - keep = horizon - 1 (default): mirrors vanilla MPPI's warm-start shape.
      - keep = 0: no warm start; re-sample around a zero nominal every step.
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
    keep: int = -1   # timesteps of the best rollout kept for the next nominal; -1 -> horizon - 1

    nominal: np.ndarray = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    last_samples: np.ndarray | None = field(init=False, default=None, repr=False)
    last_weights: np.ndarray | None = field(init=False, default=None, repr=False)
    last_best: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.sigma = np.asarray(self.sigma, dtype=np.float64).reshape(self.nu)
        if self.u_min is not None:
            self.u_min = np.asarray(self.u_min, dtype=np.float64).reshape(self.nu)
        if self.u_max is not None:
            self.u_max = np.asarray(self.u_max, dtype=np.float64).reshape(self.nu)
        self.nominal = np.zeros((self.horizon, self.nu), dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)
        if self.keep < 0:
            self.keep = self.horizon - 1
        # Clamp to [0, H-1] so the stale wrapped element from np.roll never survives.
        self.keep = int(np.clip(self.keep, 0, self.horizon - 1))

    def reset(self):
        self.nominal[:] = 0.0

    def act(
        self,
        state: np.ndarray,
        cost_fn: CostFn,
        backend: _Backend,
        refine_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        K, H, nu = self.n_samples, self.horizon, self.nu
        noise = self.rng.normal(scale=self.sigma, size=(K, H, nu))
        U = self.nominal[None, :, :] + noise
        if self.u_min is not None or self.u_max is not None:
            U = np.clip(U, self.u_min, self.u_max)

        initial = np.broadcast_to(state, (K, state.shape[0])).copy()
        states, sensordata = backend.rollout(initial, U)    # (K, H, nstate), (K, H, nsensordata)
        costs = np.asarray(cost_fn(states, U, sensordata), dtype=np.float64).reshape(K)

        best = int(np.argmin(costs))
        best_U = U[best:best + 1].copy()    # (1, H, nu): the single best rollout
        if refine_fn is not None:
            # Refine the best rollout (e.g. gradient descent on the cost) before it
            # becomes the nominal. refine_fn: (best (1, H, nu), state) -> (1, H, nu).
            best_U = np.asarray(refine_fn(best_U, state), dtype=np.float64).reshape(1, H, nu)
        self.nominal = best_U[0].copy()     # cache the (refined) best rollout as the nominal
        self.last_best = self.nominal       # expose for inspection

        if self.store_samples:
            self.last_samples = states
            w_norm = np.zeros(K)
            w_norm[best] = 1.0
            self.last_weights = w_norm

        u0 = self.nominal[0].copy()
        self.nominal = np.roll(self.nominal, -1, axis=0)   # shift left by one (executed action)
        self.nominal[self.keep:] = 0.0                     # keep first `keep` steps, zero the tail
        return u0
