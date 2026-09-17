"""S9 / WO-3.2 — rewrite every objective mode as (additive stage) + (terminal readout).

Why this exists
---------------
iLQR/DDP needs a cost that decomposes over time: `J = sum_t l(x_t, u_t, t) + phi(x_H)`.
The FPL objectives are NOT of that form -- `_score_fpl` collapses the whole (H+1)-step
series through one power mean, so no per-step stage cost exists in the original state.

They are, however, all **quasi-arithmetic**: every power mean is `g^-1(sum_t w_t g(m_t))`
for a fixed generator `g`. Carrying `z_t = sum_{s<t} w_s g(m_s)` as extra state makes the
accumulation additive and pushes the entire nonlinearity into a terminal readout `g^-1`
(composed with the `-log` cost bridge). This is an **exact rewrite**, not an
approximation: no linearization, no surrogate, no tuning constant.

Augmented state and block-triangularity
---------------------------------------
The augmented system is

    x_{t+1} = F(x_t, u_t)                       (untouched physics)
    z_{t+1} = z_t + phi_t(x_t, u_t)             (this module)
    J       = psi(z_H, x_H)                     (this module)

`z` never feeds back into `F`, so `d x_{t+1} / d z_t = 0` and the augmented Jacobian is
block lower-triangular. That is what keeps the iLQR backward pass on the physics block
the same size it was before augmenting -- and it is asserted in `tests/test_accumulator.py`
by autodiff through a real MJX step, not assumed here.

What `z` is, per mode
---------------------
| mode             | n_acc   | stage increment                        | terminal readout            |
|------------------|---------|----------------------------------------|-----------------------------|
| `normal`         | 1       | `dt * sum_i l_i(t)`                    | `z + sum_i L_i`             |
| `fpl_cost`, tp=None | 1    | `w_t * m_t`                            | `-log(z + w_H m_H)`         |
| `fpl_cost`, tp=q!=0 | 1    | `w_t * clip(m_t)^q`                    | `-(1/q) * log(z + ...)`     |
| `fpl_cost`, tp=0 | 1       | `w_t * log(clip(m_t))`                 | `-(z + ...)`                |
| `fpl_discounted` | n_run   | `w_t^(j) * f_j(t)`                     | `-log(power_mean(z, p, w))` |
| `fpl_layered`    | n_atoms | `w_t * f_j(t)`                         | `-log(outer(inner(z)))`     |
| `hybrid`         | 1       | `dt * (targets + fw * barriers)`       | `z + terminal targets/barriers` |

Two consequences worth stating, because they are results rather than implementation notes:

* **`time_p = 0` needs no terminal nonlinearity at all.** The generator is `log` and the
  cost bridge is `-log`, so they cancel exactly: `J = -z_H`. At that one setting the FPL
  objective is *already* an additive-cost problem and iLQR applies with no augmentation
  machinery beyond carrying the scalar.
* **For `fpl_discounted`, `z_H` is literally `traj.reward_terms`** -- the per-objective
  [0,1] vector the multi-objective controllers already consume. The augmentation is not
  introducing a new quantity there, it is exposing one the scorer already computes.

Scope
-----
`fpl_layered` and `hybrid` are reachable only on `g1_standup` / `g1_walk`: hopper and
walker define neither `fpl_groups` (so `_score_fpl_layered` raises) nor
`floor_term_indices` (so `_score_hybrid` is bit-identical to `_score_normal`). They are
also excluded from `config.JAX_COST_MODES`. Both are therefore implemented and gated here
against the **numpy** scorer only; the 1e-6 JAX gate (G7) covers `normal`, `fpl_cost` in
all three `time_p` branches, and `fpl_discounted`.

Backend injection
-----------------
Rather than duplicating this file once per array library, the three backend primitives
that actually differ (`power_mean`'s dtype handling, `discount_weights`' in-place vs
functional update, the array module itself) are injected as `ArrayOps`. The caller passes
the *same* helpers its reference scorer uses, so the atom-axis collapse is bit-identical
to the scorer by construction and the gate isolates the part this module is responsible
for: the time decomposition. `numpy_ops()` and `jax_ops()` build the two bundles, and
neither is imported at module load -- invariant 3 (no JAX in the default import path)
holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

# Mirrors the eps baked into both `power_mean` implementations. The decomposition has to
# apply the SAME lower clip at the SAME point in the expression or parity fails in the
# tail, where an atom sits at the floor.
POWER_MEAN_EPS = 1e-8
#: The clip `_score_hybrid` applies inside its log-barrier (sampling_base.py:387).
BARRIER_EPS = 1e-8


@dataclass(frozen=True)
class ArrayOps:
    """The backend primitives whose numpy and jnp spellings genuinely differ."""

    xp: Any                          # numpy or jax.numpy
    power_mean: Callable             # tasks.base.power_mean or tasks.jax_costs._base's
    discount_weights: Callable       # ditto

    def asarray(self, x):
        return self.xp.asarray(x)


def numpy_ops() -> ArrayOps:
    import numpy as np

    from analytic_mppi.tasks.base import discount_weights, power_mean

    return ArrayOps(xp=np, power_mean=power_mean, discount_weights=discount_weights)


def jax_ops() -> ArrayOps:
    import jax.numpy as jnp

    from analytic_mppi.tasks.jax_costs._base import discount_weights, power_mean

    return ArrayOps(xp=jnp, power_mean=power_mean, discount_weights=discount_weights)


def _normalized(ops: ArrayOps, weights, n: int):
    """The weight vector `power_mean` effectively uses, made explicit.

    `power_mean(terms, p, weights=None)` divides by `n` after summing; with weights it
    normalizes by `sum(w)` first. Both are `sum_t w_hat_t * g(m_t)` with `w_hat` summing
    to 1, which is exactly the form the accumulator needs, so build `w_hat` once here and
    use the weighted branch uniformly. (The two differ only in float association order --
    parts in 1e-16, against a 1e-6 gate.)
    """
    xp = ops.xp
    if weights is None:
        return xp.full((n,), 1.0 / n)
    w = xp.asarray(weights)
    return w / w.sum()


class AugmentedObjective:
    """One objective mode, split into an additive stage and a terminal readout.

    Build with `build_augmented_objective`. Three surfaces:

      * `init(lead_shape)`   -> `z_0`, shape `(*lead_shape, n_acc)`
      * `stage(t, ...)`      -> `dz` for a running step `t in [0, H)`, shape `(*lead, n_acc)`
      * `terminal(z, ...)`   -> the scalar score, shape `(*lead,)`

    `stage` is the iLQR stage cost's state-update half; `terminal` folds in the terminal
    atoms (a function of `x_H`) and reads out. Every reduction uses negative axes, so any
    leading batch shape works.
    """

    def __init__(self, *, ops: ArrayOps, mode: str, n_acc: int, stage_weights,
                 terminal_weight, generator: str, readout_scale: float,
                 collapse: Optional[Callable], atom_weights, p: float,
                 n_run: int, n_term: int, dt: float, groups, group_p,
                 floor_indices, floor_weight: float, per_term_terminal_weight,
                 p_time: float = 1.0):
        self._ops = ops
        self.mode = mode
        self.n_acc = int(n_acc)
        self._w = stage_weights                 # (H,) or (H, n_acc): per-step time weight
        self._w_term = terminal_weight          # scalar or (n_acc,): the H-th step's weight
        self._generator = generator             # "identity" | "power" | "log" | "sum"
        self._readout_scale = float(readout_scale)
        self._collapse = collapse
        self._atom_weights = atom_weights
        self._p = p
        self._n_run = int(n_run)
        self._n_term = int(n_term)
        self._dt = float(dt)
        self._groups = groups
        self._group_p = group_p
        self._floor_indices = floor_indices
        self._floor_weight = float(floor_weight)
        self._w_term_per_term = per_term_terminal_weight
        self._p_time = float(p_time)

    # ---- augmented state ----

    def init(self, lead_shape: Sequence[int] = ()):
        """`z_0`. Zero for every mode: each generator's accumulation starts empty."""
        return self._ops.xp.zeros(tuple(lead_shape) + (self.n_acc,))

    # ---- the additive half ----

    def stage(self, t, *, terms=None, terms_f=None):
        """Increment `dz` contributed by running step `t`. `z_{t+1} = z_t + dz`."""
        xp = self._ops.xp
        mode = self.mode

        if mode == "normal":
            return (terms.sum(axis=-1) * self._dt)[..., None]

        if mode == "hybrid":
            return (self._hybrid_step(terms, terms_f) * self._dt)[..., None]

        if mode in ("fpl_discounted", "fpl_layered"):
            # Purely linear accumulation, one slot per atom. The weight is per (t, atom)
            # because under `fpl_discounted` atoms WITHOUT a terminal value live on a
            # shorter series with its own normalization.
            return self._w[t] * terms_f

        # fpl_cost: collapse the atom axis first, then apply the time generator.
        m = self._collapse(terms_f, self._atom_weights)          # (*lead,)
        return (self._w[t] * self._g(m))[..., None]

    # ---- the terminal readout ----

    def terminal(self, z, *, terms=None, terms_f=None):
        """Fold in the terminal atoms, then map `z` to the score (smaller is better)."""
        return self.readout(self.fold_terminal(z, terms=terms, terms_f=terms_f))

    def fold_terminal(self, z, *, terms=None, terms_f=None):
        """Fold the terminal step's atoms into the accumulator: `z_final`.

        Additive in `z` for every mode -- the terminal contribution never depends on the
        accumulated value -- which is what lets iLQR treat this as one more stage
        increment on the final step (it is a function of `(x_H, u_{H-1})`, exactly like a
        running stage) and keep `readout` as the only nonlinearity in `z`.
        """
        xp = self._ops.xp
        mode = self.mode

        if mode == "normal":
            return z + terms.sum(axis=-1)[..., None]

        if mode == "hybrid":
            # Same target/floor split as the stage, recomputed from the TERMINAL shapes:
            # floors index normal terms, and only floors with a terminal fulfillment
            # (i < n_term_f) contribute a terminal barrier (sampling_base.py:389-393).
            n = terms.shape[-1]
            floors = [i for i in self._floor_indices if 0 <= i < n]
            targets = [i for i in range(n) if i not in floors]
            out = terms[..., targets].sum(axis=-1)
            n_term_f = 0 if terms_f is None else terms_f.shape[-1]
            term_floors = [i for i in floors if i < n_term_f]
            if term_floors:
                bt = -xp.log(xp.clip(terms_f[..., term_floors], BARRIER_EPS, 1.0))
                out = out + self._floor_weight * bt.sum(axis=-1)
            return z + out[..., None]

        if mode == "fpl_discounted":
            if self._n_term > 0:
                # Only the first n_term slots receive a terminal contribution; the rest
                # already closed their (shorter) series at step H-1.
                pad = xp.zeros(terms_f.shape[:-1] + (self._n_run - self._n_term,))
                z = z + xp.concatenate(
                    [self._w_term_per_term * terms_f, pad], axis=-1)
            return z

        if mode == "fpl_layered":
            return z + self._w_term_per_term * terms_f    # 1:1 alignment, always present

        # fpl_cost
        if self._n_term > 0:
            w_term = None if self._atom_weights is None else self._atom_weights[:self._n_term]
            m_term = self._collapse(terms_f, w_term)
            z = z + (self._w_term * self._g(m_term))[..., None]
        return z

    def readout(self, z_final):
        """Map the fully folded accumulator to the score. Pure function of `z_final` --
        for iLQR this is the entire terminal cost, and its derivatives never touch the
        physics (the value function's z-rows are analytic)."""
        xp = self._ops.xp
        mode = self.mode

        if mode in ("normal", "hybrid"):
            return z_final[..., 0]

        if mode == "fpl_discounted":
            reward = self._ops.power_mean(z_final, self._p, weights=self._atom_weights)
            return -xp.log(reward)

        if mode == "fpl_layered":
            inner_p = self._p if self._group_p is None else self._group_p
            group_scores = xp.stack(
                [self._ops.power_mean(z_final[..., idx], inner_p) for idx in self._groups],
                axis=-1)
            reward = self._ops.power_mean(group_scores, self._p,
                                          weights=self._atom_weights)
            return -xp.log(reward)

        # fpl_cost
        return self._readout(z_final[..., 0])

    # ---- generator and its inverse, composed with the -log cost bridge ----

    def _g(self, m):
        """The quasi-arithmetic generator applied to a per-step composite."""
        xp = self._ops.xp
        if self._generator == "identity":
            return m
        t = xp.clip(m, POWER_MEAN_EPS, None)       # power_mean clips before g, so do the same
        if self._generator == "log":
            return xp.log(t)
        return t ** self._p_time

    def _readout(self, acc):
        """`-log(g^-1(acc))`, simplified per generator.

        `time_p = None` -> reward = acc                     -> -log(acc)
        `time_p = q!=0` -> reward = acc**(1/q)              -> -(1/q) log(acc)
        `time_p = 0`    -> reward = exp(acc)                -> -acc      (exactly linear)

        NOTE: `NEXT_STEPS.md` S9's table gives the terminal as `-z^(1/p)`. That is the
        reward-maximizing convention; this codebase's scorer returns `-log(reward)`
        (sampling_base.py:495). Implementing the table literally scores a different
        objective than the numpy scorer and fails G7 for a reason unrelated to the
        decomposition.
        """
        xp = self._ops.xp
        if self._generator == "log":
            return -acc
        return self._readout_scale * xp.log(acc)

    # ---- helpers ----

    def _hybrid_step(self, terms, terms_f):
        xp = self._ops.xp
        n = terms.shape[-1]
        floors = [i for i in self._floor_indices if 0 <= i < n]
        targets = [i for i in range(n) if i not in floors]
        out = terms[..., targets].sum(axis=-1)
        if floors:
            barrier = -xp.log(xp.clip(terms_f[..., floors], BARRIER_EPS, 1.0))
            out = out + self._floor_weight * barrier.sum(axis=-1)
        return out

    # ---- convenience: run the recurrence end to end (parity / verification) ----

    def score(self, *, running_terms=None, terminal_terms=None,
              running_terms_f=None, terminal_terms_f=None):
        """Scan the recurrence over the horizon and read out. Shapes match the scorers.

        This is the reference driver used by the G7 parity test. Production iLQR does not
        call it -- it consumes `stage`/`terminal` directly inside its own `lax.scan`, with
        `z` living in the augmented state.
        """
        src = running_terms if running_terms is not None else running_terms_f
        H = src.shape[-2]
        z = self.init(src.shape[:-2])
        for t in range(H):
            z = z + self.stage(
                t,
                terms=None if running_terms is None else running_terms[..., t, :],
                terms_f=None if running_terms_f is None else running_terms_f[..., t, :],
            )
        return self.terminal(z, terms=terminal_terms, terms_f=terminal_terms_f)


def build_augmented_objective(
    *, ops: ArrayOps, mode: str, H: int, n_run: int, n_term: int,
    p: float = -1.0, gamma: float = 0.99, time_p: Optional[float] = None,
    time_discount: bool = False, terminal_value: bool = False,
    weights=None, dt: float = 1.0, collapse: Optional[Callable] = None,
    groups=None, group_p: Optional[float] = None,
    floor_indices: Sequence[int] = (), floor_weight: float = 1.0,
) -> AugmentedObjective:
    """Build the decomposition for one objective mode.

    `collapse(atoms, weights) -> composite` is the atom-axis reduction; pass the
    controller's `_collapse_objectives` to inherit a conjunction split, or leave it None
    for the flat `power_mean(atoms, p, weights)` the JAX path uses.
    """
    xp = ops.xp
    if n_term > n_run:
        raise ValueError(f"n_term={n_term} > n_run={n_run}; expected n_term <= n_run")
    if H <= 0:
        raise ValueError(f"H must be positive, got {H}")

    atom_weights = None if weights is None else xp.asarray(weights)
    if collapse is None:
        def collapse(atoms, w):
            return ops.power_mean(atoms, p, weights=w)

    common = dict(
        ops=ops, mode=mode, collapse=collapse, atom_weights=atom_weights, p=p,
        n_run=n_run, n_term=n_term, dt=dt, groups=groups, group_p=group_p,
        floor_indices=list(floor_indices), floor_weight=floor_weight,
    )

    if mode in ("normal", "hybrid"):
        # Already additive: the "generator" is the identity and the readout is a plain
        # sum. Carried through this class anyway so every mode presents one interface to
        # iLQR and the block-triangularity assertion covers all of them uniformly.
        return AugmentedObjective(
            n_acc=1, stage_weights=None, terminal_weight=None, generator="sum",
            readout_scale=1.0, per_term_terminal_weight=None, **common)

    if mode in ("fpl_discounted", "fpl_layered"):
        # One slot per atom, weights differing per atom only because atoms without a
        # terminal value run on a shorter, separately normalized series.
        disc_full = ops.discount_weights(H + 1, gamma, terminal_value)   # (H+1,)
        disc_run = ops.discount_weights(H, gamma, terminal_value)        # (H,)
        if mode == "fpl_layered":
            if not groups:
                raise ValueError("fpl_layered needs a non-empty `groups` partition")
            if n_term != n_run:
                raise ValueError(
                    f"layered FPL expects terminal atoms aligned 1:1 with running atoms; "
                    f"got {n_term} terminal vs {n_run} running")
            w_stage = xp.broadcast_to(disc_full[:H, None], (H, n_run))
            w_term = disc_full[H]
        else:
            cols = []
            for j in range(n_run):
                cols.append(disc_full[:H] if j < n_term else disc_run)
            w_stage = xp.stack(cols, axis=-1)                            # (H, n_run)
            w_term = disc_full[H]
        obj = AugmentedObjective(
            n_acc=n_run, stage_weights=w_stage, terminal_weight=None,
            generator="identity", readout_scale=1.0,
            per_term_terminal_weight=w_term, **common)
        return obj

    if mode != "fpl_cost":
        raise ValueError(
            f"unknown objective mode {mode!r} "
            f"(normal | fpl_cost | fpl_discounted | fpl_layered | hybrid)")

    # --- fpl_cost: one scalar accumulator, generator set by `time_p` ---
    series = H + 1 if n_term > 0 else H
    disc = ops.discount_weights(series, gamma, terminal_value)

    if time_p is None:
        # Discounted arithmetic mean: generator is the identity, weights are the
        # (already normalized) discounts themselves -- `_score_fpl` applies no extra norm.
        w_hat = disc
        generator, readout_scale = "identity", -1.0
        p_time = 1.0
    else:
        # Power mean over time. `time_discount=False` -> UNWEIGHTED, i.e. uniform 1/series
        # and gamma plays no role at all. That is the published config
        # (fpl_time_discount=False, time_p=-2.0), which is why G3 found no (1-g)/(1-g^H)
        # anywhere in the objective WO-3.3 set out to fix.
        w_hat = _normalized(ops, disc if time_discount else None, series)
        if time_p == 0:
            generator, readout_scale, p_time = "log", 1.0, 0.0
        else:
            generator, readout_scale, p_time = "power", -1.0 / time_p, float(time_p)

    return AugmentedObjective(
        n_acc=1, stage_weights=w_hat[:H], terminal_weight=(w_hat[H] if n_term > 0 else None),
        generator=generator, readout_scale=readout_scale,
        per_term_terminal_weight=None, p_time=p_time, **common)
