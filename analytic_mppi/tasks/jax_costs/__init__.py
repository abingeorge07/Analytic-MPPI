"""JAX (jnp) reimplementations of task cost functions, for the MJX planning backend.

Why a PARALLEL implementation rather than making `tasks/*.py` array-module-agnostic:
the numpy cost code is the definition of every published CPU result, and it uses
patterns that do not survive `jit`/`grad` (in-place `out[..., -1] = ...`, python-float
`np.where` branches, `np.asarray(dtype=float64)` coercions). Rewriting it in place to be
dual-mode risks silently changing a number in a result nobody would re-derive. Instead the
jnp version lives here and is pinned to the numpy version by parity tests
(`tests/test_jax_costs.py`, atol 1e-6) -- if they drift, a test fails rather than a paper
number quietly moving.

`has_jax_costs` is deliberately JAX-FREE (a name lookup against a frozenset) so
`config.resolve()` can reject `run.backend="mjx"` on an unsupported task without importing
jax -- config validation must work on a machine that never installed the `[mjx]` extra.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # pragma: no cover - typing only, never imported
    from analytic_mppi.tasks.base import Task


# Tasks with a jnp cost implementation in this package. Scope decision (see
# docs/mjx_gradient_mpc.md): the MJX/gradient path covers the walking/standing tasks the
# comparison needs, not the whole task registry.
JAX_COST_TASKS: frozenset[str] = frozenset({
    "hopper",
    "walker",
    "g1_standup",
    "g1_walk",
    "pendulum",
})


def has_jax_costs(task_name: str) -> bool:
    """Does `task_name` have a jnp cost implementation? Importable without jax."""
    return task_name in JAX_COST_TASKS


def make_jax_costs(task: "Task"):
    """Build the jnp cost object mirroring `task`'s numpy costs.

    Imports jax lazily -- only callers that actually plan on MJX pay for it. The returned
    object reads its constants (weights, sensor addresses, targets) FROM the live task, so
    a config's `task.kwargs` flow through automatically.

    Dispatch is by exact class (not isinstance): a new Task subclass must get its own
    entry here -- silently inheriting a parent's cost mirror would score the wrong
    objective (G1ReachTask overrides an atom, so it must NOT fall back to G1WalkJaxCosts).
    """
    from analytic_mppi.tasks.hopper import HopperTask
    from analytic_mppi.tasks.walker import WalkerTask
    from analytic_mppi.tasks.g1_standup import G1StandupTask
    from analytic_mppi.tasks.g1_walk import G1WalkTask
    from analytic_mppi.tasks.pendulum import PendulumTask

    from .hopper import HopperJaxCosts
    from .walker import WalkerJaxCosts
    from .g1 import G1StandupJaxCosts, G1WalkJaxCosts
    from .pendulum import PendulumJaxCosts

    registry = {
        HopperTask: HopperJaxCosts,
        WalkerTask: WalkerJaxCosts,
        G1StandupTask: G1StandupJaxCosts,
        G1WalkTask: G1WalkJaxCosts,
        PendulumTask: PendulumJaxCosts,
    }
    cls = registry.get(type(task))
    if cls is None:
        raise TypeError(
            f"no jnp cost mirror for {type(task).__name__} "
            f"(supported: {sorted(t.__name__ for t in registry)})"
        )
    return cls(task)
