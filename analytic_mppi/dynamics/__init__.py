from analytic_mppi.dynamics.mujoco_backend import MujocoBackend, apply_perturbation

__all__ = ["MujocoBackend", "apply_perturbation", "MJXBackend"]


def __getattr__(name):
    # MJXBackend imports jax; load it lazily so the CPU path never pays for (or
    # requires) the [mjx] extra.
    if name == "MJXBackend":
        from analytic_mppi.dynamics.mjx_backend import MJXBackend
        return MJXBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
