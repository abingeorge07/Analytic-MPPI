from analytic_mppi.dynamics.mujoco_backend import MujocoBackend

__all__ = ["MujocoBackend", "GymBackend"]


def __getattr__(name):
    # Lazy import so `import analytic_mppi.dynamics` does not require gymnasium/Box2D
    # (only the LunarLander env pulls them in).
    if name == "GymBackend":
        from analytic_mppi.dynamics.gym_backend import GymBackend
        return GymBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
