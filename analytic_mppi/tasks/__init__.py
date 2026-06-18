"""Task registry for the CPU sampling-based MPC framework."""
from __future__ import annotations

from typing import Dict, Type

from .base import Task, power_mean
from .pendulum import PendulumTask
from .walker import WalkerTask
from .cube import CubeRotationTask
from .g1_standup import G1StandupTask
from .hopper import HopperTask


TASKS: Dict[str, Type[Task]] = {
    "pendulum": PendulumTask,
    "walker": WalkerTask,
    "cube": CubeRotationTask,
    "g1_standup": G1StandupTask,
    "hopper": HopperTask,
}


def make_task(name: str, **kwargs) -> Task:
    """Instantiate a Task by registry name."""
    if name not in TASKS:
        raise KeyError(
            f"Unknown task {name!r}. Known: {sorted(TASKS)}"
        )
    return TASKS[name](**kwargs)


__all__ = [
    "Task",
    "PendulumTask",
    "WalkerTask",
    "CubeRotationTask",
    "G1StandupTask",
    "HopperTask",
    "TASKS",
    "make_task",
    "power_mean",
]
