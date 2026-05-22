from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_run(
    states_history: np.ndarray,
    controls_history: np.ndarray,
    goal_xy: Sequence[float],
    xy_idx: Sequence[int] = (1, 2),
    theta_idx: int | None = 3,
    save_path: str | Path | None = None,
):
    """Plot a closed-loop run: XY trajectory + heading arrows + control time series."""
    xy = states_history[:, list(xy_idx)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.6, color="tab:blue", label="trajectory")
    ax.plot(xy[0, 0], xy[0, 1], "o", ms=8, color="black", label="start")
    ax.plot(goal_xy[0], goal_xy[1], "*", ms=16, color="tab:red", label="goal")
    if theta_idx is not None:
        step = max(1, len(states_history) // 20)
        for i in range(0, len(states_history), step):
            th = states_history[i, theta_idx]
            ax.arrow(xy[i, 0], xy[i, 1], 0.10 * np.cos(th), 0.10 * np.sin(th),
                     head_width=0.04, color="tab:blue", alpha=0.5,
                     length_includes_head=True)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Trajectory")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    t = np.arange(controls_history.shape[0])
    for i in range(controls_history.shape[1]):
        ax.plot(t, controls_history[:, i], lw=1.2, label=f"u[{i}]")
    ax.set_xlabel("step"); ax.set_ylabel("control")
    ax.set_title("Applied controls")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
