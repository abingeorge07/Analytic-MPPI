"""MuJoCo-based visualization for closed-loop MPPI runs.

Two entry points:
  * run_live(...)   -- interactive `mujoco.viewer.launch_passive` window,
                       optionally with MPPI sample-rollout overlays.
  * run_record(...) -- headless offscreen rendering to mp4/gif via
                       mujoco.Renderer + imageio.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import mujoco


# ---- low-level helpers to append geoms onto an MjvScene -----------------

def _append_line(scene, p0, p1, rgba, width: float = 3.0) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        int(mujoco.mjtGeom.mjGEOM_LINE),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g,
        int(mujoco.mjtGeom.mjGEOM_LINE),
        float(width),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scene.ngeom += 1
    return True


def _append_sphere(scene, pos, radius: float, rgba) -> bool:
    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g,
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1
    return True


def _draw_overlays(
    scene,
    goal_xy,
    samples: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    xy_idx: Sequence[int] = (1, 2),
    top_n: int = 32,
    z: float = 0.06,
    clear: bool = True,
) -> None:
    """Append goal marker + top-N sample rollouts to `scene`.

    `clear=True` resets scene.ngeom first -- correct for `viewer.user_scn`
    (a dedicated user-geom scene). For `Renderer.scene`, pass `clear=False`
    so the floor / robot geoms populated by `update_scene` are preserved.
    """
    if clear:
        scene.ngeom = 0
    _append_sphere(scene, [goal_xy[0], goal_xy[1], z], 0.08,
                   [0.95, 0.15, 0.15, 0.95])
    if samples is None or weights is None:
        return
    K, H, _ = samples.shape
    idx = list(xy_idx)
    n = min(top_n, K)
    order = np.argpartition(-weights, n - 1)[:n] if n < K else np.arange(K)
    xy = samples[order][..., idx]                 # (n, H, 2)
    w = weights[order]
    w_norm = w / max(float(w.max()), 1e-12)
    for s in range(n):
        alpha = 0.12 + 0.65 * float(w_norm[s])
        rgba = [1.0, 0.55, 0.0, alpha]
        for t in range(H - 1):
            ok = _append_line(
                scene,
                [xy[s, t, 0], xy[s, t, 1], z],
                [xy[s, t + 1, 0], xy[s, t + 1, 1], z],
                rgba,
                width=2.5,
            )
            if not ok:
                return


# ---- public entry points ------------------------------------------------

def run_live(
    backend,
    controller,
    cost_fn: Callable,
    goal_xy,
    n_steps: int,
    *,
    xy_idx: Sequence[int] = (1, 2),
    draw_rollouts: bool = False,
    on_step: Callable | None = None,
) -> None:
    """Run closed-loop MPPI with the live MuJoCo viewer (real-time paced)."""
    import mujoco.viewer as mj_viewer

    if draw_rollouts:
        controller.store_samples = True

    state = backend.get_state()
    if on_step is not None:
        on_step(0, state, None)

    with mj_viewer.launch_passive(backend.model, backend.data) as viewer:
        dt = backend.dt
        for step in range(n_steps):
            if not viewer.is_running():
                break
            t0 = time.perf_counter()
            u = controller.act(state, cost_fn, backend)
            state = backend.step(u)
            if on_step is not None:
                on_step(step + 1, state, u)

            _draw_overlays(
                viewer.user_scn,
                goal_xy,
                samples=controller.last_samples if draw_rollouts else None,
                weights=controller.last_weights if draw_rollouts else None,
                xy_idx=xy_idx,
            )
            viewer.sync()

            elapsed = time.perf_counter() - t0
            sleep_for = dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


def run_record(
    backend,
    controller,
    cost_fn: Callable,
    goal_xy,
    n_steps: int,
    save_path,
    *,
    xy_idx: Sequence[int] = (1, 2),
    draw_rollouts: bool = False,
    width: int = 720,
    height: int = 540,
    fps: int | None = None,
    camera: str | int = "topdown",
    on_step: Callable | None = None,
) -> Path:
    """Closed-loop MPPI rendered headlessly to mp4/gif.

    File extension drives format: `.mp4` / `.m4v` use ffmpeg via imageio;
    anything else (e.g. `.gif`) uses imageio's default writer.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise RuntimeError(
            "Recording needs imageio. Install with:  pip install -e \".[viz]\""
        ) from e

    if draw_rollouts:
        controller.store_samples = True

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if fps is None:
        fps = max(1, int(round(1.0 / backend.dt)))

    state = backend.get_state()
    if on_step is not None:
        on_step(0, state, None)

    frames: list[np.ndarray] = []
    renderer = mujoco.Renderer(backend.model, width=width, height=height)
    try:
        for step in range(n_steps):
            u = controller.act(state, cost_fn, backend)
            state = backend.step(u)
            if on_step is not None:
                on_step(step + 1, state, u)
            renderer.update_scene(backend.data, camera=camera)
            _draw_overlays(
                renderer.scene,
                goal_xy,
                samples=controller.last_samples if draw_rollouts else None,
                weights=controller.last_weights if draw_rollouts else None,
                xy_idx=xy_idx,
                clear=False,  # preserve world geoms set by update_scene
            )
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    suffix = save_path.suffix.lower()
    if suffix in (".mp4", ".m4v"):
        imageio.mimsave(save_path, frames, fps=fps, codec="libx264", quality=8,
                        macro_block_size=None)
    else:
        imageio.mimsave(save_path, frames, fps=fps)
    return save_path
