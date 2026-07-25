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
    z: float = 0.06,
    clear: bool = True,
) -> None:
    """Append goal marker + the single best (highest-weight) rollout to `scene`.

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
    best = int(np.argmax(weights))
    idx = list(xy_idx)
    xy = samples[best][..., idx]                  # (H, 2)
    rgba = [1.0, 0.55, 0.0, 0.95]
    H = xy.shape[0]
    for t in range(H - 1):
        ok = _append_line(
            scene,
            [xy[t, 0], xy[t, 1], z],
            [xy[t + 1, 0], xy[t + 1, 1], z],
            rgba,
            width=3.0,
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
    """Run closed-loop MPPI with the live MuJoCo viewer (real-time paced).

    Shuts down `mujoco.rollout`'s thread pool before the viewer exits to avoid
    a teardown race that segfaults at close on Linux/GLFW.
    """
    import mujoco.viewer as mj_viewer
    from mujoco import rollout as mj_rollout

    if draw_rollouts:
        controller.store_samples = True

    state = backend.get_state()
    if on_step is not None:
        on_step(0, state, None)

    # Hide the left (settings) and right (info) UI panels — toggle back at runtime with Tab.
    viewer = mj_viewer.launch_passive(backend.model, backend.data,
                                      show_left_ui=False, show_right_ui=False)
    try:
        dt = backend.dt
        for step in range(n_steps):
            if not viewer.is_running():
                break
            t0 = time.perf_counter()

            # Rollouts touch only thread-local MjData (not backend.data) and
            # read the const model -- safe to run outside viewer.lock().
            u = controller.act(state, cost_fn, backend)

            # backend.step mutates backend.data, and viewer.user_scn is read
            # by the render thread -- both must be protected by viewer.lock().
            with viewer.lock():
                state = backend.step(u)
                _draw_overlays(
                    viewer.user_scn,
                    goal_xy,
                    samples=controller.last_samples if draw_rollouts else None,
                    weights=controller.last_weights if draw_rollouts else None,
                    xy_idx=xy_idx,
                )

            if on_step is not None:
                on_step(step + 1, state, u)
            viewer.sync()

            elapsed = time.perf_counter() - t0
            sleep_for = dt - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        pass
    finally:
        # Order matters: tear down the rollout thread pool BEFORE the viewer
        # (and before the MjModel ref count drops at scope exit). Otherwise
        # rollout's atexit handler can fire after the model is freed.
        try:
            mj_rollout.shutdown_persistent_pool()
        except Exception:
            pass
        try:
            viewer.close()
        except Exception:
            pass


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
