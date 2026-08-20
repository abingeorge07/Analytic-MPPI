"""Render the feasibility jump (no obstacle yet) to an mp4 — visual proof Barkour can jump
under sampling MPC. Reuses JumpProbe from quadruped_jump_probe.py.

Run: python verification/com_study/quadruped_jump_render.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio

from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.controllers import MPPIv2
from analytic_mppi.eval import init_barkour_stand
from quadruped_jump_probe import JumpProbe


def main():
    task = JumpProbe(jump_h=0.5)
    backend = MujocoBackend(task.model_path, nthread=8)
    ctrl = MPPIv2(task, backend, num_samples=384, num_knots=8, plan_horizon=0.5,
                  spline_type="zero", noise_level=0.9, temperature=0.08, seed=0)
    init_barkour_stand(backend)
    state = backend.get_state()

    cam = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    renderer = mujoco.Renderer(backend.model, width=640, height=400)
    frames = []
    try:
        for _ in range(260):
            u = ctrl.act(state)
            state = backend.step(u)
            renderer.update_scene(backend.data, camera=(cam if cam >= 0 else -1))
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    out = Path(__file__).resolve().parent / "quadruped_jump.mp4"
    fps = int(round(1.0 / backend.dt))
    imageio.mimsave(str(out), frames, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    print(f"saved -> {out}  ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
