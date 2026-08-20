"""Side-by-side comparison videos: FPL vs conventional, synchronized in one frame.

The paper's money shot. For each environment we render THREE controllers running the SAME
task from the SAME seed, composited left-to-right into a single synchronized clip:

    [ FPL (min-fulfillment) ] [ best-SAFE linear ] [ aggressive linear ]

Everything is identical across panels except the objective composition (same MPPI sampler,
budget, horizon, atoms, temporal weakest-link). The story the viewer sees directly:
  * FPL is as upright as the safe linear weight AND clearly faster/further.
  * The aggressive linear weight (tuned to match FPL's speed) falls / drops — the price the
    linear family pays to go fast. No single linear weight is both fast and safe; FPL is both.

A live caption tracks each panel's measured progress + upright/hold so the numbers match the
pixels. Writes runs/paper/{env}_sidebyside.mp4.

Run:  .venv/bin/python verification/fpl_vs_linear_video.py [env ...]
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import ENVS as ENVSPEC, sampler_kwargs, cost_kwargs, linear_weights  # noqa: E402
from analytic_mppi.eval import make_controller  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "runs" / "paper"
_FONT = fm.findfont("DejaVu Sans")
_FONT_B = fm.findfont("DejaVu Sans:bold")
PANEL_W = 420

# Per-env render config: camera, panel height, #steps (>=~8s playback), slow-motion, and the
# two linear weights to contrast (safe vs aggressive). FPL is always fpl_p=-1.
VIDEO = {
    "walker": dict(height=460, steps=500, slowmo=1.0, camera="floating", cam_cfg=None,
                   safe_wv=4.0, aggr_wv=16.0, note=""),
    "hopper": dict(height=460, steps=200, slowmo=2.5, camera="track", cam_cfg=None,
                   safe_wv=0.5, aggr_wv=2.0, note="2.5x slow-motion"),
    "cube": dict(height=360, steps=700, slowmo=1.0, camera=None,
                 cam_cfg=None, safe_wv=1.0, aggr_wv=8.0, note=""),
    "quadruped": dict(height=440, steps=3000, slowmo=1.0, camera=None,
                      cam_cfg=dict(track="chassis", azimuth=120, elevation=-8, distance=1.8),
                      safe_wv=1.0, aggr_wv=8.0, note=""),
}


def _fit_font(text, font_path, max_size, width, min_size=9):
    for sz in range(max_size, min_size - 1, -1):
        f = ImageFont.truetype(font_path, sz)
        if f.getlength(text) <= width - 12:
            return f
    return ImageFont.truetype(font_path, min_size)


def _panel_controller(env, cost_label):
    """Build a (task, backend, ctrl) for one panel. cost_label in {'fpl','safe','aggr'}."""
    spec = ENVSPEC[env]
    vid = VIDEO[env]
    if cost_label == "fpl":
        cost = "fpl"
    elif cost_label == "safe":
        cost = f"lin:{vid['safe_wv']}"
    else:
        cost = f"lin:{vid['aggr_wv']}"
    build = dict(num_samples=256, plan_horizon=spec["horizon"], num_knots=spec["knots"],
                 spline_type="zero", **sampler_kwargs("mppi", env, 256), **cost_kwargs(cost, env))
    task, backend, ctrl = make_controller(
        spec["task"], "mppi", cost_mode="fpl_cost", seed=0,
        task_kwargs={spec["difficulty_key"]: spec["difficulty"]}, **build)
    if spec["init"] is not None:
        spec["init"](backend)
    return task, backend, ctrl


def _make_cam(backend, env):
    vid = VIDEO[env]
    cc = vid.get("cam_cfg")
    if cc is not None:
        bid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, cc["track"])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = bid
        cam.azimuth = cc["azimuth"]; cam.elevation = cc["elevation"]; cam.distance = cc["distance"]
        return cam
    if vid["camera"] is not None:
        cid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, vid["camera"])
        return cid if cid >= 0 else -1
    return -1


def _progress_str(env, task, sd_row):
    """A short live read of this panel's outcome from the latest sensordata row."""
    if ENVSPEC[env]["metric"] in ("locomotion",):
        vx = sd_row[task._vel_adr]; up = sd_row[task._zax_adr + 2]
        return f"vx={vx:+.2f}  up={up:.2f}"
    if ENVSPEC[env]["metric"] == "quadruped":
        return f"vx={task._torso_vel_x(sd_row):+.2f}  up={task._torso_up(sd_row):.2f}"
    ang = np.degrees(task._angle_err(sd_row[None]))[0]
    rot = np.degrees(task.target_angle) - ang
    return f"rot={rot:.0f}deg  drift={task._hold_xy(sd_row[None])[0]:.02f}"


def render(env):
    spec = ENVSPEC[env]; vid = VIDEO[env]
    panels = [("FPL (min-fulfillment)", "fpl"),
              (f"best-safe linear (wv={vid['safe_wv']:g})", "safe"),
              (f"aggressive linear (wv={vid['aggr_wv']:g})", "aggr")]
    H = vid["height"]
    stride = 1
    # run each panel, collect frames + per-frame progress captions
    all_frames, all_caps, tasks = [], [], []
    for _title, cost_label in panels:
        task, backend, ctrl = _panel_controller(env, cost_label)
        cam = _make_cam(backend, env)
        renderer = mujoco.Renderer(backend.model, width=PANEL_W, height=H)
        state = backend.get_state()
        dt = float(backend.dt)
        sub = max(1, int(round((1.0 / dt) / 45.0)))
        frames, caps = [], []
        try:
            for t in range(vid["steps"]):
                u = ctrl.act(state); state = backend.step(u)
                if t % sub == 0:
                    renderer.update_scene(backend.data, camera=cam)
                    frames.append(renderer.render().copy())
                    caps.append(_progress_str(env, task, np.asarray(backend.data.sensordata)))
        finally:
            renderer.close()
        all_frames.append(frames); all_caps.append(caps); tasks.append(task)
        print(f"  {env}: rendered {cost_label} ({len(frames)} frames)")

    n = min(len(f) for f in all_frames)
    # composite frame-by-frame
    title_h = 46
    banner_h = 30
    W = PANEL_W * len(panels)
    tfont = _fit_font("aggressive linear (wv=16)", _FONT_B, 18, PANEL_W)
    cfont = ImageFont.truetype(_FONT, 15)
    out = []
    for i in range(n):
        canvas = Image.new("RGB", (W, H + title_h + banner_h), (17, 17, 20))
        d = ImageDraw.Draw(canvas)
        for p, (title, _c) in enumerate(panels):
            x0 = p * PANEL_W
            im = Image.fromarray(all_frames[p][i]).convert("RGB")
            canvas.paste(im, (x0, title_h))
            col = (240, 120, 110) if p == 0 else (150, 170, 210)
            d.text((x0 + 8, 12), title, font=tfont, fill=col)
            cap = all_caps[p][i]
            d.rectangle([x0, title_h + H, x0 + PANEL_W, title_h + H + banner_h], fill=(0, 0, 0))
            d.text((x0 + 8, title_h + H + 6), cap, font=cfont, fill=(230, 230, 235))
            if p > 0:
                d.line([(x0, title_h), (x0, title_h + H)], fill=(60, 60, 66), width=2)
        out.append(np.asarray(canvas))

    # figure out fps from the model dt of the first panel
    _t, b0, _c = _panel_controller(env, "fpl")
    dt0 = float(b0.dt); sub = max(1, int(round((1.0 / dt0) / 45.0)))
    fps = max(1, int(round((1.0 / dt0) / sub / vid["slowmo"])))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{env}_sidebyside.mp4"
    imageio.mimsave(str(path), out, fps=fps, codec="libx264", quality=8, macro_block_size=None)
    print(f"saved {path}  ({n} frames @ {fps} fps{', ' + vid['note'] if vid['note'] else ''})")
    return path


def main():
    envs = sys.argv[1:] or ["walker", "hopper"]
    for env in envs:
        print(env)
        render(env)


if __name__ == "__main__":
    main()
