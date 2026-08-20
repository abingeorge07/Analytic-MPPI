"""Supplementary videos for the paper: for each of the four winning environments, one clip of
the best FPL spec and one of the best *safe* linear-weight MPPI baseline (the strongest linear
weight that still survives ~always — a fair "both stay up" comparison, not the aggressive weight
that face-plants). Identical MPPIv2 sampler / atoms / budget / horizon for both clips of a pair;
only the scalarization differs (FPL = power-mean p=-1 conjunction; linear = p=1 weighted sum).

Every clip uses the exact study parameters (num_samples / horizon / knots / noise / temperature /
fpl_time_p) and difficulty from that env's `*_pareto_sweep.py`, the model's tracking camera so a
translating robot stays framed, and a caption with this clip's own measured outcome. All clips
are >=10 s (frames subsampled to ~45 fps); the hopper is a labelled 3.3x slow-motion because
sustained fast hopping is inherently unstable for this short-horizon MPC.

Writes runs/paper/{env}_fpl.mp4 and runs/paper/{env}_linear.mp4.

Run:  .venv/bin/python verification/paper_videos.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

from analytic_mppi.eval import (make_controller, init_hopper_stand, init_barkour_stand, init_cube)

OUT = Path(__file__).resolve().parent.parent / "runs" / "paper"
_FONT = fm.findfont("DejaVu Sans")
_FONT_B = fm.findfont("DejaVu Sans:bold")

# Per-env: study params (matched to the pareto sweep), difficulty, best-safe linear weight, the
# tracking camera, and the 30-seed paper numbers for the caption. `weights` puts the swept weight
# on the velocity/alignment atom (index 2), matching each sweep's build_configs.
# steps chosen for >=10 s of real playback (steps*dt), except the hopper: sustained fast
# hopping is inherently unstable for this short-horizon MPC (it hops vigorously for ~3 s then
# tumbles — that IS the dynamic-competition nature), so the hopper shows its real aggressive
# tv=2.0 regime as a labelled 3.3x slow-motion (3 s real -> 10 s). The others are real-time.
# The long-horizon "safe" linear is the CONSERVATIVE weight (quadruped w_v=1, not the study's
# short-horizon best-safe w_v=2 which tumbles over 11 s); both clips of a pair survive the clip.
ENVS = [
    dict(env="hopper", pretty="Hopper", init=init_hopper_stand, camera="track",
         steps=150, height=480, seed=3, slowmo=3.33, note="3.3x slow-motion",
         build=dict(num_samples=256, plan_horizon=0.6, num_knots=4, spline_type="zero",
                    noise_level=0.3, temperature=0.2, fpl_time_p=-2.0),
         task_kwargs=dict(target_velocity=2.0), cmd="2.0 m/s",
         lin_weights=[1.0, 1.0, 0.5, 0.5], lin_name="w_v=0.5",
         fpl_val="1.2 m/s hopping", lin_val="0.6 m/s"),
    dict(env="walker", pretty="Walker2d", init=None, camera="floating",
         steps=1100, height=480, seed=0, slowmo=1.0,
         build=dict(num_samples=256, plan_horizon=0.6, num_knots=6, spline_type="zero",
                    noise_level=0.8, temperature=0.2, fpl_time_p=-2.0),
         task_kwargs=dict(target_velocity=5.0), cmd="5.0 m/s",
         lin_weights=[1.0, 1.0, 4.0, 0.5], lin_name="w_v=4",
         fpl_val="1.7 m/s running", lin_val="0.4 m/s"),
    dict(env="quadruped", pretty="Barkour quadruped", init=init_barkour_stand, camera=None,
         cam_cfg=dict(track="chassis", azimuth=120, elevation=-8, distance=1.8),
         steps=5500, height=480, seed=0, slowmo=1.0,   # 11 s at the model's 0.002 dt
         build=dict(num_samples=256, plan_horizon=0.3, num_knots=5, spline_type="zero",
                    noise_level=0.5, temperature=0.2, fpl_time_p=-2.0),
         task_kwargs=dict(target_velocity=1.5), cmd="1.5 m/s",
         lin_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5], lin_name="w_v=1",
         fpl_val="1.2 m/s trotting", lin_val="0.8 m/s"),
    dict(env="cube", pretty="LEAP hand: in-hand cube", init=init_cube, camera=None,
         steps=1100, height=360, seed=0, slowmo=1.0,
         build=dict(num_samples=128, plan_horizon=0.3, num_knots=4, spline_type="zero",
                    noise_level=0.4, temperature=0.1, fpl_time_p=-2.0),
         task_kwargs=dict(target_angle=1.2), cmd="69 deg roll",
         lin_weights=[1.0, 1.0, 1.0, 0.5], lin_name="w_a=1",
         fpl_val="62 deg rolled", lin_val="18 deg rolled"),
]

WIDTH = 640


def _fit_font(text, font_path, max_size, width, min_size=10):
    """Largest font size (<= max_size) whose rendered `text` fits within `width`."""
    for sz in range(max_size, min_size - 1, -1):
        f = ImageFont.truetype(font_path, sz)
        if f.getlength(text) <= width - 16:
            return f
    return ImageFont.truetype(font_path, min_size)


def _caption(frames, line1, line2, width):
    """Draw a two-line caption in a translucent banner at the top of each frame (in place).
    Font sizes auto-shrink to fit the frame width so captions never overflow."""
    f1 = _fit_font(line1, _FONT_B, 20, width)
    f2 = _fit_font(line2, _FONT, 16, width)
    h1, h2 = f1.size, f2.size
    banner_h = 8 + h1 + 4 + h2 + 8
    out = []
    for arr in frames:
        im = Image.fromarray(arr).convert("RGB")
        d = ImageDraw.Draw(im, "RGBA")
        d.rectangle([0, 0, im.width, banner_h], fill=(0, 0, 0, 150))
        d.text((10, 8), line1, font=f1, fill=(255, 255, 255, 255))
        d.text((10, 8 + h1 + 4), line2, font=f2, fill=(205, 217, 235, 255))
        out.append(np.asarray(im))
    return out


def render_clip(spec, kind):
    """kind in {'fpl','linear'}: build the controller with the matched study params, run the
    closed loop with the env's tracking camera, caption, and save a real-time mp4."""
    is_fpl = kind == "fpl"
    algo = dict(**spec["build"])
    if is_fpl:
        algo.update(fpl_p=-1.0)
    else:
        algo.update(fpl_p=1.0, fpl_weights=spec["lin_weights"])
    task, backend, ctrl = make_controller(
        spec["env"], "mppi", cost_mode="fpl_cost", seed=spec["seed"],
        task_kwargs=spec["task_kwargs"], **algo)
    if spec["init"] is not None:
        spec["init"](backend)
    state = backend.get_state()

    # Camera: either a named model camera, the free camera (None), or a custom low tracking
    # camera (cam_cfg) that follows a body at a chosen azimuth/elevation/distance — the built-in
    # barkour 'track' cam sits 0.9 m up and looks down, which flattens the gait; a low side-on
    # tracking view shows the robot clearly walking on its legs.
    render_cam = -1
    cc = spec.get("cam_cfg")
    if cc is not None:
        bid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_BODY, cc["track"])
        render_cam = mujoco.MjvCamera()
        render_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        render_cam.trackbodyid = bid
        render_cam.azimuth = cc["azimuth"]
        render_cam.elevation = cc["elevation"]
        render_cam.distance = cc["distance"]
    elif spec["camera"] is not None:
        cid = mujoco.mj_name2id(backend.model, mujoco.mjtObj.mjOBJ_CAMERA, spec["camera"])
        render_cam = cid if cid >= 0 else -1

    renderer = mujoco.Renderer(backend.model, width=WIDTH, height=spec["height"])
    dt = float(backend.dt)
    slowmo = float(spec.get("slowmo", 1.0))
    stride = max(1, int(round((1.0 / dt) / 45.0)))       # subsample toward ~45 fps
    out_fps = max(1, int(round((1.0 / dt) / stride / slowmo)))  # slowmo<-> lower fps = stretched
    frames, sd_hist = [], []
    try:
        for t in range(spec["steps"]):
            u = ctrl.act(state)
            state = backend.step(u)
            sd_hist.append(np.asarray(backend.data.sensordata, dtype=np.float64).copy())
            if t % stride == 0:
                renderer.update_scene(backend.data, camera=render_cam)
                frames.append(renderer.render().copy())
    finally:
        renderer.close()

    # Line 1 marks which clip this is; line 2 (identical on both clips of a pair) gives THIS
    # clip's own measured FPL-vs-linear outcome, so the numbers match what the viewer sees.
    this = ("FPL (min-fulfillment, p=-1)" if is_fpl
            else f"best safe linear MPPI ({spec['lin_name']})")
    note = f"   [{spec['note']}]" if spec.get("note") else ""
    line1 = f"{spec['pretty']}  —  {this}{note}"
    line2 = (f"commanded {spec['cmd']}   |   this clip:  "
             f"FPL {spec['fpl_val']}   vs   best-safe linear {spec['lin_val']}   (both stay stable)")
    frames = _caption(frames, line1, line2, WIDTH)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{spec['env']}_{kind}.mp4"
    imageio.mimsave(str(path), frames, fps=out_fps,
                    codec="libx264", quality=8, macro_block_size=None)

    # survival / achieved diagnostic (so we can confirm both clips of a pair actually "succeed")
    sd = np.asarray(sd_hist)
    diag = _diagnose(spec["env"], task, sd)
    print(f"  {path.name:22s} {diag}")
    return path


def _diagnose(env, task, sd):
    if env in ("hopper", "walker"):
        vx = sd[:, task._vel_adr]
        up = sd[:, task._zax_adr + 2]
        return f"mean vx={vx.mean():.2f}  min upright={up.min():.2f}  (fell if <~0.5)"
    if env == "quadruped":
        vx = task._torso_vel_x(sd)
        up = task._torso_up(sd)
        return f"mean vx={vx.mean():.2f}  min upright={up.min():.2f}  (fell if <0.5)"
    # cube
    ang = np.degrees(task._angle_err(sd))
    xy = task._hold_xy(sd)
    achieved = np.degrees(task.target_angle) - ang[-10:].mean()
    return f"rotated={achieved:.1f}deg  max xy-drift={xy.max():.3f} (drop if >0.06)"


def main():
    for spec in ENVS:
        print(spec["pretty"])
        render_clip(spec, "fpl")
        render_clip(spec, "linear")
    print(f"\nAll clips in {OUT}")


if __name__ == "__main__":
    main()
