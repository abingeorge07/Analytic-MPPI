"""Render the cube reorientation win: FPL tumbles the cube toward the goal and keeps it in
hand, while the aggressive linear weight flings it out of the palm. Writes two mp4s under
runs/ for visual verification (the metric alone can be gamed; watch the cube).

Run:  .venv/bin/python verification/cube_render.py
"""
from __future__ import annotations

from pathlib import Path

from analytic_mppi.eval import render_video, init_cube

TA = 1.8               # rad ≈ 103° forward roll (the regime where linear drops 100%)
STEPS = 200
SHARED = dict(num_samples=256, plan_horizon=0.3, num_knots=4, spline_type="zero",
              noise_level=0.4, temperature=0.1, task_kwargs=dict(target_angle=TA))
OUT = Path(__file__).resolve().parent.parent / "runs"


def main():
    r = render_video("cube", "mppi", steps=STEPS, out_path=OUT / "cube_FPL.mp4",
                     seed=0, cost_mode="fpl_cost", init_fn=init_cube,
                     fpl_p=-1.0, fpl_time_p=-2.0, width=640, height=360, **SHARED)
    print("FPL   ->", r["path"])
    r = render_video("cube", "mppi", steps=STEPS, out_path=OUT / "cube_linear.mp4",
                     seed=0, cost_mode="fpl_cost", init_fn=init_cube,
                     fpl_p=1.0, fpl_weights=[1.0, 1.0, 8.0, 0.5], fpl_time_p=-2.0,
                     width=640, height=360, **SHARED)
    print("linear->", r["path"])


if __name__ == "__main__":
    main()
