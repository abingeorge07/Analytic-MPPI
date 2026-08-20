"""Can FPL clear the hurdle, and does the CoM clearance guidance help? Feasibility gate 2
for the jump-over-obstacle 2x2. Runs FPL on quadruped_jump with com_guidance ON and OFF and
reports how far forward the robot gets (obstacle at x_obs=0.55), peak CoM height, and whether
it stayed upright. If +guidance clears while -guidance gets stuck at the wall, the CoM
clearance atom is doing real work and the full 2x2 is worth running.

Run: python verification/com_study/quadruped_jump_clear_probe.py
"""
from __future__ import annotations

import numpy as np

from analytic_mppi.eval import run_episode, init_barkour_stand
from analytic_mppi.tasks import make_task

STEPS = 650
SHARED = dict(num_samples=256, plan_horizon=0.45, num_knots=8, spline_type="zero",
              noise_level=0.7, temperature=0.15)
FPL = dict(cost_mode="fpl_cost", fpl_p=-1.0, fpl_time_p=-2.0, fpl_gamma=0.99)


def main():
    for guidance in [True, False]:
        task = make_task("quadruped_jump", com_guidance=guidance)
        x_obs = task.x_obs
        print(f"\n=== com_guidance={guidance}  (obstacle at x={x_obs}) ===")
        for seed in range(2):
            res = run_episode("quadruped_jump", "mppi", steps=STEPS, seed=seed,
                              init_fn=init_barkour_stand,
                              task_kwargs=dict(com_guidance=guidance), **SHARED, **FPL)
            sd = res["sd"]
            x = task._torso_pos_x(sd)
            h = task._torso_height(sd)
            up = task._torso_up(sd)
            cleared = x.max() > x_obs + 0.08
            # height AT the moment the CoM is closest to the obstacle x
            near = np.argmin(np.abs(x - x_obs))
            print(f"  seed {seed}: max_x={x.max():.2f}  cleared={cleared}  "
                  f"peak_h={h.max():.2f}  h@obstacle={h[near]:.2f}  up_min={up.min():.2f}")


if __name__ == "__main__":
    main()
