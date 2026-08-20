"""Guardrail step 1 for the quadruped CoM 2x2: (a) confirm FPL is COMPETENT on the base
Barkour walk under the published sweep settings, and (b) MEASURE the natural CoM-height bob
during a good gait — so the upcoming vertical anti-bob atom can be shaped as a FLOOR wide
enough not to fight the gait's legitimate bob (the pendulum 'the floor must clear the control
the task actually needs' lesson).

Settings mirror verification/quadruped_pareto_sweep.py (fpl_cost, fpl_p=-1, fpl_time_p=-2,
temp=0.2, K=128, horizon=0.3). Reports per-gait: forward speed, uprightness (fall check),
CoM-height mean/std/min/max, and the mean FPL atom vector (want all ~high during the gait).

Run:  python verification/com_study/quadruped_base_probe.py
"""
from __future__ import annotations

import numpy as np

from analytic_mppi.eval import run_episode, init_barkour_stand
from analytic_mppi.tasks import make_task

TV = 2.0
STEPS = 250
SHARED = dict(num_samples=128, plan_horizon=0.3, num_knots=5, spline_type="zero",
              noise_level=0.5, temperature=0.2)
FPL = dict(cost_mode="fpl_cost", fpl_p=-1.0, fpl_time_p=-2.0, fpl_gamma=0.99)


def main() -> None:
    task = make_task("quadruped", target_velocity=TV)
    atom_names = task.cost_term_names_f

    for seed in range(3):
        res = run_episode("quadruped", "mppi", steps=STEPS, seed=seed,
                          init_fn=init_barkour_stand, task_kwargs=dict(target_velocity=TV),
                          **SHARED, **FPL)
        sd = res["sd"]                        # (STEPS, nsensordata)
        h = task._torso_height(sd)
        vx = task._torso_vel_x(sd)
        up = task._torso_up(sd)
        # settle: ignore the first 60 steps (start transient), read the gait
        g = slice(60, None)
        # mean atom vector over the gait
        states = res["states"]
        qpos = task.qpos_of(states[1:])       # align with sd/ctrls
        atoms = task.running_cost_terms_f(qpos[g], None, sd[g], res["ctrls"][g])
        amean = atoms.mean(axis=0)
        print(f"seed {seed}: vx={vx[g].mean():.2f} m/s  up_min={up.min():.2f} "
              f"(fall<0.5)  h: mean={h[g].mean():.3f} std={h[g].std():.3f} "
              f"min={h[g].min():.3f} max={h[g].max():.3f}  (h*={task.target_height})")
        print("          atoms " + "  ".join(f"{n.split('_')[0]}={v:.2f}"
                                              for n, v in zip(atom_names, amean)))

    print("\nCoM anti-bob band should be ~flat (>=~1) across [h_min, h_max] of a good gait,")
    print("dropping only for bob BEYOND that — else it fights the natural gait.")


if __name__ == "__main__":
    main()
