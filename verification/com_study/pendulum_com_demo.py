"""CoM-reference pendulum: does an MPPI cost on the Cartesian CoM position drive the
pendulum to a commanded 'intended angle' in one shot?

The `pendulum_com` task (analytic_mppi/tasks/pendulum.py::PendulumComTask) costs the
distance of the link CoM to a reference CoM derived from a target angle theta*. This is
the single-link stand-in for "derive the desired CoM behaviour, then use it as an MPPI
cost". We command a few target angles from the hang-down start and measure:

  * final |theta - theta*|            -> did it REACH the intended angle?
  * final |theta_dot| and a hold check -> did it HOLD there (feasible), or just pass through?

Feasibility is the point. Max actuator torque is 2 N*m; peak gravity torque is m*g*l ~=
4.9 N*m, so the pendulum can only statically hold within ~24 deg of straight-down/up
(|sin theta*| <= 2/4.9 ~= 0.41). A CoM reference in the un-holdable band is dynamically
INFEASIBLE for this actuator: MPPI pulls toward it but cannot settle. Upright (theta*=pi)
is holdable but only reachable via swing-up (a dynamic maneuver), not a direct one-shot.
That contrast is exactly the "a kinematic CoM reference must also be dynamically feasible"
lesson that motivates a template-derived reference on the real robots.

Run:  python verification/pendulum_com_demo.py
"""
from __future__ import annotations

import numpy as np

from analytic_mppi.eval import run_episode, init_hang_down


# holdable band edge for this actuator: |sin theta*| <= tau_max / (m g l) = 2 / 4.905
_TAU_MAX, _MGL = 2.0, 1.0 * 9.81 * 0.5
_HOLDABLE_SIN = _TAU_MAX / _MGL

# (label, target angle in radians from hang-down). 0 = down, pi = upright.
TARGETS = [
    ("near-bottom  (~17 deg, holdable)", np.deg2rad(17.0)),
    ("shallow      (~24 deg, edge)",     np.arcsin(min(0.999, _HOLDABLE_SIN))),
    ("horizontal   (90 deg, infeasible)", np.deg2rad(90.0)),
    ("upright      (180 deg, swing-up)",  np.pi),
]

RUN = dict(
    controller="mppi",
    cost_mode="normal",
    num_samples=256,
    num_knots=8,
    plan_horizon=1.0,
    spline_type="zero",
    noise_level=0.6,
    temperature=0.1,
    steps=200,          # 200 * 0.02 s = 4 s
    seed=0,
)


def _wrap_err(theta: np.ndarray, target: float) -> np.ndarray:
    """Wrap-aware |theta - target| in [0, pi]."""
    return np.abs(((theta - target + np.pi) % (2.0 * np.pi)) - np.pi)


def main() -> None:
    dt = 0.02
    print(f"holdable band: |sin theta*| <= {_HOLDABLE_SIN:.3f}  "
          f"(theta* within ~{np.rad2deg(np.arcsin(_HOLDABLE_SIN)):.0f} deg of down/up)\n")
    header = f"{'target':38s} {'final err (deg)':>16s} {'final |w| (rad/s)':>18s} {'held?':>7s}"
    print(header)
    print("-" * len(header))

    for label, target in TARGETS:
        res = run_episode(
            "pendulum_com",
            RUN["controller"],
            steps=RUN["steps"],
            seed=RUN["seed"],
            cost_mode=RUN["cost_mode"],
            init_fn=init_hang_down,
            task_kwargs=dict(target_angle=float(target)),
            num_samples=RUN["num_samples"],
            num_knots=RUN["num_knots"],
            plan_horizon=RUN["plan_horizon"],
            spline_type=RUN["spline_type"],
            noise_level=RUN["noise_level"],
            temperature=RUN["temperature"],
        )
        states = res["states"]                 # (T+1, nstate)
        theta = states[:, 1]                   # qpos[0]
        omega = states[:, 2]                   # qvel[0]
        # look at the last 0.5 s to judge whether it SETTLED at the target
        tail = slice(-25, None)
        err_tail = _wrap_err(theta[tail], target)
        held = bool(err_tail.mean() < np.deg2rad(10.0) and np.abs(omega[tail]).mean() < 0.5)
        print(f"{label:38s} {np.rad2deg(err_tail.mean()):16.1f} "
              f"{np.abs(omega[tail]).mean():18.2f} {('yes' if held else 'no'):>7s}")


if __name__ == "__main__":
    main()
