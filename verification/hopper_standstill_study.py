"""Hopper stand-still-refusal study — the first competitive read on FPL vs linear.

Everything is held identical across configs EXCEPT the power-mean exponent `p`:
  * p = +1   -> arithmetic mean of the fulfillment atoms == a LINEAR scalarization
               (the baseline: a linear cost that is content to stand still).
  * p <= 0   -> FPL conjunction (min-fulfillment floor): standing still tanks the
               speed atom -> composite ~0 -> the controller is forced to move.

Same atoms, same true-MPPI softmax-weighted update (cost = -log u), same sampler,
same seeds. So any difference is attributable to the scalarization alone.

Headline question: does p=+1 stand still while p<=-1 walks forward, at equal or
lower fall rate?

Run:
    .venv/bin/python verification/hopper_standstill_study.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from analytic_mppi.controllers import MPPIv2
from analytic_mppi.eval import (
    Config, run_study, init_hopper_stand, make_task,
    walker_metrics, walker_panels, plot_study,
)

# --- experiment knobs (offline: be generous with samples) ------------------
STEPS = 150            # 3.0 s at dt=0.02
N_EPISODES = 8         # enough seeds that per-config means/fall-counts are trustworthy
NUM_SAMPLES = 256
PLAN_HORIZON = 0.6
NUM_KNOTS = 4
NOISE_LEVEL = 0.3
TEMPERATURE = 0.2      # λ for the fpl_cost (-log u) configs; watch ESS
NORMAL_TEMPERATURE = 1.0  # the legacy quadratic cost has a larger scale than -log u,
                          # so it needs a higher temp (at 0.05 its ESS collapses to ~1)
STANDSTILL_VX = 0.1    # |vx| below this counts as "standing still"
FALL_UPRIGHT = 0.6     # min torso zaxis_z below this over an episode counts as a fall

P_SWEEP = [
    ("linear  (p=+1)", 1.0),
    ("geom    (p= 0)", 0.0),
    ("fpl harm(p=-1)", -1.0),
    ("fpl     (p=-2)", -2.0),
]

# Soft-min over time (power-mean q<=0 on the time axis): a rollout's value = its
# WORST moment, so the min-fulfillment floor holds across the trajectory and a
# go-fast-then-faceplant rollout is penalized. Compared head-to-head with the
# time-average configs above to test whether it cuts FPL's fall rate.
TIME_SOFTMIN_Q = -2.0
SOFTMIN_SWEEP = [
    ("fpl+smin(p=-1)", -1.0),
    ("fpl+smin(p=-2)", -2.0),
]


def build_configs():
    # Reference: vanilla MPPI on the legacy hand-tuned quadratic cost (DIFFERENT
    # objectives than the fulfillment atoms, and its own temperature). Not part of
    # the controlled "only p differs" comparison — that's the p-sweep below.
    normal = Config("normal (quad cost)", MPPIv2, cost_mode="normal",
                    kwargs=dict(noise_level=NOISE_LEVEL, temperature=NORMAL_TEMPERATURE))
    p_sweep = [
        Config(label, MPPIv2, cost_mode="fpl_cost",
               kwargs=dict(noise_level=NOISE_LEVEL, temperature=TEMPERATURE),
               fpl_p=p)
        for label, p in P_SWEEP
    ]
    softmin = [
        Config(label, MPPIv2, cost_mode="fpl_cost",
               kwargs=dict(noise_level=NOISE_LEVEL, temperature=TEMPERATURE,
                           fpl_time_p=TIME_SOFTMIN_Q),
               fpl_p=p)
        for label, p in SOFTMIN_SWEEP
    ]
    return [normal] + p_sweep + softmin


def locomotion_summary(study, task):
    """Scalar headline metrics per config, aggregated over episodes."""
    rows = []
    for label, res in study.items():
        sd = res["sd"]                              # (N_EP, T, nsd)
        ess = res["ess"]                            # (N_EP, T)
        px = sd[..., task._pos_adr + 0]             # torso x     (N_EP, T)
        vx = sd[..., task._vel_adr]                 # torso vx    (N_EP, T)
        zax = sd[..., task._zax_adr + 2]            # uprightness (N_EP, T)

        fwd = px[:, -1] - px[:, 0]                  # per-episode forward displacement (N_EP,)
        fell = zax.min(axis=1) < FALL_UPRIGHT       # per-episode fall flag (N_EP,)
        rows.append((
            label,
            fwd.mean(), fwd.std(),                  # mean ± std across seeds
            vx.mean(),
            (np.abs(vx) < STANDSTILL_VX).mean(),
            int(fell.sum()), int(fell.size),        # fell in n of N seeds
            np.nanmean(ess),
        ))
    return rows


def print_summary(rows):
    hdr = (f"{'config':16s}  {'fwd_dist(m)':>14s}  {'mean_vx':>8s}  "
           f"{'standstill':>10s}  {'fell':>7s}  {'mean_ESS':>8s}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for label, fwd, fwd_sd, vx, ss, nfell, ntot, ess in rows:
        print(f"{label:16s}  {fwd:6.2f} ± {fwd_sd:4.2f}  {vx:8.3f}  "
              f"{ss:10.2f}  {nfell:2d}/{ntot:<3d}  {ess:8.1f}")


def main():
    task = make_task("hopper")
    print(f"hopper stand-still study: target_velocity={task.target_velocity} "
          f"num_samples={NUM_SAMPLES} steps={STEPS} n_ep={N_EPISODES} temp={TEMPERATURE}")

    study = run_study(
        "hopper", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
        init_fn=init_hopper_stand,
        num_samples=NUM_SAMPLES, plan_horizon=PLAN_HORIZON, num_knots=NUM_KNOTS,
        spline_type="zero",
    )

    rows = locomotion_summary(study, task)
    print_summary(rows)

    out = Path(__file__).resolve().parent / "hopper_standstill.png"
    fig = plot_study(study, lambda r: walker_metrics(r, task), walker_panels(),
                     dt=task.mj_model.opt.timestep,
                     title="Hopper: linear (p≥0) vs FPL conjunction (p<0) — identical atoms & update")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
