"""G1 humanoid WALK flagship — the competing-objective payoff for the FPL thesis.

HONEST framing (revised after a rendered-video check exposed a rz-only-metric overclaim):
Neither controller achieves clean TALL walking from scratch — both sink from the ~0.98 m
standing height. The real, structural difference is the FAILURE MODE:
  * The FPL min-fulfillment floor keeps the torso ORIENTED (up-vector rz never inverts,
    stays >~0.7): the humanoid degrades to a low CROUCH-SHUFFLE but keeps moving and never
    face-plants.
  * The linear-weight family, at EVERY forward weight, INVERTS the torso (rz -> negative)
    and collapses to the floor (min height ~0.05-0.1): it walks straight into a topple.
So the claim is "FPL prevents the catastrophic topple/inversion the linear scalarization
walks into," NOT "FPL walks tall." (Tightening the height atom to force tallness makes FPL
topple too — orientation and height floors are in direct tension for this MPC setup.)

Metrics reported: mean/min torso height, min uprightness rz (KEY: FPL stays >0, linear
inverts), forward speed, and `no_invert` = fraction of episodes whose torso never inverts.

Fairness: linear family and FPL use the SAME layered mode, SAME sampler, SAME budget and
iterations — only the OUTER composition differs (p=1 + per-group weights = linear family;
p<0 uniform = FPL). The linear baseline is given the SAME `iterations` as FPL, so the
difference is not "FPL got more compute."

Run:  .venv/bin/python verification/g1_walk_study.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic_mppi.eval import Config, run_study, init_g1_stand, make_task, g1_walk_metrics

STEPS = 200
N_EPISODES = 6
NUM_SAMPLES = 128
NOISE = 0.3
TEMP = 0.1
GP = 1.0            # posture-group inner power-mean (arithmetic) — isolate the OUTER comp.
ITERS = 3          # gradient iterations, given to BOTH linear and FPL
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.5, num_knots=4, spline_type="zero")
TARGET_VELS = [0.3, 0.5]
TOPPLE_UPRIGHT = 0.3   # min torso up-vector z below this over an episode = a topple

# 6 layered groups: [orientation, height, posture, control, forward, wobble].
def lin_weights(wf):
    return [1.0, 1.0, 1.0, 1.0, wf, 1.0]


def build_configs():
    return [
        Config("linear wf=1", "mppi", "fpl_layered",
               dict(noise_level=NOISE, temperature=TEMP, fpl_weights=lin_weights(1.0),
                    fpl_group_p=GP), fpl_p=1.0),
        Config("linear wf=4", "mppi", "fpl_layered",
               dict(noise_level=NOISE, temperature=TEMP, fpl_weights=lin_weights(4.0),
                    fpl_group_p=GP), fpl_p=1.0),
        Config("FPL p=-1", "mppi", "fpl_layered",
               dict(noise_level=NOISE, temperature=TEMP, fpl_group_p=GP), fpl_p=-1.0),
        Config("FPL + adaptive-sampling", "fpl_adaptive", "fpl_layered",
               dict(noise_level=NOISE, temperature=TEMP, fpl_group_p=GP,
                    steer_mode="binding", explore_mode="absolute",
                    explore_scale_hi=0.4, explore_scale_lo=0.12), fpl_p=-1.0),
    ]


def stats(res, task):
    m = g1_walk_metrics(res, task)
    sd = res["sd"]
    x = sd[..., task._torso_pos_adr]                      # torso x (forward)
    fdist = x[:, -1] - x[:, 0]                            # per-episode forward distance
    up = m["upright"]                                     # torso up-vector z (rz)
    h = task._torso_height(sd)                            # torso height
    # KEY honest metric: did the torso INVERT (rz < 0 = tipped past horizontal)? FPL's floor
    # prevents this; the linear family walks into it. `no_invert` = never inverted.
    no_invert = (up.min(axis=1) >= 0.0)
    late_fv = m["fwd_vel"][:, -60:].mean(axis=1)          # sustained late-episode speed
    return dict(fdist=fdist, no_invert=no_invert, late_fv=late_fv, up=up, h=h,
                min_h=h.min(axis=1), rz_min=up.min(axis=1), fwd_vel=m["fwd_vel"])


def main():
    per_tv = {}
    for tv in TARGET_VELS:
        task = make_task("g1_walk", target_velocity=tv)
        study = run_study("g1_walk", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_g1_stand, task_kwargs=dict(target_velocity=tv),
                          iterations=ITERS, progress=False, **SHARED)
        per_tv[tv] = {label: stats(res, task) for label, res in study.items()}
        print(f"tv={tv} done")

    print("\n" + "=" * 82)
    print(f"G1 WALK | K={NUM_SAMPLES} steps={STEPS} eps={N_EPISODES} iters={ITERS} "
          f"| layered, only OUTER p/weights differ")
    print("=" * 82)
    for tv in TARGET_VELS:
        print(f"\n-- target_velocity = {tv} " + "-" * 42)
        print(f"{'config':26s}{'mean_h':>8s}{'min_h':>7s}{'rz_min':>8s}"
              f"{'fwd_vx':>8s}{'no_invert':>10s}")
        for label, s in per_tv[tv].items():
            print(f"{label:26s}{s['h'].mean():8.2f}{s['min_h'].mean():7.2f}"
                  f"{s['rz_min'].mean():8.2f}{s['fwd_vel'].mean():8.2f}"
                  f"{s['no_invert'].mean():10.2f}")

    # ---- time-course plot at the primary speed: FPL walks-and-stays-up vs linear falls ----
    tv = TARGET_VELS[-1]
    task = make_task("g1_walk", target_velocity=tv)
    dt = task.mj_model.opt.timestep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, s) in enumerate(per_tv[tv].items()):
        c = cmap[i % len(cmap)]
        for ax, key in ((ax1, "fwd_vel"), (ax2, "up")):
            arr = s[key]
            xs = np.arange(arr.shape[1]) * dt
            ax.plot(xs, arr.mean(0), color=c, lw=1.5, label=label if key == "fwd_vel" else None)
            ax.fill_between(xs, arr.mean(0) - arr.std(0), arr.mean(0) + arr.std(0),
                            color=c, alpha=0.12, lw=0)
    ax1.axhline(tv, ls="--", c="k", lw=0.8, alpha=0.6)
    ax1.set(xlabel="time (s)", ylabel="forward speed (m/s)", title=f"G1 walk @ {tv} m/s: forward speed")
    ax2.axhline(TOPPLE_UPRIGHT, ls="--", c="r", lw=0.8, alpha=0.6)
    ax2.set(xlabel="time (s)", ylabel="uprightness rot(ẑ)·ẑ", title="Uprightness (red = topple line)")
    ax1.legend(fontsize=8, loc="best"); ax1.grid(alpha=0.3); ax2.grid(alpha=0.3)
    out = Path(__file__).resolve().parent / "g1_walk.png"
    fig.tight_layout(); fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nsaved time-course plot -> {out}")


if __name__ == "__main__":
    main()
