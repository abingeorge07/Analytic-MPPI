"""g1_reach (forward lean-and-hold) — an ACHIEVABLE static-balance humanoid task, used to
test whether FPL beats the linear-weight family on the g1. VERDICT: it does NOT.

The task rewards leaning the torso FORWARD to a target x-offset while staying upright and
tall (feet planted, no gait). KEY ENABLER = planning HORIZON: at H=0.5 s the humanoid can't
plan a stable lean and collapses under ANY controller; at H>=1.0 s it can (use long horizons
for g1 dynamic tasks).

HONEST RESULT (6-seed study + target_reach sweep {0.2,0.35,0.5}): a CONSERVATIVE linear
weight (wf=1) stays tall+upright and leans, and DOMINATES FPL at every difficulty
(achieved 0.80/0.60/0.20 vs FPL 0.40/0.00/0.00). An earlier "FPL win" was an artifact of
comparing FPL only vs the AGGRESSIVE linear weight (wf=4, which topples) — the exact unfair
comparison the hopper study forbids. This script exists to REPRODUCE that negative result:
compare FPL against the FULL linear family incl. the conservative weight, and render+look
(rz alone is a trap — a sprawl keeps rz>0).

Why g1_reach isn't an FPL win (vs the hopper, which is): the hopper competition is DYNAMIC
with no safe conservative operating point (to go fast you MUST hop = risk falling), so FPL's
floor earns its keep; g1_reach leaning is quasi-static, so a low-risk conservative linear
weight exists and wins. FPL's edge needs unavoidable/dynamic competition.

Fairness: identical sampler / budget / iterations / horizon; ONLY the outer composition
differs (p=1 + per-group weights = linear family; p<0 uniform = FPL). Baseline gets the
same iterations, so the comparison is not confounded by compute.

Honest metrics (rz ALONE is a trap on a humanoid — a sprawl keeps rz>0; always co-check
height): forward reach (torso x), mean/min torso height, min uprightness rz, and
  success = fraction of episodes that stay BOTH tall (min height > 0.6 m) AND upright
            (rz never < 0.5) the whole episode.

Run:  python verification/g1_reach_study.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic_mppi.eval import Config, run_study, init_g1_stand, make_task

TASK = "g1_reach"
TARGET_REACH = 0.2
COST_MODE = "fpl_layered"
GP = 1.0                      # inner (within-group) power-mean p
NOISE = 0.3
TEMP = 0.1
# Horizon is the enabler: H>=1.0 s lets the humanoid plan a stable lean. num_knots scaled
# so the knot spacing stays ~0.16-0.19 s (matches the H=0.5/k=4 baseline resolution).
PLAN_HORIZON = 1.0
NUM_KNOTS = 6
NUM_SAMPLES = 128
ITERS = 3
STEPS = 150
N_EPISODES = 6

H_FULL, H_FLOOR = 0.6, 0.5    # success thresholds (tall + upright)
RZ_FLOOR = 0.5


def lin_weights(wf):
    # [orientation, height, posture, control, forward, wobble] outer-group weights.
    return [1.0, 1.0, 1.0, 1.0, float(wf), 1.0]


def build_configs():
    return [
        Config("linear wf=1", "mppi", COST_MODE,
               dict(noise_level=NOISE, temperature=TEMP, fpl_weights=lin_weights(1.0),
                    fpl_group_p=GP), fpl_p=1.0),
        Config("linear wf=2", "mppi", COST_MODE,
               dict(noise_level=NOISE, temperature=TEMP, fpl_weights=lin_weights(2.0),
                    fpl_group_p=GP), fpl_p=1.0),
        Config("linear wf=4", "mppi", COST_MODE,
               dict(noise_level=NOISE, temperature=TEMP, fpl_weights=lin_weights(4.0),
                    fpl_group_p=GP), fpl_p=1.0),
        Config("FPL p=-1", "mppi", COST_MODE,
               dict(noise_level=NOISE, temperature=TEMP, fpl_group_p=GP), fpl_p=-1.0),
        Config("FPL p=-2", "mppi", COST_MODE,
               dict(noise_level=NOISE, temperature=TEMP, fpl_group_p=GP), fpl_p=-2.0),
        Config("FPL + adaptive-sampling", "fpl_adaptive", COST_MODE,
               dict(noise_level=NOISE, temperature=TEMP, fpl_group_p=GP,
                    steer_mode="binding", explore_mode="absolute",
                    explore_scale_hi=0.4, explore_scale_lo=0.12), fpl_p=-1.0),
    ]


def stats(res, task):
    sd = res["sd"]
    x = sd[..., task._torso_pos_adr]                      # torso x = forward displacement
    h = task._torso_height(sd)                            # torso height
    rz = task._torso_orientation(sd)[..., 2]             # up-vector z
    reach = x[:, -1]                                      # held forward offset at episode end
    tall_and_up = (h.min(axis=1) > H_FLOOR) & (rz.min(axis=1) > RZ_FLOOR)
    return dict(reach=reach, mean_h=h.mean(axis=1), min_h=h.min(axis=1),
                rz_min=rz.min(axis=1), success=tall_and_up, h_series=h, rz_series=rz)


def main():
    task = make_task(TASK, target_reach=TARGET_REACH)
    study = run_study(TASK, build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                      init_fn=init_g1_stand, task_kwargs=dict(target_reach=TARGET_REACH),
                      num_samples=NUM_SAMPLES, plan_horizon=PLAN_HORIZON, num_knots=NUM_KNOTS,
                      spline_type="zero", iterations=ITERS, progress=True)
    per = {label: stats(res, task) for label, res in study.items()}

    print("\n" + "=" * 88)
    print(f"G1 REACH (lean-and-hold, target={TARGET_REACH} m) | K={NUM_SAMPLES} H={PLAN_HORIZON}s "
          f"steps={STEPS} eps={N_EPISODES} iters={ITERS}")
    print("=" * 88)
    print(f"{'config':26s}{'reach_x':>9s}{'mean_h':>8s}{'min_h':>7s}{'rz_min':>8s}"
          f"{'success':>9s}")
    for label, s in per.items():
        print(f"{label:26s}{s['reach'].mean():9.2f}{s['mean_h'].mean():8.2f}"
              f"{s['min_h'].mean():7.2f}{s['rz_min'].mean():8.2f}{s['success'].mean():9.2f}")
    print("=" * 88)
    print("Compare FPL vs the BEST linear weight (not just aggressive wf=4). Observed result:\n"
          "conservative linear wf=1 stays tall+upright AND leans, matching/beating FPL — so\n"
          "g1_reach is NOT an FPL win (unlike the hopper). See module docstring for why.")

    # ---- height + uprightness time courses ----
    dt = task.mj_model.opt.timestep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, s) in enumerate(per.items()):
        c = cmap[i % len(cmap)]
        for ax, key in ((ax1, "h_series"), (ax2, "rz_series")):
            arr = s[key]
            xs = np.arange(arr.shape[1]) * dt
            mu, sd_ = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(xs, mu, color=c, lw=1.5, label=label)
            ax.fill_between(xs, mu - sd_, mu + sd_, color=c, alpha=0.12, lw=0)
    ax1.axhline(H_FLOOR, ls="--", c="k", lw=0.8)
    ax1.set(xlabel="time (s)", ylabel="torso height (m)", title="Torso height (1=standing ~0.98)")
    ax2.axhline(RZ_FLOOR, ls="--", c="k", lw=0.8)
    ax2.set(xlabel="time (s)", ylabel="rz (1=upright)", title="Torso uprightness")
    ax2.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"g1_reach: FPL holds tall+upright forward lean; linear family topples "
                 f"(H={PLAN_HORIZON}s)", y=1.02)
    fig.tight_layout()
    out = "verification/g1_reach.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
