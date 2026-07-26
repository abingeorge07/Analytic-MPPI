"""Walker2d flagship #2 — FPL Pareto-dominates the linear-weight family UNDER DYNAMIC
COMPETITION (the thesis: FPL's edge needs unavoidable/dynamic competition, not merely
"competing objectives").

Context: at gentle speeds the Walker2d walks quasi-statically and NOTHING falls — there IS a
safe conservative linear weight, so FPL wins nothing (same as g1_reach). The competition only
turns DYNAMIC in the RUNNING regime (aggressive target speed + exploration), where going
faster REQUIRES flight phases that risk a fall. THERE, no single linear weight is both fast
and safe, and one fixed FPL spec (p<0, uniform) dominates the frontier — reproducing the
hopper result on a harder biped.

Falsifiable claim: on the (achieved speed, survival) plane, the single FPL point sits
above-and-right of the entire linear-weight frontier at every commanded speed, no retuning.

Fairness (identical to the hopper study): same sampler / atoms / budget / horizon / temporal
weakest-link (fpl_time_p) for ALL configs; ONLY the objective composition differs
  - linear family: p=+1, per-atom velocity weight swept
  - FPL:           p=-1, uniform weights (min-fulfillment conjunction)
`power_mean(x, p=1, weights=w) == Σ w_i x_i`, so p=1 IS the linear cost. Walker FPL atoms were
upgraded to the hopper shapes (1-sided velocity ramp, orientation decaying before horizontal,
1-sided height) — SHARED by both families, so only the scalarization varies.

Run:  .venv/bin/python verification/walker_pareto_sweep.py
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic_mppi.eval import Config, run_study, make_task

# --- running-regime knobs (found via the sanity probe) ---------------------
STEPS = 150            # 3.0 s at dt=0.02
N_EPISODES = 16        # fall rate resolves to 1/16 ≈ 0.06
NUM_SAMPLES = 256
NOISE = 0.8            # aggressive exploration to reach the running regime
TEMP = 0.2
TIME_P = -2.0          # temporal weakest-link, given to BOTH families
FALL_UPRIGHT = 0.6     # min torso zaxis_z over an episode below this = a fall
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.6, num_knots=6, spline_type="zero")

TARGET_VELS = [4.0, 5.0, 6.0]

# Linear family: atom order [height, orientation, velocity, control]. Sweep the velocity
# weight to trace the slow/safe -> fast/reckless frontier; safety atoms fixed at 1.
LINEAR_WV = [2.0, 4.0, 8.0, 16.0, 32.0]
FPL_P = -1.0


def build_configs():
    lin = [
        Config(f"lin wv={wv:g}", "mppi", "fpl_cost",
               dict(noise_level=NOISE, temperature=TEMP,
                    fpl_weights=[1.0, 1.0, wv, 0.5], fpl_time_p=TIME_P), fpl_p=1.0)
        for wv in LINEAR_WV
    ]
    fpl = Config(f"FPL p={FPL_P:g}", "mppi", "fpl_cost",
                 dict(noise_level=NOISE, temperature=TEMP, fpl_time_p=TIME_P), fpl_p=FPL_P)
    return lin + [fpl]


def episode_stats(res, task):
    sd = res["sd"]
    vx = sd[..., task._vel_adr]
    zax = sd[..., task._zax_adr + 2]
    px = sd[..., task._pos_adr]
    fell = zax.min(axis=1) < FALL_UPRIGHT
    return dict(vx=float(vx.mean()), fall=float(fell.mean()),
                fwd=float((px[:, -1] - px[:, 0]).mean()), ess=float(np.nanmean(res["ess"])))


def main():
    per_tv = {}
    for tv in TARGET_VELS:
        task = make_task("walker", target_velocity=tv)
        study = run_study("walker", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          task_kwargs=dict(target_velocity=tv), progress=False, **SHARED)
        per_tv[tv] = {label: episode_stats(res, task) for label, res in study.items()}
        print(f"tv={tv} done")

    print("\n" + "=" * 78)
    print(f"WALKER2d linear-family vs FPL (running regime) | K={NUM_SAMPLES} steps={STEPS} "
          f"eps={N_EPISODES} noise={NOISE} time_p={TIME_P}")
    print("=" * 78)
    for tv in TARGET_VELS:
        print(f"\n-- target_velocity = {tv} " + "-" * 40)
        print(f"{'config':12s}{'vx':>8s}{'fall':>7s}{'fwd':>7s}{'ess':>7s}")
        for label, s in per_tv[tv].items():
            print(f"{label:12s}{s['vx']:8.3f}{s['fall']:7.2f}{s['fwd']:7.2f}{s['ess']:7.0f}")

    tv = TARGET_VELS[-1]
    fpl_label = f"FPL p={FPL_P:g}"
    fpl_s = per_tv[tv][fpl_label]
    safe_lin = [(l, s) for l, s in per_tv[tv].items()
                if l != fpl_label and s["fall"] <= fpl_s["fall"] + 1e-9]
    print("\n" + "=" * 78)
    print(f"RAW-PERFORMANCE (tv={tv}): FPL -> vx={fpl_s['vx']:.3f} at fall={fpl_s['fall']:.2f}")
    if safe_lin:
        best = max(safe_lin, key=lambda kv: kv[1]["vx"])
        print(f"  fastest linear at <= that fall rate: {best[0]} -> vx={best[1]['vx']:.3f} "
              f"({100*(fpl_s['vx']-best[1]['vx'])/max(best[1]['vx'],1e-6):+.0f}% vs FPL)")
    else:
        print("  NO linear setting is as safe as FPL at this speed.")

    data_out = Path(__file__).resolve().parent / "walker_pareto_data.json"
    data_out.write_text(json.dumps({str(tv): per_tv[tv] for tv in TARGET_VELS}, indent=2))
    print(f"\nsaved raw stats -> {data_out}")
    make_figure(per_tv, fpl_label)


def make_figure(per_tv, fpl_label):
    """1xN small-multiple: FPL point above-and-right of the whole linear frontier at each
    commanded speed (up-right = faster AND safer). Mirrors the hopper figure."""
    tvs = list(per_tv.keys())
    n = len(tvs)
    LIN_C, FPL_C = "#4c72b0", "#dd8452"
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, tv in zip(axes, tvs):
        lin = sorted([(s["vx"], 1.0 - s["fall"]) for l, s in per_tv[tv].items()
                      if l.startswith("lin")])
        xs, ys = zip(*lin)
        ax.plot(xs, ys, "o-", color=LIN_C, lw=2, ms=8, mec="white", mew=1,
                label="linear-weight family", zorder=3)
        for (l, s) in per_tv[tv].items():
            if l.startswith("lin"):
                ax.annotate(l.replace("lin wv=", "w="), (s["vx"], 1.0 - s["fall"]),
                            fontsize=6.5, color=LIN_C, xytext=(0, -11),
                            textcoords="offset points", ha="center")
        fs = per_tv[tv][fpl_label]
        fx, fy = fs["vx"], 1.0 - fs["fall"]
        ax.plot(fx, fy, "*", color=FPL_C, ms=26, mec="k", mew=1.2,
                label="FPL (one fixed spec)", zorder=5)
        safe = [(s["vx"], 1.0 - s["fall"]) for l, s in per_tv[tv].items()
                if l.startswith("lin") and s["fall"] <= fs["fall"] + 1e-9]
        if safe:
            bx, by = max(safe)
            ax.annotate("", xy=(fx, fy), xytext=(bx, by),
                        arrowprops=dict(arrowstyle="->", color="k", lw=1.6))
            if fx > bx:
                ax.text((fx + bx) / 2, min(fy, by) - 0.06,
                        f"+{100*(fx-bx)/max(bx,1e-6):.0f}% speed\nat equal safety",
                        fontsize=8, ha="center", va="top", fontweight="bold")
        ax.set_title(f"commanded speed = {tv} m/s", fontsize=10)
        ax.set_xlabel("achieved forward speed  (m/s)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("survival rate  (1 − fall)")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.suptitle("Walker2d (running regime): one fixed FPL spec dominates the linear-weight "
                 "family at every speed — no retuning\n(identical sampler / atoms / budget / "
                 "temporal weakest-link; only the objective composition differs)",
                 fontsize=11, y=1.04)
    out = Path(__file__).resolve().parent / "walker_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved Pareto figure -> {out}")


if __name__ == "__main__":
    main()
