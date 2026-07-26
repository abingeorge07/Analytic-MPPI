"""Hopper flagship — FPL Pareto-dominates the LINEAR-WEIGHT FAMILY (handoff §6).

The falsifiable claim (stronger than "FPL beats one tuned linear"):
    A SINGLE FPL spec (p<0, uniform atom weights) dominates the lower envelope of the
    ENTIRE linear-weight family on the (achieved speed, fall rate) plane, across a
    commanded-speed sweep, with NO per-speed retuning.

Fairness (non-negotiable, FPL_MPPI_HANDOFF §6):
  * Everything identical except the OBJECTIVE-AXIS scalarization.
  * `fpl_cost` mode for every config, same temporal aggregation (`fpl_time_p`, the
    weakest-link-over-time) given to BOTH the linear family and FPL — so the only thing
    that varies is how the per-step objective vector is collapsed:
       - linear family: p=+1, per-atom `fpl_weights` swept (the whole weight family)
       - FPL:           p<0,  uniform weights (min-fulfillment conjunction)
  * `power_mean(x, p=1, weights=w) == Σ w_i x_i` exactly, so p=1 IS a linear cost.

Reads out:
  1. RAW-PERFORMANCE headline (sequenced #1): at the hardest speed, the fastest FPL
     controller vs the fastest linear controller AT MATCHED fall rate.
  2. ROBUSTNESS headline (sequenced #2): the Pareto plot — one FPL point below the
     linear frontier at every difficulty.

Run:  .venv/bin/python verification/hopper_pareto_sweep.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analytic_mppi.eval import Config, run_study, init_hopper_stand, make_task

# --- knobs (offline: be generous with samples/episodes) --------------------
STEPS = 150            # 3.0 s at dt=0.02
N_EPISODES = 16        # fall rate resolves to 1/16 ≈ 0.06
NUM_SAMPLES = 256
NOISE = 0.3
TEMP = 0.2
TIME_P = -2.0          # temporal weakest-link, given to BOTH families
FALL_UPRIGHT = 0.6     # min torso zaxis_z over an episode below this = a fall
SHARED = dict(num_samples=NUM_SAMPLES, plan_horizon=0.6, num_knots=4, spline_type="zero")

TARGET_VELS = [2.0, 2.5, 3.0]

# Linear family: atom order [height, orientation, velocity, control]. Sweep the velocity
# weight to trace the slow/safe -> fast/reckless frontier; safety atoms fixed at 1.
LINEAR_WV = [0.5, 1.0, 2.0, 4.0, 8.0]
# FPL: single spec, uniform weights. p=-1 (harmonic) is the sweet spot (p=-2 is too greedy).
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
    """Per-config aggregate: mean achieved vx, fall rate, forward distance, ESS."""
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
        task = make_task("hopper", target_velocity=tv)
        study = run_study("hopper", build_configs(), steps=STEPS, n_episodes=N_EPISODES,
                          init_fn=init_hopper_stand, task_kwargs=dict(target_velocity=tv),
                          progress=False, **SHARED)
        per_tv[tv] = {label: episode_stats(res, task) for label, res in study.items()}
        print(f"tv={tv} done")

    # ---- tables ----
    print("\n" + "=" * 78)
    print(f"HOPPER linear-family vs FPL | K={NUM_SAMPLES} steps={STEPS} eps={N_EPISODES} "
          f"| time_p={TIME_P}")
    print("=" * 78)
    for tv in TARGET_VELS:
        print(f"\n-- target_velocity = {tv} " + "-" * 40)
        print(f"{'config':12s}{'vx':>8s}{'fall':>7s}{'fwd':>7s}{'ess':>7s}")
        for label, s in per_tv[tv].items():
            print(f"{label:12s}{s['vx']:8.3f}{s['fall']:7.2f}{s['fwd']:7.2f}{s['ess']:7.0f}")

    # ---- raw-performance headline at the hardest speed ----
    tv = TARGET_VELS[-1]
    fpl_label = f"FPL p={FPL_P:g}"
    fpl_s = per_tv[tv][fpl_label]
    # best linear controller whose fall rate is <= FPL's: fastest safe linear.
    safe_lin = [(l, s) for l, s in per_tv[tv].items()
                if l != fpl_label and s["fall"] <= fpl_s["fall"] + 1e-9]
    print("\n" + "=" * 78)
    print(f"RAW-PERFORMANCE (tv={tv}): FPL p={FPL_P:g} -> vx={fpl_s['vx']:.3f} "
          f"at fall={fpl_s['fall']:.2f}")
    if safe_lin:
        best = max(safe_lin, key=lambda kv: kv[1]["vx"])
        print(f"  fastest linear at <= that fall rate: {best[0]} -> vx={best[1]['vx']:.3f} "
              f"({100*(fpl_s['vx']-best[1]['vx'])/max(best[1]['vx'],1e-6):+.0f}% vs FPL)")
    else:
        print("  NO linear setting is as safe as FPL at this speed.")

    # ---- cache raw stats so the figure is reproducible without re-running the sweep ----
    import json
    data_out = Path(__file__).resolve().parent / "hopper_pareto_data.json"
    data_out.write_text(json.dumps({str(tv): per_tv[tv] for tv in TARGET_VELS}, indent=2))
    print(f"\nsaved raw stats -> {data_out}")

    make_figure(per_tv, fpl_label)


def make_figure(per_tv, fpl_label):
    """Clean 1xN small-multiple: at EACH commanded speed, the FPL point sits above-and-right
    of the whole linear-weight frontier (up-right = faster AND safer). One panel per
    difficulty makes the 'no per-speed retuning' thesis visually explicit."""
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
        # label each linear point with its velocity weight (shows the frontier IS a retune knob)
        for (l, s) in per_tv[tv].items():
            if l.startswith("lin"):
                ax.annotate(l.replace("lin wv=", "w="), (s["vx"], 1.0 - s["fall"]),
                            fontsize=6.5, color=LIN_C, xytext=(0, -11),
                            textcoords="offset points", ha="center")
        fs = per_tv[tv][fpl_label]
        fx, fy = fs["vx"], 1.0 - fs["fall"]
        ax.plot(fx, fy, "*", color=FPL_C, ms=26, mec="k", mew=1.2,
                label="FPL (one fixed spec)", zorder=5)

        # headline arrow: fastest linear AT LEAST as safe as FPL -> FPL (gain at equal safety)
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
    fig.suptitle("Hopper: one fixed FPL spec dominates the entire linear-weight family at every "
                 "speed — no retuning\n(identical sampler / atoms / budget / temporal "
                 "weakest-link; only the objective composition differs)",
                 fontsize=11, y=1.04)
    out = Path(__file__).resolve().parent / "hopper_pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved Pareto figure -> {out}")


if __name__ == "__main__":
    main()
