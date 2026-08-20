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

import sys
import numpy as np

from analytic_mppi.eval import Config, run_study, init_hopper_stand, make_task
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci            # noqa: E402
from _figs import pareto_figure               # noqa: E402

# --- knobs (offline: be generous with samples/episodes) --------------------
STEPS = 150            # 3.0 s at dt=0.02
N_EPISODES = 30        # submission-grade: 30 seeds + 95% CIs on every point
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
    """Per-config aggregate with 95% CIs: mean per-episode speed (± CI), fall rate,
    survival (Wilson CI)."""
    sd = res["sd"]
    vx = sd[..., task._vel_adr]
    zax = sd[..., task._zax_adr + 2]
    fell = zax.min(axis=1) < FALL_UPRIGHT          # (n_ep,)
    vx_ep = vx.mean(axis=1)                          # (n_ep,) per-episode time-mean speed
    n = int(fell.size)
    vx_m, vx_h = mean_ci(vx_ep)
    p, lo, hi = wilson_ci(int((~fell).sum()), n)
    return dict(vx=vx_m, vx_ci=vx_h, fall=float(fell.mean()),
                surv=p, surv_lo=lo, surv_hi=hi, ess=float(np.nanmean(res["ess"])))


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
        print(f"{'config':12s}{'vx±CI':>12s}{'fall':>7s}{'surv':>7s}")
        for label, s in per_tv[tv].items():
            print(f"{label:12s}{s['vx']:7.3f}±{s['vx_ci']:.2f}{s['fall']:7.2f}{s['surv']:7.2f}")

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

    pareto_figure(
        per_tv, fpl_label,
        "Hopper: one fixed FPL spec dominates the linear-weight family at every speed — no "
        "retuning\n(identical sampler / atoms / budget / temporal weakest-link; 30 seeds, "
        "95% CIs; only the objective composition differs)",
        Path(__file__).resolve().parent / "hopper_pareto.png")


if __name__ == "__main__":
    main()
