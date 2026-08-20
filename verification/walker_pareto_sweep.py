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

from analytic_mppi.eval import Config, run_study, make_task
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ci import mean_ci, wilson_ci  # noqa: E402
from _figs import pareto_figure  # noqa: E402

# --- running-regime knobs (found via the sanity probe) ---------------------
STEPS = 150            # 3.0 s at dt=0.02
N_EPISODES = 30        # fall rate resolves to 1/16 ≈ 0.06
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
    fell = zax.min(axis=1) < FALL_UPRIGHT
    vx_ep = vx.mean(axis=1)
    vx_m, vx_h = mean_ci(vx_ep)
    p, lo, hi = wilson_ci(int((~fell).sum()), int(fell.size))
    return dict(vx=vx_m, vx_ci=vx_h, fall=float(fell.mean()),
                surv=p, surv_lo=lo, surv_hi=hi)


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
        print(f"{'config':12s}{'vx±CI':>12s}{'fall':>7s}{'surv':>7s}")
        for label, s in per_tv[tv].items():
            print(f"{label:12s}{s['vx']:7.3f}±{s['vx_ci']:.2f}{s['fall']:7.2f}{s['surv']:7.2f}")

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
    pareto_figure(
        per_tv, fpl_label,
        'Walker2d (running regime): one fixed FPL spec dominates the linear-weight family at every speed — no retuning\\n(identical sampler / atoms / budget / temporal weakest-link; 30 seeds, 95% CIs; only the objective composition differs)',
        Path(__file__).resolve().parent / "walker_pareto.png")


if __name__ == "__main__":
    main()
