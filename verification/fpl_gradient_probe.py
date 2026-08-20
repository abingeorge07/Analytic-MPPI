"""FPL-native sampler attempt #3: gradient-guided refinement at LOW budget.

Colored noise + the absolute-scale schedule did not beat plain FPL+MPPI (see
fpl_sampler_race.py) — consistent with the prior negative: for warm-started receding-horizon
search the proposal barely matters. The one mechanism that is categorically different is LOCAL
OPTIMIZATION: take the top-mu rollouts and push them downhill on the analytic FPL reward
gradient (BPTT through MuJoCo FD Jacobians; cost_gd.py). Its best shot is a SCARCE budget,
where a few gradient-refined elites might match many more random samples — the sample-
efficiency regime that matters for real-time / onboard control.

Question: does FPL+MPPI with 2 gradient steps on 8 elites at K=16/32 reach the productive
speed of plain FPL+MPPI at K=256? If yes, that is a genuine FPL-native-optimizer win at low
budget. If no, the objective — not the search — is where all the value is (clean negative).

Checkpointed. Run:  .venv/bin/python verification/fpl_gradient_probe.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _experiment import run_trial                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci                                    # noqa: E402

EXP = "gradient_probe"
ENVS = ["walker", "hopper"]
GD = dict(gd_iterations=2, gd_lr=0.05, num_refine=8)
# (label, K, cost_gd). Gradient refinement is ~100-300x slower per step (FD Jacobians), so
# this is a small rigorous confirmation: can a few gradient steps at the SCARCE budget (K=16)
# reach plain MPPI's generous-budget (K=256) productive speed? Timing already says no; these
# seeds give the CI. (K=32 grad dropped — same story at 2x the cost.)
CONFIGS = [
    ("plain K=16", 16, None),
    ("grad  K=16", 16, GD),
    ("plain K=256 (ref)", 256, None),
]
SEEDS = list(range(8))
CKPT = Path(__file__).resolve().parent / "checkpoints" / "gradient_probe.jsonl"


def run():
    ckpt = Checkpoint(CKPT)
    for env in ENVS:
        for label, K, gd in CONFIGS:
            for seed in SEEDS:
                fields = dict(exp=EXP, env=env, label=label, K=K, seed=seed,
                              grad=bool(gd))
                if ckpt.has(fields):
                    continue
                try:
                    m = run_trial(env, "mppi", "fpl", seed=seed, K=K, cost_gd=gd)
                except Exception as e:
                    m = dict(error=f"{type(e).__name__}: {e}", prod=None, survived=None)
                ckpt.record(fields, m)
            print(f"  {env} / {label} done", flush=True)
    print(ckpt.summary())
    return ckpt


def report(ckpt):
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 74)
    print(f"GRADIENT PROBE — productive speed (mean±CI, %surv), {len(SEEDS)} seeds")
    print("=" * 74)
    for env in ENVS:
        print(f"\n== {env} ==")
        for label, K, _gd in CONFIGS:
            v = [r for r in rows if r["env"] == env and r["label"] == label
                 and r.get("prod") is not None]
            if not v:
                print(f"  {label:20s}  (no data)"); continue
            prod = np.array([r["prod"] for r in v]); surv = np.array([r["survived"] for r in v])
            m, h = mean_ci(prod)
            print(f"  {label:20s}  prod={m:.2f}±{h:.2f}  surv={surv.mean()*100:.0f}%")


def main():
    ckpt = run()
    report(ckpt)


if __name__ == "__main__":
    main()
