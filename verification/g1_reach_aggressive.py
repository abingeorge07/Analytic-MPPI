"""G1 humanoid timebox — does an AGGRESSIVE forward-reach target turn the (null) g1_reach task
into a genuine dynamic-competition FPL win?

`g1_reach` at a MODEST target (0.2 m) is a documented null: a conservative linear weight leans a
little and holds, so no dynamic competition and FPL wins nothing (`g1_reach_study.py`). The
hypothesis here: push the target FORWARD (0.4–0.7 m) into the regime where a safe static lean
can't reach it — reaching then requires leaning the torso COM toward the edge of support (a
height/topple risk), which is the hopper structure applied to posture. If FPL then reaches
farther while staying tall+upright than the best linear weight, g1 becomes a 5th robot; if the
conservative linear weight still wins (or everything collapses), it's an honest null.

Checkpointed. Run:  .venv/bin/python verification/g1_reach_aggressive.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

from analytic_mppi.eval import run_episode, make_task, init_g1_stand
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _checkpoint import Checkpoint       # noqa: E402
from _ci import mean_ci, wilson_ci        # noqa: E402

EXP = "g1_reach_aggr"
TARGETS = [0.3, 0.5, 0.7]
SEEDS = list(range(6))
H_FLOOR, RZ_FLOOR = 0.5, 0.5             # tall + upright thresholds (rz alone is a trap)
BUILD = dict(num_samples=128, plan_horizon=1.0, num_knots=6, spline_type="zero",
             iterations=3, noise_level=0.3, temperature=0.1, fpl_group_p=1.0)
# (label, fpl_p, fpl_weights)  — layered cost; weights order [orient,height,posture,ctrl,fwd,wobble]
CONFIGS = [
    ("FPL p=-1", -1.0, None),
    ("lin wf=1", 1.0, [1, 1, 1, 1, 1, 1]),
    ("lin wf=2", 1.0, [1, 1, 1, 1, 2, 1]),
    ("lin wf=4", 1.0, [1, 1, 1, 1, 4, 1]),
]
CKPT = Path(__file__).resolve().parent / "checkpoints" / "g1_reach_aggr.jsonl"


def run():
    ckpt = Checkpoint(CKPT)
    for tr in TARGETS:
        task = make_task("g1_reach", target_reach=tr)
        for label, fpl_p, w in CONFIGS:
            for seed in SEEDS:
                fields = dict(exp=EXP, target=tr, label=label, seed=seed)
                if ckpt.has(fields):
                    continue
                kw = dict(BUILD, fpl_p=fpl_p)
                if w is not None:
                    kw["fpl_weights"] = w
                try:
                    res = run_episode("g1_reach", "mppi", steps=150, seed=seed,
                                      cost_mode="fpl_layered", init_fn=init_g1_stand,
                                      task_kwargs=dict(target_reach=tr), **kw)
                    sd = res["sd"]
                    x = sd[..., task._torso_pos_adr]
                    h = task._torso_height(sd)
                    rz = task._torso_orientation(sd)[..., 2]
                    safe = bool(h.min() > H_FLOOR and rz.min() > RZ_FLOOR)
                    reach = float(x[-1])
                    m = dict(reach=reach, min_h=float(h.min()), rz_min=float(rz.min()),
                             safe=(1.0 if safe else 0.0),
                             prod=(reach if safe else 0.0))
                except Exception as e:
                    m = dict(error=f"{type(e).__name__}: {e}", reach=None, safe=None, prod=None)
                ckpt.record(fields, m)
            print(f"  target={tr} {label} done", flush=True)
    print(ckpt.summary())
    return ckpt


def report(ckpt):
    rows = ckpt.rows(exp=EXP)
    print("\n" + "=" * 80)
    print("G1 AGGRESSIVE REACH — reach_x (m) at tall+upright safety, 6 seeds")
    print("=" * 80)
    verdict = []
    for tr in TARGETS:
        print(f"\n-- target_reach = {tr} m --")
        print(f"{'config':12s}{'reach':>8s}{'min_h':>8s}{'rz_min':>8s}{'safe%':>7s}{'prod':>8s}")
        cells = {}
        for label, _p, _w in CONFIGS:
            v = [r for r in rows if r["target"] == tr and r["label"] == label
                 and r.get("reach") is not None]
            if not v:
                continue
            reach = np.array([r["reach"] for r in v]); safe = np.array([r["safe"] for r in v])
            prod = np.array([r["prod"] for r in v])
            minh = np.mean([r["min_h"] for r in v]); rzm = np.mean([r["rz_min"] for r in v])
            cells[label] = dict(reach=reach.mean(), safe=safe.mean(), prod=prod.mean())
            print(f"{label:12s}{reach.mean():8.2f}{minh:8.2f}{rzm:8.2f}"
                  f"{safe.mean()*100:6.0f}%{prod.mean():8.2f}")
        # verdict: does FPL have the highest productive (safe) reach?
        if cells:
            best = max(cells, key=lambda k: cells[k]["prod"])
            verdict.append((tr, best, cells))
    print("\n" + "=" * 80)
    fpl_wins = [tr for tr, best, _ in verdict if best.startswith("FPL")]
    print(f"FPL has the highest SAFE reach at targets: {fpl_wins if fpl_wins else 'NONE'}")
    if fpl_wins:
        print("=> g1 shows dynamic competition in the aggressive-reach regime (candidate 5th robot).")
    else:
        print("=> honest NULL: conservative linear still wins or all collapse (horizon-limited).")


def main():
    ckpt = run()
    report(ckpt)


if __name__ == "__main__":
    main()
