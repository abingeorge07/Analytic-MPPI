"""GATE G3 follow-through — do the headline hopper results survive WO-3.3's terminal value?

The runbook says: on G3, re-run the studies with the terminal fix on. That turned out not to
be a one-line change, because **WO-3.3 as specified does not reach the published objective.**

The published hopper/walker spec is `fpl_cost` + `fpl_time_p = -2.0` + `fpl_time_discount =
False`. In that path `_score_fpl` calls `power_mean(per_step, time_p, weights=None)` -- an
UNWEIGHTED soft-min over time. There is no `(1-g)/(1-g^H)` renormalization in it at all, so
the thing WO-3.3 replaces is not present and `terminal_value=True` is inert. (The controller
now raises rather than silently no-op, per invariant 11.5.) WO-3.3's premise applies to the
`time_p=None` and `fpl_discounted` paths, which the published results do not use.

So a single on/off comparison is impossible; reaching the terminal value from the published
spec requires moving TWO knobs. This script separates them:

    A  published    time_p=-2, time_discount=False, terminal_value=False
    B  +discount    time_p=-2, time_discount=True,  terminal_value=False
    C  +tail        time_p=-2, time_discount=True,  terminal_value=True

A->B isolates discount-weighting the soft-min; B->C isolates WO-3.3's tail term. Only B->C is
the terminal-value question. A->B is a confounder that has to be reported separately or the
tail gets credit/blame for the discounting.

Run:  ./a-mppi/bin/python verification/s4_terminal_value_g3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _experiment                                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci, wilson_ci                        # noqa: E402

import fpl_portability                                    # noqa: E402
import fpl_zero_tuning                                    # noqa: E402

ENV = "hopper"
FLOOR = 1e-8                                              # held fixed; G1/G2 covered the floor
OUT = Path(__file__).resolve().parent / "checkpoints" / "s4_terminal_value"

ARMS = [("A_published", False, False),
        ("B_discount",  True,  False),
        ("C_tail",      True,  True)]

STUDIES = [(fpl_portability, "portability", "cost"),
           (fpl_zero_tuning, "zero_tuning", "obj")]


def run_arm(mod, name: str, arm: str, time_discount: bool, tv: bool) -> Checkpoint:
    _experiment.set_atom_floor(FLOOR)
    _experiment.set_time_discount(time_discount)
    _experiment.set_terminal_value(tv)
    saved = (mod.ENVS, mod.CKPT)
    mod.ENVS, mod.CKPT = [ENV], OUT / f"{name}_{arm}.jsonl"
    try:
        return mod.run(progress=False)
    finally:
        mod.ENVS, mod.CKPT = saved
        _experiment.set_time_discount(False)
        _experiment.set_terminal_value(False)


def _stat(rows, kf, cfg):
    v = [r for r in rows if r.get(kf) == cfg and r.get("prod") is not None]
    if not v:
        return None
    prod = np.array([r["prod"] for r in v], float)
    surv = np.array([r["survived"] for r in v], float)
    m, h = mean_ci(prod)
    _, s_lo, s_hi = wilson_ci(int(surv.sum()), len(surv))
    return dict(m=m, h=h, s=float(surv.mean()), slo=s_lo, shi=s_hi, n=len(v))


def report() -> None:
    print("=" * 100)
    print("GATE G3 follow-through — hopper, floor 1e-8, 16 seeds")
    print("  A published | B +discount-weighted soft-min | C +WO-3.3 tail")
    print("  Only B->C is the terminal-value question; A->B is the confounder.")
    print("=" * 100)
    for mod, name, kf in STUDIES:
        print(f"\n### {name}")
        print(f"  {'arm':12s} {'FPL prod':>17s} {'best-linear prod':>21s} {'GAP':>9s}  "
              f"{'FPL surv':>9s}  separation")
        for arm, _, _ in ARMS:
            p = OUT / f"{name}_{arm}.jsonl"
            if not p.exists():
                print(f"  {arm:12s} (not run)")
                continue
            rows = [r for r in Checkpoint(p).rows() if r.get("env") == ENV]
            f = _stat(rows, kf, "fpl")
            lins = [(c, _stat(rows, kf, c)) for c in sorted({r.get(kf) for r in rows})
                    if c != "fpl"]
            lins = [(c, s) for c, s in lins if s is not None]
            if f is None or not lins:
                print(f"  {arm:12s} (no rows)")
                continue
            bc, bs = max(lins, key=lambda t: t[1]["m"])
            sep = "clears" if f["m"] - f["h"] > bs["m"] + bs["h"] else "OVERLAPS"
            print(f"  {arm:12s} {f['m']:8.4f}+-{f['h']:<6.4f} {bs['m']:8.4f}+-{bs['h']:<6.4f}"
                  f" [{bc}] {f['m']-bs['m']:+9.4f}  {f['s']:9.2f}  {sep}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for arm, td, tv in ARMS:
        for mod, name, _ in STUDIES:
            print(f"[run] {name} / {arm} (time_discount={td}, terminal_value={tv})", flush=True)
            run_arm(mod, name, arm, td, tv)
    report()


if __name__ == "__main__":
    main()
