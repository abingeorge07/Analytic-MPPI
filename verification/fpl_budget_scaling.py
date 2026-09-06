"""Does spending MORE samples on the FPL objective make the hopper safer or less safe?

The capability sweep found survival FALLING as K rose (256 -> 512 -> 1024) at a fixed
objective — the third of the three "optimize harder, survive less" axes, alongside `fpl_p`
and `fpl_time_p`. If the objective's argmin is a progress atom, or if the conjunction
saturates at its clipped zero, then a bigger sample budget just finds a better optimum of
the wrong thing, and more samples should hurt. Repairing the objective should make budget
scaling monotone again.

So this sweeps K for the original cost and for the repaired one (`both` = soft atom floors
+ progress atom outside the conjunction), holding sampler / horizon / knots / seeds /
episode length fixed.

    python verification/fpl_budget_scaling.py --seeds 100 --Ks 256 512 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment import run_trial  # noqa: E402
from _ci import wilson_ci  # noqa: E402
from fpl_monotonicity import GEOM, VARIANTS  # noqa: E402

OUT = REPO / "runs" / "diagnostics" / "fpl_monotonicity"


def _job(payload):
    geom, variant, cell_kw, seed, K, steps, nthread = payload
    build, tkw = VARIANTS[variant]
    extra = dict(GEOM[geom], nthread=nthread)
    extra.update(build)
    extra.update(cell_kw)
    return run_trial("hopper", "mppi", "fpl", seed=seed, K=K, steps=steps,
                     extra=extra, task_extra=dict(tkw) or None)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", default="h09", choices=list(GEOM))
    p.add_argument("--variants", nargs="+", default=["orig", "both"])
    p.add_argument("--Ks", nargs="+", type=int, default=[256, 512, 1024])
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--workers", type=int, default=26)
    p.add_argument("--nthread", type=int, default=1)
    p.add_argument("--cell", default="p=-1,tp=-2")
    p.add_argument("--tag", default="budget")
    args = p.parse_args()

    cell_kw = dict(fpl_p=-1.0, fpl_time_p=-2.0)
    jobs, keys = [], []
    for variant in args.variants:
        for K in args.Ks:
            for s in range(args.seeds):
                jobs.append((args.geom, variant, cell_kw, s, K, args.steps, args.nthread))
                keys.append((variant, K))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=1))

    agg: Dict[Any, List[Dict[str, Any]]] = {}
    for k, r in zip(keys, res):
        agg.setdefault(k, []).append(r)

    rows = []
    print(f"\nbudget scaling, geometry {args.geom}, cell {args.cell}, "
          f"{args.seeds} seeds, {args.steps} steps")
    for variant in args.variants:
        print(f"\n  --- {variant} ---")
        print(f"  {'K':>5s} {'surv':>5s} {'Wilson95':>13s} {'vx':>7s} {'prod':>7s} "
              f"{'ess':>6s} {'minup':>6s}")
        for K in args.Ks:
            rs = agg[(variant, K)]
            n = len(rs)
            k = int(sum(r["survived"] for r in rs))
            _, lo, hi = wilson_ci(k, n)
            row = dict(geom=args.geom, variant=variant, K=K, n=n, survived=k / n,
                       lo=float(lo), hi=float(hi),
                       vx=float(np.mean([r["vx"] for r in rs])),
                       prod=float(np.mean([r["prod"] for r in rs])),
                       ess=float(np.mean([r["ess"] for r in rs])),
                       minup=float(np.mean([r["minup"] for r in rs])))
            rows.append(row)
            print(f"  {K:>5d} {row['survived']:5.2f} [{lo:.2f},{hi:.2f}] "
                  f"{row['vx']:>+7.3f} {row['prod']:>+7.3f} {row['ess']:>6.1f} "
                  f"{row['minup']:>6.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.tag}.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {OUT / f'{args.tag}.json'}")


if __name__ == "__main__":
    main()
