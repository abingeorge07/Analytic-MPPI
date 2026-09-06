"""Is hopper survival MONOTONE in FPL conjunction strength? And what makes it so?

Background
----------
`fpl_binding_probe.py` shows that on hopper the min-conjunction's binding atom is almost
never the safety atom (orientation): it is the height BAND at a 0.6 s horizon, and forward
SPEED at a 0.9 s horizon. Sharpening the conjunction therefore pours optimization pressure
into non-safety objectives, and survival goes DOWN as `fpl_p` / `fpl_time_p` go down.

Two candidate repairs, tested here as a 2x2:

  soft   `atom_soft_floor=0.05` — atoms get a strictly monotone exponential tail below
         their floor instead of a flat clip at 0. Fixes the ABSORBING-ZERO failure: with
         a hard clip, power_mean(p<0) is pinned at ~eps as soon as any atom clips, so the
         composite stops discriminating exactly in the failure region.

  split  `fpl_conj_indices=[0,1,3]` — the p<0 conjunction runs over the CONSTRAINT atoms
         (height, orientation, control) only; the PROGRESS atom (velocity) is composed
         with the conjunction's scalar at an outer geometric mean. Fixes the BINDING-ATOM
         INVERSION: a progress objective has no floor to hold, so putting it inside a
         "raise the least-satisfied objective" operator makes a sharper p mean "go faster".

Each is measured on its own and together, sweeping conjunction strength on both axes, at
both the myopic (0.6 s / 4 knot) and the known-good (0.9 s / 6 knot) geometry. Sampler,
budget, horizon, knots, seeds and episode length are held fixed within each geometry; only
the cost composition varies.

    python verification/fpl_monotonicity.py --geom h06 h09 --seeds 30
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment import run_trial  # noqa: E402
from _ci import wilson_ci  # noqa: E402

OUT = REPO / "runs" / "diagnostics" / "fpl_monotonicity"

GEOM: Dict[str, Dict[str, Any]] = {
    "h06": dict(spline_type="cubic", temperature=0.05, noise_level=0.6,
                plan_horizon=0.6, num_knots=4),
    "h09": dict(spline_type="cubic", temperature=0.1, noise_level=0.6,
                plan_horizon=0.9, num_knots=6),
}

# hopper atom order: 0 height, 1 orientation, 2 velocity, 3 control.
# Constraints = {height, orientation, control}; progress = {velocity}.
CONJ = [0, 1, 3]
# Outer level: harmonic mean of [conjunction, velocity] with the conjunction carrying the
# 3-atom mass it had before the split. This is chosen so the split is an EXACT no-op at the
# base point -- with inner p = outer p = -1 and weights (3,1),
#   1 / (3/4 * 1/M + 1/4 * 1/v),  M = 3 / (1/h + 1/o + 1/c)
#     = 4 / (1/h + 1/o + 1/c + 1/v) = power_mean([h,o,v,c], -1),
# the original objective, atom for atom. So the only thing the split changes is WHERE a
# sharper `fpl_p` acts: on the constraint atoms alone, instead of on all four. That makes
# it a clean control -- any difference in the sweep is attributable to the redirection of
# conjunction pressure, not to a different operating point.
SPLIT = dict(fpl_conj_indices=CONJ, fpl_outer_p=-1.0)
SOFT = dict(atom_soft_floor=0.05, atom_tail_tau=0.5)

# `margin` gives the ONLY true safety atom (orientation) room above the fall line. At the
# default the atom reaches 0 at z = 0.60, which is exactly the episode's fall threshold, so
# it is still reporting "fine" while the fall is already unrecoverable and reaches "violated"
# only once the episode is lost. A p<0 conjunction can only act on an atom that moves before
# the failure; with zero margin there is nothing for a sharper p to grip. 0.75 is ~cos(41 deg),
# comfortably inside the recoverable region. The fall threshold itself is untouched.
MARGIN = dict(orientation_floor=0.75)

VARIANTS: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {
    # name : (build kwargs, task kwargs)
    "orig":         ({}, {}),
    "soft":         ({}, dict(SOFT)),
    "soft01":       ({}, dict(atom_soft_floor=0.01, atom_tail_tau=0.5)),
    "split":        (dict(SPLIT), {}),
    "both":         (dict(SPLIT), dict(SOFT)),
    "margin":       ({}, dict(MARGIN)),
    "margin_split": (dict(SPLIT), dict(MARGIN)),
}

# Conjunction-strength sweep. The two axes are swept one at a time from the shared
# base point (fpl_p=-1, fpl_time_p=-2), so each row changes exactly one number.
P_SWEEP = [-1.0, -2.0, -4.0, -8.0]
TP_SWEEP = [-2.0, -4.0, -8.0, -16.0]


def cells() -> List[Tuple[str, Dict[str, Any]]]:
    out = [("p=-1,tp=-2", dict(fpl_p=-1.0, fpl_time_p=-2.0))]
    out += [(f"p={p:g}", dict(fpl_p=p, fpl_time_p=-2.0)) for p in P_SWEEP[1:]]
    out += [(f"tp={t:g}", dict(fpl_p=-1.0, fpl_time_p=t)) for t in TP_SWEEP[1:]]
    return out


def _job(payload):
    geom, variant, cell_kw, seed, K, steps, nthread = payload
    build, tkw = VARIANTS[variant]
    # `nthread` flows through run_trial -> run_episode -> make_controller. It only sets how
    # many threads mujoco.rollout uses; each rollout is independent, so results are bitwise
    # identical for any value. Kept small here because the outer loop is already process-
    # parallel over seeds and the default (one thread per core) would oversubscribe badly.
    extra = dict(GEOM[geom], nthread=nthread); extra.update(build); extra.update(cell_kw)
    return run_trial("hopper", "mppi", "fpl", seed=seed, K=K, steps=steps,
                     extra=extra, task_extra=dict(tkw) or None)


def run(geom: str, variants: List[str], seeds: int, K: int, steps: int,
        workers: int, nthread: int = 2) -> List[Dict[str, Any]]:
    jobs, keys = [], []
    for variant in variants:
        for label, kw in cells():
            for s in range(seeds):
                jobs.append((geom, variant, kw, s, K, steps, nthread))
                keys.append((variant, label))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=1))

    agg: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for k, r in zip(keys, res):
        agg.setdefault(k, []).append(r)

    rows = []
    for (variant, label), rs in agg.items():
        n = len(rs)
        k = int(sum(r["survived"] for r in rs))
        _, lo, hi = wilson_ci(k, n)
        rows.append(dict(geom=geom, variant=variant, cell=label, n=n,
                         survived=k / n, lo=float(lo), hi=float(hi),
                         vx=float(np.mean([r["vx"] for r in rs])),
                         prod=float(np.mean([r["prod"] for r in rs])),
                         ess=float(np.mean([r["ess"] for r in rs])),
                         minup=float(np.mean([r["minup"] for r in rs]))))
    return rows


def report(rows: List[Dict[str, Any]], variants: List[str]) -> None:
    order = [c[0] for c in cells()]
    by = {(r["variant"], r["cell"]): r for r in rows}
    geom = rows[0]["geom"]
    print(f"\n################  geometry {geom}: {GEOM[geom]}  ################")
    for variant in variants:
        print(f"\n  --- {variant} ---")
        print(f"  {'cell':12s} {'surv':>5s} {'Wilson95':>14s} {'vx':>7s} "
              f"{'prod':>7s} {'ess':>6s} {'minup':>6s}")
        seq = []
        for c in order:
            r = by.get((variant, c))
            if r is None:
                continue
            seq.append(r["survived"])
            print(f"  {c:12s} {r['survived']:5.2f} [{r['lo']:.2f},{r['hi']:.2f}]".ljust(40)
                  + f"{r['vx']:>+7.3f} {r['prod']:>+7.3f} {r['ess']:>6.1f} {r['minup']:>6.2f}")
        # Monotonicity is judged separately per axis: cells 0..3 sweep fpl_p, cells
        # 0 then 4..6 sweep fpl_time_p. "Non-decreasing" means no cell is worse than the
        # base point by more than sampling noise (compared on the Wilson bounds).
        p_seq = seq[:4]
        tp_seq = [seq[0]] + seq[4:]
        base_lo = by[(variant, order[0])]["lo"]
        worst_p = min(p_seq[1:]) if len(p_seq) > 1 else float("nan")
        worst_tp = min(tp_seq[1:]) if len(tp_seq) > 1 else float("nan")
        drop_p = by[(variant, order[int(np.argmin(p_seq[1:])) + 1])]
        drop_tp = by[(variant, order[4 + int(np.argmin(tp_seq[1:]))])]
        print(f"    fpl_p      : base {seq[0]:.2f} -> worst {worst_p:.2f}   "
              f"{'NON-DECREASING' if drop_p['hi'] >= base_lo and worst_p >= seq[0] - 1e-9 else ('within-noise' if drop_p['hi'] >= base_lo else 'DEGRADES')}")
        print(f"    fpl_time_p : base {seq[0]:.2f} -> worst {worst_tp:.2f}   "
              f"{'NON-DECREASING' if drop_tp['hi'] >= base_lo and worst_tp >= seq[0] - 1e-9 else ('within-noise' if drop_tp['hi'] >= base_lo else 'DEGRADES')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geom", nargs="+", default=["h06", "h09"], choices=list(GEOM))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("-K", type=int, default=256)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--workers", type=int, default=13)
    p.add_argument("--nthread", type=int, default=2)
    p.add_argument("--tag", default="mono")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    for g in args.geom:
        rows = run(g, args.variants, args.seeds, args.K, args.steps, args.workers,
                   nthread=args.nthread)
        report(rows, args.variants)
        allrows += rows
    (OUT / f"{args.tag}.json").write_text(json.dumps(allrows, indent=2))
    print(f"\nwrote {OUT / f'{args.tag}.json'}")


if __name__ == "__main__":
    main()
