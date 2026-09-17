"""S2 / GATE G1 — do the published hopper results survive the fulfillment-atom floor?

NEXT_STEPS S2: re-run the headline studies with the floor ON and OFF, same seeds, same
configs, and report the delta. WO-3.4's point is that a floor is needed only for p<0
(`dM_p/dx_i` diverges as an atom -> 0) and is a no-op at p=1, so introducing it moves the
FPL arm and leaves the linear arm alone unless it is applied to both. This script measures
how much that costs on the numbers already written up.

Two things this does NOT do, deliberately:

  * It does not touch `verification/checkpoints/*.jsonl`. Those files are the record of
    the published campaign and are left byte-for-byte alone. Results go to
    `checkpoints/s2_atom_floor/<study>_floor<eps>.jsonl` -- a separate file per floor, so
    a floor-on trial can never be served from a floor-off cache. (Trial keys do not
    contain the floor, so sharing one file WOULD silently do exactly that.)

  * It does not reuse the August checkpoints as the floor-off baseline. They do not
    reproduce under the current environment -- they were produced under an unbounded
    `mujoco>=3.2` pin (resolving to 3.8.1); HEAD pins `mujoco>=3.2,<3.6` and the venv is
    on 3.5.0, which changes fixed-seed hopper trajectories. Invariant 11.3 warns about
    exactly this. Both arms of the A/B are therefore recomputed here, under one
    environment, so the delta measures the FLOOR and nothing else.

Run:  ./a-mppi/bin/python verification/s2_atom_floor_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _experiment                                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci, wilson_ci                        # noqa: E402
from analytic_mppi.tasks.base import (ATOM_FLOOR_LEGACY,  # noqa: E402
                                      ATOM_FLOOR_RECOMMENDED)

import fpl_portability                                    # noqa: E402
import fpl_sampler_race                                   # noqa: E402
import fpl_zero_tuning                                    # noqa: E402

ENV = "hopper"
FLOORS = [ATOM_FLOOR_LEGACY, ATOM_FLOOR_RECOMMENDED]      # 1e-8 (incumbent) vs 1e-3
OUT = Path(__file__).resolve().parent / "checkpoints" / "s2_atom_floor"

# (module, study name, the module globals to pin to hopper-only)
STUDIES = [
    (fpl_portability,  "portability"),
    (fpl_sampler_race, "sampler_race"),
    (fpl_zero_tuning,  "zero_tuning"),
]


def _tag(floor: float) -> str:
    return f"{floor:.0e}".replace("-0", "-")


def _path(name: str, floor: float, tv: bool = False) -> Path:
    """One checkpoint file per (study, floor, terminal_value). The `_tv` suffix is added
    only when terminal_value is ON, so the floor-only files written for G1/G2 keep their
    names and stay valid (they were all produced with terminal_value=False)."""
    return OUT / f"{name}_floor{_tag(floor)}{'_tv' if tv else ''}.jsonl"


def run_study_at(mod, name: str, floor: float, tv: bool = False) -> Checkpoint:
    """Run one study, hopper only, at one (floor, terminal_value), into its own file."""
    _experiment.set_atom_floor(floor)
    _experiment.set_terminal_value(tv)
    # Pin the study to hopper and redirect its checkpoint. Restored afterwards so
    # importing this driver never leaves a study module mutated.
    saved = (mod.ENVS, mod.CKPT)
    mod.ENVS, mod.CKPT = [ENV], _path(name, floor, tv)
    try:
        ckpt = mod.run(progress=False)
    finally:
        mod.ENVS, mod.CKPT = saved
        _experiment.set_atom_floor(ATOM_FLOOR_LEGACY)
        _experiment.set_terminal_value(False)
    return ckpt


# ---------------------------------------------------------------- aggregation

def _rows(name: str, floor: float, tv: bool = False) -> list[dict]:
    p = _path(name, floor, tv)
    if not p.exists():
        return []
    return [r for r in Checkpoint(p).rows() if r.get("env") == ENV]


def _key_field(name: str) -> str:
    return "obj" if name == "zero_tuning" else ("cost" if name == "portability" else "label")


def _stat(rows, kf, cfg):
    v = [r for r in rows if r.get(kf) == cfg and r.get("prod") is not None]
    if not v:
        return None
    prod = np.array([r["prod"] for r in v], float)
    surv = np.array([r["survived"] for r in v], float)
    m, h = mean_ci(prod)
    _, s_lo, s_hi = wilson_ci(int(surv.sum()), len(surv))
    return dict(m=m, h=h, s=float(surv.mean()), slo=s_lo, shi=s_hi, n=len(v))


def _disjoint(a_lo, a_hi, b_lo, b_hi) -> bool:
    return (a_hi < b_lo) or (b_hi < a_lo)


def report() -> None:
    lo, hi = FLOORS
    print("=" * 104)
    print(f"GATE G1 — hopper, floor {lo:.0e} (incumbent) vs {hi:.0e}, "
          f"{len(fpl_portability.SEEDS)} seeds.  prod = achieved speed x survival.  CI = 95%.")
    print("=" * 104)

    for mod, name in STUDIES:
        kf = _key_field(name)
        rows_lo, rows_hi = _rows(name, lo), _rows(name, hi)
        if not rows_lo or not rows_hi:
            print(f"\n[{name}] no rows — run() first")
            continue
        # The sampler race holds the COST fixed at FPL and varies the proposal, so its
        # rows are not an FPL-vs-linear contrast at all -- labelling them as arms would
        # misread the study.
        note = "   [cost held FIXED at FPL; the SAMPLER varies]" if name == "sampler_race" else ""
        print(f"\n### {name}{note}")
        print(f"{'config':24s} {'prod @1e-8':>16s} {'prod @1e-3':>16s} {'Dprod':>8s}  "
              f"{'surv @1e-8':>17s} {'surv @1e-3':>17s}  flag")
        for cfg in sorted({r.get(kf) for r in rows_lo}):
            a, b = _stat(rows_lo, kf, cfg), _stat(rows_hi, kf, cfg)
            if a is None or b is None:
                continue
            flags = []
            if _disjoint(a["m"] - a["h"], a["m"] + a["h"], b["m"] - b["h"], b["m"] + b["h"]):
                flags.append("PROD-MOVED")
            if _disjoint(a["slo"], a["shi"], b["slo"], b["shi"]):
                flags.append("SURV-MOVED")
            print(f"{str(cfg):24s} {a['m']:7.4f}+-{a['h']:<6.4f} {b['m']:7.4f}+-{b['h']:<6.4f} "
                  f"{b['m']-a['m']:+8.4f}  {a['s']:5.2f}[{a['slo']:.2f},{a['shi']:.2f}] "
                  f"{b['s']:5.2f}[{b['slo']:.2f},{b['shi']:.2f}]  {','.join(flags)}")

    # The per-config table is diagnostic. THIS is the quantity G1 gates: the FPL claim is
    # "FPL beats the whole linear-weight family", so the statistic is FPL minus the BEST
    # linear weight -- not any single linear config's delta.
    print("\n" + "=" * 104)
    print("HEADLINE — FPL vs the BEST linear weight. This gap is what G1 actually gates.")
    print("=" * 104)
    for mod, name in STUDIES:
        if name == "sampler_race":
            continue                      # no linear arm in this study
        kf = _key_field(name)
        print(f"\n{name}:")
        for floor in FLOORS:
            rows = _rows(name, floor)
            if not rows:
                continue
            f = _stat(rows, kf, "fpl")
            lins = [(c, _stat(rows, kf, c)) for c in sorted({r.get(kf) for r in rows})
                    if c != "fpl"]
            lins = [(c, s) for c, s in lins if s is not None]
            if f is None or not lins:
                continue
            bc, bs = max(lins, key=lambda t: t[1]["m"])
            sep = ("FPL CI clears best-linear CI"
                   if f["m"] - f["h"] > bs["m"] + bs["h"] else "CIs OVERLAP")
            print(f"  floor {floor:.0e}:  FPL {f['m']:.4f}+-{f['h']:.4f} (surv {f['s']:.2f})   "
                  f"best-lin[{bc}] {bs['m']:.4f}+-{bs['h']:.4f} (surv {bs['s']:.2f})   "
                  f"GAP {f['m']-bs['m']:+.4f}   [{sep}]")


def sweep_report(floors: list[float]) -> None:
    """S3 / GATE G2: is the FPL advantage a function of the floor?

    The objection this answers is the successor to lambda/p confounding -- 'FPL only wins
    because you floored the atoms, and the floor is only needed for p<0'. If the FPL-minus-
    best-linear gap is flat across decades of epsilon, that objection is answered with a
    number. If it grows as epsilon shrinks, the advantage is partly a conditioning artifact
    and must be reported as one.
    """
    print("\n" + "=" * 104)
    print("GATE G2 — FPL advantage vs the atom floor (hopper)")
    print("=" * 104)
    for mod, name in STUDIES:
        if name == "sampler_race":
            continue                          # no linear arm; not an FPL-vs-linear contrast
        kf = _key_field(name)
        print(f"\n{name}:")
        print(f"  {'floor':>8s} {'FPL prod':>18s} {'best-linear prod':>24s} {'GAP':>9s}  separation")
        for floor in floors:
            rows = _rows(name, floor)
            if not rows:
                print(f"  {floor:8.0e}  (not run)")
                continue
            f = _stat(rows, kf, "fpl")
            lins = [(c, _stat(rows, kf, c)) for c in sorted({r.get(kf) for r in rows})
                    if c != "fpl"]
            lins = [(c, s) for c, s in lins if s is not None]
            if f is None or not lins:
                continue
            bc, bs = max(lins, key=lambda t: t[1]["m"])
            sep = "clears" if f["m"] - f["h"] > bs["m"] + bs["h"] else "OVERLAPS"
            print(f"  {floor:8.0e} {f['m']:8.4f}+-{f['h']:<6.4f} "
                  f"{bs['m']:10.4f}+-{bs['h']:<6.4f} [{bc}] {f['m']-bs['m']:+9.4f}  {sep}")


def tv_report(floors: list[float]) -> None:
    """GATE G3 follow-through: do the headline results survive the terminal-value fix?

    WO-3.3 is not a small correction at the study horizon -- with H=30 and gamma=0.99 the
    post-horizon tail is 75% of the discount mass, so the terminal step's weight goes
    0.029 -> 0.747. If the FPL-vs-linear claim depends on the horizon truncation it will
    show up here, and the 2x2 separates it from the floor.
    """
    print("\n" + "=" * 104)
    print("GATE G3 (follow-through) — headline gap under terminal_value OFF vs ON")
    print("=" * 104)
    for mod, name in STUDIES:
        if name == "sampler_race":
            continue
        kf = _key_field(name)
        print(f"\n{name}:")
        print(f"  {'floor':>8s} {'tv':>4s} {'FPL prod':>18s} {'best-linear prod':>22s} "
              f"{'GAP':>9s}  separation")
        for floor in floors:
            for tv in (False, True):
                rows = _rows(name, floor, tv)
                if not rows:
                    print(f"  {floor:8.0e} {str(tv):>4s}  (not run)")
                    continue
                f = _stat(rows, kf, "fpl")
                lins = [(c, _stat(rows, kf, c)) for c in sorted({r.get(kf) for r in rows})
                        if c != "fpl"]
                lins = [(c, s) for c, s in lins if s is not None]
                if f is None or not lins:
                    continue
                bc, bs = max(lins, key=lambda t: t[1]["m"])
                sep = ("clears" if f["m"] - f["h"] > bs["m"] + bs["h"] else "OVERLAPS")
                print(f"  {floor:8.0e} {str(tv):>4s} {f['m']:8.4f}+-{f['h']:<6.4f} "
                      f"{bs['m']:8.4f}+-{bs['h']:<6.4f} [{bc}] {f['m']-bs['m']:+9.4f}  {sep}")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--floors", nargs="+", type=float, default=None,
                    help="atom floors to run (default: the G1 A/B pair)")
    ap.add_argument("--studies", nargs="+", default=None,
                    help="subset of study names to run")
    ap.add_argument("--sweep", action="store_true",
                    help="also print the GATE G2 floor-sensitivity table")
    ap.add_argument("--terminal-value", action="store_true",
                    help="run with WO-3.3's terminal value ON as well, and print the "
                         "GATE G3 floor x terminal_value table")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args(argv)

    floors = a.floors if a.floors else FLOORS
    studies = ([s for s in STUDIES if s[1] in set(a.studies)] if a.studies else STUDIES)
    tvs = (False, True) if a.terminal_value else (False,)

    OUT.mkdir(parents=True, exist_ok=True)
    if not a.report_only:
        for floor in floors:
            for tv in tvs:
                for mod, name in studies:
                    print(f"[run] {name} @ floor {floor:.0e} terminal_value={tv}", flush=True)
                    run_study_at(mod, name, floor, tv)
    report()
    if a.sweep:
        sweep_report(sorted(set(FLOORS) | set(floors), reverse=True))
    if a.terminal_value:
        tv_report(sorted(set(floors), reverse=True))


if __name__ == "__main__":
    main()
