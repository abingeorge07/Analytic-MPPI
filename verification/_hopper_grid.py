"""Direct multiprocessing grid search over hopper cost+controller configs.

Replaces the LLM-agent search (which stalled on background-poll). Each config is evaluated
by _hopper_search.evaluate in its own worker process (nthread=1 so N workers = N cores),
ranked by the survival-gated speed+periodicity composite. Deterministic, no polling.

Usage:  .venv/bin/python verification/_hopper_grid.py <stage> [n_workers]
Stages build progressively finer grids (edit STAGES below). Results saved to
verification/checkpoints/hopper_grid_stage<N>.json (ranked).
"""
from __future__ import annotations
import sys, json, itertools, time, warnings
from pathlib import Path
from multiprocessing import Pool

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _eval_one(cfg):
    from _hopper_search import evaluate           # import inside worker
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    np.seterr(all="ignore")
    try:
        m = evaluate(dict(cfg))
    except Exception as e:                         # never let one config kill the pool
        return dict(cfg=cfg, error=str(e), composite=-1)
    return dict(cfg=cfg, **{k: m[k] for k in
                ("survive_min", "survive_frac", "minup", "meanvx", "travel",
                 "periodicity", "n_hops", "composite")})


def grid(base, **axes):
    """Cartesian product of axes merged onto base. axes values are lists."""
    keys = list(axes); out = []
    for combo in itertools.product(*[axes[k] for k in keys]):
        c = dict(base); c.update(dict(zip(keys, combo))); out.append(c)
    return out


def build(stage):
    if stage == 1:
        base = dict(z_full=0.99, hop_w=1, hop_vz=1, H=0.8, knots=6, K=512,
                    time_p=-2.0, fpl_p=-1.0, steps=750, seeds=[0, 1, 2], nthread=1)
        cfgs = []
        for hf, hfl in [(1.0, 0.6), (1.05, 0.7), (1.1, 0.8), (1.05, 0.6), (1.1, 0.7)]:
            cfgs += grid(dict(base, h_full=hf, h_floor=hfl),
                         z_floor=[0.90, 0.93, 0.96], TV=[1.0, 1.5, 2.0])
        return cfgs
    if stage == 2:
        # Refine the winning region (1.05/0.7 & 1.1/0.8 bands, z_floor 0.9-0.96, TV1.0)
        # with more budget/horizon (push survival toward 15s) + two-sided velocity for speed.
        base = dict(z_full=0.99, hop_w=1, hop_vz=1, knots=6, time_p=-2.0, fpl_p=-1.0,
                    steps=750, seeds=[0, 1, 2], nthread=1)
        cfgs = []
        # survival/quality push: budget + horizon on the best bands (one-sided, TV1.0)
        for hf, hfl in [(1.05, 0.7), (1.1, 0.8), (1.1, 0.7)]:
            cfgs += grid(dict(base, h_full=hf, h_floor=hfl, TV=1.0),
                         z_floor=[0.9, 0.96], K=[768, 1024], H=[1.0])
        # speed push: two-sided velocity (cap top speed -> less lunge-topple) at higher TV
        for hf, hfl in [(1.05, 0.7), (1.1, 0.8)]:
            cfgs += grid(dict(base, h_full=hf, h_floor=hfl, vel_mode="twoside",
                              hi_over=1.3, K=768, H=1.0),
                         z_floor=[0.93, 0.96], TV=[1.5, 2.0])
        return cfgs
    if stage == 3:
        # Final verification of the best candidates over 6 seeds (robust, not 3-seed noise),
        # in the winning regime (K512, H0.8), plus finer control resolution (knots=8).
        base = dict(z_full=0.99, hop_w=1, hop_vz=1, knots=6, H=0.8, K=512,
                    time_p=-2.0, fpl_p=-1.0, steps=750, seeds=[0, 1, 2, 3, 4, 5], nthread=1)
        picks = [
            dict(h_full=1.05, h_floor=0.7, z_floor=0.90, TV=1.0),               # stage-1 best survival
            dict(h_full=1.10, h_floor=0.8, z_floor=0.96, TV=1.0),               # never-topple
            dict(h_full=1.10, h_floor=0.8, z_floor=0.90, TV=1.0),               # more vx
            dict(h_full=1.10, h_floor=0.7, z_floor=0.93, TV=1.5),               # high minup, periodicity
            dict(h_full=1.05, h_floor=0.7, z_floor=0.93, TV=1.0),               # interpolate
            dict(h_full=1.05, h_floor=0.7, z_floor=0.90, TV=1.0, knots=8),      # finer control
            dict(h_full=1.10, h_floor=0.8, z_floor=0.93, TV=1.0, knots=8),      # finer control
        ]
        return [dict(base, **p) for p in picks]
    if stage == 4:
        # Controller-smoothness sweep (never explored before): noise / temperature / spline_type
        # on the two best cost configs. Cleaner control may extend sustained hopping.
        cfgs = []
        for hf, hfl, zfl in [(1.1, 0.8, 0.90), (1.05, 0.7, 0.90)]:
            base = dict(h_full=hf, h_floor=hfl, z_full=0.99, z_floor=zfl, TV=1.0,
                        hop_w=1, hop_vz=1, knots=6, H=0.8, K=512, time_p=-2.0, fpl_p=-1.0,
                        steps=750, seeds=[0, 1, 2, 3], nthread=1)
            cfgs += grid(base, noise=[0.15, 0.3], temp=[0.1, 0.2],
                         spline_type=["zero", "linear"])
        return cfgs
    if stage == 5:
        # CEILING PUSH: structural levers never tried — MPPI refinement `iterations`,
        # bigger sample budget K, and longer horizon with SCALED knots (fine resolution).
        # One-factor-at-a-time from the best cost config, plus aggressive combos. 4 seeds.
        best = dict(h_full=1.1, h_floor=0.8, z_full=0.99, z_floor=0.9, TV=1.0,
                    hop_w=1, hop_vz=1, knots=6, H=0.8, K=512, noise=0.3, temp=0.1,
                    time_p=-2.0, fpl_p=-1.0, steps=750, seeds=[0, 1, 2, 3], nthread=1)
        variants = [
            dict(),                                       # reference
            dict(iterations=2),                           # MPPI refinement
            dict(iterations=3),
            dict(K=1024),                                 # budget
            dict(K=2048),
            dict(H=1.0, knots=8),                         # longer horizon, scaled knots
            dict(H=1.2, knots=10),
            dict(iterations=2, K=1024),                   # combos
            dict(iterations=2, H=1.0, knots=8),
            dict(iterations=3, K=1024, H=1.0, knots=8),   # aggressive
        ]
        return [dict(best, **v) for v in variants]
    raise ValueError(f"stage {stage} not defined yet")


def main():
    stage = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    cfgs = build(stage)
    print(f"stage {stage}: {len(cfgs)} configs on {nw} workers", flush=True)
    t0 = time.time()
    done = []
    outdir = HERE / "checkpoints"; outdir.mkdir(exist_ok=True)
    out = outdir / f"hopper_grid_stage{stage}.json"
    with Pool(nw) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_one, cfgs), 1):
            done.append(r)
            out.write_text(json.dumps(sorted(done, key=lambda x: -x.get("composite", -1)), indent=2))  # incremental save
            c = r["cfg"]
            tag = (f"h{c.get('h_full')}/{c.get('h_floor')} z_fl{c.get('z_floor')} TV{c.get('TV')} "
                   f"K{c.get('K')} H{c.get('H')}/kn{c.get('knots')} it{c.get('iterations',1)}"
                   + ("" if "cost_gd" not in c else " gd"))
            if "error" in r:
                print(f"[{i}/{len(cfgs)}] ERR {tag}: {r['error'][:80]}", flush=True)
            else:
                print(f"[{i}/{len(cfgs)}] {tag:40s} surv_min={r['survive_min']:3d} minup={r['minup']:+.2f} "
                      f"vx={r['meanvx']:+.2f} per={r['periodicity']:.2f} comp={r['composite']:.1f}", flush=True)
    done.sort(key=lambda r: -r.get("composite", -1))
    outdir = HERE / "checkpoints"; outdir.mkdir(exist_ok=True)
    out = outdir / f"hopper_grid_stage{stage}.json"
    out.write_text(json.dumps(done, indent=2))
    print(f"\n=== TOP 12 (of {len(cfgs)}) in {time.time()-t0:.0f}s -> {out} ===")
    for r in done[:12]:
        if "error" in r: continue
        c = r["cfg"]
        print(f"comp={r['composite']:6.1f} surv_min={r['survive_min']:3d} minup={r['minup']:+.2f} "
              f"vx={r['meanvx']:+.2f} per={r['periodicity']:.2f} | "
              f"h{c['h_full']}/{c['h_floor']} z{c['z_full']}/{c['z_floor']} TV{c['TV']} "
              f"vm={c.get('vel_mode','one')} K{c['K']} H{c['H']} kn{c['knots']} "
              f"n{c.get('noise',0.3)} t{c.get('temp',0.2)} sp{c.get('spline_type','zero')}")


if __name__ == "__main__":
    main()
