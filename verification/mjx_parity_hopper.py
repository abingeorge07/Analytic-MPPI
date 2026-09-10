"""Phase-1 exit gate: behavioral parity of MJX vs CPU *planning* on hopper.

Runs the published hopper config N seeds twice -- run.backend="mujoco" and "mjx" --
with everything else identical (same seeds, same controller, same CPU closed loop).
MJX is not bit-identical to CPU MuJoCo (f32, contact solve), so single trajectories
diverge; the claim under test is DISTRIBUTIONAL: survival and mean forward speed
should be statistically indistinguishable. Human-reviewed table, not an assert.

Note on wall-clock: each episode builds a fresh MJXBackend, and a fresh device model
means a fresh XLA compile (~25-30 s). The compile is planning-warmup, not physics, so
it is excluded from any timing shown here.

Usage:  python3 -m verification.mjx_parity_hopper [--seeds 10]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from analytic_mppi.config import load_config_file, resolve
from analytic_mppi.eval import run_episode
from analytic_mppi.tasks import make_task
from verification._experiment import _metrics


def run_arm(backend: str, seeds: range) -> list[dict]:
    cfg = load_config_file("configs/env/hopper.py").with_(**{"run.backend": backend})
    r = resolve(cfg)
    task = make_task(r.task_name, **r.task_kwargs)
    out = []
    for s in seeds:
        t0 = time.perf_counter()
        res = run_episode(r.task_name, r.controller, steps=r.steps, seed=s,
                          init_fn=r.init_fn, task_kwargs=r.task_kwargs, **r.build)
        m = _metrics("hopper", res, task)
        m["wall_s"] = time.perf_counter() - t0
        out.append(m)
        print(f"  seed {s}: survived={int(m['survived'])} vx={m['vx']:+.2f} "
              f"minup={m['minup']:.2f}  ({m['wall_s']:.0f}s)")
    return out


def summarize(name: str, ms: list[dict]) -> str:
    surv = np.array([m["survived"] for m in ms])
    vx = np.array([m["vx"] for m in ms])
    return (f"{name:8s}  survival {int(surv.sum())}/{len(ms)}   "
            f"vx {vx.mean():+.2f} ± {vx.std():.2f}   "
            f"vx|survived {vx[surv > 0].mean():+.2f}" if surv.any() else
            f"{name:8s}  survival 0/{len(ms)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = range(args.seeds)

    print("== CPU planning (run.backend=mujoco) ==")
    cpu = run_arm("mujoco", seeds)
    print("== MJX planning (run.backend=mjx) ==")
    gpu = run_arm("mjx", seeds)

    print("\n== summary ==")
    print(summarize("cpu", cpu))
    print(summarize("mjx", gpu))


if __name__ == "__main__":
    main()
