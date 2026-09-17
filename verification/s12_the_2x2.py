"""S12 / GATE G11 — the primary result: {FPL, linear} x {MPPI, iLQR}, hopper + walker.

The point: the iLQR column has NO lambda — no temperature, no softmax, no `-log(u)`
bridge. If FPL's advantage survives there, "you found a better temperature" is
eliminated structurally rather than by sweeping around it.

Cells (10 seeds each, per env):
  mppi x {fpl, linear}   — exactly the campaign's `run_trial` (published spec,
                           zero-order spline, K=256, 1 iteration/step).
  ilqr x {fpl, linear}   — S11's architecture with a TEMPERATURE-FREE warm start:
                           PredictiveSampling (argmax over the Gaussian cloud, mean
                           always included; no softmax anywhere in this column),
                           handing its mean to iLQR each MPC step. Linear spline
                           (iLQR's projection requirement; shared inside the column).

MATCHED BUDGET — the definition of record (NEXT_STEPS S12 says make it once, write it
down): equal DYNAMICS-EVALUATION budget per MPC step, counted in length-H rollout
equivalents. MPPI: K=256 rollouts. iLQR column: K_ws=128 warm-start rollouts + per
iLQR iteration [(2*nv + nu) jacfwd tangent rollouts + 8 line-search rollouts + 1
projection re-evaluation]:
  hopper (nv=6, nu=3): 128 + 5*(15+8+1)/... -> iters=5: 128 + 5*24 = 248 ~ 256
  walker (nv=9, nu=6): 128 + 4*(24+8+1)     -> iters=4: 128 + 4*32 = 256
Wall-clock is NOT matched (CPU sampler vs GPU solver); the budget is model
evaluations, the standard for sampler-vs-gradient comparisons.

Config of record, identical across all four cells of an env: atom floor 1e-3
(deliberate; G1 measured -5/-6% of the gap; identical in both arms), published
horizon/knots/noise/time_p, fpl_gamma default, terminal_value=False (G3),
mujoco==3.5.0 baseline. FPL: p=-1. Linear: p=+1 with the published best-safe weight
vector (the same "linear" arm the campaign's studies use).

GATE G11 read-out (report distributions, not means):
  PASS   FPL >= linear in the iLQR column too    -> lambda/p confound resolved BY
         CONSTRUCTION; mark it in docs/mppi_math.md §8 and say so in the paper.
  FAIL   FPL loses without a temperature         -> the single most informative
         outcome in the plan: the advantage was an interaction with the sampling
         update rule. Diagnose which; reframe. DO NOT BURY.
  MIXED  per-task reporting, no aggregate claim.

Run:  ./a-mppi/bin/python verification/s12_the_2x2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _experiment                                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402
from _ci import mean_ci, wilson_ci                        # noqa: E402

from analytic_mppi.controllers import PredictiveSampling   # noqa: E402
from analytic_mppi.controllers.ilqr import ILQRMPC, WarmStartedILQR  # noqa: E402
from analytic_mppi.dynamics import MujocoBackend           # noqa: E402
from analytic_mppi.dynamics.mjx_backend import MJXBackend  # noqa: E402
from analytic_mppi.tasks import make_task                  # noqa: E402

EXP = "the_2x2"
ENVS_RUN = ["hopper", "walker"]
SEEDS = list(range(10))
FLOOR = 1e-3
K_MPPI = 256
K_WS = 128
ILQR_ITERS = {"hopper": 5, "walker": 4}
COSTS = ["fpl", "linear"]
OUT = Path(__file__).resolve().parent / "checkpoints" / "s12_the_2x2"
CKPT = OUT / "the_2x2.jsonl"


def _cost_flags(env: str, cost: str) -> dict:
    spec = _experiment.ENVS[env]
    base = dict(use_fpl_cost=True, fpl_time_p=spec["time_p"], fpl_atom_floor=FLOOR)
    if cost == "fpl":
        return dict(base, fpl_p=-1.0)
    return dict(base, fpl_p=1.0, fpl_weights=_experiment.linear_weights(env, 1.0))


# ------------------------------- the MPPI column -------------------------------

def run_mppi_cells(ckpt: Checkpoint) -> None:
    _experiment.set_atom_floor(FLOOR)
    try:
        for env in ENVS_RUN:
            for cost in COSTS:
                for seed in SEEDS:
                    fields = dict(exp=EXP, env=env, opt="mppi", cost=cost,
                                  seed=seed, K=K_MPPI, floor=FLOOR)
                    if ckpt.has(fields):
                        continue
                    try:
                        m = _experiment.run_trial(env, "mppi", cost, seed=seed, K=K_MPPI)
                    except Exception as e:                 # record, never lose a cell
                        m = dict(error=f"{type(e).__name__}: {e}",
                                 prod=None, survived=None)
                    ckpt.record(fields, m)
                print(f"[mppi] {env}/{cost} done", flush=True)
    finally:
        _experiment.set_atom_floor(1e-8)                   # restore the module default


# ------------------------------- the iLQR column -------------------------------

def run_ilqr_cells(ckpt: Checkpoint) -> None:
    for env in ENVS_RUN:
        spec = _experiment.ENVS[env]
        grid = dict(num_knots=spec["knots"], plan_horizon=spec["horizon"],
                    spline_type="linear")
        task = make_task(spec["task"], **{spec["difficulty_key"]: spec["difficulty"]})
        mjxb = MJXBackend(task.model_path)
        for cost in COSTS:
            cfg = _cost_flags(env, cost)
            print(f"[ilqr] {env}/{cost}: compiling "
                  f"(iters={ILQR_ITERS[env]})...", flush=True)
            tic = time.perf_counter()
            ilqr = ILQRMPC(task, mjxb, iterations=ILQR_ITERS[env], warmup=True,
                           **grid, **cfg)
            print(f"[ilqr] {env}/{cost}: warmup {time.perf_counter() - tic:.0f}s",
                  flush=True)
            for seed in SEEDS:
                fields = dict(exp=EXP, env=env, opt="ilqr", cost=cost, seed=seed,
                              K=K_WS, ilqr_iters=ILQR_ITERS[env], floor=FLOOR)
                if ckpt.has(fields):
                    continue
                try:
                    m = _one_ilqr_episode(env, task, ilqr, cfg, grid, seed)
                except Exception as e:
                    m = dict(error=f"{type(e).__name__}: {e}",
                             prod=None, survived=None)
                ckpt.record(fields, m)
                pm = m.get("prod")
                print(f"[ilqr] {env}/{cost} seed {seed}: "
                      f"prod {pm if pm is None else round(pm, 3)} "
                      f"({m.get('wall_s', float('nan')):.0f}s)", flush=True)


def _one_ilqr_episode(env, task, ilqr, cfg, grid, seed) -> dict:
    spec = _experiment.ENVS[env]
    plan_cpu = MujocoBackend(task.model_path)
    # Temperature-free warm start: argmax over the Gaussian cloud, mean included.
    sampler = PredictiveSampling(task, plan_cpu, num_samples=K_WS,
                                 noise_level=spec["noise"], seed=seed, iterations=1,
                                 **grid, **cfg)
    wrapped = WarmStartedILQR(sampler, ilqr)
    wrapped.reset()

    step_backend = MujocoBackend(task.model_path)
    if spec["init"] is not None:
        spec["init"](step_backend)
    state = step_backend.get_state()

    sd, acc, lam, resid, jpre, jpost = [], [], [], [], [], []
    tic = time.perf_counter()
    for _ in range(spec["steps"]):
        u = wrapped.act(state)
        state = step_backend.step(u)
        sd.append(np.asarray(step_backend.data.sensordata).copy())
        acc.append(ilqr.last_accept_rate)
        lam.append(float(np.max(ilqr.last_lambda_trace)))
        resid.append(ilqr.last_proj_resid)
        jpre.append(ilqr.last_J_pre)
        jpost.append(ilqr.last_J_post)
    wall = time.perf_counter() - tic

    m = _experiment._metrics(env, dict(sd=np.asarray(sd)), task)
    m.update(accept_rate=float(np.mean(acc)), lam_max=float(np.max(lam)),
             resid_mean=float(np.mean(resid)),
             jgap_mean=float(np.mean(np.asarray(jpost) - np.asarray(jpre))),
             wall_s=float(wall))
    return m


# ----------------------------------- report -----------------------------------

def _cell_rows(rows, env, opt, cost):
    v = [r for r in rows if r.get("env") == env and r.get("opt") == opt
         and r.get("cost") == cost and r.get("prod") is not None]
    return v


def report(ckpt: Checkpoint) -> None:
    rows = ckpt.rows()
    print("\n" + "=" * 100)
    print(f"S12 — the 2x2, floor {FLOOR}, {len(SEEDS)} seeds/cell "
          f"(budget: {K_MPPI} rollout-eq/step both columns)")
    print("=" * 100)
    verdicts = {}
    for env in ENVS_RUN:
        print(f"\n### {env}")
        print(f"  {'cell':16s} {'prod mean±CI':>18s}  {'survival':>16s}  per-seed prod")
        cell_stats = {}
        for opt in ("mppi", "ilqr"):
            for cost in COSTS:
                v = _cell_rows(rows, env, opt, cost)
                if not v:
                    print(f"  {opt}/{cost:8s} (no rows)")
                    continue
                prod = np.array([r["prod"] for r in v], float)
                surv = np.array([r["survived"] for r in v], float)
                pm, ph = mean_ci(prod)
                _, slo, shi = wilson_ci(int(surv.sum()), len(surv))
                cell_stats[(opt, cost)] = (pm, ph, prod)
                print(f"  {opt}/{cost:8s} {pm:9.3f}±{ph:<7.3f}  "
                      f"{surv.mean():5.2f} [{slo:.2f},{shi:.2f}]  "
                      f"{np.round(np.sort(prod), 2).tolist()}")
        # G11 read per env: FPL vs linear WITHIN the iLQR column.
        if ("ilqr", "fpl") in cell_stats and ("ilqr", "linear") in cell_stats:
            f_m, f_h, _ = cell_stats[("ilqr", "fpl")]
            l_m, l_h, _ = cell_stats[("ilqr", "linear")]
            if f_m - f_h > l_m + l_h:
                verdicts[env] = "FPL"
            elif l_m - l_h > f_m + f_h:
                verdicts[env] = "LINEAR"
            else:
                verdicts[env] = "OVERLAP"
            print(f"  -> iLQR column: FPL {f_m:.3f}±{f_h:.3f} vs linear "
                  f"{l_m:.3f}±{l_h:.3f}  [{verdicts[env]}]")

    print("\n" + "=" * 100)
    if verdicts:
        vals = set(verdicts.values())
        if vals == {"FPL"}:
            print("GATE G11: PASS — FPL survives an optimizer with no temperature on "
                  "every env. lambda/p confounding is resolved BY CONSTRUCTION; update "
                  "docs/mppi_math.md §8 and the paper.")
        elif vals == {"LINEAR"}:
            print("GATE G11: FAIL — FPL loses without a temperature. This is the most "
                  "informative outcome in the plan: the advantage was an interaction "
                  "with the sampling update rule. Diagnose (lambda? proposal? -log "
                  "bridge?) and reframe. DO NOT BURY.")
        else:
            print(f"GATE G11: MIXED / OVERLAPPING ({verdicts}) — per-task reporting, "
                  f"no aggregate claim.")
    print("=" * 100)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(CKPT)
    run_mppi_cells(ckpt)
    run_ilqr_cells(ckpt)
    report(ckpt)


if __name__ == "__main__":
    main()
