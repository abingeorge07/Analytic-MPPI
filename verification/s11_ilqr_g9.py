"""GATES G9 + G10 — is iLQR usable on hopper, and does the knot basis handicap FPL?

G9, all three criteria (NEXT_STEPS):
  1. >= 20x faster than GradientMPC at matched final cost. Measured at the PLANNER
     level (`_plan` calls, post-warmup, same nominal, same objective, same knot grid):
     the time GradientMPC's Adam trace takes to first reach iLQR's final executed cost
     J_post, over the time of one iLQR solve. Cross-evaluated through GradientMPC's own
     loss so the two arms are compared by ONE evaluator (they linearize on
     warm-start-off vs -on models whose forward solutions agree; the evaluator choice
     removes even that).
  2. line-search acceptance rate > 50% with warm start — closed loop, MPPI-warm-started
     (S11's WarmStartedILQR), mean of per-step acceptance rates.
  3. λ trace does not diverge — max λ over every closed-loop step stays off the 1e8 cap.

On fail (2)/(3): expected — linearization disagreeing across a contact-mode boundary;
the numbers are recorded and become the "before" half of the diffmjx ablation (Phase 5).
On fail (1): profile before changing anything (recompilation per step / non-batched
Jacobians are the usual culprits).

G10 — projection residual `‖(I-WW^+)u*‖/‖u*‖` and J_pre vs J_post, SPLIT BY ARM
(fpl vs linear), from the same closed-loop runs. FLAG if FPL loses materially more
(plausible: p<0 wants sharp switching near contact; knots cannot represent it) — that
would mean the thesis is being UNDERSTATED.

Config of record: atom floor 1e-3 (deliberate, G1 measured −5/−6% on the gap; identical
in both arms), published hopper spec otherwise (horizon 0.6, 4 knots, time_p=-2), spline
"linear" in BOTH arms of every comparison here (iLQR's projection needs it; the sampler
must share the knot grid to hand over its mean).

Run:  ./a-mppi/bin/python verification/s11_ilqr_g9.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _experiment                                        # noqa: E402
from _checkpoint import Checkpoint                        # noqa: E402

from analytic_mppi.controllers import GradientMPC, MPPIv2          # noqa: E402
from analytic_mppi.controllers.ilqr import ILQRMPC, WarmStartedILQR  # noqa: E402
from analytic_mppi.dynamics import MujocoBackend                   # noqa: E402
from analytic_mppi.dynamics.mjx_backend import MJXBackend          # noqa: E402
from analytic_mppi.tasks import make_task                          # noqa: E402

ENV = "hopper"
SPEC = _experiment.ENVS[ENV]
FLOOR = 1e-3
OUT = Path(__file__).resolve().parent / "checkpoints" / "s11_ilqr_g9"
CKPT = OUT / "g9_g10.jsonl"

GRID = dict(num_knots=SPEC["knots"], plan_horizon=SPEC["horizon"], spline_type="linear")


def arm_cfg(arm: str) -> dict:
    base = dict(use_fpl_cost=True, fpl_time_p=SPEC["time_p"], fpl_atom_floor=FLOOR)
    if arm == "fpl":
        return dict(base, fpl_p=-1.0)
    if arm == "linear":
        return dict(base, fpl_p=1.0,
                    fpl_weights=_experiment.linear_weights(ENV, 1.0))
    raise ValueError(arm)


def _task():
    return make_task(SPEC["task"], **{SPEC["difficulty_key"]: SPEC["difficulty"]})


# --------------------------- Part A: G9 criterion 1 (speed) ---------------------------

def part_a_speed(ckpt: Checkpoint, arm: str = "fpl",
                 ilqr_iters: int = 10, grad_iters: int = 400) -> None:
    import jax

    task = _task()
    mjxb = MJXBackend(task.model_path)
    cfg = arm_cfg(arm)

    # Test states: settled stand + three states along one seeded MPPI trajectory, so
    # the comparison covers flight/stance mode variety rather than one lucky pose.
    cpu = MujocoBackend(task.model_path)
    _experiment.ENVS[ENV]["init"](cpu)
    mppi = MPPIv2(task, cpu, num_samples=64, noise_level=SPEC["noise"],
                  temperature=SPEC["temp"], seed=0, iterations=1, **GRID, **cfg)
    states = {0: cpu.get_state().copy()}
    st = states[0]
    for i in range(60):
        st = cpu.step(mppi.act(st))
        if i in (9, 29, 59):
            states[i + 1] = st.copy()

    print(f"[A] building iLQR ({ilqr_iters} iters) + GradientMPC ({grad_iters} iters)...",
          flush=True)
    ilqr = ILQRMPC(task, mjxb, iterations=ilqr_iters, warmup=True, **GRID, **cfg)
    grad = GradientMPC(task, mjxb, iterations=grad_iters, learning_rate=0.01,
                       warmup=True, **GRID, **cfg)
    print(f"[A] warmup: ilqr {ilqr.warmup_s:.1f}s, grad {grad.warmup_s:.1f}s", flush=True)
    jnp = ilqr._jnp
    mean0 = jnp.asarray(ilqr._mean_init)          # identical nominal for both arms

    for si, st in states.items():
        fields = dict(exp="g9_speed", env=ENV, arm=arm, state=int(si),
                      floor=FLOOR, ilqr_iters=ilqr_iters, grad_iters=grad_iters)
        if ckpt.has(fields):
            continue
        t0, q0, v0 = st[0], st[mjxb.qpos_slice], st[mjxb.qvel_slice]

        best_ti = np.inf
        for _ in range(2):                        # time the better of two steady calls
            tic = time.perf_counter()
            knots_star, stats = ilqr._plan(t0, q0, v0, mean0, jnp.asarray(ilqr.lam_init))
            jax.block_until_ready((knots_star, stats))
            best_ti = min(best_ti, time.perf_counter() - tic)
        stats = {k: np.asarray(v) for k, v in
                 ((k, jax.device_get(v)) for k, v in stats.items())}

        best_tg = np.inf
        for _ in range(2):
            tic = time.perf_counter()
            gk, losses = grad._plan(t0, q0, v0, mean0)
            jax.block_until_ready((gk, losses))
            best_tg = min(best_tg, time.perf_counter() - tic)
        losses = np.asarray(jax.device_get(losses), dtype=np.float64)

        # One evaluator for the crossing point: GradientMPC's own loss of the iLQR
        # executed plan (the knot-projected plan — J_post is the headline, G10).
        j_star_eval = float(grad._loss(knots_star, t0, q0, v0))
        reach = np.nonzero(losses <= j_star_eval)[0]
        per_adam = best_tg / grad_iters
        if reach.size:
            t_match = float((reach[0] + 1) * per_adam)
            speedup, matched = t_match / best_ti, True
        else:
            t_match = float(best_tg)              # grad never got there: lower bound
            speedup, matched = best_tg / best_ti, False

        m = dict(t_ilqr=float(best_ti), t_grad_total=float(best_tg),
                 t_grad_match=t_match, speedup=float(speedup), matched=matched,
                 J0=float(stats["cost_curve"][0]),
                 J_pre=float(stats["J_pre"]), J_post=float(stats["J_post"]),
                 J_post_grad_eval=j_star_eval,
                 grad_best=float(losses.min()), grad_final=float(losses[-1]),
                 resid=float(stats["proj_resid"]),
                 accept_rate=float(np.mean(stats["accepted"])),
                 lam_max=float(np.max(stats["lambda_trace"])))
        ckpt.record(fields, m)
        print(f"[A] state {si}: ilqr {best_ti:.3f}s J_post {m['J_post']:.4f} | "
              f"grad reach {'#%d' % reach[0] if reach.size else 'never'} "
              f"-> speedup {speedup:.1f}x{'' if matched else ' (lower bound)'}", flush=True)


# ------------------- Part B: G9 criteria 2-3 + G10 (closed loop) -------------------

def part_b_closed_loop(ckpt: Checkpoint, arm: str, seeds=(0, 1, 2), steps: int = 80,
                       ilqr_iters: int = 4, k_ws: int = 128) -> None:
    task = _task()
    cfg = arm_cfg(arm)
    mjxb = MJXBackend(task.model_path)
    print(f"[B/{arm}] building iLQR (iters={ilqr_iters})...", flush=True)
    ilqr = ILQRMPC(task, mjxb, iterations=ilqr_iters, warmup=True, **GRID, **cfg)
    print(f"[B/{arm}] warmup {ilqr.warmup_s:.1f}s", flush=True)

    for seed in seeds:
        fields = dict(exp="g9_closed_loop", env=ENV, arm=arm, seed=int(seed),
                      floor=FLOOR, ilqr_iters=ilqr_iters, k_ws=k_ws, steps=steps)
        if ckpt.has(fields):
            continue
        plan_cpu = MujocoBackend(task.model_path)
        sampler = MPPIv2(task, plan_cpu, num_samples=k_ws, noise_level=SPEC["noise"],
                         temperature=SPEC["temp"], seed=seed, iterations=1,
                         **GRID, **cfg)
        wrapped = WarmStartedILQR(sampler, ilqr)
        wrapped.reset()

        step_backend = MujocoBackend(task.model_path)
        SPEC["init"](step_backend)
        state = step_backend.get_state()

        rows = dict(acc=[], lam=[], alpha=[], resid=[], jpre=[], jpost=[], sd=[])
        tic = time.perf_counter()
        for _ in range(steps):
            u = wrapped.act(state)
            state = step_backend.step(u)
            rows["sd"].append(np.asarray(step_backend.data.sensordata).copy())
            rows["acc"].append(ilqr.last_accept_rate)
            rows["lam"].append(float(np.max(ilqr.last_lambda_trace)))
            rows["alpha"].append(ilqr.last_alpha_used)
            rows["resid"].append(ilqr.last_proj_resid)
            rows["jpre"].append(ilqr.last_J_pre)
            rows["jpost"].append(ilqr.last_J_post)
        wall = time.perf_counter() - tic

        met = _experiment._metrics(ENV, dict(sd=np.asarray(rows["sd"])), task)
        lam = np.asarray(rows["lam"])
        m = dict(met,
                 accept_rate=float(np.mean(rows["acc"])),
                 alpha_median=float(np.nanmedian(rows["alpha"])),
                 lam_max=float(lam.max()), lam_median=float(np.median(lam)),
                 lam_at_cap_frac=float(np.mean(lam >= 1e8 * 0.99)),
                 resid_mean=float(np.mean(rows["resid"])),
                 resid_max=float(np.max(rows["resid"])),
                 jpre_mean=float(np.mean(rows["jpre"])),
                 jpost_mean=float(np.mean(rows["jpost"])),
                 jgap_mean=float(np.mean(np.asarray(rows["jpost"])
                                         - np.asarray(rows["jpre"]))),
                 wall_s=float(wall))
        ckpt.record(fields, m)
        print(f"[B/{arm}] seed {seed}: accept {m['accept_rate']:.2f} "
              f"lam_max {m['lam_max']:.1e} resid {m['resid_mean']:.3f} "
              f"Jgap {m['jgap_mean']:+.4f} vx {m['vx']:.2f} surv {m['survived']:.0f} "
              f"({wall:.0f}s)", flush=True)


# ----------------------------------- report -----------------------------------

def report(ckpt: Checkpoint) -> None:
    rows = ckpt.rows()
    print("\n" + "=" * 100)
    print("GATE G9 — iLQR usable on hopper (floor 1e-3, linear spline, published spec)")
    print("=" * 100)

    sp = [r for r in rows if r.get("exp") == "g9_speed"]
    if sp:
        sps = np.array([r["speedup"] for r in sp], float)
        allm = all(r["matched"] for r in sp)
        print(f"[1] speedup vs GradientMPC at matched final cost: "
              f"min {sps.min():.1f}x  median {np.median(sps):.1f}x  "
              f"({'all matched' if allm else 'some are lower bounds (grad never reached J_post)'})")
        print(f"    -> {'PASS' if sps.min() >= 20 else 'FAIL'} (criterion: >= 20x)")

    cl = [r for r in rows if r.get("exp") == "g9_closed_loop" and r.get("arm") == "fpl"]
    if cl:
        acc = np.array([r["accept_rate"] for r in cl], float)
        lam = np.array([r["lam_max"] for r in cl], float)
        cap = np.array([r["lam_at_cap_frac"] for r in cl], float)
        print(f"[2] warm-started acceptance rate: mean {acc.mean():.2f} "
              f"(per seed: {np.round(acc, 2).tolist()}) -> "
              f"{'PASS' if acc.mean() > 0.5 else 'FAIL'} (criterion: > 0.5)")
        print(f"[3] lambda trace: max {lam.max():.2e}, at-cap fraction "
              f"{cap.max():.2f} -> {'PASS' if lam.max() < 1e8 * 0.99 else 'FAIL'} "
              f"(criterion: bounded away from the 1e8 cap)")

    print("\n" + "=" * 100)
    print("GATE G10 — knot-basis handicap, split by arm")
    print("=" * 100)
    for arm in ("fpl", "linear"):
        a = [r for r in rows if r.get("exp") == "g9_closed_loop" and r.get("arm") == arm]
        if not a:
            print(f"  {arm:8s} (not run)")
            continue
        resid = np.array([r["resid_mean"] for r in a], float)
        jgap = np.array([r["jgap_mean"] for r in a], float)
        print(f"  {arm:8s} resid {resid.mean():.4f}  J_post-J_pre {jgap.mean():+.5f}  "
              f"(n={len(a)})")
    print("  FLAG if the fpl row loses materially more than linear (understated thesis).")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(CKPT)
    part_a_speed(ckpt)
    for arm in ("fpl", "linear"):
        part_b_closed_loop(ckpt, arm)
    report(ckpt)


if __name__ == "__main__":
    main()
