"""Diagnostic: WHICH fulfillment atom binds the min-conjunction, and when.

Question this answers
---------------------
On hopper, sharpening the FPL conjunction (`fpl_p` more negative) or the temporal
weakest-link (`fpl_time_p` more negative) REDUCES survival. A power-mean with p<0 is an
"improve the least-satisfied objective" operator, so it is a safety mechanism only if the
least-satisfied atom is a SAFETY atom. If instead the argmin is the VELOCITY atom during
normal hopping, the conjunction is saying "go faster, ignore posture" -- and sharpening p
amplifies exactly that.

So this script instruments a closed-loop episode and records, per control step:

  * the atoms at the REALIZED state (what the robot actually did),
  * the softmax-weighted distribution of `argmin_j f_j(k,h)` over the whole sampled cloud
    (K x H) -- i.e. which atom is binding in the rollouts the update is actually riding,
  * the same restricted to the elite (top-decile weight) rollouts,
  * the per-atom mean over the cloud, and the time index of the temporal soft-min,
  * ESS of the softmax weighting.

Nothing here changes the controller: the probe reads `ctrl.last_trajectory` after each
`act()` call, so the episode is bit-identical to `_experiment.run_trial` with the same
build kwargs (verified by `--check-parity`).

Usage
-----
    python verification/fpl_binding_probe.py --arms base p-4 --seeds 10 --steps 400
    python verification/fpl_binding_probe.py --list
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _experiment import ENVS, cost_kwargs, sampler_kwargs  # noqa: E402

OUT = REPO / "runs" / "diagnostics" / "fpl_binding"

# The base config from the anomaly table: cubic spline, temp 0.05, noise 0.6, 0.6 s
# horizon, 4 knots. Every arm below is this with ONE thing changed, so the comparison
# holds sampler / budget / horizon / knots / seeds / episode length fixed.
BASE = dict(spline_type="cubic", temperature=0.05, noise_level=0.6,
            plan_horizon=0.6, num_knots=4, fpl_time_p=-2.0, fpl_p=-1.0)

# The known-good geometry (horizon 0.9 + 6 knots + temp 0.1). Used to test whether the
# anti-monotonicity is horizon myopia rather than a property of the objective.
GOOD_GEOM = dict(BASE, plan_horizon=0.9, num_knots=6, temperature=0.1)

ARMS: Dict[str, Dict[str, Any]] = {
    "base":       dict(BASE),
    "p-2":        dict(BASE, fpl_p=-2.0),
    "p-4":        dict(BASE, fpl_p=-4.0),
    "tp-4":       dict(BASE, fpl_time_p=-4.0),
    "tp-8":       dict(BASE, fpl_time_p=-8.0),
    # same conjunction sweep, run on the 0.9 s / 6-knot geometry
    "h9":         dict(GOOD_GEOM),
    "h9_p-2":     dict(GOOD_GEOM, fpl_p=-2.0),
    "h9_p-4":     dict(GOOD_GEOM, fpl_p=-4.0),
    "h9_tp-4":    dict(GOOD_GEOM, fpl_time_p=-4.0),
    "h9_tp-8":    dict(GOOD_GEOM, fpl_time_p=-8.0),
}

ATOMS = ["height", "orient", "veloc", "control"]


def build_kwargs(env: str, sampler: str, cost: str, K: int,
                 extra: Dict[str, Any]) -> Dict[str, Any]:
    """Identical construction to _experiment.run_trial (fairness protocol in one place)."""
    spec = ENVS[env]
    build = dict(num_samples=K, plan_horizon=spec["horizon"], num_knots=spec["knots"],
                 spline_type="zero",
                 **sampler_kwargs(sampler, env, K), **cost_kwargs(cost, env))
    build.update(extra)
    return build


def probe_episode(env: str, seed: int, K: int, steps: int, extra: Dict[str, Any],
                  *, sampler: str = "mppi", cost: str = "fpl",
                  nthread: Optional[int] = None) -> Dict[str, Any]:
    """One closed-loop episode with per-step binding-atom instrumentation."""
    from analytic_mppi.eval import make_controller

    spec = ENVS[env]
    build = build_kwargs(env, sampler, cost, K, extra)
    task, backend, ctrl = make_controller(
        spec["task"], sampler, cost_mode="fpl_cost", seed=seed, nthread=nthread,
        task_kwargs={spec["difficulty_key"]: spec["difficulty"]}, **build,
    )
    if spec["init"] is not None:
        spec["init"](backend)

    temperature = float(build["temperature"])
    n_a = len(task.cost_term_names_f)

    state = backend.get_state()
    sd_hist, u_hist, ess_hist = [], [], []
    dist_all = np.zeros((steps, n_a))     # weighted P(argmin atom = j) over cloud
    dist_elite = np.zeros((steps, n_a))   # same, top-decile-weight rollouts only
    atom_mean = np.zeros((steps, n_a))    # weighted mean atom value over cloud
    dist_h0 = np.zeros((steps, n_a))      # argmin at the FIRST rollout step only
    dist_hT = np.zeros((steps, n_a))      # argmin at the LAST rollout step only
    tmin_frac = np.zeros(steps)           # where in the horizon the temporal min sits
    margin = np.zeros(steps)              # (2nd-smallest - smallest) atom gap, weighted
    # --- conjunction saturation ---
    # A clipped-linear atom is FLAT at 0 below its floor. Once any atom clips, power_mean
    # with p<0 is pinned at ~eps regardless of every other atom, so the cost stops
    # discriminating between rollouts exactly in the failure region p<0 is meant to guard.
    frac_zero = np.zeros((steps, n_a))    # frac of (k,h) cells with atom j clipped to 0
    frac_any_zero = np.zeros(steps)       # frac of (k,h) cells with ANY atom clipped
    frac_dead = np.zeros(steps)           # frac of ROLLOUTS whose FPL reward is at the floor
    reward_spread = np.zeros(steps)       # IQR of log-reward across the cloud (discrimination)

    for t in range(steps):
        u = ctrl.act(state)
        traj = ctrl.last_trajectory
        rf = np.asarray(traj.running_terms_f)              # (K, H, n_a)
        scores = np.asarray(traj.scores)                   # (K,) smaller better
        z = -(scores - scores.min()) / temperature
        w = np.exp(z)
        w = w / w.sum() if np.isfinite(w.sum()) and w.sum() > 0 else np.full_like(w, 1.0 / w.size)

        order = np.argsort(rf, axis=-1)                    # (K, H, n_a)
        amin = order[..., 0]                               # (K, H)
        srt = np.take_along_axis(rf, order, axis=-1)
        gap = srt[..., 1] - srt[..., 0]                    # (K, H)

        H = rf.shape[1]
        onehot = (amin[..., None] == np.arange(n_a))       # (K, H, n_a)
        dist_all[t] = np.einsum("k,khj->j", w, onehot) / H
        dist_h0[t] = np.einsum("k,kj->j", w, onehot[:, 0])
        dist_hT[t] = np.einsum("k,kj->j", w, onehot[:, -1])
        atom_mean[t] = np.einsum("k,khj->j", w, rf) / H
        margin[t] = float(np.einsum("k,kh->", w, gap) / H)

        n_el = max(1, rf.shape[0] // 10)
        el = np.argsort(-w)[:n_el]
        we = w[el] / w[el].sum()
        dist_elite[t] = np.einsum("k,khj->j", we, onehot[el]) / H

        # Which timestep is the rollout's weakest moment (what fpl_time_p sharpens onto)?
        per_step = np.min(rf, axis=-1)                     # (K, H) proxy for the composite
        tmin_frac[t] = float(np.einsum("k,k->", w, np.argmin(per_step, axis=1)) / max(1, H - 1))

        zero = rf <= 1e-6                                  # (K, H, n_a)
        frac_zero[t] = zero.mean(axis=(0, 1))
        frac_any_zero[t] = float(zero.any(axis=-1).mean())
        rew = np.asarray(traj.reward)                      # (K,) in (0,1]
        frac_dead[t] = float((rew <= 1e-6).mean())
        lr = np.log(np.clip(rew, 1e-300, None))
        reward_spread[t] = float(np.percentile(lr, 75) - np.percentile(lr, 25))

        ess_hist.append(float(getattr(ctrl, "last_ess", np.nan) or np.nan))
        state = backend.step(u)
        sd_hist.append(np.asarray(backend.data.sensordata, dtype=np.float64).copy())
        u_hist.append(np.asarray(u, dtype=np.float64).copy())

    sd = np.asarray(sd_hist)
    uu = np.asarray(u_hist)
    realized = np.asarray(task.running_cost_terms_f(None, None, sd, uu))  # (T, n_a)
    up = sd[..., task._zax_adr + 2]
    vx = sd[..., task._vel_adr]
    h = sd[..., task._pos_adr + 2]
    fell = bool(up.min() < spec["fall"])
    fall_step = int(np.argmax(up < spec["fall"])) if fell else -1

    return dict(
        seed=seed, fell=fell, fall_step=fall_step, survived=(0.0 if fell else 1.0),
        vx=float(vx.mean()), minup=float(up.min()), minh=float(h.min()),
        ess=float(np.nanmean(ess_hist)),
        dist_all=dist_all, dist_elite=dist_elite, dist_h0=dist_h0, dist_hT=dist_hT,
        atom_mean=atom_mean, realized=realized, tmin_frac=tmin_frac, margin=margin,
        up=up, vx_t=vx, h_t=h, ess_t=np.asarray(ess_hist),
        frac_zero=frac_zero, frac_any_zero=frac_any_zero, frac_dead=frac_dead,
        reward_spread=reward_spread,
    )


def _worker(payload):
    env, seed, K, steps, extra, sampler, cost, nthread = payload
    r = probe_episode(env, seed, K, steps, extra, sampler=sampler, cost=cost,
                      nthread=nthread)
    return r


def run_arm(env: str, arm: str, seeds: int, K: int, steps: int, *,
            sampler: str = "mppi", cost: str = "fpl", workers: int = 8,
            nthread: int = 3) -> List[Dict[str, Any]]:
    extra = ARMS[arm]
    jobs = [(env, s, K, steps, extra, sampler, cost, nthread) for s in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_worker, jobs))


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (centre - half, centre + half)


def summarize(arm: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    k = int(sum(r["survived"] for r in rows))
    lo, hi = wilson(k, n)
    surv = [r for r in rows if not r["fell"]]
    fall = [r for r in rows if r["fell"]]

    def stack(rs, key):
        return np.concatenate([r[key] for r in rs], axis=0) if rs else np.zeros((0, 4))

    out = dict(
        arm=arm, n=n, survived=k / n, wilson=(lo, hi),
        vx=float(np.mean([r["vx"] for r in rows])),
        ess=float(np.mean([r["ess"] for r in rows])),
        minup=float(np.mean([r["minup"] for r in rows])),
        dist_all=stack(rows, "dist_all").mean(axis=0).tolist(),
        dist_elite=stack(rows, "dist_elite").mean(axis=0).tolist(),
        dist_h0=stack(rows, "dist_h0").mean(axis=0).tolist(),
        dist_hT=stack(rows, "dist_hT").mean(axis=0).tolist(),
        atom_mean=stack(rows, "atom_mean").mean(axis=0).tolist(),
        realized=stack(rows, "realized").mean(axis=0).tolist(),
        dist_surv=stack(surv, "dist_all").mean(axis=0).tolist() if surv else None,
        dist_fall=stack(fall, "dist_all").mean(axis=0).tolist() if fall else None,
        margin=float(np.mean(np.concatenate([r["margin"] for r in rows]))),
        tmin_frac=float(np.mean(np.concatenate([r["tmin_frac"] for r in rows]))),
        n_surv=len(surv), n_fall=len(fall),
        frac_zero=stack(rows, "frac_zero").mean(axis=0).tolist(),
        frac_any_zero=float(np.mean(np.concatenate([r["frac_any_zero"] for r in rows]))),
        frac_dead=float(np.mean(np.concatenate([r["frac_dead"] for r in rows]))),
        reward_spread=float(np.mean(np.concatenate([r["reward_spread"] for r in rows]))),
    )
    return out


def _fmt(v):
    return "  ".join(f"{x:5.3f}" for x in v) if v is not None else "     --"


def print_summary(s: Dict[str, Any]) -> None:
    lo, hi = s["wilson"]
    print(f"\n=== {s['arm']}  n={s['n']}  survival={s['survived']:.2f} "
          f"[{lo:.2f},{hi:.2f}]  vx={s['vx']:+.3f}  ess={s['ess']:.1f}  "
          f"minup={s['minup']:.2f}")
    print(f"    atoms            {'  '.join(f'{a:>5s}' for a in ATOMS)}")
    print(f"    argmin  all      {_fmt(s['dist_all'])}")
    print(f"    argmin  elite    {_fmt(s['dist_elite'])}")
    print(f"    argmin  h=0      {_fmt(s['dist_h0'])}")
    print(f"    argmin  h=H      {_fmt(s['dist_hT'])}")
    print(f"    cloud mean       {_fmt(s['atom_mean'])}")
    print(f"    realized         {_fmt(s['realized'])}")
    print(f"    argmin surviving {_fmt(s['dist_surv'])}  (n={s['n_surv']})")
    print(f"    argmin falling   {_fmt(s['dist_fall'])}  (n={s['n_fall']})")
    print(f"    frac clipped→0   {_fmt(s['frac_zero'])}")
    print(f"    binding margin={s['margin']:.3f}  temporal-min position={s['tmin_frac']:.2f}")
    print(f"    SATURATION: any-atom-zero cells={s['frac_any_zero']:.3f}  "
          f"dead rollouts={s['frac_dead']:.3f}  log-reward IQR={s['reward_spread']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="hopper")
    p.add_argument("--arms", nargs="+", default=["base", "p-4"])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("-K", type=int, default=256)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--sampler", default="mppi")
    p.add_argument("--cost", default="fpl")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--nthread", type=int, default=3)
    p.add_argument("--tag", default="probe")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for k, v in ARMS.items():
            print(f"{k:12s} {v}")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for arm in args.arms:
        rows = run_arm(args.env, arm, args.seeds, args.K, args.steps,
                       sampler=args.sampler, cost=args.cost,
                       workers=args.workers, nthread=args.nthread)
        s = summarize(arm, rows)
        s["extra"] = ARMS[arm]
        print_summary(s)
        summaries.append(s)
        np.savez_compressed(
            OUT / f"{args.tag}_{args.env}_{arm}.npz",
            **{f"{k}_{r['seed']}": r[k] for r in rows
               for k in ("dist_all", "realized", "up", "ess_t", "atom_mean", "vx_t")},
            fell=np.array([r["fell"] for r in rows]),
        )

    (OUT / f"{args.tag}_{args.env}_summary.json").write_text(
        json.dumps(summaries, indent=2, default=str))
    print(f"\nwrote {OUT / f'{args.tag}_{args.env}_summary.json'}")


if __name__ == "__main__":
    main()