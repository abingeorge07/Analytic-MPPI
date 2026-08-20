"""Parametrized hopper cost/controller evaluation engine for the 15s no-topple search.

Reads a JSON param blob from argv[1] (or stdin), monkeypatches HopperTask fulfillment
atoms + runs closed-loop episodes over `steps` x `seeds`, prints JSON scores.

Objective (user requirement): the hopper must NOT topple over, sustained for the full
15 s (750 steps), and the whole-body COM height (com_z) should trace CLEAN PERIODIC hops
(low noise) while com_x advances steadily. So the score is survival-gated periodicity:
  - survive_frac : fraction of steps that are upright AND not collapsed (must be ~1.0)
  - periodicity  : normalized autocorrelation peak of detrended com_z (1 = perfectly periodic)
  - n_hops       : number of com_z peaks (hop count)
  - travel/vx    : forward progress
  - composite    : survival-gated blend agents/rankers optimize
"""
from __future__ import annotations
import sys, json
import numpy as np
import mujoco
from analytic_mppi.eval import run_episode, make_task, init_hopper_stand
from analytic_mppi.tasks.hopper import HopperTask

DT = 0.02

def _band(x, full, floor):
    return np.clip((x - floor) / (full - floor), 0.0, 1.0)

def make_atoms(p):
    hf_full, hf_floor = p["h_full"], p["h_floor"]
    zf_full, zf_floor = p["z_full"], p["z_floor"]
    TV = p["TV"]
    vel_mode = p.get("vel_mode", "oneside")
    hi_over = p.get("hi_over", 1.4)
    steady_w = p.get("steady_w", 0.0)     # >0: replace control atom with pitch-rate steadiness
    steady_rate = p.get("steady_rate", 3.0)
    ctrl_coef = p.get("ctrl_coef", 0.25)
    hop_w = p.get("hop_w", 0.0)           # >0: add a vertical-velocity "hop" atom
    hop_vz = p.get("hop_vz", 1.0)         # com upward-vel giving full hop credit

    def _hf(self, sd): return _band(self._torso_height(sd), hf_full, hf_floor)
    def _of(self, sd): return _band(self._torso_zaxis_z(sd), zf_full, zf_floor)
    def _vf(self, sd):
        vx = self._torso_vel_x(sd)
        if vel_mode == "oneside":
            return np.clip(vx / TV, 0.0, 1.0)
        up = np.clip(vx / (0.5 * TV), 0, 1)
        down = np.clip((hi_over * TV - vx) / ((hi_over - 1.0) * TV), 0, 1)
        return np.minimum(up, down)
    def _cf(self, u): return np.clip(1.0 - ctrl_coef * np.mean(u ** 2, axis=-1), 0.0, 1.0)
    def _steady(qvel): return np.clip(1.0 - np.abs(qvel[..., 2]) / steady_rate, 0.0, 1.0)

    def run_f(self, qpos, qvel, sd, u):
        cols = [_hf(self, sd), _of(self, sd), _vf(self, sd)]
        cols.append(_steady(qvel) if steady_w > 0 else _cf(self, u))
        if hop_w > 0:
            vz = sd[..., self._vel_adr + 2]
            cols.append(np.clip(np.abs(vz) / hop_vz, 0.0, 1.0))
        return np.stack(cols, axis=-1)
    def term_f(self, qpos, qvel, sd):
        cols = [_hf(self, sd), _of(self, sd), _vf(self, sd)]
        if steady_w > 0:
            cols.append(_steady(qvel))
        if hop_w > 0:
            vz = sd[..., self._vel_adr + 2]
            cols.append(np.clip(np.abs(vz) / hop_vz, 0.0, 1.0))
        return np.stack(cols, axis=-1)
    return dict(_height_fulfillment=_hf, _orientation_fulfillment=_of,
                _velocity_fulfillment=_vf, running_cost_terms_f=run_f,
                terminal_cost_terms_f=term_f)

def _com_traj(task, states):
    """Whole-body COM (com_x, com_z) per step via mj_forward over stored qpos."""
    m = task.mj_model
    d = mujoco.MjData(m)
    qpos = task.qpos_of(states)                     # (T+1, nq)
    comx = np.empty(len(qpos)); comz = np.empty(len(qpos))
    for i, q in enumerate(qpos):
        d.qpos[:] = q; d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        c = d.subtree_com[1]                         # torso subtree = whole robot
        comx[i] = c[0]; comz[i] = c[2]
    return comx, comz

def periodicity(z, lag_lo=8, lag_hi=120):
    """Normalized autocorrelation peak of detrended z in [lag_lo, lag_hi] -> (peak, period_steps)."""
    if len(z) < lag_hi + 5:
        return 0.0, 0
    zc = z - np.convolve(z, np.ones(31) / 31, mode="same")   # remove slow drift
    zc = zc[15:-15]
    zc = zc - zc.mean()
    denom = np.sum(zc * zc) + 1e-12
    lags = list(range(lag_lo, min(lag_hi, len(zc) - 1)))
    ac = [np.sum(zc[:-l] * zc[l:]) / denom for l in lags]
    if not ac:
        return 0.0, 0
    k = int(np.argmax(ac))
    return float(max(0.0, ac[k])), int(lags[k])

def count_hops(z, prominence=0.05):
    zc = z - np.convolve(z, np.ones(31) / 31, mode="same")
    zc = zc[15:-15]
    peaks = 0
    for i in range(1, len(zc) - 1):
        if zc[i] > zc[i - 1] and zc[i] >= zc[i + 1] and zc[i] > prominence:
            peaks += 1
    return peaks

def evaluate(p, want_traj=False):
    steps = p.get("steps", 750)
    seeds = p.get("seeds", [0, 1, 2, 3])
    up_ok = p.get("up_ok", 0.6)
    h_ok = p.get("h_ok", 0.7)
    TV = p["TV"]
    atoms = make_atoms(p)
    orig = {k: getattr(HopperTask, k) for k in atoms}
    per = []; trajs = []
    try:
        for k, fn in atoms.items(): setattr(HopperTask, k, fn)
        for seed in seeds:
            kw = dict(num_samples=p.get("K", 256), plan_horizon=p["H"], num_knots=p["knots"],
                      spline_type=p.get("spline_type", "zero"), noise_level=p.get("noise", 0.3),
                      temperature=p.get("temp", 0.2), fpl_time_p=p.get("time_p", -2.0),
                      fpl_p=p.get("fpl_p", -1.0), iterations=p.get("iterations", 1),
                      nthread=p.get("nthread", 1))
            if "cost_gd" in p:
                kw["cost_gd"] = p["cost_gd"]
            if "weights" in p: kw["fpl_weights"] = p["weights"]
            res = run_episode("hopper", "mppi", steps=steps, seed=seed, cost_mode="fpl_cost",
                              init_fn=init_hopper_stand, task_kwargs=dict(target_velocity=TV), **kw)
            task = make_task("hopper", target_velocity=TV)
            sd = res["sd"]; states = res["states"]
            up = sd[..., task._zax_adr + 2]; h = sd[..., task._pos_adr + 2]
            comx, comz = _com_traj(task, states)
            alive = (up > up_ok) & (h > h_ok)
            bad = ~alive
            first_bad = int(np.argmax(bad)) if bad.any() else steps
            # periodicity/hops measured on the ALIVE prefix (post-topple flailing is meaningless)
            az = comz[:max(first_bad, 40)]
            per_peak, per_lag = periodicity(az)
            per.append(dict(
                survive_frac=float(alive.mean()), survive_steps=first_bad,
                minup=float(up.min()), minh=float(h[10:].min()),
                periodicity=per_peak, period_s=per_lag * DT, n_hops=count_hops(az),
                travel=float(comx[-1] - comx[0]), meanvx=float(sd[..., task._vel_adr].mean()),
                comz_std=float(np.std(comz[:max(first_bad, 40)])),
                toppled=bool(up.min() < 0.3)))
            if want_traj:
                trajs.append(dict(comx=comx.tolist(), comz=comz.tolist(), up=up.tolist(),
                                  survive_steps=first_bad, seed=seed))
    finally:
        for k, fn in orig.items(): setattr(HopperTask, k, fn)
    keys = ["survive_frac", "survive_steps", "minup", "minh", "periodicity", "period_s",
            "n_hops", "travel", "meanvx", "comz_std"]
    agg = {k: float(np.mean([r[k] for r in per])) for k in keys}
    agg["survive_min"] = int(min(r["survive_steps"] for r in per))
    agg["toppled_any"] = int(sum(r["toppled"] for r in per))
    agg["n_seeds"] = len(seeds)
    agg["per_seed_survive"] = [r["survive_steps"] for r in per]
    # survival-gated composite: full survival is the gate; then reward clean periodic hopping
    # AND forward speed (the user wants a lively, not stationary, hopper). Survival dominates
    # (gate = survive_frac**2), then periodicity + forward speed contribute among survivors.
    surv = agg["survive_frac"]
    gate = surv ** 2                              # heavily punish any toppling
    agg["composite"] = float(gate * (100.0
                             + 30.0 * agg["periodicity"]
                             + 30.0 * float(np.clip(agg["meanvx"], 0.0, 1.0))
                             + 2.0 * float(np.clip(agg["travel"], 0.0, 15.0))))
    if want_traj:
        agg["trajs"] = trajs
    return agg

if __name__ == "__main__":
    blob = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    p = json.loads(blob)
    out = evaluate(p, want_traj=bool(p.get("want_traj")))
    print(json.dumps(out))
