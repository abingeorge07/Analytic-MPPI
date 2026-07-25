"""Inspect the INDIVIDUAL objective values along a chosen controller's trajectory.

The point: `normal` and the FPL controllers are NOT optimizing the same thing.
  * normal         -> minimizes a weighted SUM of penalty terms (different scales/weights).
  * fpl_discounted -> per objective, discount-sums its [0,1] fulfillment over the horizon
                      (FQ-value), then takes a POWER-MEAN across objectives (weakest-link
                      when p<0), fed to the MPPI update as cost = -log(power_mean).
  * fpl_nominal    -> the same fpl_discounted pipeline, but scoring ONLY a hand-picked
                      subset of fulfillment atoms (default: nominal_fulfillment).
  * fpl_layered    -> two-level: discount-sum each atom, inner power-mean the per-joint
                      atoms into one "posture" score, then outer power-mean it against
                      orientation & height (fpl_group_p = inner p, fpl_p = outer p).
  * fpl_wip        -> scratch slot for WIP experiments; currently orientation + height
                      only (drops the joint-angle nominal term to remove the posture-vs-
                      height fight).
  * fpl_cma / fpl_cma_layered -> FplValueCMA optimizer (adaptive covariance + log-rank ×
                      exp(beta·reward) weighting) on fpl_discounted / fpl_layered. The
                      rank weighting is scale-invariant, so it doesn't wash out on FPL's
                      narrow reward band the way MPPI's softmax does — this is what keeps
                      the robot from sinking into a crouch.
  * composed / composed_layered -> ComposedGradientMPPI: form a separate ES gradient per
                      objective and compose them toward the WORST-satisfied one (α∝s^{p-1}).
                      composed = flat atoms, composed_layered = g1 groups. On g1 it holds
                      the orientation floor / worst group above the scalar softmax-FPL score.
So the same physical state gets a very different score under each. This script runs
ONE chosen controller and prints, per step, BOTH representations of every objective
for the state it actually visited — so you can see what each controller was chasing.

Usage:
  python verification/inspect_costs.py --controller normal
  python verification/inspect_costs.py --controller fpl_discounted --live   # MuJoCo viewer
  python verification/inspect_costs.py --controller composed_layered --live # multi-obj g1
  python verification/inspect_costs.py --controller fpl_nominal --every 5
  # options: --task g1_standup|hopper|... --shove 4.0 --steps 150 --every 15 \
  #          --fpl-p -2 --temperature 0.1 --num-samples 128 --fpl-terms nominal_fulfillment

Live viewer keys (--live):
  SPACE / P  pause & resume — on pause it prints the full cost breakdown for the
             exact frame frozen on screen, so you can eyeball the pose and compare
             it against the FPL fulfillments / normal penalties side by side.
  N / ->     single-step one control step while paused (prints costs each step).
  Q          quit the run.
  (while paused the MuJoCo scene stays interactive — rotate/zoom to inspect the pose.)
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import numpy as np
import mujoco

from analytic_mppi.tasks import make_task
from analytic_mppi.tasks.base import power_mean
from analytic_mppi.eval import run_episode, make_controller, init_hopper_stand, init_hang_down
from analytic_mppi.controllers import MPPIv2, ComposedGradientMPPI
from analytic_mppi.controllers.experimental import FplValueCMA


# The controllers we're comparing. cost_mode + its default hyperparams.
CONTROLLERS = {
    "normal":         dict(cost_mode="normal", params=dict(temperature=1.0)),
    # fpl_discounted: per objective, discount-sum its fulfillment over the horizon, then
    # power-mean across ALL objectives ("discount then compose" — timing-flexible).
    "fpl_discounted": dict(cost_mode="fpl_discounted", params=dict(fpl_p=0.1, temperature=0.01)),
    # fpl_nominal: the same fpl_discounted pipeline, but scoring ONLY the hand-picked
    # nominal_fulfillment atom. `term_names` is resolved to fpl_term_indices at runtime;
    # override the picked atoms with --fpl-terms.
    "fpl_nominal":    dict(cost_mode="fpl_discounted", term_names=["nominal_fulfillment"],
                          params=dict(fpl_p=0.1, temperature=0.1)),
    # fpl_layered: hierarchical composition over task.fpl_groups. Discount-sum each atom,
    # inner power-mean the per-joint atoms into one "posture" score (fpl_group_p), then
    # outer power-mean {orientation, height, posture} (fpl_p). fpl_group_p=None -> = fpl_p.
    "fpl_layered":    dict(cost_mode="fpl_layered",
                          params=dict(fpl_p=0.1, temperature=0.1, fpl_group_p=-2.0)),
    # fpl_wip: scratch slot for work-in-progress FPL experiments. CURRENTLY: drop the
    # posture/joint-angle nominal term entirely and score ONLY orientation + height +
    # control — "just stay upright & tall while moving as little as possible", no target
    # joint pose. Tests whether the nominal term was fighting height/orientation (the robot
    # must leave the standing pose to gain height). fpl_discounted over those three atoms.
    "fpl_wip":        dict(cost_mode="fpl_discounted",
                          term_names=["orientation_fulfillment", "height_fulfillment",
                                      "control_fulfillment"],
                          params=dict(fpl_p=0.1, temperature=0.1)),
    # fpl_cma: FplValueCMA (adaptive covariance + log-rank × exp(beta·reward) value
    # weighting) on flat fpl_discounted. The rank weighting is scale-invariant, so it
    # doesn't wash out on FPL's narrow [0,1] reward band the way MPPI's softmax does —
    # this is what stops the robot sinking into a crouch. `cls` selects the controller.
    "fpl_cma":        dict(cost_mode="fpl_discounted", cls=FplValueCMA,
                          params=dict(sigma_init=0.3, sigma_floor=0.1, use_cov_update=True,
                                      beta=10.0, fpl_p=0.1)),
    # fpl_cma_layered: the same value-CMA optimizer on the layered composition — combines
    # the per-joint layered structure with the rank-weighted adaptive search.
    "fpl_cma_layered": dict(cost_mode="fpl_layered", cls=FplValueCMA,
                          params=dict(sigma_init=0.3, sigma_floor=0.1, use_cov_update=True,
                                      beta=10.0, fpl_p=0.01, fpl_group_p=0.01)),
    # composed / composed_layered: ComposedGradientMPPI. Instead of collapsing the
    # per-objective FPL vector into ONE scalar, form a separate ES gradient per objective
    # (per-objective softmax-weighted mean of knots) and compose them toward the WORST-
    # satisfied objective (compose="worst_first", alpha_mode="power_p" -> α_j ∝ s_j^{p-1},
    # the natural-gradient weights on power_mean(s, fpl_p)). obj_weighting defaults to the
    # SHARP "softmax" (w_kj ∝ v_kj^{1/temperature}); "proportional" is mushy and topples g1.
    # "composed" scores the flat atoms (works on any task); "composed_layered" scores the
    # g1 groups {orientation, height, posture}. On g1 this holds the orientation floor and
    # the worst group's fulfillment ABOVE the scalar softmax-FPL baseline.
    "composed":         dict(cost_mode="fpl_discounted", cls=ComposedGradientMPPI,
                          params=dict(fpl_p=0.1, temperature=0.01,
                                      compose="worst_first", alpha_mode="power_p")),
    "composed_layered": dict(cost_mode="fpl_layered", cls=ComposedGradientMPPI,
                          params=dict(fpl_p=0.1, temperature=0.01, fpl_group_p=0.1,
                                      compose="worst_first", alpha_mode="power_p")),
}


def make_shove(vx):
    """g1_standup: start standing, then apply an initial forward torso velocity."""
    def _init(backend):
        kf = backend.model.keyframe("stand")
        backend.data.qpos[:] = kf.qpos
        backend.data.qvel[:] = 0.0
        backend.data.qvel[0] = vx
        mujoco.mj_forward(backend.model, backend.data)
    return _init


def pick_init(task_name, shove):
    if task_name == "g1_standup":
        return make_shove(shove)
    if task_name == "hopper":
        return init_hopper_stand
    if task_name == "pendulum":
        return init_hang_down
    return None


# ---- shared cost/atom computation + pretty-printing ------------------------

def _terms(task, qpos, qvel, sd, ctrls, disp_p, fpl_sel=None):
    nct = task.running_cost_terms(qpos, qvel, sd, ctrls)       # (..., n) penalties, lower=better
    fct = task.running_cost_terms_f(qpos, qvel, sd, ctrls)     # (..., m) fulfillments in [0,1]
    # Honor the scored subset (--fpl-terms / preset term_names): only the atoms the
    # controller actually optimizes are shown and composited, so excluded atoms (e.g.
    # nominal) don't clutter the display or skew the pmean.
    if fpl_sel is not None:
        fct = fct[..., fpl_sel]
    fcomp = power_mean(fct, disp_p)                            # (...,) FPL composite over shown atoms
    return nct, fct, fcomp


def _fmt(names, row):
    return "  ".join(f"{n[:10]}={v:7.3f}" for n, v in zip(names, row))


def _print_step(tag, task, nct_t, fct_t, fcomp_t, fpl_names=None):
    names_f = task.cost_term_names_f if fpl_names is None else fpl_names
    print(f"[{tag}]  NORMAL: {_fmt(task.cost_term_names, nct_t)}   SUM={nct_t.sum():7.3f}")
    print(f"         FPL:    {_fmt(names_f, fct_t)}   pmean={fcomp_t:.3f}  -log={-np.log(fcomp_t):.3f}")


# ---- layered-FPL breakdown (per-atom -> discounted sum -> group -> final) ---
# Reproduces _score_fpl_layered on the EXECUTED trajectory so every intermediate is
# visible: this is an episode-level diagnostic, NOT a single planning-horizon rollout
# score the controller optimized. Only tasks that define `fpl_groups` (g1_standup).

def _layered_atom_meta(task, qpos, sd):
    """Per grouped-atom metadata aligned by atom index: (group, term, raw_series|None,
    reference|None). g1-aware (orientation rz / torso height / per-joint angle); falls
    back to generic names + blank raw for any other task that defines fpl_groups."""
    m = task.mj_model
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
              for j in range(m.njnt)]
    n = sum(len(idx) for idx in task.fpl_groups)
    meta = [None] * n
    for gname, idx in zip(task.fpl_group_names, task.fpl_groups):
        for local, a in enumerate(idx):
            if gname == "orientation" and hasattr(task, "_torso_orientation"):
                meta[a] = (gname, "orientation_rz", task._torso_orientation(sd)[..., 2], 1.0)
            elif gname == "height" and hasattr(task, "_torso_height"):
                meta[a] = (gname, "torso_height", task._torso_height(sd),
                           getattr(task, "target_height", None))
            elif gname == "posture" and hasattr(task, "qstand"):
                jn = jnames[local + 1] if local + 1 < len(jnames) else f"joint_{local:02d}"
                meta[a] = (gname, jn, qpos[..., 7 + local], float(task.qstand[7 + local]))
            else:
                meta[a] = (gname, f"{gname}_{local}", None, None)
    return meta


def layered_breakdown(task, qpos, qvel, sd, ctrls, gamma, inner_p, outer_p):
    """Full layered decomposition of the executed trajectory. Returns per-step grouped
    fulfillments, per-atom discounted sums (FQ-values), inner group scores, and the final
    outer power-mean + its -log cost."""
    fg = task.running_cost_terms_f_grouped(qpos, qvel, sd, ctrls)          # (T, n)
    T, n = fg.shape
    meta = _layered_atom_meta(task, qpos, sd)
    disc = gamma ** np.arange(T, dtype=np.float64)
    norm = (1.0 - gamma) / (1.0 - gamma ** T) if gamma < 1.0 else 1.0 / max(T, 1)
    fq = (fg * disc[:, None]).sum(axis=0) * norm                           # (n,) FQ-values
    group_scores = [float(power_mean(fq[idx], inner_p)) for idx in task.fpl_groups]
    reward = float(power_mean(np.asarray(group_scores), outer_p))
    cost = -float(np.log(max(reward, 1e-12)))
    return dict(T=T, n=n, fg=fg, meta=meta, fq=fq, group_scores=group_scores,
                reward=reward, cost=cost, gamma=gamma, inner_p=inner_p, outer_p=outer_p)


def write_layered_csv(path, task, bd):
    """Tidy long-format CSV: per-step atom rows, then discounted-sum rows, then group
    scores, then the final fpl_score + cost. `row_type` separates the four sections."""
    fg, meta, fq = bd["fg"], bd["meta"], bd["fq"]
    cols = ["row_type", "step", "group", "term", "raw", "reference",
            "fulfillment", "discounted_sum", "fpl_score"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        # 1) per horizon step: raw value + [0,1] fulfillment for every atom
        for t in range(bd["T"]):
            for a in range(bd["n"]):
                g, term, raw_s, ref = meta[a]
                rawv = "" if raw_s is None else f"{float(raw_s[t]):.6f}"
                refv = "" if ref is None else f"{float(ref):.6f}"
                w.writerow(["perstep", t, g, term, rawv, refv, f"{float(fg[t, a]):.6f}", "", ""])
        # 2) discounted sum (FQ-value) per atom
        for a in range(bd["n"]):
            g, term, _, _ = meta[a]
            w.writerow(["discounted_sum", "", g, term, "", "", "", f"{float(fq[a]):.6f}", ""])
        # 3) inner power-mean per group
        for gname, idx, gs in zip(task.fpl_group_names, task.fpl_groups, bd["group_scores"]):
            w.writerow(["group_score", "", gname,
                        f"inner pmean(p={bd['inner_p']:g}) over {len(idx)} atoms",
                        "", "", "", "", f"{float(gs):.6f}"])
        # 4) final outer power-mean + its -log cost
        w.writerow(["final", "", "ALL", f"fpl_score = outer pmean(p={bd['outer_p']:g})",
                    "", "", "", "", f"{bd['reward']:.6f}"])
        w.writerow(["final", "", "ALL", "cost = -log(fpl_score)",
                    "", "", "", "", f"{bd['cost']:.6f}"])


def print_layered_summary(task, bd, top_k=8):
    fq, meta, fg, ip = bd["fq"], bd["meta"], bd["fg"], bd["inner_p"]
    print(f"LAYERED FPL breakdown  (executed trajectory, discounted γ={bd['gamma']:g})")
    # `discounted` is γ-weighted (early-dominated), so it can HIDE a late fall — `final`
    # (last step, undiscounted) and `min` (worst step) expose the actual end state.
    print(f"  {'group':<12} {'discounted':>10} {'final':>8} {'min':>8}   atoms  (inner p={ip:g})")
    for gname, idx, gs in zip(task.fpl_group_names, task.fpl_groups, bd["group_scores"]):
        per_step = power_mean(fg[:, idx], ip)                 # (T,) undiscounted per-step
        print(f"  {gname:<12} {gs:>10.4f} {per_step[-1]:>8.4f} {per_step.min():>8.4f}   {len(idx)}")
    print(f"  {'FPL score':<12} {bd['reward']:>10.4f} {'':>8} {'':>8}   "
          f"outer p={bd['outer_p']:g}  cost=-log={bd['cost']:.4f}")
    order = np.argsort(fq)
    k = min(top_k, len(fq))
    print(f"  weakest {k} atoms by discounted fulfillment (these bind the score):")
    for a in order[:k]:
        g, term, raw_s, ref = meta[a]
        extra = ""
        if raw_s is not None and ref is not None:
            extra = f"   last raw={float(raw_s[-1]):+.3f}  ref={float(ref):+.3f}"
        print(f"    {g:<11} {term:<24} FQ={fq[a]:.3f}{extra}")


def write_rollout_video(path, task, states, fps, height=480, width=640):
    """Offscreen-render the EXECUTED trajectory to an MP4 (no live viewer needed).
    Replays each visited state through a free camera tracking the root body."""
    import imageio.v2 as imageio  # v2 API: mimsave(path, frames, fps=...)

    model = task.mj_model
    data = mujoco.MjData(model)
    qpos_all, qvel_all = task.qpos_of(states), task.qvel_of(states)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    track_bid = 1 if model.nbody > 1 else 0        # root/pelvis: keeps the robot framed
    renderer = mujoco.Renderer(model, height, width)
    try:
        frames = []
        for i in range(len(states)):
            data.qpos[:] = qpos_all[i]
            data.qvel[:] = qvel_all[i]
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.xpos[track_bid]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())  # render() reuses its buffer
    finally:
        renderer.close()
    imageio.mimsave(path, frames, fps=fps)
    return len(frames)


def _build_kwargs(args, params, cls):
    kw = dict(
        num_samples=args.num_samples, plan_horizon=args.plan_horizon,
        num_knots=args.num_knots, spline_type="zero", iterations=args.iterations,
        **params,
    )
    # MPPIv2 and its subclasses (e.g. ComposedGradientMPPI) take a fixed isotropic
    # `noise_level`; the CMA controllers bring their own sampling scale (sigma_init /
    # initial_noise_level) via `params`.
    if issubclass(cls, MPPIv2):
        kw["noise_level"] = args.noise_level
    return kw


# ---- live loop (opens a MuJoCo viewer, prints costs as it runs) -------------

def run_live(args, spec, params, init, disp_p):
    import mujoco.viewer as mj_viewer
    from mujoco import rollout as mj_rollout

    cls = spec.get("cls", MPPIv2)
    task, backend, ctrl = make_controller(
        args.task, cls, cost_mode=spec["cost_mode"], seed=args.seed,
        **_build_kwargs(args, params, cls),
    )
    if init is not None:
        init(backend)
    state = backend.get_state()

    # Show only the scored fulfillment atoms (if a subset is selected), matching main().
    fpl_sel = params.get("fpl_term_indices")
    fpl_names = None if fpl_sel is None else [task.cost_term_names_f[i] for i in fpl_sel]

    states_hist, ctrls_hist, sd_hist = [state.copy()], [], []

    # Interactive control shared with the viewer's key callback. The callback runs
    # on the viewer's UI thread, so it only flips flags here — all the cost
    # computation / printing happens in the main loop below.
    ctl = {"paused": False, "step_once": False, "print_now": False, "quit": False}

    def key_cb(keycode):
        try:
            ch = chr(keycode).upper()
        except ValueError:
            ch = ""
        if keycode == 32 or ch == "P":            # SPACE / P -> toggle pause
            ctl["paused"] = not ctl["paused"]
            if ctl["paused"]:
                ctl["print_now"] = True           # dump costs for the frozen frame
        elif ch == "N" or keycode == 262:         # N / right-arrow -> single step
            ctl["step_once"] = True
        elif ch == "Q":                           # Q -> stop the run
            ctl["quit"] = True

    def print_costs(tag):
        # Costs for the frame currently displayed: latest control/sensordata + state.
        u_last = ctrls_hist[-1] if ctrls_hist else np.zeros(backend.model.nu)
        sd_now = sd_hist[-1] if sd_hist else np.asarray(backend.data.sensordata,
                                                        dtype=np.float64)
        qpos, qvel = task.qpos_of(state), task.qvel_of(state)
        nct, fct, fcomp = _terms(task, qpos, qvel, sd_now, u_last, disp_p, fpl_sel)
        _print_step(tag, task, nct, fct, float(fcomp), fpl_names)

    viewer = mj_viewer.launch_passive(backend.model, backend.data, key_callback=key_cb,
                                      show_left_ui=False, show_right_ui=False)
    print("(live viewer — SPACE/P pause & dump costs, N single-step, Q quit)\n")
    try:
        dt = backend.dt
        step = 0
        while step < args.steps and viewer.is_running() and not ctl["quit"]:
            # Frozen: keep the scene interactive, print the breakdown once on entry,
            # and idle until the user resumes or single-steps.
            if ctl["paused"] and not ctl["step_once"]:
                if ctl["print_now"]:
                    ctl["print_now"] = False
                    print_costs(f"t={step:3d} PAUSED")
                viewer.sync()
                time.sleep(0.02)
                continue
            stepping_while_paused = ctl["step_once"]
            ctl["step_once"] = False

            t0 = time.perf_counter()
            u = ctrl.act(state)
            with viewer.lock():
                state = backend.step(u)
            sd = np.asarray(backend.data.sensordata, dtype=np.float64).copy()
            states_hist.append(state.copy()); ctrls_hist.append(u.copy()); sd_hist.append(sd)
            if step % max(1, args.every) == 0 or stepping_while_paused:
                qpos, qvel = task.qpos_of(state), task.qvel_of(state)
                nct, fct, fcomp = _terms(task, qpos, qvel, sd, u, disp_p, fpl_sel)
                tag = f"t={step:3d}" + (" STEP" if stepping_while_paused else "")
                _print_step(tag, task, nct, fct, float(fcomp), fpl_names)
            viewer.sync()
            step += 1
            elapsed = time.perf_counter() - t0
            if not stepping_while_paused and dt - elapsed > 0:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            mj_rollout.shutdown_persistent_pool()
        except Exception:
            pass
        try:
            viewer.close()
        except Exception:
            pass
    return dict(states=np.asarray(states_hist), ctrls=np.asarray(ctrls_hist),
                sd=np.asarray(sd_hist))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", choices=list(CONTROLLERS), default="fpl_layered")
    ap.add_argument("--task", default="g1_standup")
    ap.add_argument("--shove", type=float, default=4.0, help="g1_standup only: initial torso vx")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=15, help="print every Nth step")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--plan-horizon", type=float, default=1.0)
    ap.add_argument("--num-knots", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=1,
                    help="inner optimizer steps per act() (sample->score->update cycles); "
                         ">1 = a few cycles of gradient descent before committing each action")
    ap.add_argument("--noise-level", type=float, default=0.3)
    ap.add_argument("--fpl-p", type=float, default=None, help="override the preset power-mean p")
    ap.add_argument("--fpl-group-p", type=float, default=None,
                    help="fpl_layered only: inner power-mean p across per-joint atoms")
    ap.add_argument("--temperature", type=float, default=None, help="override the preset temperature")
    ap.add_argument("--fpl-weighting", choices=["softmax", "proportional", "exp"], default=None,
                    help="FPL update rule: softmax(-log u) | proportional w_k=u_k/Σu_j | exp w_k∝exp(u_k/λ)")
    ap.add_argument("--fpl-terms", default=None,
                    help="comma-separated fulfillment atom names to score (subset of "
                         "task.cost_term_names_f), e.g. 'nominal_fulfillment'. Overrides the preset.")
    ap.add_argument("--live", action="store_true", help="open an interactive MuJoCo viewer while running")
    ap.add_argument("--csv", default=None,
                    help="write a per-step layered-FPL breakdown CSV to this path (raw + "
                         "fulfillment per atom per step, then discounted sums, group & final scores)")
    ap.add_argument("--fpl-gamma", type=float, default=0.99,
                    help="discount γ for the FPL discounted sums (and the CSV breakdown)")
    ap.add_argument("--video", default=None,
                    help="save an offscreen-rendered MP4 of the rollout (default: alongside "
                         "--csv with a .mp4 extension, so saving a CSV also saves a video)")
    ap.add_argument("--no-video", action="store_true",
                    help="disable the automatic video that accompanies --csv")
    ap.add_argument("--fps", type=int, default=None, help="video frames/sec (default: 1/dt)")
    args = ap.parse_args()

    task = make_task(args.task)
    spec = CONTROLLERS[args.controller]
    cls = spec.get("cls", MPPIv2)
    params = dict(spec["params"])
    if args.fpl_p is not None and spec["cost_mode"] != "normal":
        params["fpl_p"] = args.fpl_p
    if args.fpl_group_p is not None and spec["cost_mode"] == "fpl_layered":
        params["fpl_group_p"] = args.fpl_group_p
    if spec["cost_mode"] != "normal":
        params.setdefault("fpl_gamma", args.fpl_gamma)
    # temperature / fpl_weighting are MPPIv2 knobs; only override where the spec uses them
    # (the CMA controllers weight by rank/value, not a softmax temperature).
    if args.temperature is not None and "temperature" in params:
        params["temperature"] = args.temperature
    if args.fpl_weighting is not None and cls is MPPIv2 and spec["cost_mode"] == "fpl_discounted":
        params["fpl_weighting"] = args.fpl_weighting
    # Restrict the FPL reward to a subset of fulfillment atoms. --fpl-terms (names)
    # overrides the preset's term_names; both resolve against task.cost_term_names_f.
    term_names = spec.get("term_names")
    if args.fpl_terms is not None:
        term_names = [s.strip() for s in args.fpl_terms.split(",") if s.strip()]
    if term_names and spec["cost_mode"] == "fpl_discounted":
        missing = [n for n in term_names if n not in task.cost_term_names_f]
        if missing:
            raise SystemExit(f"unknown fulfillment atom(s) {missing}; "
                             f"task {args.task} has {task.cost_term_names_f}")
        params["fpl_term_indices"] = [task.cost_term_names_f.index(n) for n in term_names]
    # p used to DISPLAY the fpl composite (even for non-fpl controllers, so you can
    # see what FPL *would* have scored the trajectory they produced).
    disp_p = params.get("fpl_p", -2.0)

    init = pick_init(args.task, args.shove)

    print(f"\ncontroller = {args.controller}   ({spec['cost_mode']}, params={params})")
    print(f"task = {args.task}   steps = {args.steps}   seed = {args.seed}   "
          f"iterations = {args.iterations}"
          + (f"   shove = {args.shove}" if args.task == 'g1_standup' else ""))
    print("=" * 100)
    print("NORMAL cost terms (penalty, LOWER=better)  ->  sum is what `normal` minimizes")
    print("FPL atoms in [0,1] (HIGHER=better)         ->  power_mean(p=%.1f); `fpl` maximizes it, cost=-log(pmean)" % disp_p)
    print("=" * 100)

    if args.live:
        res = run_live(args, spec, params, init, disp_p)
    else:
        res = run_episode(
            args.task, cls, steps=args.steps, seed=args.seed,
            cost_mode=spec["cost_mode"], init_fn=init, **_build_kwargs(args, params, cls),
        )

    # Executed trajectory: pair sensordata (recorded AFTER each step) with states[1:].
    states, ctrls, sd = res["states"], res["ctrls"], res["sd"]
    if len(ctrls) == 0:
        print("(no steps ran)")
        return
    qpos = task.qpos_of(states)[1:]
    qvel = task.qvel_of(states)[1:]
    # Show only the scored fulfillment atoms (if --fpl-terms / a preset selected a subset).
    fpl_sel = params.get("fpl_term_indices")
    fpl_names = None if fpl_sel is None else [task.cost_term_names_f[i] for i in fpl_sel]
    nct, fct, fcomp = _terms(task, qpos, qvel, sd, ctrls, disp_p, fpl_sel)

    layered_ok = bool(getattr(task, "fpl_groups", None))
    if args.csv and not layered_ok:
        raise SystemExit(f"--csv needs a task with fpl_groups (layered FPL); "
                         f"{args.task} defines none")

    # Layered breakdown: the clean, debuggable view (per-joint -> discounted -> score).
    if layered_ok:
        gamma_csv = float(params.get("fpl_gamma", args.fpl_gamma))
        outer_p = float(params.get("fpl_p", disp_p))
        inner_p = params.get("fpl_group_p", None)
        inner_p = float(outer_p if inner_p is None else inner_p)
        bd = layered_breakdown(task, qpos, qvel, sd, ctrls, gamma_csv, inner_p, outer_p)
        print("=" * 100)
        print_layered_summary(task, bd)
        if args.csv:
            write_layered_csv(args.csv, task, bd)
            print(f"\n  wrote layered breakdown CSV -> {args.csv}   "
                  f"({bd['T']} steps x {bd['n']} atoms, + discounted sums + scores)")
        print("=" * 100)

    # Offscreen video of the rollout (no live viewer needed). Default: whenever a CSV is
    # saved, drop an MP4 next to it (override path with --video, disable with --no-video).
    video_path = args.video
    if video_path is None and args.csv and not args.no_video:
        video_path = os.path.splitext(args.csv)[0] + ".mp4"
    if video_path and args.live:
        print("  (skipping offscreen video in --live mode; the viewer is already rendering)")
    elif video_path:
        fps = args.fps or max(1, round(1.0 / float(task.mj_model.opt.timestep)))
        try:
            nfr = write_rollout_video(video_path, task, states, fps)
            print(f"  wrote rollout video -> {video_path}   ({nfr} frames @ {fps} fps)")
        except Exception as e:
            print(f"  (video render failed: {type(e).__name__}: {e})")

    # Compact cross-representation means (what `normal` minimizes vs the flat fpl composite).
    print("-" * 100)
    _print_step("MEAN ", task, nct.mean(0), fct.mean(0), fcomp.mean(), fpl_names)

    # Verbose per-step dump only when NOT layered (layered detail lives in the CSV / summary).
    if not layered_ok and not args.live:
        print("-" * 100)
        for t in range(0, len(ctrls), args.every):
            _print_step(f"t={t:3d}", task, nct[t], fct[t], fcomp[t], fpl_names)
        print("=" * 100)
        print("Read: compare the per-objective values the chosen controller actually achieved.")
        print("If `fpl_discounted` leaves one atom (e.g. height) low while its pmean is 'ok', the weakest-link")
        print("aggregation is satisfied and stops pushing that objective — whereas `normal`'s")
        print("quadratic term keeps pulling it. Run all controllers and compare the MEAN rows.")


if __name__ == "__main__":
    main()
