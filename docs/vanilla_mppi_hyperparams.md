# Vanilla-MPPI hyperparameter directory (hopper)

**Scope:** the hyperparameters relevant to plain MPPI with the vanilla (normal) cost — i.e.
`objective.mode = "normal"`, which runs `_score_normal` in
[`sampling_base.py`](../analytic_mppi/controllers/sampling_base.py) over the quadratic terms in
[`tasks/hopper.py`](../analytic_mppi/tasks/hopper.py), dispatched via `("gaussian",
"path_integral")` in [`config.py`](../analytic_mppi/config.py) to `MPPIv2`. None of the `fpl_*`
params are active in this mode; they are omitted below (see `config.py`'s `Objective` dataclass
for the FPL-mode axis).

This is a reference for the sensitivity analysis and LHS sweeps — the source of truth for values
and constraints is always `config.py`/`hopper.py`, not this file. Update it if those drift.

---

## 1. Core optimizer hyperparameters

| Param | Config field | Type / hard constraint | Values seen in repo | Step-2 working config |
|---|---|---|---|---|
| Samples *K* | `proposal.num_samples` | int, > 0 | 128 – 256 across verification scripts | 256 (explicit override — `HOPPER` itself leaves this at the schema default of 128, see note below) |
| Knots | `proposal.num_knots` | int, > 0 | 4 – 6 | 4 |
| Horizon (s) | `proposal.plan_horizon` | float, > 0 | 0.3 – 1.0 | 0.6 |
| Noise σ | `proposal.noise_level` | float, unconstrained (used as Gaussian scale) | 0.3 – 0.8 | 0.4 |
| Temperature λ | `update.temperature` | float, > 0 | 0.05 – 0.2 (`capability_probe.py` sweeps [0.05, 0.1, 0.2]) | 0.15 |
| Spline type | `run.spline_type` | `"zero"` \| `"linear"` \| `"cubic"`, validated in `Run.__post_init__` | mostly `"zero"`, some `"cubic"` | `"zero"` |

**`num_samples` baseline correction:** `configs/env/hopper.py`'s `HOPPER` does not set
`proposal.num_samples`, so it resolves to the `Proposal` schema default of **128**, not the 256
that `verification/hopper_pareto_sweep.py` uses as its own standalone script constant. The step-2
search (below) was run and validated at 256 throughout, so
`configs/exp/hopper_vanilla_mppi.py` sets `proposal.num_samples: 256` explicitly rather than
silently inheriting a different, unvalidated default.

**Spline-type gap (closed 2026-08-28):** `Run.__post_init__` previously rejected `"cubic"` even
though `spline.py`'s `SPLINE_INTERPOLATORS` implements it (and a few verification scripts built
`MPPIv2` directly with `spline_type="cubic"`, bypassing `config.py`). `_check` now allows all three
spline types; `tests/test_config.py` updated to match (`test_cubic_spline_type_is_valid`, and the
schema-validation bad-input case swapped to the genuinely-invalid `"quad"`).

## 2. Cost-term weights — tunable, but NOT part of the sensitivity/LHS sweep

Added as ctor kwargs on `HopperTask` (2026-08-28) so a config's `task.kwargs` can tune them while
searching for a working vanilla setup. Defaults reproduce the original hardcoded values exactly —
no existing run is affected.

| Param | Default | Applies to |
|---|---|---|
| `height_weight` | 10.0 | `height_cost = height_weight * (height - target_height)**2` |
| `orientation_weight` | 50.0 | `orientation_cost = orientation_weight * (1 - zaxis_z)**2` |
| `velocity_weight` | 5.0 | `velocity_cost = velocity_weight * (vel - target_velocity)**2` |
| `control_weight` | 0.3 | `control_cost = control_weight * sum(u**2)` |

**Decision:** these four are fair game while hunting for a working vanilla-MPPI config (step 2),
but are held fixed at their defaults for the sensitivity analysis (step 3) and Latin Hypercube
sweep (step 4) — only the params in §1 are varied there.

## 3. Held fixed — not swept

| Param | Value | Notes |
|---|---|---|
| `run.steps` | 500 (= 10.0 s at dt = 0.02) | episode length; matches the §4 protocol length (raised from the 150/3.0s used only as a cheap search proxy — see §5's proxy-mismatch note) |
| `run.iterations` | 1 | optimizer iterations per MPC step |
| `proposal.include_mean` | `False` | forcing `True` reroutes to Predictive Sampling (argmax), not MPPI |
| `task.kwargs.target_velocity` | 2.0 (hopper baseline; task default 1.0) | high enough that standing still fails the speed objective |
| `task.kwargs.target_height` | 1.0 (unset from task default) | |

`orientation_floor`, `atom_soft_floor`, `atom_tail_tau` (also `HopperTask.__init__` kwargs) only
shape the FPL fulfillment atoms — irrelevant to vanilla cost, not listed here.

---

## 4. Step-3 sensitivity analysis protocol

One-at-a-time sweep: for each §1 param, vary it alone across its range (§1 "Values seen in repo"
column) while holding every other param at the step-2 working config.

- **Metric — reward term vs. hyperparam, normalized to [0, 1].** Vanilla cost mode only computes
  unbounded quadratic terms (`running_terms`/`terminal_terms`), which have no natural [0, 1]
  scale to plot on. Rather than inventing an ad-hoc min-max normalization (which would only be
  comparable within one sweep, not across params or later runs), reuse the task's existing
  fulfillment atoms — `running_cost_terms_f` / `terminal_cost_terms_f` in `tasks/hopper.py`
  (height/orientation/velocity/control fulfillment, each already bounded to [0, 1] by
  construction). These are computed straight from the rollout (qpos/qvel/sensordata/controls)
  independently of which cost mode drove the controller, so they can be evaluated post-hoc on the
  vanilla-MPPI trajectories with no extra plumbing. One plot per atom: atom fulfillment (mean over
  the episode, y-axis, [0, 1]) vs. the swept hyperparameter value (x-axis).
- **X-axis — min-max normalized to [0, 1], not raw units.** Each param's own tested range from §1
  (e.g. num_samples 64–1024) is rescaled so its min → 0 and max → 1
  (`x_norm = (x - x_min) / (x_max - x_min)`), instead of plotting raw values (64, 128, ..., 1024).
  This keeps relative spacing between the swept points intact and lets every sensitivity plot
  (num_samples, num_knots, horizon, noise, temperature, spline_type) share one x-axis scale
  regardless of the param's natural units — useful for eyeballing which param the reward is most
  sensitive to. `spline_type` is categorical, not min-max normalizable; plot it as a bar/strip
  over its 3 categories rather than forcing it onto [0, 1]. Keep the raw value in a tick label or
  hover/legend so the plot stays readable.
- **Rollout duration:** 10 s per episode → `run.steps = 500` at dt = 0.02 (scene.xml
  `timestep="0.02"`). Overrides the step-2/hopper-baseline `run.steps = 150`.
- **Episodes per point:** 5 seeds (`run.seeds`) per hyperparameter value, aggregated (mean ± CI
  as elsewhere in `verification/*.py`, e.g. `mean_ci`/`wilson_ci` in `verification/_ci.py`).
- **Output layout:** `results/sens_mppi/hopper/vanilla_vs_fpl/{file names}` — one file per
  swept param: `{param}.png` (overlay: vanilla + FPL, 8 fulfillment-atom lines / hatched bars)
  + `{param}.json` (raw per-seed atom means, CI, grid, x_norm, nested per controller) for
  `num_samples`, `num_knots`, `plan_horizon`, `noise_level`, `temperature`, `spline_type`, plus
  `p_value.png`/`.json` (FPL-only — `objective.p` has no vanilla counterpart). Per-arm
  `{vanilla,fpl}_config.resolved.json` and a shared `provenance.json`. One folder, not split by
  controller pair, since this is meant to grow into "every MPPI variant vs. its FPL
  counterpart" — see `verification/hopper_fpl_vs_vanilla_sweep.py`'s module docstring for the
  per-arm baseline / shared-protocol-length design decisions (2026-08-28).

## 5. Step-2 search results

Config: [`configs/exp/hopper_vanilla_mppi.py`](../configs/exp/hopper_vanilla_mppi.py)
(`VANILLA_MPPI` / `CONFIG`). Full search log and reasoning are in that file's module
docstring; summary:

- **Proxy-length mismatch (the key pitfall).** The search started at 150 steps (3.0 s) as a
  cheap proxy. A candidate that looked good there (`height_weight=25, velocity_weight=1.0`:
  85% survival at 3 s) collapsed to **10% survival** at the real 500-step (10 s) protocol
  length — height sags gradually and only produces a fall later in the episode. Every
  number below is measured at the full 500-step length; the 150-step numbers earlier in the
  search were proxy-only and are superseded.
- **Root cause.** `HopperTask`'s default weights (`height_weight=10`, `orientation_weight=50`,
  `velocity_weight=5`) let the optimizer trade torso height for speed cheaply: sagging 0.5 m
  in height only costs `10 * 0.5² = 2.5`, while a full tip-over costs `50 * 1² = 50` — so
  nothing in the vanilla cost stops the leg from folding as long as the torso stays upright.
  This is the same failure mode `tasks/hopper.py`'s FPL height-fulfillment comment describes.
- **Fix.** Grid search over `height_weight` × `velocity_weight` at the full 500-step length
  (20 seeds/cell) found lowering `velocity_weight` — not raising `height_weight` — is what
  actually buys survival: it removes the incentive to sprint at the expense of stance, rather
  than trying to out-penalize a sag that's already priced too cheaply.
- **Working config:** `height_weight=10.0` (default), `velocity_weight=0.15` (down from
  default 5.0), `noise_level=0.4`, `temperature=0.15`, `num_samples=256`. 30-seed
  confirmation: survival **0.80** (Wilson95 lower bound 0.63), vx **1.07 m/s** (54% of
  `target_velocity=2.0`), mean torso height 0.83.
- **Trade-off, stated plainly:** this reaches roughly 2/3 the speed of a well-tuned FPL/
  linear-family hopper config at similar survival (`hopper_pareto_sweep.py`'s linear family
  clears >85% survival near the same target velocity) — a quadratic cost has no per-atom
  floor to protect a minimum-fulfillment guarantee the way FPL's power-mean does, so the only
  lever is turning DOWN the reward for speed. That gap is expected, not a tuning failure to
  chase further — it's the baseline the eventual FPL comparison (the original request's
  4th bullet) is meant to be run against.

## Pipeline status

- [x] **Step 1** — hyperparameter directory (this document)
- [x] **Step 2** — working `objective.mode="normal"` config found on hopper: see §5 and
      `configs/exp/hopper_vanilla_mppi.py`
- [x] **Step 3** — OAT sensitivity sweep per §4, extended to overlay the FPL-cost arm (same 6
      params, each swept around its OWN working config — `configs/exp/hopper_vanilla_mppi.py`
      vs `configs/exp/hopper_fpl_mppi.py`, the latter's `run.steps` raised 400->500 to match
      the shared protocol length; see `verification/hopper_fpl_vs_vanilla_sweep.py`'s
      docstring for the survival spot-check justifying that) plus a 7th, FPL-only param:
      `objective.p` (the power-mean/conjunction exponent, grid `[-0.5,-1,-2,-4,-8]`, no vanilla
      counterpart). 5 seeds/point/arm, full 500-step episodes, fulfillment-atom plots + raw
      JSON under `results/sens_mppi/hopper/vanilla_vs_fpl/`. (The original vanilla-only
      artifacts under `.../vanilla_mppi/` were superseded and deleted once confirmed
      bit-identical to the "vanilla" arm here — `verification/hopper_sensitivity_sweep.py`
      still exists standalone and reproduces them if ever needed.)
      Qualitative read:
        - **Vanilla is capability-limited, not just slower, across the whole grid** — at every
          param's own baseline, FPL's height/velocity fulfillment sit far above vanilla's (e.g.
          baseline points: height 0.30 vanilla vs 0.96 FPL, velocity 0.54 vs 0.77), consistent
          with §5's survival/speed gap; `orientation_fulfillment`/`control_fulfillment` stay
          near-saturated (>=0.85) for BOTH arms everywhere swept, since neither was ever the
          binding constraint.
        - **`num_knots` and `plan_horizon` are the two params both arms are genuinely sensitive
          to**, but FPL keeps climbing where vanilla plateaus/degrades: `num_knots` 2->8 —
          vanilla velocity 0.09->0.72, FPL 0.22->0.80; `plan_horizon` — vanilla velocity peaks
          at its own baseline (0.6s, 0.54) then falls to 0.21 by 1.0s, while FPL's height/
          velocity broadly rise with horizon (noisily — e.g. a dip at 0.7s) to a joint high
          right around its own baseline (0.9s: height 0.96, velocity 0.77, though velocity's
          single-point max is 0.6s at 0.80) — each arm's step-2/flagship search independently
          landed near its own sensitivity peak, not at a shared optimum.
        - **`objective.p`** is comparatively flat from -8 to -1 (conjunction sharpening costs
          little here, matching `fpl_binding_probe.py`'s finding that hopper's binding atom is
          rarely the safety atom); weakening toward -0.5 gives the first real dip (velocity
          0.77->0.71, height 0.96->0.89), i.e. moving away from the harmonic-mean conjunction
          costs more than sharpening it further.
        - `num_samples`/`noise_level`/`temperature` remain comparatively flat for both arms
          within their tested ranges.
- [x] **Step 4** — Latin Hypercube Sampling: one 192-point, 6-D design (num_samples,
      num_knots, plan_horizon, noise_level, temperature, spline_type; hand-rolled with
      numpy since scipy isn't installed in this repo's venv) shared by BOTH arms, so
      each of the 192 samples puts vanilla and FPL at the exact same param combination —
      a paired comparison of joint effects, vs. step 3's one-at-a-time sweep. Built as
      two independently-drawn LHS batches unioned together (64 points at design_seed=0,
      then a further 128 via `--append --design-seed 1` once 64 turned out too few to
      pin down the noisier correlations — see `hopper_lhs_sweep.py --append`, which
      reuses the already-computed points instead of re-running them). Each arm still
      keeps its OWN baseline (`hopper_vanilla_mppi.py` / `hopper_fpl_mppi.py`) for
      everything not swept. 5 seeds/point/arm, full 500-step episodes; the same 4
      fulfillment atoms plus mean forward velocity (vx) and survival (fall line 0.6 on
      min torso-zaxis-z, matching `verification/_experiment.py`'s convention exactly).
      Spearman rank correlation (numpy `rankdata`+`corrcoef`, no scipy) per param per
      metric per arm, plus per-category means for the categorical `spline_type`.
      Script: `verification/hopper_lhs_sweep.py`; outputs (design, raw results,
      correlation summary, joint-effect scatter) under
      `results/sens_mppi/hopper/vanilla_vs_fpl_lhs/`.
      Qualitative read (numbers below are the 192-point read; the initial 64-point pass
      overstated a couple of weak effects as noise — e.g. `num_samples`'s vanilla-
      orientation rho dropped 0.49→0.32, survival rho 0.35→0.16 — settling out the
      correlations below is exactly why the design was grown):
        - **`plan_horizon` is the single most jointly-influential param for both arms,
          with opposite signs on speed.** Across the full 192-point cloud, vanilla's
          velocity/vx correlate strongly NEGATIVELY with horizon (rho -0.58 / -0.64) —
          confirming step 3's OAT finding that vanilla's velocity peaks near its own
          short baseline (0.6s) and degrades at longer horizons — while FPL's
          height/orientation correlate strongly POSITIVELY with horizon (rho 0.53 /
          0.67), and FPL's control_fulfillment falls with horizon even more sharply
          than vanilla's (rho -0.62 vs. -0.33). Both arms' survival rises with horizon
          (rho 0.50-0.52).
        - **`num_knots` is FPL's dominant joint-effect lever** (vx rho 0.55, velocity
          0.53, control 0.45, height 0.29) and, with the larger sample, now shows a
          real secondary effect for vanilla too (velocity 0.33, vx 0.34) — weaker than
          FPL's but no longer "flat". This reproduces step 3's OAT finding that knots is
          one of the two params FPL is most sensitive to, and refines it: vanilla is
          knot-sensitive as well, just less so, a distinction the 64-point pass wasn't
          quite powered to resolve (FPL's control-fulfillment rho alone moved from 0.32
          to 0.45 with the extra 128 points).
        - **FPL's survival advantage is concentrated near its own tuned corner of the
          space, not uniform across it.** Averaged over the full LHS cloud, FPL's mean
          survival by `spline_type` (0.21 zero / 0.37 linear / 0.35 cubic) is markedly
          LOWER than vanilla's (0.70-0.75 across all three) — the opposite of the step-3
          OAT / step-2 flagship comparison, where FPL clears 100% survival at ITS
          baseline vs. vanilla's 80%. The `num_knots`/`plan_horizon` interaction explains
          this: FPL needs both high knots AND long horizon together to reach its
          survival ceiling (its own baseline sits at knots=6, horizon=0.9s), and the LHS
          design samples that joint region only rarely, whereas vanilla's flatter,
          capability-limited response stays comparatively uniform across the whole
          space. This is the joint-effect signal step 3's one-at-a-time sweep could not
          see: FPL isn't unconditionally more robust, it's more PEAKED — high reward at
          its sweet spot, more fragile than vanilla away from it. (FPL's linear-vs-cubic
          survival gap that looked clean at 64 points, 0.35 vs. 0.43, closed to a
          statistical wash at 192, 0.37 vs. 0.35 — not a real spline preference beyond
          "zero is worse".)
        - `noise_level` mainly hits `control_fulfillment` for vanilla (rho -0.63,
          tracking control effort's sensitivity to injected exploration noise) and is
          otherwise comparatively flat for both arms, matching step 3.
        - `num_samples` and `temperature` remain the least influential dims for both
          arms across the joint design (|rho| <= 0.32 and <= 0.28 respectively at 192
          points — both dropped from their 64-point highs, confirming they were the
          noisiest reads, not the strongest), consistent with step 3's "flat" read for
          those two.
