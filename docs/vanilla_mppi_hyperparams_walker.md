# Vanilla-MPPI hyperparameter directory (walker)

**Scope:** the hyperparameters relevant to plain MPPI with the vanilla (normal) cost — i.e.
`objective.mode = "normal"`, which runs `_score_normal` in
[`sampling_base.py`](../analytic_mppi/controllers/sampling_base.py) over the quadratic terms in
[`tasks/walker.py`](../analytic_mppi/tasks/walker.py), dispatched via `("gaussian",
"path_integral")` in [`config.py`](../analytic_mppi/config.py) to `MPPIv2`. None of the `fpl_*`
params are active in this mode; they are omitted below (see `config.py`'s `Objective` dataclass
for the FPL-mode axis). Same shape as [`docs/vanilla_mppi_hyperparams.md`](vanilla_mppi_hyperparams.md)
(hopper's version of this document); read that one first if you have not, since the pipeline and
its pitfalls are common to both.

This is a reference for the sensitivity analysis and LHS sweeps — the source of truth for values
and constraints is always `config.py`/`walker.py`, not this file. Update it if those drift.

**A note on units before anything else:** walker's model timestep is `dt = 0.01`
(`analytic_mppi/envs/walker/walker.xml`, `option timestep="0.01"`) — HALF hopper's `dt = 0.02`.
Every `run.steps` value below is walker-specific; do not carry a raw step count over from the
hopper doc without re-deriving the number of seconds it represents.

---

## 1. Core optimizer hyperparameters

Walker's own pre-existing verification scripts (`verification/walker_pareto_sweep.py`,
`verification/mismatch_robustness_walker.py`) and `verification/_experiment.ENVS["walker"]` all
use ONE fixed value per param — unlike hopper, where different scripts used a genuine range
(e.g. `num_samples` 128–256). So there is no "range seen in repo" to read the step-3 sweep grid
off of here; the grids below are deliberately WIDENED around that single flagship value so the
step-2 baseline sits in the MIDDLE of its grid, not at an edge (baseline values below match
step 2's working config, §5 — every baseline coincides with the flagship value already in use,
since step 2 did not need to touch any of these six params, only the cost weights in §2).

| Param | Config field | Type / hard constraint | Flagship value (used identically across walker_pareto_sweep.py / mismatch_robustness_walker.py / `_experiment.ENVS["walker"]`) | Step-3 sweep grid (this doc; baseline centered) |
|---|---|---|---|---|
| Samples *K* | `proposal.num_samples` | int, > 0 | 256 | 64, 128, **256**, 512, 1024 |
| Knots | `proposal.num_knots` | int, > 0 | 6 | 2, 3, 4, 5, **6**, 7, 8, 9, 10 |
| Horizon (s) | `proposal.plan_horizon` | float, > 0 | 0.6 | 0.2, 0.3, 0.4, 0.5, **0.6**, 0.7, 0.8, 0.9, 1.0 |
| Noise σ | `proposal.noise_level` | float, unconstrained (Gaussian scale) | 0.8 | 0.4, 0.5, 0.6, 0.7, **0.8**, 0.9, 1.0, 1.1, 1.2 |
| Temperature λ | `update.temperature` | float, > 0 | 0.2 | 0.05, 0.1, 0.15, **0.2**, 0.25, 0.3, 0.35 |
| Spline type | `run.spline_type` | `"zero"` \| `"linear"` \| `"cubic"`, validated in `Run.__post_init__` | `"zero"` | all 3 (categorical) |

**`num_samples` — no baseline-correction footgun here.** Hopper's `HOPPER` env config left
`proposal.num_samples` at the schema default (128) while every verification script actually used
256, a mismatch only caught and documented after the fact (see the hopper doc §1). Walker's
[`configs/env/walker.py`](../configs/env/walker.py) bakes `proposal.num_samples: 256` into
`WALKER` directly and explicitly, so `configs/exp/walker_vanilla_mppi.py` /
`walker_fpl_mppi.py` inherit the correct value with no override needed and no silent drift is
possible. Decided up front, not discovered later.

**Noise/temperature grids intentionally extend ABOVE the flagship value**, not just around it
symmetrically in a narrow band — e.g. `noise_level` 0.8 sits at the flagship's own operating
point, but climbing the whole way to 1.2 asks a genuinely open question (does more exploration
noise help or hurt at this difficulty) rather than only interpolating below a value nobody has
tried exceeding.

## 2. Cost-term weights — tunable, but NOT part of the sensitivity/LHS sweep

Added as ctor kwargs on `WalkerTask` (this task) so a config's `task.kwargs` can tune them while
searching for a working vanilla-MPPI setup — mirrors `HopperTask`'s identical ctor-kwarg pattern.
Defaults reproduce the original hardcoded values exactly — no existing run is affected (verified
bit-for-bit against the pre-change hardcoded formulas before anything else in this task was
touched).

| Param | Default | Applies to |
|---|---|---|
| `height_weight` | 10.0 | `height_cost = height_weight * (height - target_height)**2` |
| `orientation_weight` | 10.0 | `orientation_cost = orientation_weight * (zaxis_z - 1)**2` |
| `velocity_weight` | 1.0 | `velocity_cost = velocity_weight * (vel - target_velocity)**2` |
| `control_weight` | 0.001 | `control_cost = control_weight * sum(u**2)` |

**Decision:** these four are fair game while hunting for a working vanilla-MPPI config (step 2),
but are held fixed at their defaults for the sensitivity analysis (step 3) and Latin Hypercube
sweep (step 4) — only the params in §1 are varied there. (In the event, only `velocity_weight`
ended up moving — see §5.)

**Ratio note (from the original task brief, resolved empirically):** walker's
height:velocity weight ratio (10:1) is 5x more height-protective than hopper's ORIGINAL ratio
(10:5 = 2:1) was before hopper's own fix. Empirically (§5) this did NOT make walker's default
vanilla cost meaningfully more survivable out of the box — raising `height_weight` further
never helped in the step-2 grid search, and `velocity_weight` was still the only lever that moved
survival, exactly as it was for hopper. The larger height:velocity ratio was a red herring, not
a head start.

## 3. Held fixed — not swept

| Param | Value | Notes |
|---|---|---|
| `run.steps` | 1000 (= 10.0 s at dt = 0.01) | episode length; matches the §4 protocol length (raised from `WALKER`'s own 150-step/1.5s flagship length, which was used only as a cheap search proxy at an even shorter 300-step/3.0s setting — see §5's proxy-mismatch note) |
| `run.iterations` | 1 | optimizer iterations per MPC step |
| `proposal.include_mean` | `False` | forcing `True` reroutes to Predictive Sampling (argmax), not MPPI |
| `task.kwargs.target_velocity` | 5.0 (walker "running regime" flagship; task default 1.5) | aggressive enough to require flight-phase running, not walking — this is deliberately a much harder target than hopper's 2.0 (see `verification/walker_pareto_sweep.py`'s module docstring) |
| `task.kwargs.target_height` | 1.2 (task default, unset) | |

`WalkerTask` has no `orientation_floor` / `atom_soft_floor` / `atom_tail_tau` ctor kwargs (unlike
`HopperTask`) — its FPL atoms use a hard clip only (`np.clip`, not `tasks/base.soft_ramp`), so
there is nothing analogous to list here.

---

## 4. Step-3 sensitivity analysis protocol

One-at-a-time sweep: for each §1 param, vary it alone across its range (§1 grid column) while
holding every other param at the step-2 working config.

- **Metric — reward term vs. hyperparam, normalized to [0, 1].** Vanilla cost mode only computes
  unbounded quadratic terms (`running_cost_terms`/`terminal_cost_terms`), which have no natural
  [0, 1] scale to plot on. Reuse the task's existing fulfillment atoms —
  `running_cost_terms_f` / `terminal_cost_terms_f` in `tasks/walker.py` (height/orientation/
  velocity/control fulfillment, each already bounded to [0, 1] by construction). These are
  computed straight from the rollout (qpos/qvel/sensordata/controls) independently of which cost
  mode drove the controller, so they can be evaluated post-hoc on the vanilla-MPPI trajectories
  with no extra plumbing. One plot per atom: atom fulfillment (mean over the episode, y-axis,
  [0, 1]) vs. the swept hyperparameter value (x-axis).
- **X-axis — min-max normalized to [0, 1], not raw units.** Each param's own tested range from
  §1 is rescaled so its min → 0 and max → 1 (`x_norm = (x - x_min) / (x_max - x_min)`), instead
  of plotting raw values. This keeps relative spacing between the swept points intact and lets
  every sensitivity plot share one x-axis scale regardless of the param's natural units.
  `spline_type` is categorical, not min-max normalizable; plot it as a bar/strip over its 3
  categories rather than forcing it onto [0, 1]. Keep the raw value in a tick label or
  hover/legend so the plot stays readable.
- **Rollout duration:** 10 s per episode → `run.steps = 1000` at dt = 0.01 (walker.xml
  `timestep="0.01"`). Overrides the step-2/flagship-baseline `run.steps = 150` (1.5 s). NOT
  hopper's 500 — copying hopper's step count here would silently run a 5-second episode instead
  of a 10-second one.
- **Episodes per point:** 5 seeds (`run.seeds` in the protocol), aggregated with
  `verification/_ci.py`'s `mean_ci` (95% CI on the per-episode mean).
- **Output layout:** `results/sens_mppi/walker/vanilla_vs_fpl/{file names}` — one file per swept
  param: `{param}.png` (overlay: vanilla + FPL, 8 fulfillment-atom lines / hatched bars) +
  `{param}.json` (raw per-seed atom means, CI, grid, x_norm, nested per controller) for
  `num_samples`, `num_knots`, `plan_horizon`, `noise_level`, `temperature`, `spline_type`, plus
  `p_value.png`/`.json` (FPL-only — `objective.p` has no vanilla counterpart). Per-arm
  `{vanilla,fpl}_config.resolved.json` and a shared `provenance.json`. Mirrors
  `verification/hopper_fpl_vs_vanilla_sweep.py`'s layout exactly (see that module's docstring
  for the per-arm-baseline / shared-protocol-length design decisions, which apply unchanged
  here — only the numbers differ).

## 5. Step-2 search results

Configs: [`configs/exp/walker_vanilla_mppi.py`](../configs/exp/walker_vanilla_mppi.py)
(`VANILLA_MPPI` / `CONFIG`) and
[`configs/exp/walker_fpl_mppi.py`](../configs/exp/walker_fpl_mppi.py) (`FPL_MPPI` / `CONFIG`).
Full search log and reasoning are in `walker_vanilla_mppi.py`'s module docstring; summary:

- **Proxy-length mismatch, faster than hopper's.** Hopper's proxy (150 steps / 3.0 s) looked
  fine and only collapsed at the full 500-step (10 s) length. On walker, the SAME default
  weights already collapse to 12% survival (1/8) by just 300 steps (3.0 s) — the failure mode
  shows up much earlier here, so even the cheap search proxy had to be a genuine 3.0 s, not a
  shorter smoke-test length, to be predictive of the full 1000-step (10 s) protocol result.
- **Root cause — same lever as hopper, one difference.** `WalkerTask`'s default
  `velocity_weight=1.0` lets the optimizer trade stance for speed cheaply, exactly like hopper's
  `velocity_weight=5.0` did. UNLIKE hopper, raising `height_weight` (10 → 20) did NOT help on
  walker at any `velocity_weight` tested — it was neutral-to-harmful in every cell of a 2-D grid.
  `velocity_weight` alone is the lever that moves survival.
- **Search.** 3.0 s proxy grid over `velocity_weight` (15 seeds/cell after an 8-seed
  neighborhood-finding pass): the safe region is `velocity_weight <= ~0.15` (survival ≥ 0.93);
  0.35–0.7 is a noisy, no-clean-optimum middle region (survival 0.47–0.73). Three candidates
  (`velocity_weight` = 0.10 / 0.15 / 0.20) were then validated at the FULL 1000-step (10 s)
  protocol length (12 seeds/cell): 0.10 → 100% survival (12/12), vx 2.20 m/s (44% of target);
  0.15 → 75% survival, vx 2.56 m/s (51%); 0.20 → 50% survival, vx 2.78 m/s (56%).
- **Working config:** `height_weight=10.0` (default, unchanged), `velocity_weight=0.10` (down
  from default 1.0). 30-seed confirmation @1000 steps: survival **0.97** (29/30, Wilson95 lower
  bound 0.833), vx **2.19 m/s** (44% of `target_velocity=5.0`). Bundle:
  `runs/capability/walker/mppi_normal_K256_10s_weighttuned/`.
- **FPL flagship confirmation** (`walker_fpl_mppi.py`, no cost-weight search needed — the FPL
  atoms/composition are the arm being compared against, not tuned): 30-seed result @1000 steps:
  survival **1.00** (30/30, Wilson95 lower bound 0.89), vx **1.71 m/s** (34% of target). Bundle:
  `runs/capability/walker/mppi_fpl_K256_10s/`.
- **Notable, and DIFFERENT from hopper: FPL is slower here, not faster.** Hopper's FPL flagship
  clearly beat its tuned vanilla arm on speed (83% vs 54% of target) at comparable-or-better
  survival. On walker, FPL is actually SLOWER than the tuned vanilla config (1.71 m/s / 34% vs.
  vanilla's 2.19 m/s / 44%) while only marginally safer (1.00 vs. 0.97 survival). At
  `target_velocity=5.0` (flight-phase running — a much harder ask than hopper's 2.0), FPL's
  min-fulfillment conjunction appears to protect the orientation/height atoms at more cost to
  the velocity atom than vanilla's tuned-down quadratic sum does: vanilla, once
  `velocity_weight` is turned down enough to survive, can still push speed opportunistically in
  a way the conjunction does not reward. This is reported as an empirical result to carry into
  steps 3/4 (where it may or may not hold across the swept hyperparameter ranges), not
  papered over to match the hopper narrative.
- **Trade-off, stated plainly (the part that DOES match hopper):** walker's
  `target_velocity=5.0` is a substantially harder ask than hopper's 2.0, so neither arm's
  fraction-of-target here is on the same absolute scale as hopper's numbers. A quadratic cost
  still has no per-atom floor to protect a minimum-fulfillment guarantee the way FPL's
  power-mean does — vanilla's survival (0.97, not 1.00) is bought by turning DOWN the reward for
  speed, and it is still the less robust arm by that measure even though it happens to average a
  higher speed at this particular difficulty setting.

## Pipeline status

- [x] **Step 1** — hyperparameter directory (this document); `height_weight` /
      `orientation_weight` / `velocity_weight` / `control_weight` ctor kwargs added to
      `WalkerTask` (defaults verified bit-for-bit identical to the pre-change hardcoded
      formulas); `configs/env/walker.py` (`WALKER` / `WALKER_META`) created and verified to
      match `verification/_experiment.ENVS["walker"]` exactly.
- [x] **Step 2** — working `objective.mode="normal"` config found on walker: see §5 and
      `configs/exp/walker_vanilla_mppi.py`. FPL flagship confirmed at the same protocol length
      in `configs/exp/walker_fpl_mppi.py`.
- [ ] **Step 3** — OAT sensitivity sweep per §4, vanilla vs. FPL overlay (6 shared params +
      FPL-only `objective.p`) — `verification/walker_sensitivity_sweep.py` /
      `verification/walker_fpl_vs_vanilla_sweep.py`.
- [ ] **Step 4** — Latin Hypercube Sampling joint-effects sweep over the 6 shared params,
      paired vanilla/FPL — `verification/walker_lhs_sweep.py`.