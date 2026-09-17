# Analytic-MPPI — State of the Repo

*Snapshot: 2026-09-10, branch `abg-mjx`, HEAD `8a9957e "first version: very slow"`.*
*Sections 1–8 describe the repo as built. Sections 9–13 are the current work brief — read
those before touching anything.*

**What this repo is:** a MuJoCo-in-the-loop sampling-MPC testbed built to answer one
research question — does **FPL** (Fulfillment / Priority Logic: bounded per-objective
fulfillment aggregated by a power-mean) beat a conventional weighted-sum cost for
morphology-agnostic legged control? Everything else in the repo — the controller zoo, the
config system, the MJX arm — exists to make that comparison fair and reproducible.

---

## 1. Architecture — four orthogonal axes

One `ExperimentConfig` fully determines a run (same config + seed → same trajectory). The
schema mirrors the four stages of `SamplingController.act()`:

| axis | config node | maps to | question it answers |
|---|---|---|---|
| **proposal** | `proposal.kind` | `sample_knots()` | how are candidate plans drawn? |
| **objective** | `objective.mode` | `_score_*()` | how does a rollout become one scalar? |
| **update** | `update.rule` | `update_mean()` | how do scores become the new plan? |
| **run** | `run.*` | outer loop | steps, seeds, spline, backend, viewer |

A `(proposal.kind, update.rule)` pair resolves through `_DISPATCH`
([config.py:290](analytic_mppi/config.py#L290)) to a concrete controller class. Pairs
nobody wrote a class for **raise at load time** rather than silently running something
else. `accepted_params()` derives the legal kwargs by MRO introspection — a parameter the
chosen controller cannot accept is an error, not a silent drop.

```
configs/*.py  →  ExperimentConfig  →  resolve()  →  (task, backend, controller)  →  run_episode
   (Python)         (frozen dataclasses)              ↑ errors on any unrepresentable combo
                            ↓
                    to_json() = provenance record (hashable, diffable)
```

---

## 2. MPPI variants — what we have

All 13 live in `SAMPLING_CONTROLLERS` ([controllers/__init__.py](analytic_mppi/controllers/__init__.py)).
Every one subclasses `SamplingController` ([sampling_base.py](analytic_mppi/controllers/sampling_base.py)),
so they share the knot-spline loop, the scoring pipeline, and warm-starting — only
`sample_knots()` and `update_mean()` differ.

### Baseline / textbook

| key | class | what it does |
|---|---|---|
| `vanilla` (legacy) | [`MPPI`](analytic_mppi/controllers/mppi.py) | textbook full-horizon MPPI, callable cost. Used only by the unicycle env |
| `mppi` | [`MPPIv2`](analytic_mppi/controllers/mppi_v2.py) | **the workhorse.** Knot-spline MPPI, softmax `w_k ∝ exp(−(S_k − min S)/λ)` |
| `predictive_sampling` | [`PredictiveSampling`](analytic_mppi/controllers/predictive_sampling.py) | greedy argmax, no softmax; current mean injected as a sample so it can't regress |
| `cem` | [`CEM`](analytic_mppi/controllers/cem.py) | cross-entropy: refit a diagonal Gaussian to the top-`num_elites` |
| `mppi_cma` | [`MppiCma`](analytic_mppi/controllers/mppi_cma.py) | block-diagonal CMA; per-knot covariance with an eigenvalue clamp |
| `dial` | [`DIAL`](analytic_mppi/controllers/dial.py) | diffusion-inspired annealing, `σ[i,h] = σ₀·exp(−i/(β₁N) − (H−1−h)/(β₂H))` |

### FPL-native variants (the research contributions)

Each attacks a *different stage* of MPPI-as-inference using the one thing FPL provides
that a scalar cost cannot: a **bounded, absolutely-calibrated** reward in `[0,1]`.

| key | class | stage attacked | status |
|---|---|---|---|
| `fpl_adaptive` | [`FplAdaptiveMPPI`](analytic_mppi/controllers/fpl_adaptive.py) | **proposal** — adapt covariance; widen variance along dims that move the *binding* objective | clean negative (documented) |
| `fpl_colored` | [`FplColoredMPPI`](analytic_mppi/controllers/fpl_colored.py) | **proposal** — power-law `f^{−β}` colored noise (iCEM-style) + absolute-scale explore/exploit schedule | negative; `β=0, use_absolute_scale=False` is bit-compatible with plain MPPI |
| `fpl_tempered` | [`FplTemperedMPPI`](analytic_mppi/controllers/fpl_tempered.py) | **weighting** — set λ on an absolute scale; ESS-targeting bisection, feasibility gate | active |
| `fpl_shielded` | [`FplShieldedMPPI`](analytic_mppi/controllers/fpl_shielded.py) | **selection** — lexicographic safety: hard feasibility filter (`min over time & safety atoms ≥ τ`), softmax on performance among survivors, graceful fallback | **the strongest FPL-only claim** — a weighted sum provably can't encode strict priority |
| `composed_grad` | [`ComposedGradientMPPI`](analytic_mppi/controllers/composed_gradient.py) | **update** — per-objective ES gradients composed with weights favouring the worst-satisfied objective; stays inside the sample hull | active |
| `fpl_gmm` | [`FplGmmSampler`](analytic_mppi/controllers/experimental.py) | multi-modal proposal | experimental |

Plus four notebook-grade controllers in [experimental.py](analytic_mppi/controllers/experimental.py)
(`UniformGD`, `GaussianGD`, `RankCMA`, `FplValueCMA`).

---

## 3. MPC (non-sampling) — what we have

| what | where | method | status |
|---|---|---|---|
| **Gradient MPC** | [`GradientMPC`](analytic_mppi/controllers/gradient_mpc.py) | projected Adam on spline knots, gradient via `jax.grad` through the differentiable MJX rollout | **deprecated for contact tasks — see §9.1.** Keep as a baseline arm; do not invest in speeding it up |
| **Cost-GD refinement** | [`cost_gd.py`](analytic_mppi/controllers/cost_gd.py) | BPTT through `mjd_transitionFD`. Wraps any sampling controller: sample → roll out → refine top-N by gradient descent. Free joints handled in tangent space via `mj_integratePos` | same BPTT caveat, **but its FD-Jacobian plumbing is the reusable asset** — WO-1 and WO-2 both need it |
| **iLQR / iLQG** | — | *not implemented.* Seam exists: subclass `GradientMPC._build_plan_fn`, register `("gradient", "ilqg")` | **WO-1: this is now the primary gradient arm** |

The design point of `GradientMPC` is that it descends **exactly the objective MPPI ranks
by** — `jax_scoring.score_*` are formula-for-formula mirrors of the numpy scorers, so
"sampling vs. gradient" is a clean controlled comparison rather than two different problems.
**This property is non-negotiable and iLQR must preserve it** (see §11).

Bounds are handled by **projection** (clip knots to `[u_min, u_max]` after each Adam step),
matching the sampling stack's clip. A tanh reparameterization was considered and rejected
(needs atanh on warm-start plans, distorts step sizes for asymmetric ranges).

---

## 4. The math

### 4.1 Control parameterization
Plans are **knot splines**: `num_knots` knots uniformly on `[0, plan_horizon]`,
interpolated to the sim grid (`zero` / `linear` / `cubic`). Warm start = `shift_plan`
(shift the spline by `dt` and hold the tail). The knot→controls map is **linear**, so
gradient MPC precomputes it as a basis matrix `W` by running the numpy interpolator on
one-hot knots — identical interpolation by construction, one einsum inside `jit`.

### 4.2 The scoring pipeline — power means everywhere

The central object is the **generalized (power) mean**
([tasks/base.py:26](analytic_mppi/tasks/base.py#L26)):

$$M_p(x; w) = \left(\sum_i w_i x_i^{\,p}\right)^{1/p}, \qquad M_0 = \exp\!\left(\sum_i w_i \log x_i\right)$$

This single knob spans the whole baseline↔FPL axis:
- **`p = 1`** with weights `w` → exactly a **linear scalarization** `Σ wᵢxᵢ` (sweep `w` → the "linear-weight family", the baseline arm)
- **`p < 0`** with uniform weights → **min-fulfillment conjunction** (soft weakest-link, the FPL arm)
- **`p → −∞`** → hard `min`

Applied on **two axes**, and the order matters — that's what distinguishes the modes:

| `objective.mode` | aggregation order | additive in time? |
|---|---|---|
| `normal` | `Σ_t Σ_j c_j(t)·dt + terminal` — plain quadratic cost | yes |
| `fpl_cost` | power-mean **across atoms** per step → aggregate **over time** | yes, if time agg is a sum |
| `fpl_discounted` | discount **over time** per atom → power-mean **across atoms** | **no** — needs WO-3 |
| `fpl_layered` | inner power-mean within atom *groups* (`group_p`) → outer power-mean across groups | **no** — needs WO-3 |
| `hybrid` | power-mean + a `−log` **log-barrier** on designated floor atoms | depends on placement |

Time aggregation is either a γ-discounted normalized sum `(1−γ)/(1−γ^H)·Σγᵗ` or, when
`time_p ≤ 0`, a **weakest-link-over-time** power mean (hopper/walker use `time_p = −2.0`).
The `time_p ≤ 0` case is **not** additive in time and also needs WO-3.

### 4.3 The bridge from reward to MPPI
FPL produces a reward `u_k ∈ (0,1]`, larger-is-better; MPPI wants a cost. The scorer
returns **`S_k = −log(u_k)`**, which turns the standard softmax into

$$w_k \;\propto\; \left(\frac{u_k}{\max_j u_j}\right)^{1/\lambda}$$

— a genuine MPPI update over the FPL composite, so **the FPL and linear arms share an
identical update rule and differ only in scoring.** That is the fairness protocol.

Alternative weightings on `MPPIv2`: `proportional` (`w_k = u_k/Σu_j`, no temperature) and
`exp` (`w_k ∝ exp(r_k/λ)`).

### 4.4 Fulfillment atoms
Each task exposes 3–4 atoms mapping a physical quantity to `[0,1]` via a **one-sided
ramp** — e.g. hopper height: full credit ≥ 1.15 m, linear to 0 at 0.85 m.
[`soft_ramp`](analytic_mppi/tasks/base.py#L53) optionally replaces the hard clip with an
exponential tail, because a flat-zero region is **fatal under a `p<0` conjunction**: the
composite pins at ε as soon as any atom clips and stops distinguishing "just below floor"
from "catastrophic" — exactly the region the conjunction exists to guard.

### 4.5 Known theoretical gaps
[docs/mppi_math.md](docs/mppi_math.md) is an honest audit, not marketing. It documents:
the implemented update is **not** Williams' update (a term is missing); **clipping breaks
the change of measure** (asymmetric confound); **λ and `p` are confounded**, worst exactly
where the results live; FPL is characterized as a **second-order correction to a uniform
linear scalarization**; the Jensen gap on the mean update; and two live bugs-in-waiting
(`fpl_discounted` order-of-operations optimism, the ε-plateau on walker/quadruped/cube).

**New entry, added by this brief — the `p<0` derivative singularity.** Since

$$\frac{\partial M_p}{\partial x_i} = w_i\, x_i^{\,p-1} M_p^{\,1-p}$$

the gradient diverges as `x_i → 0` for any `p < 1`, and the Hessian diverges faster.
Sampling never evaluates a derivative so this was invisible; **any second-order method
(WO-1) hits it immediately** and it will show up as `Quu` indefiniteness on exactly the
failing rollouts. `soft_ramp`'s exponential tail keeps `x > 0` but exponentially small,
which is not enough. See WO-3.4 for the mandatory floor and the fairness consequence.

---

## 5. Dynamics backends

| backend | class | role |
|---|---|---|
| `mujoco` (default) | [`MujocoBackend`](analytic_mppi/dynamics/mujoco_backend.py) | CPU. `mujoco.rollout` (multithreaded) for parallel planning rollouts; `step`/`get_state` for the closed loop |
| `mjx` (opt-in) | [`MJXBackend`](analytic_mppi/dynamics/mjx_backend.py) | GPU. `jit(vmap(scan(mjx.step)))`, **planning-only** |
| `diffmjx` (planned) | — | **WO-2.** MJX fork with informative contact gradients |

`MJXBackend` deliberately implements no `step`/`get_state`: **the closed loop is always
stepped by the CPU backend**, so the physics of record, the viewer, `--print-costs` and
video rendering stay exactly what produced every existing result. An MJX run differs from
its CPU baseline only in which model the *controller believes* — the same structure as the
model-mismatch studies. It runs float32 on GPU (f64 is ~64× slower on a consumer card and
the sampler's noise dwarfs f32 error).

Two faces: the numpy `rollout()` (drop-in for every sampling controller) and the jax-native
`rollout_jax()` / `step_fn()` that `jax.grad` flows through for gradient MPC.

`apply_perturbation()` builds a mismatched "true" model (mass / gain / friction / damping
scales) for robustness studies — you cannot retune weights for an unknown drift, which is
the point.

---

## 6. Tasks

11 registered in [`TASKS`](analytic_mppi/tasks/__init__.py): `pendulum`, `pendulum_com`,
`walker`, `hopper`, `cube`, `g1_standup`, `g1_walk`, `g1_reach`, `quadruped`,
`quadruped_com`, `quadruped_jump`. MJCF lives under `analytic_mppi/envs/<name>/`.

Every task defines costs **twice** — `running/terminal_cost_terms` (quadratic) and
`*_terms_f` (fulfillment atoms) — in batched numpy that broadcasts over `(K, H, …)`.

**JAX cost mirrors** ([tasks/jax_costs/](analytic_mppi/tasks/jax_costs/)) exist for 4 tasks
only: `hopper`, `walker`, `g1_standup`, `g1_walk`. These are *parallel* implementations,
not a refactor — the numpy code is the definition of every published number and uses
patterns that don't survive `jit`/`grad`. They're pinned by parity tests at **1e-6**;
`has_jax_costs()` is deliberately jax-free so config validation works without the extra.

Note for WO-4: these 11 tasks are **11 distinct topologies**, which is the wrong shape for
a morphology sweep under MJX. See §9.5.

---

## 7. Libraries

| library | version | why |
|---|---|---|
| **numpy** | 2.2.6 | every cost, every controller, the whole CPU path |
| **mujoco** | **3.5.0 (pinned `>=3.2,<3.6`)** | physics + `mujoco.rollout` + `mjd_transitionFD`. The upper bound is deliberate: 3.12 changes fixed-seed hopper trajectories, silently breaking comparability |
| **matplotlib** | — | `viz/plot.py`, study panels |
| **jax / jaxlib** | 0.6.2 (CUDA 12) | `[mjx]` extra only — GPU rollouts + autodiff |
| **mujoco-mjx** | **==3.5.0 exact** | pinned exactly because it pip-depends on its same-version `mujoco`; an unpinned `>=3.5` would upgrade the core CPU sim as a side effect |
| **pytest** | — | `[dev]` |
| **imageio[ffmpeg]** | ≥2.30 | `[viz]` extra, mp4 recording |
| **diffmjx** *(planned)* | git, `[diffmjx]` extra | WO-2. Pulls a **fork of mujoco-mjx** plus `mjx_diffrax` and `softjax`. **Version-collision risk is the first thing to check** — see WO-2.0 |

**No JAX in the default path.** `dynamics/__init__.py` lazy-loads `MJXBackend` via
`__getattr__`; `gradient_mpc.py` imports jax only in `__init__`. The controller registry
and all config validation work on a machine that never installed the extra.
**Preserve this for diffmjx: a third extra, lazy-loaded, never imported at config time.**

**Environment note:** the `a-mppi/` venv now has the full `[mjx]` stack (jax 0.6.2 CUDA,
mujoco-mjx 3.5.0, mujoco downgraded 3.8.1 → 3.5.0). A known workaround is baked into
`mjx_backend.py`: `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false`, because XLA's Triton
autotuner hard-crashes compiling the **backward** pass of mjx's constraint solver on
jax 0.6.2 / RTX 4070.

---

## 8. Experiment / eval layer

- **[eval.py](analytic_mppi/eval.py)** — `make_controller` (fresh backend per call, so
  episodes are independent), `run_episode`, `run_study` (multi-seed), per-task metric and
  panel functions, `render_video`. Named init states (`hopper_stand`, `g1_stand`,
  `cube`, `barkour_stand`, `hang_down`) settle the robot before the task begins so the
  opening transient doesn't confound results.
- **[configs/](configs/)** — `base.py` (all defaults) → `env/*.py` (published per-env
  values, asserted equal to `_experiment.ENVS` by a test) → `exp/*.py` (experiments,
  ablations, sweeps).
- **[verification/](verification/)** — **45 study scripts**: pareto sweeps, LHS/grid
  searches, sensitivity, mismatch robustness, sampler races, portability, zero-tuning,
  shield studies, video renderers, and `mjx_parity_hopper.py`.
- **Tests — 78 total**: `test_config.py` (33), `test_mppi_smoke.py` (25),
  `test_gradient_mpc.py` (8), `test_jax_costs.py` (7), `test_mjx_backend.py` (5).

### Findings already written up
- [FPL_FINDINGS.md](FPL_FINDINGS.md), [FPL_MPPI_CAMPAIGN.md](FPL_MPPI_CAMPAIGN.md),
  [FPL_MPPI_HANDOFF.md](FPL_MPPI_HANDOFF.md)
- Headline results: sampler race = **clean negative**; portability across samplers =
  **win**; zero-tuning = **win**; mismatch robustness portable = **win**; g1 aggressive
  reach = **honest null**.
- [docs/vanilla_mppi_hyperparams.md](docs/vanilla_mppi_hyperparams.md) — the baseline
  search log (hopper vanilla lands at 80% survival, 54% of target speed).
- [paper/](paper/) — `fpl_publishability.tex`, annotated bibliography, deep research report.

---

# 9. Current diagnosis and plan

## 9.0 What the gradient arm is actually for — read this first

**The comparison axis of this repo is scoring: FPL vs. weighted-sum, controller held
fixed.** It is *not* sampling vs. gradient. Every decision below follows from that, and
misreading it leads to optimizing the wrong things.

So iLQR is not being built because it is a faster or better controller. It is being built
as an **instrument that breaks the λ/`p` confound** — the most damaging open criticism of
the FPL claim, logged in [docs/mppi_math.md](docs/mppi_math.md) and §4.5 as "worst exactly
where the results live." A skeptic can always say the FPL arm found a better temperature
rather than a better objective.

**iLQR has no λ.** No softmax, no temperature, no `S_k = −log(u_k)` bridge (§4.3) — the
power mean enters directly as a cost and the optimizer descends it. A 2×2 of
`{FPL, linear} × {MPPI, iLQR}` eliminates the temperature explanation *structurally*
rather than by sweeping around it.

Second benefit: the existing portability win is across **samplers**, which all share the
knot-spline loop, the noise structure, and the same update family. iLQR is a genuinely
different optimizer class — deterministic, second-order, different search space. FPL
holding up there is much stronger evidence that the effect lives in the objective rather
than in an interaction with some proposal distribution.

**Consequence for prioritization:** any change that perturbs the FPL and linear arms
**asymmetrically** is a threat to the headline claim and outranks everything else. Two such
changes exist and are now the top of the queue — see WO-3.3 and WO-3.4. Conversely, the
controller-engineering experiments (T2.1, T2.2) justify the *implementation* and belong in
an appendix; they are not paper figures for this project.

## 9.1 Why the gradient arm is slow — and why speeding it up is the wrong fix

`GradientMPC` takes `jax.grad` of the total cost through a 30-step MJX rollout. That is
**BPTT**: it chains 30 one-step Jacobians into a single scalar gradient. Two separate
problems, only one of which is about speed.

**Speed.** Each Adam step needs a full forward + full reverse pass. 30 Adam steps × 30
sim steps = 900 differentiated steps per control decision. 1.35 s/step is roughly what
that costs. Cutting Adam iterations trades the gap for solution quality and doesn't
change the asymptotics.

**Correctness, which is worse.** Through contact, the product of many Jacobians is
ill-conditioned — the gradient becomes chaotic or vanishes. Worse, **stock MJX returns a
zero contact gradient whenever bodies are not currently touching**, so the optimizer gets
no signal at all about a contact it has not yet made. On hopper that means the gradient
arm can refine a gait it already has and cannot discover a footfall it doesn't.

**iLQR fixes both.** It only ever needs *one-step* Jacobians `(A_t, B_t)` — obtainable
in one `vmap` across the horizon, fully parallel — and never multiplies them together.
The Riccati recursion composes them *with regularization* (`Quu + λI`), and the forward
pass re-simulates through the **true nonlinear dynamics** with a line search. A bad
Jacobian at one timestep degrades one step of one iteration instead of poisoning a single
global gradient. Expect one to two orders of magnitude on wall-clock and a qualitative
change in contact behaviour.

**Do not delete `GradientMPC`.** It becomes the BPTT baseline arm for WO-5's headline
ablation. Just stop optimizing it.

## 9.2 What iLQR does and does not do with contact

It never needs a contact schedule; there is no combinatorial object anywhere in the
algorithm. The mode sequence is whatever the forward rollout produced, and mode changes
happen in the **forward pass** — if the new controls land the foot a step earlier, the new
nominal trajectory simply has a different mode sequence and nothing had to be told.

The limitation is that the linearization **cannot anticipate** a mode change. `A_t, B_t`
just before touchdown contain no information that touchdown is coming. This is exactly
the myopia that makes a hopper retract its leg to avoid a locally costly impact.

> **iLQR handles contacts. It does not discover them.**

Three mitigations, all the same knob in different clothes: informative contact gradients
(WO-2), a terminal value that sees past the horizon (WO-3.3), and warm-starting from the
sampler (WO-1.5). The last one is why the target architecture is **MPPI → iLQR**, not
iLQR alone: the sampler is the only component that can discover a mode, and iLQR is the
only one that can produce feedback gains.

## 9.3 Order of work

```
WO-3.3/3.4  the two asymmetric confounds     ← START HERE. Threatens published results.
WO-0        environment + parity gates       (blocks all gradient work)
WO-3.1/3.2  FPL cost made DDP-compatible     (blocks WO-1 on fpl_discounted/layered/time_p<0)
WO-1        iLQR behind the existing seam    (the main build)
WO-2        diffmjx integration              (upgrade; WO-1 must work on stock MJX first)
WO-4        parametric morphology family     (independent, can run in parallel)
WO-5        experiment protocol              (consumes all of the above)
```

**WO-3.3 and WO-3.4 come before the build.** They are not iLQR prerequisites — they are
confounds sitting directly on the FPL-vs-linear claim, and they affect numbers already
published in `FPL_FINDINGS.md`. Landing them first means the re-run required by §11.6
happens once, against a stable baseline, instead of being tangled up with a new controller.

**WO-1 must reach a working state on stock MJX before WO-2 starts.** If iLQR and diffmjx
land together and results are bad, you cannot tell which one is at fault.

---

## 10. Work orders

### WO-0 — Environment and gates

**0.1** Enable `jax_enable_x64` for all Jacobian-comparison tests. The default float32
produces ~1e-3 discrepancies against `mjd_transitionFD` that look like real bugs and are
not. Keep production rollouts in f32 (see §5) but gate correctness tests on f64.

**0.2** Add a Jacobian parity test — `tests/test_jacobians.py`, target ≤1e-6 relative
Frobenius error between `jax.jacobian(mjx.step)` and `mjd_transitionFD`, on:
pendulum, double pendulum, **a free-floating body** (this is where quaternion / tangent-space
handling with `nq ≠ nv` silently breaks — the error localizes to the rotational block),
and a 7-DoF arm in free space. All smooth, no contact. `cost_gd.py` already has the
tangent-space `mj_integratePos` plumbing; reuse it rather than rewriting.

**0.3** Add a **solver-unroll sensitivity** test. MJX autodiff differentiates the
*algorithm*, so you get the derivative of a partially converged solve. Sweep `iterations`
/ `ls_iterations` and assert `‖A_ad − A_fd‖` decreases monotonically and flattens. Record
the iteration count where it flattens; that number becomes a config default. If it does
**not** converge, stop and report — that means implicit differentiation is needed and the
plan changes.

**0.4** Verify gradients stay correct when the **active contact set changes**. MJX pads
the contact array; it is easy to end up differentiating through padding. A two-sphere
approach/separate test is enough.

**Acceptance:** all four green on f64, documented in `docs/jacobian_audit.md`.

---

### WO-1 — iLQR behind the `("gradient", "ilqg")` seam

**Files:** new `analytic_mppi/controllers/ilqr.py`; register in `_DISPATCH`; extend
`GradientMPC._build_plan_fn` or subclass it.

**1.1 — Structure.** Standard DDP/iLQR with Gauss-Newton Hessians:
- backward pass as a `lax.scan` over `(Q_x, Q_u, Q_xx, Q_uu, Q_ux)`, producing `(k_t, K_t)`
- Levenberg regularization `Q̄_uu = Q_uu + λI` with adaptive λ (increase on failed
  factorization or failed line search, decrease on success)
- forward pass re-simulating **true dynamics** with backtracking on α
- Jacobians `(A_t, B_t)` via `vmap(jax.jacobian(step_fn))` across the horizon — one
  batched call, not a loop

**1.2 — Parameterization: run iLQR in per-step control space. Settled, do not revisit.**

The repo's sampling stack optimizes over knots via the linear basis `W` (§4.1); iLQR will
optimize over per-step controls and project back with `θ = W⁺u*`. On hopper that is 12
DOF vs 90 — the two optimizers do **not** search the same set.

That would be fatal if the claim were "gradient beats sampling." It is not (§9.0). The
claim is FPL vs. weighted-sum with the controller fixed, so both FPL and linear arms sit
in the same parameterization and the comparison is clean. The difference between arms is
scoring and nothing else. Take the simple option.

Note also that `W` is not lossless at footfall: contact-rich optimal control wants
near-bang-bang torque, and 4 linearly-interpolated knots cannot represent a switch
mid-interval. iLQR will find one, the projection discards it.

**Therefore log the projection residual — per arm — as self-protection, not fairness:**

```
resid = ‖(I − WW⁺)u*‖ / ‖u*‖        # logged separately for FPL and linear
J_pre  = J(u*)                       # what iLQR's line search accepted on
J_post = J(WW⁺u*)                    # what the robot actually executes
```

A `p<0` conjunction punishes the weakest atom, and near-contact atoms are usually the weak
ones — so the FPL optimum plausibly wants *sharper* switching than the linear one, and
would therefore lose more to the projection. If both arms lose ~3%, ignore it. **If FPL
loses 15% and linear loses 3%, the knot basis is handicapping your own thesis** and you are
understating your result. Two extra scalars, no design change, and you want to know.

Report `J_post` as the headline metric (it is what the robot experiences) and keep `J_pre`
only as an iLQR convergence diagnostic — iLQR's monotone-descent guarantee applies to
`J_pre`, and cost can increase across the projection.

Minor: for `run.spline ∈ {zero, linear}`, `Wθ` is a convex combination of adjacent knots so
knot-feasible implies control-feasible. For `cubic` it does not — splines overshoot — so a
second clip after projection is needed, which reintroduces the "clipping breaks the change
of measure" confound. Prefer `linear` for the iLQR arm, or document the extra clip.

**1.3 — Control bounds.** Use box-constrained iLQR (projected-Newton on the `Q_uu`
sub-problem) rather than clip-after-update. Clipping after an unconstrained Newton step
produces a direction that isn't a descent direction and will trip the line search
constantly. This is a real deviation from the sampling stack's clip; document it in
`docs/mppi_math.md` alongside the existing "clipping breaks the change of measure" entry,
because it is the same class of confound.

**1.4 — Instrumentation is not optional.** Log per MPC step, from the first commit:
line-search acceptance rate, final α, regularization λ trace, iteration count, cost
reduction per iteration, and `Q_uu` condition number. With contact these are the only
convergence diagnostics you have, and they are *also* WO-5 measurements (§12.2). Persist
them into the existing provenance record.

**1.5 — Warm-start from a sampling controller.** Mirror the `cost_gd.py` wrapper pattern:
accept any `SamplingController` instance, take its mean plan as the iLQR nominal. This is
the MPPI → iLQR architecture and it is what makes iLQR usable on contact tasks at all.

**Acceptance:**
- pendulum and cart-pole: converges to the known LQR solution, ≤1e-4 cost gap
- hopper: ≥20× faster than `GradientMPC` at matched final cost
- line-search acceptance rate >50% on hopper with warm start
- 8+ tests in `tests/test_ilqr.py` matching the density of `test_gradient_mpc.py`

---

### WO-3 — FPL cost: two confounds, then DDP compatibility

**Split by urgency.** 3.3 and 3.4 are **not** iLQR work — they are defects that perturb the
FPL and linear arms *asymmetrically* and therefore sit directly on the claim in §9.0. They
affect numbers already in `FPL_FINDINGS.md` and come first, before any controller work.
3.1 and 3.2 are the DDP-compatibility build and block WO-1 for three of the five objective
modes.

The asymmetry is the whole point and is easy to miss. For `p = 1`, `∂M_p/∂x_i = w_i` —
bounded, well-conditioned, insensitive. For `p < 0` the same quantity diverges and the
composite is dominated by the weakest atom. **Anything that touches small fulfillment
values, or truncates the horizon, moves the FPL arm and leaves the linear arm alone.**
That is indistinguishable from an FPL effect unless it is fixed identically in both.

**3.1 — The problem.** DDP requires an additively separable stage cost. `fpl_discounted`,
`fpl_layered`, and any mode with `time_p ≤ 0` apply a nonlinearity *after* time
aggregation, so there is no `ℓ(x_t, u_t)` to expand.

**3.2 — The fix is exact, not an approximation.** Every power mean is a quasi-arithmetic
mean — of the form `g⁻¹(Σ g(x))` — so it decomposes into an additive accumulator plus a
terminal nonlinearity. Augment the state with `z_t ∈ ℝⁿ`:

| mode | accumulator (additive stage "cost") | terminal nonlinearity |
|---|---|---|
| `time_p = p ≤ 0` over one atom | `z ← z + w_t · x_t^p` | `−(z)^{1/p}` |
| `p = 0` (geometric) | `z ← z + w_t · log x_t` | `−exp(z)` |
| `fpl_discounted` (n atoms) | `z_j ← z_j + γᵗ c_j(t)`, `z ∈ ℝⁿ` | `−M_p(z)` |
| `fpl_layered` | one accumulator per group | nested `M_{p_out}(M_{p_in}(·))` |

This is **exact** — no Frank-Wolfe, no reweighting approximation. The augmented dynamics
are **block-triangular** (`x` does not depend on `z`), so the extra Jacobian blocks are
`∂z'/∂x`, `∂z'/∂u`, `∂z'/∂z = I` and `∂x'/∂z = 0`. Cost in the backward pass is
negligible; `n` is 3–4 atoms.

**Bonus worth logging:** `V_z` at each timestep is the vector of *marginal values* of each
objective at the current fulfillment level. For `p < 0` these automatically upweight the
worst-satisfied objective. That is FPL's core semantics emerging inside the Riccati
recursion as a time-varying weighting, and it is a directly reportable result — it is the
gradient-side analogue of what `composed_grad` does by sampling.

**3.3 — Terminal value: fix the horizon truncation. `[PRIORITY — do before WO-0]`**

The current time aggregation `(1−γ)/(1−γ^H)·Σγᵗ` **renormalizes** to `[0,1]` rather than
adding a tail term. That silently assumes post-horizon fulfillment equals the in-horizon
average — optimistic, and very likely the same defect already flagged as "`fpl_discounted`
order-of-operations optimism" in `docs/mppi_math.md §8`.

**Why this is a confound and not just a bug:** truncation makes every atom read low.
A `p = 1` sum absorbs that as a uniform scale factor. A `p < 0` conjunction does not — it
weights low fulfillments heavily, so truncation distorts the *composition*, not just the
level. The renormalization therefore biases the FPL arm and leaves the linear arm alone.
Replace with an explicit terminal estimate:

```
FV = (1−γ)·Σ_{t<H} γᵗ r⃗_t  +  γ^H · v⃗(x_H)
```

Cheapest `v⃗`: hold `r⃗_{H}` constant. Better: a fitted terminal fulfillment estimate. This
matters far more for iLQR than for MPPI because the horizon is short and `p<0` weights low
fulfillments heavily — an objective that *would* be satisfied just past the horizon reads
as unfulfilled and distorts the whole priority ordering. **Apply the same change to the
sampling scorers**, or the two arms stop sharing an objective (§11.1).

**3.4 — Floor the fulfillments. `[PRIORITY — highest single item in this brief]`**

Per §4.5, `∂M_p/∂x_i = w_i x_i^{p−1} M_p^{1−p}` diverges as `x_i → 0` for any `p < 1`.
Clamp atoms to `[ε, 1]`, `ε ≈ 1e-3`, before the power mean.

**Why this outranks everything else.** For `p = 1` the derivative is just `w_i` — bounded,
no singularity, **no floor needed at all**. The floor exists *only* because `p < 0`
diverges. So introducing it changes the FPL objective and leaves the linear objective
untouched. That is a direct, mechanical confound to the exact comparison this repo makes,
and it is currently invisible because sampling never evaluates a derivative. It is also
almost certainly the ε-plateau already logged for walker/quadruped/cube in
`docs/mppi_math.md §8`.

`soft_ramp`'s exponential tail keeps `x > 0` but exponentially small, which is not enough —
exponentially small still diverges under `x^{p−1}`.

**Protocol, in order:**
1. Apply the identical floor in **both** the numpy and JAX paths and in **both** arms.
   A floor applied only where it is numerically needed is the confound.
2. Re-run `sampler race`, `zero-tuning`, `portability`, and `mismatch robustness`.
3. If the numbers hold, the floor is a free correctness fix and you have removed a
   criticism. If they move, **that is a finding about the published results** and gets
   reported. Do not tune ε until the numbers come back.
4. Sweep ε over `{1e-2, 1e-3, 1e-4}` and report FPL-vs-linear sensitivity to it. If the
   FPL advantage depends on ε, say so — it is the same class of objection as λ/`p`
   confounding and better to own than to have found.

**Acceptance for WO-3 as a whole:** parity test at 1e-6 between the augmented-state
terminal formulation and the existing numpy scorer, for all five modes, on hopper and
walker. Extend `tests/test_jax_costs.py`. Plus: headline studies re-run and their delta
recorded in `FPL_FINDINGS.md`, whatever it is.

---

### WO-2 — diffmjx integration

Repo: `martius-lab/diffmjx` (ICLR 2026, Paulus et al., *Differentiable Simulation of Hard
Contacts with Soft Gradients*). Three independent mechanisms.

**2.0 — Check feasibility before writing any code.** In this order:
1. The README lists **SSH access to private component repositories** as a prerequisite.
   Confirm `martius-lab/mujoco`, `martius-lab/mjx_diffrax`, and `a-paulus/softjax` are all
   clonable. If not, stop and request access — everything below is blocked.
2. It ships a **fork of mujoco-mjx**. §7 pins `mujoco-mjx==3.5.0` exactly *because* it
   pip-depends on same-version `mujoco`, and the CPU sim is the physics of record. Install
   the fork in a **separate venv** first and confirm it does not drag the core `mujoco`
   off 3.5.0. If it does, the `[diffmjx]` extra must be mutually exclusive with `[mjx]`
   and that needs to be enforced in `pyproject.toml`, not by convention.
3. Check the existing `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` workaround still
   suppresses the backward-pass crash on the fork.

**2.1 — CFD (Contact Force from a Distance).** Contact constraints apply miniscule forces
even when no contact is active, guiding the optimizer toward contact configurations.
**Straight-through estimation** means CFD forces affect *only the gradient*; the forward
simulation stays untouched hard contact. This is the direct fix for the zero-gradient
problem in §9.1 and it does **not** cost you physical fidelity — which matters because
your score comes from the forward rollout.

**2.2 — The CFD catch, and the one implementation detail that must not be missed.**
Straight-through means the gradient is a deliberately **biased surrogate**, not the
derivative of the function being evaluated.

- *Good news:* iLQR tolerates this far better than BPTT, because the line search only
  accepts a step if the **true** cost decreased. The DP structure gives a correctness check
  against real dynamics every iteration. Expect more `Q_uu` indefiniteness and more
  regularization — which is exactly what WO-1.4's instrumentation measures.
- *You lose the stationarity claim.* iLQR with biased Jacobians converges to wherever the
  line search stops improving, not to a stationary point. Say so in the paper.
- **Critical:** `K_t = −Q_uu⁻¹ Q_ux` computed from CFD-biased Jacobians are gains for a
  robot that feels contacts from a distance — which yours does not. **Run two modes:
  CFD on for the search direction, CFD off (or `mjd_transitionFD`) for one final backward
  pass on the converged trajectory to synthesize the gains you actually deploy.** One extra
  backward pass. Validate via the mismatch protocol in WO-5 T5.1; the failure is obvious
  when it happens.

**2.3 — Adaptive integration (`mjx_diffrax`, Tsit5/Dopri5).** Fixed-stepsize integrators
cause gradient oscillation from discretization error. Treat this as a **morphology-invariance**
mechanism, not just a gradient-quality one — it is the physics-side counterpart to WO-4.2's
time normalization, and without it a `dt` tuned for one scale shows up as
morphology-correlated noise in the WO-4 sweep.

**2.4 — `scan_loop` and softjax collision.** `scan_loop` replaces `lax.while_loop` in the
constraint solver, which makes the solver iteration count an explicit config parameter —
wire it to WO-0.3's finding. Softjax smooths `plane-{cylinder,cube}` and `cube-cube`;
`plane-{sphere,ellipsoid,capsule}` and `sphere-sphere` are already smooth in MJX.
**Mesh-mesh collision is unsupported** — keep all morphology geometry on primitives
(WO-4.1), which you want anyway for a parametric family.

**2.5 — Compilation time is a first-class metric, not a footnote.** The README warns that
MJX gradient compilation is slow and gets *slower* with adaptive integration, and ships a
dedicated timing experiment (`04_time-toss`) — the authors consider it a primary caveat.
For a morphology loop this is potentially the binding constraint. Log compile time and run
time separately in the provenance record from day one.

**Acceptance:** hopper iLQR with diffmjx beats hopper iLQR with stock MJX on success rate
at matched compute; the gradient-sweep plot (WO-5 T1.1) shows nonzero pre-contact
sensitivity; core `mujoco` still at 3.5.0 and all 78 existing tests green.

---

### WO-4 — Parametric morphology family

**4.1 — Fixed topology, varying parameters. This is the single most important design
decision in the brief.** MJX recompiles on *structural* change (body count, geom count,
contact pairs) but not on parameter change. The current 11 tasks are 11 topologies, so a
sweep over them means a gradient recompile per design — with WO-2.5's warning that is
likely fatal to the loop.

Build a **single parametric MJCF** (start from hopper) exposing link lengths, masses,
inertias, and actuator gears as parameters over a fixed skeleton. Then compile once and
`vmap` the whole population. If varying topology is genuinely required later, bucket by
topology and amortize compilation across every design in a bucket.

Geometry stays on primitives (capsules, spheres, boxes, planes) per WO-2.4.

**4.2 — Normalizations, so no per-morphology tuning is required.** Every one of these
turns a tuned quantity into a derived one:
- **Time:** scale `plan_horizon` and `dt` by `τ = √(L/g)` with `L` a characteristic length.
  Horizon = a fixed number of `τ`, not a fixed number of seconds. Highest-leverage item here.
- **Actions:** sample and optimize in normalized torque space `[−1,1]^m`, scale by each
  actuator's limit inside the rollout.
- **Cost:** FPL already handles this — fulfillments are bounded in `[0,1]` and comparable
  across morphologies by construction. This is the repo's existing thesis and WO-4 is its
  strongest test. Express the underlying physical quantities dimensionlessly anyway
  (Froude number `v/√(gL)` for speed, height as a fraction of leg length, cost of transport
  for energy) so the *atoms* are morphology-fair, not just their aggregate.
- **Selection rule:** prefer CEM's elite fraction over MPPI's `λ` where possible —
  "top 10%" is scale-free by construction. `fpl_tempered`'s ESS-targeting bisection already
  does the equivalent for MPPI; make it the default for morphology sweeps.

**4.3 — Enforce a strictly fixed compute budget per candidate.** Any "run until converged"
rule reintroduces morphology-dependent bias through the back door.

**Acceptance:** 100+ morphologies sampled over ≥2 orders of magnitude in leg length, mass
ratio, and actuator strength; **zero retuning** — one FPL spec, one normalized horizon,
one compliance schedule; a single compile for the whole population.

---

## 11. Invariants — do not break these

**11.1 — The fairness protocol, stated precisely.** The controlled variable is **scoring**.
FPL and linear arms must differ in the objective and in nothing else — same optimizer, same
parameterization, same budget, same seeds, same floor, same terminal treatment (§4.3, §9.0).

What this does **not** require: that the gradient arm and the sampling arm search the same
set. They don't, and that is fine (WO-1.2) — the 2×2 is `{FPL, linear} × {MPPI, iLQR}` and
each cell's comparison is within-controller. Do not spend effort making iLQR search knots.

What it **does** require: any change to the objective — WO-3.3's terminal term, WO-3.4's
floor, any new atom — lands in the numpy path, the JAX path, and both `p` settings
**simultaneously**, with the 1e-6 parity test extended to cover it. A change that lands
asymmetrically invalidates every published number in `FPL_FINDINGS.md`, and asymmetry is
easy to introduce by accident because `p<0` is the only setting that is ever numerically
forced (§4.5).

**11.2 — The closed loop is always stepped by the CPU backend.** No exceptions for iLQR or
diffmjx. The physics of record does not change.

**11.3 — Version pins.** `mujoco` stays at 3.5.0 (3.12 changes fixed-seed hopper
trajectories). `mujoco-mjx==3.5.0` exact. WO-2.0 exists to protect this.

**11.4 — No JAX in the default path.** Lazy-load `diffmjx` the same way `MJXBackend` is
lazy-loaded. Config validation must work on a machine with no extras installed.

**11.5 — Unrepresentable configs raise at load time.** New `(proposal, update)` pairs go
in `_DISPATCH` or they error. Do not add silent fallbacks.

**11.6 — Existing results stay reproducible.** Re-run `sampler race`, `zero-tuning`,
`portability`, and `mismatch robustness` after WO-3 lands. If numbers move, report it.

---

## 12. Experiment protocol (WO-5)

Tiered. **Do not advance a tier until the one below passes** — a T4 failure that is really
a T0 bug costs weeks.

**Read the tiers against §9.0 before allocating effort.** The claim is FPL vs. weighted-sum,
so most of what follows is infrastructure validation, not results:

| tier | what it establishes | where it goes |
|---|---|---|
| **T0, T1** | the derivatives are correct | **gate** — a wrong Jacobian invalidates the 2×2 silently, so these are mandatory but invisible in the paper |
| **T2.1, T2.2, T2.3, T2.4** | the gradient arm is implemented competently | **appendix.** These are controller engineering. Necessary to justify the implementation, not evidence for FPL |
| **T3.1** | MPPI → iLQR beats either alone | **appendix**, same reason |
| **the 2×2** | `{FPL, linear} × {MPPI, iLQR}` — FPL survives an optimizer with **no λ** | **primary result.** This is what §9.0 built the gradient arm for. It is not currently a numbered tier; add it as T6 and treat it as the paper's core table |
| **T4** | zero-retuning across a morphology family | **headline.** The strongest existing claim, now under a second optimizer |
| **T5.1, T5.2** | feedback gains, stabilizability as a design metric | **secondary** — real, but it is a gradient-arm capability claim, not an FPL claim |

Concretely: if time runs out, ship T0/T1 + the 2×2 + T4. Everything else is supporting.

### 12.1 T0–T1: is the derivative even right
- **T0.1–T0.4** — WO-0 above.
- **T1.1 — the headline plot.** Point mass or single finger approaching a plane,
  parameterized by gap `φ`. Plot `∂(next state)/∂u` vs `φ`, overlaying: **stock MJX**
  (flat zero then a jump — the failure mode), FD at several `ε`, **CFD at several
  strengths**, and dense-sampled ground truth. Deliverable claim: *"CFD strength c gives
  anticipation range X mm"* — a designed quantity, not a tuning artifact. This is the
  argument for autodiff over FD (FD's range is set by `ε`, which is a numerical artifact
  you cannot defend).
- **T1.2 — estimator quality vs stiffness.** Bias and variance of the first-order estimator
  vs a zeroth-order one, as a function of contact stiffness. Report the crossover. A
  reviewer will ask; showing it unprompted makes the first-order claims credible instead
  of naive, and it tells you where in the morphology family autodiff stops being trusted.

### 12.2 T2–T3: does the optimizer work
- **T2.1 — derivative-source ablation.** iLQR + diffmjx vs iLQR + FD vs iLQR + stock MJX
  autodiff. Report cost-vs-iteration, wall-clock, success over ≥20 inits, **line-search
  acceptance rate**, and the **λ trace**. The last two are the informative ones — they say
  *how* it failed.
- **T2.2 — iLQR vs BPTT at matched derivatives.** Success rate vs horizon, 10 → 500 steps.
  `GradientMPC` is the BPTT arm. Expect iLQR roughly flat and BPTT collapsing past some
  horizon. **This is probably the single best figure in the paper** — it isolates "the DP
  structure matters" from "the derivative source matters" and justifies the architecture in
  one image. Run it even though CFD helps BPTT: CFD fixes the *zero-gradient* problem, not
  the *conditioning* problem, so BPTT should still collapse.
- **T2.3 — diffmjx leave-one-out.** CFD / softjax collision / adaptive integration are
  three independent switches. Report iLQR success with each disabled. Tells readers which
  piece carries the result.
- **T2.4 — straight-through bias.** Quantify CFD-gradient deviation from the true gradient;
  show iLQR converges anyway. Doubles as justification for WO-2.2's two-mode gain synthesis.
- **T3.1 — the honest negative.** A task requiring a contact absent from initialization.
  Show iLQR-from-scratch fails, MPPI succeeds, **MPPI → iLQR succeeds and beats MPPI alone
  on final cost.** Include it. The failure is the argument for the architecture.

### 12.3 T4: morphology invariance — the actual claim
- **T4.1** success rate vs morphology parameter. Should be smooth; cliffs are suspicious.
- **T4.2 — solver-failure correlation. The experiment that makes or breaks co-design
  credibility, and almost nobody runs it.** Regress line-search rejection rate, λ magnitude,
  and iteration count against the morphology parameters. If solver difficulty correlates
  with design parameters, the ranking is measuring the optimizer, not the designs. Report
  the coefficients even when inconvenient. WO-1.4's logging exists for this.
- **T4.3 — compute ceiling.** Rank 20 morphologies at 1× and 100× budget; report Spearman ρ.
  Low ρ means the screen is measuring compute noise.
- **T4.4 — seed variance vs between-morphology variance.** Report the ratio. Comparable
  means no signal.
- **T4.5 — gold-standard correlation.** On a small subset, run something expensive and
  trusted (huge-budget CMA-ES, or CITO with a full relaxation homotopy) and check rank
  correlation. **Without this, T4.1–T4.4 only show the loop is self-consistent, not that
  it is right.**

### 12.4 T5: what sampling structurally cannot give
- **T5.1 — feedback gains under mismatch.** Use the existing `apply_perturbation()`
  machinery. Plan on nominal, execute on a perturbed model. Show `K_t` recovers where
  open-loop playback does not. **This is also the validation for WO-2.2** — if the gains
  were synthesized with CFD on, this test fails.
- **T5.2 — stabilizability as a design metric.** Sweep max recoverable perturbation across
  the morphology family. Check rank correlation against open-loop cost. If they rank the
  same, `K_t` added nothing and say so. **If they diverge, you have a design criterion
  sampling MPC cannot produce** — the strongest possible justification for the gradient arm
  existing at all.

---

## 13. Open gaps carried forward

- JAX cost mirrors cover 4 of 11 tasks. WO-4 reduces the pressure here (one parametric
  family instead of 11 topologies) but `g1_reach`, `cube`, and the quadruped variants stay
  CPU-only unless someone writes the mirrors.
- `fpl_layered` / `hybrid` / conjunction-split collapse remain CPU-only by scope. WO-3.2
  gives the recipe for `fpl_layered`; `hybrid`'s `−log` barrier needs its own analysis
  because the barrier interacts with WO-3.4's floor.
- The two bugs-in-waiting in [docs/mppi_math.md §8](docs/mppi_math.md) are still live.
  **WO-3.3 probably fixes the first one** (`fpl_discounted` optimism) — confirm and close
  it explicitly rather than letting it drift. WO-3.4 addresses the ε-plateau on
  walker/quadruped/cube; that one needs the existing studies re-run per §11.6.
- λ / `p` confounding (§4.5) — **this brief closes it, and that is the point of WO-1.**
  The gradient arm has no temperature at all, so the 2×2 in §12 removes "you found a better
  λ" as an explanation structurally rather than by sweeping. When it lands, update
  `docs/mppi_math.md §8` to mark the entry resolved-by-construction rather than open, and
  say so explicitly in the paper — it is the strongest methodological upgrade here.
- Replacing it as the leading open objection: **ε sensitivity** (WO-3.4). The floor is
  needed only for `p<0`, so "FPL wins because of the floor" is the new version of the same
  criticism. The ε sweep in WO-3.4 step 4 exists to answer it; run it before publication,
  not after review.
