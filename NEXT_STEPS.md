# NEXT_STEPS — execution runbook

*Companion to [REPO_STATE.md](REPO_STATE.md). That file is context and rationale; this file
is the ordered list of what to do. Read §9.0 of REPO_STATE once, then work from here.*

**One rule: never skip a gate.** Every `GATE` below has a pass criterion and an explicit
on-fail branch. A gate that fails means stop and take the branch — not "note it and carry
on." Most of the expensive failure modes in this project are silent, and the gates are the
only place they surface.

**Sizes:** `S` ≈ half a day, `M` ≈ 1–3 days, `L` ≈ 1–2 weeks.

---

## START HERE

> **S1.** Add an ε floor to the fulfillment atoms in the numpy scorer. One line. Then run
> the four headline studies and see whether the published numbers move.

That is the whole first task. It is small, it is not glamorous, and it can invalidate
results already written up — which is exactly why it goes first, before any new code
exists to confuse the diagnosis.

---

# PHASE 0 — Close the two asymmetric confounds

**Why first:** these perturb the FPL arm and not the linear arm, which is
indistinguishable from an FPL effect. They touch numbers already in `FPL_FINDINGS.md`.
Landing them now means the re-run happens once, against a stable baseline.

### S1 — ε floor, numpy path `[S]`
- **Do:** clamp every atom to `[ε, 1]` with `ε = 1e-3` immediately before the power mean in
  `tasks/base.py`. Applies to **both** `p<0` and `p=1` paths — a floor applied only where
  it is numerically needed *is* the confound.
- **Done when:** `pytest` green, floor is a named constant not a magic number, and it is
  exposed in `ExperimentConfig` so it lands in the provenance record.

### S2 — Re-run the headline studies `[M]`
- **Do:** `sampler race`, `zero-tuning`, `portability`, `mismatch robustness`. Same seeds,
  same configs, floor on vs. floor off.

### 🔶 GATE G1 — do the published results survive the floor?
- **Check:** delta in each headline metric, floor-on vs. floor-off.
- **PASS** (deltas within seed noise): free correctness fix, one criticism removed. Record
  the deltas in `FPL_FINDINGS.md` anyway and continue to S3.
- **FAIL** (any headline flips sign or loses significance): **stop feature work.** This is a
  finding about the existing results, not a bug to route around. Write it up, decide with
  your advisor whether the affected claim is retracted or re-scoped, *then* continue. Do not
  tune ε to recover the old numbers.

### S3 — ε sensitivity sweep `[S]`
- **Do:** sweep `ε ∈ {1e-2, 1e-3, 1e-4, 1e-6}` on hopper and walker. Report FPL-vs-linear
  gap as a function of ε.
- **Done when:** you can state in one sentence whether the FPL advantage depends on ε.

### 🔶 GATE G2 — is the FPL advantage ε-dependent?
- **PASS** (gap roughly flat over 2+ decades): quote it in the paper as robustness. This
  pre-empts the successor to the λ/`p` objection.
- **FAIL** (gap grows as ε shrinks): the advantage is partly a conditioning artifact. Not
  fatal, but it must be reported as a limitation and the mechanism explained. Add it to
  `docs/mppi_math.md §8`.

### S4 — Terminal value replaces renormalization `[M]`
- **Do:** replace `(1−γ)/(1−γ^H)·Σγᵗ` with `FV = (1−γ)Σ_{t<H} γᵗ r⃗_t + γ^H·v⃗(x_H)`.
  Start with the cheap `v⃗` = hold `r⃗_H` constant. Numpy path first.
- **Done when:** implemented behind a config flag so old behaviour is still reproducible.

### 🔶 GATE G3 — does the terminal fix close the logged bug?
- **Check:** the "`fpl_discounted` order-of-operations optimism" entry in
  `docs/mppi_math.md §8`. Construct the case that exhibits it; confirm it no longer does.
- **PASS:** mark the entry resolved with a pointer to the commit. Re-run S2's four studies
  once more with the terminal fix on.
- **FAIL:** the optimism has a second, independent cause. Do not proceed to Phase 1 —
  diagnose it now, because it will contaminate every iLQR result downstream.

> **✅ MILESTONE 0** — the objective is now confound-free and both arms score identically.
> Everything after this is additive. Tag the commit.

---

# PHASE 1 — Earn trust in the derivatives

**Why before any optimizer:** a wrong Jacobian produces plausible-looking results that are
wrong, and you will not detect it downstream. This phase produces no features.

### S5 — f64 test harness `[S]`
- **Do:** `jax_enable_x64` for Jacobian tests only. Production rollouts stay f32.
- **Note:** without this you will chase ~1e-3 phantom errors against `mjd_transitionFD`.

### S6 — Jacobian parity test `[M]`
- **Do:** `tests/test_jacobians.py`. Compare `jax.jacobian(mjx.step)` against
  `mjd_transitionFD` on: pendulum → double pendulum → **free-floating body** → 7-DoF arm.
  All smooth, no contact. Reuse the tangent-space `mj_integratePos` plumbing already in
  `cost_gd.py` rather than rewriting it.

### 🔶 GATE G4 — smooth-system Jacobian agreement
- **Pass criterion:** relative Frobenius error **≤ 1e-6** on all four systems.
- **On fail, in this order:** (1) confirm f64 is actually on; (2) if the error localizes to
  the rotational block of the free-floating body, it is `nq ≠ nv` tangent-space handling —
  fix there; (3) if it is diffuse, the integrator or the `W` basis differs between paths.
- **Do not proceed on a "close enough" 1e-4.** That is a real bug and it will grow.

### S7 — Solver unroll sensitivity `[S]`
- **Do:** sweep MJX solver `iterations` / `ls_iterations`, plot `‖A_ad − A_fd‖` against it.

### 🔶 GATE G5 — does the algorithmic derivative converge to the true one?
- **PASS** (monotone decrease, then flat): record the flattening iteration count and pin it
  as a config default. Continue.
- **FAIL** (drifts, oscillates, or plateaus high): MJX autodiff is giving you the derivative
  of a partially converged solve and it does not approach the true one. **The plan changes:**
  implicit differentiation is required, or fall back to `mjd_transitionFD` for the iLQR
  Jacobians. Decide before writing iLQR, not after.

### S8 — Contact-set change test `[S]`
- **Do:** two-sphere approach/separate. Confirm gradients stay correct as the active
  contact set changes — MJX pads the contact array and it is easy to differentiate padding.

### 🔶 GATE G6 — gradients survive contact-set changes
- **PASS:** continue.
- **FAIL:** everything about contact downstream is unreliable. Fix here.

> **✅ MILESTONE 1** — derivatives are trustworthy. Write `docs/jacobian_audit.md`. Tag.

---

# PHASE 2 — Make the FPL cost DDP-compatible

### S9 — Additive-accumulator decomposition `[M]`
- **Do:** implement the state augmentation from REPO_STATE WO-3.2. Every power mean is
  quasi-arithmetic (`g⁻¹(Σ g(x))`), so this is **exact**, not an approximation.

  | mode | accumulator | terminal |
  |---|---|---|
  | `time_p = p ≤ 0` | `z ← z + w_t x_t^p` | `−z^{1/p}` |
  | `p = 0` | `z ← z + w_t log x_t` | `−exp(z)` |
  | `fpl_discounted` | `z_j ← z_j + γᵗ c_j(t)` | `−M_p(z)` |
  | `fpl_layered` | one accumulator per group | nested |

- **Done when:** augmented dynamics are block-triangular (`∂x'/∂z = 0`) and that is asserted
  in a test, not just assumed.

### 🔶 GATE G7 — augmented formulation matches the numpy scorer
- **Pass criterion:** **1e-6** parity, all five `objective.mode` values, hopper and walker.
  Extend `tests/test_jax_costs.py`.
- **On fail:** the decomposition is wrong for that mode. `fpl_layered` is the likely
  offender (nested means). Do not special-case — rederive.

> **✅ MILESTONE 2** — every objective mode is expressible as (additive stage) + (terminal
> nonlinearity). iLQR is now unblocked. Tag.

---

# PHASE 3 — Build iLQR

### S10 — Core iLQR `[L]`
- **File:** `analytic_mppi/controllers/ilqr.py`, registered at `("gradient", "ilqg")`.
- **Do:** `lax.scan` backward pass; Levenberg `Q̄_uu = Q_uu + λI` with adaptive λ;
  forward pass on **true dynamics** with backtracking on α; Jacobians via
  `vmap(jax.jacobian(step_fn))` in one batched call.
- **Parameterization:** per-step control space, project to knots with `θ = W⁺u*`. Settled —
  see REPO_STATE WO-1.2. Do not spend time making iLQR search knots.
- **Bounds:** box-constrained iLQR (projected-Newton on the `Q_uu` sub-problem), **not**
  clip-after-update.
- **Instrumentation from the first commit** — these are also your Phase 6 measurements:
  line-search acceptance rate, final α, λ trace, iteration count, cost reduction per
  iteration, `Q_uu` condition number, and **per-arm** projection residual
  `‖(I − WW⁺)u*‖/‖u*‖` plus both `J(u*)` and `J(WW⁺u*)`.

### 🔶 GATE G8 — iLQR is correct on problems with known answers
- **Pass criterion:** pendulum and cart-pole converge to the analytic LQR solution,
  cost gap **≤ 1e-4**.
- **On fail:** it is almost always sign convention in the backward pass or the regularization
  schedule. Debug here, on a linear problem, not on hopper.

### S11 — Warm start from a sampler `[M]`
- **Do:** mirror the `cost_gd.py` wrapper pattern — accept any `SamplingController`, use its
  mean plan as the iLQR nominal. This is the MPPI → iLQR architecture.

### 🔶 GATE G9 — iLQR is usable on hopper
- **Pass criteria, all three:**
  1. **≥ 20×** faster than `GradientMPC` at matched final cost
  2. line-search acceptance rate **> 50%** with warm start
  3. λ trace does not diverge
- **On fail (2) or (3):** the linearization is disagreeing with true dynamics across a mode
  boundary. Expected — this is what Phase 5 fixes. **Record the numbers and continue**;
  they become the "before" half of the diffmjx ablation. Do not try to tune around it.
- **On fail (1):** profile before changing anything. Usual culprits are recompilation per
  MPC step or a non-batched Jacobian loop.

### 🔶 GATE G10 — is the knot basis handicapping FPL?
- **Check:** projection residual and `J_pre` vs `J_post`, **split by arm**.
- **PASS** (both arms lose similar, small amounts): note it and move on.
- **FLAG** (FPL loses materially more — plausible, since `p<0` favours sharp switching near
  contact and knots cannot represent a mid-interval switch): you are **understating** your
  own result. Report both numbers. Consider a knot-count ablation as a follow-up.

> **✅ MILESTONE 3** — a working λ-free optimizer. Tag.

---

# PHASE 4 — The primary result

**This is the paper.** It needs only stock MJX, hopper and walker, and Phases 0–3. Nothing
from diffmjx or the morphology family. Do it now, before the infrastructure grows.

### S12 — The 2×2 `[M]`
- **Do:** `{FPL, linear} × {MPPI, iLQR}`, matched budget, ≥10 seeds per cell,
  report distributions not means.
- **The point:** iLQR has **no λ** — no temperature, no softmax, no `S_k = −log(u_k)` bridge.
  If FPL wins in the iLQR column too, "you found a better temperature" is eliminated
  *structurally* rather than by sweeping around it. That is the strongest methodological
  upgrade available to this project.

### 🔶 GATE G11 — does FPL survive an optimizer with no temperature?
- **PASS:** you have the result. Update `docs/mppi_math.md §8` to mark λ/`p` confounding
  **resolved by construction**, and say so explicitly in the paper. Continue to Phase 5
  only if time permits.
- **FAIL — FPL loses in the iLQR column:** this is the single most informative outcome in
  the whole plan. It means the FPL advantage was an interaction with the sampling update
  rule, not a property of the objective. **Do not bury it.** Diagnose which (λ setting?
  proposal distribution? the `−log` bridge?), then reframe the contribution around the
  sampler interaction — that is still a real, publishable finding, just a different paper.
- **MIXED** (FPL wins on one task, not the other): report per-task, no aggregate claim.

> **✅ MILESTONE 4 — MINIMUM PUBLISHABLE RESULT.** If everything after this gets cut for
> time, you still have a paper. Tag, and write the draft now while it is fresh.

---

# PHASE 5 — diffmjx (upgrade, not prerequisite)

### S13 — Feasibility check, before any code `[S]`
1. Confirm `martius-lab/mujoco`, `martius-lab/mjx_diffrax`, `a-paulus/softjax` are all
   clonable — the README lists SSH access to private repos as a prerequisite.
2. Install the mujoco-mjx **fork** in a **separate venv**. Confirm it does not drag core
   `mujoco` off 3.5.0.
3. Confirm the `XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` workaround still suppresses
   the backward-pass crash on the fork.

### 🔶 GATE G12 — can diffmjx coexist with the pins?
- **PASS:** add a `[diffmjx]` extra, lazy-loaded, mutually exclusive with `[mjx]` enforced
  in `pyproject.toml` — not by convention.
- **FAIL** (fork upgrades core `mujoco`): **do not install it in the main venv.** 3.12
  changes fixed-seed hopper trajectories and silently breaks comparability with every
  existing result. Run diffmjx in an isolated env and treat it as a separate study, or drop
  it. Phase 4 is already done; this is optional.

### S14 — CFD integration `[M]`
- **Do:** enable Contact Force from a Distance. Forward sim stays hard contact
  (straight-through), only the gradient sees the reaching-out forces.
- **Mandatory two-mode gain synthesis:** CFD **on** for the search direction, CFD **off**
  (or `mjd_transitionFD`) for one final backward pass to synthesize the `K_t` you deploy.
  CFD-biased gains are gains for a robot that feels contact from a distance — yours does not.

### 🔶 GATE G13 — the gradient sweep plot
- **Do:** point mass approaching a plane, parameterized by gap `φ`. Plot `∂(next state)/∂u`
  vs `φ`: stock MJX (flat zero then a jump), FD at several `ε`, CFD at several strengths,
  dense-sampled ground truth.
- **Pass criterion:** you can state *"CFD strength c gives anticipation range X mm"* as a
  designed quantity. This is the whole argument for autodiff over FD.
- **On fail** (CFD gradient is zero pre-contact too): CFD is not actually enabled. Check
  the straight-through path.

### S15 — Adaptive integration + softjax `[M]`
- **Do:** `mjx_diffrax` (Tsit5/Dopri5) and `col_soft_enable`. Keep all geometry on
  primitives — mesh-mesh is unsupported.
- **Log compile time and run time separately** in the provenance record. The README warns
  gradient compilation is slow and gets slower with adaptive integration; for Phase 6 this
  is potentially the binding constraint.

### 🔶 GATE G14 — does diffmjx actually help?
- **Check:** re-run G9's three criteria with diffmjx on. Leave-one-out over
  {CFD, softjax, adaptive}.
- **PASS:** report the ablation table — it tells readers which piece carries the result.
- **FAIL** (no improvement, or compile time dominates): **report the negative.** "We
  integrated informative contact gradients and they did not change the FPL conclusion" is a
  legitimate and useful appendix result. Phase 4 stands regardless.

---

# PHASE 6 — Morphology family

### S16 — Parametric MJCF, fixed topology `[L]`
- **Do:** one parametric hopper exposing link lengths, masses, inertias, actuator gears over
  a **fixed skeleton**. Primitives only.
- **Why fixed topology:** MJX recompiles on structural change, not parameter change. Fixed
  topology means **one compile, then `vmap` the whole population.** With S15's compile-time
  warning, varying topology is likely fatal to the loop.

### 🔶 GATE G15 — does the population compile once?
- **Check:** compile time for N=1 vs N=100 morphologies.
- **PASS** (N=100 ≈ N=1 plus rollout time): continue.
- **FAIL** (recompile per morphology): something in the parameterization is structural.
  Find it — this gate decides whether Phase 6 is feasible at all.

### S17 — Normalizations `[M]`
- Time: `plan_horizon` and `dt` scaled by `τ = √(L/g)`; horizon = fixed number of `τ`.
- Actions: normalized torque space `[−1,1]^m`, scaled by actuator limits inside the rollout.
- Atoms: dimensionless physical quantities (Froude `v/√(gL)`, height as fraction of leg
  length, cost of transport).
- Selection: `fpl_tempered`'s ESS-targeting bisection as the MPPI default; CEM elite
  fraction where possible.
- **Strictly fixed compute budget per candidate.** No "run until converged."

### S18 — The sweep `[M]`
- 100+ morphologies, ≥2 orders of magnitude in leg length, mass ratio, actuator strength.
- **Zero retuning.** One FPL spec, one normalized horizon, one schedule.

### 🔶 GATE G16 — is the sweep measuring morphology or measuring the optimizer?
Four checks. **All four must pass**, and G16b is the one nobody runs.
- **a. Success vs. parameter** — smooth. Cliffs are suspicious.
- **b. Solver-failure correlation** — regress line-search rejection rate, λ magnitude, and
  iteration count against the morphology parameters. **If solver difficulty correlates with
  design parameters, the ranking is measuring the optimizer.** Report the coefficients even
  when inconvenient.
- **c. Compute ceiling** — rank 20 morphologies at 1× and 100× budget, report Spearman ρ.
  Low ρ = measuring compute noise.
- **d. Seed variance vs. between-morphology variance** — report the ratio. Comparable = no
  signal.
- **On any fail:** the sweep does not support a morphology claim yet. Fix the specific
  failure; do not report the ranking.

### 🔶 GATE G17 — is the cheap screen actually right?
- **Do:** on a small subset, run something expensive and trusted (huge-budget CMA-ES, or
  CITO with a full relaxation homotopy). Check rank correlation.
- **Why it matters:** without this, G16 only shows the loop is **self-consistent**, not that
  it is **correct**. This is the only experiment that validates the screen as a proxy.
- **On fail:** report the loop as a consistency result, not a design-search result.

---

# PHASE 7 — Secondary results (only if Phases 0–4 are written up)

### S19 — Feedback gains under mismatch `[M]`
Use the existing `apply_perturbation()`. Plan on nominal, execute on perturbed. Show `K_t`
recovers where open-loop playback does not. **This also validates S14's two-mode gain
synthesis** — if gains were built with CFD on, this fails visibly.

### S20 — Stabilizability as a design metric `[M]`
Sweep max recoverable perturbation across the morphology family. Rank-correlate against
open-loop cost. **If they rank the same, `K_t` added nothing — say so.** If they diverge,
you have a design criterion sampling MPC cannot produce.

---

## Kill criteria

Stop and re-plan rather than pushing through, if:

- **G1 fails hard** — headline results do not survive the floor. The claim needs re-scoping
  before more infrastructure is built on it.
- **G5 fails** — MJX autodiff does not converge to the true Jacobian. Switch the whole
  gradient arm to `mjd_transitionFD` (slower, but `cost_gd.py` already has the plumbing) or
  invest in implicit differentiation. Decide before S10.
- **G11 fails** — FPL loses without a temperature. Reframe the contribution around the
  sampler interaction. Do not keep adding phases hoping the effect reappears.
- **G12 fails** — diffmjx cannot coexist with the pins. Drop Phase 5. It is optional and
  Milestone 4 does not depend on it.
- **G15 fails** — per-morphology recompilation. Phase 6 is infeasible as designed; either
  reduce to a handful of hand-built morphologies or drop the co-design framing.

## Invariants — true at every step

1. The closed loop is **always** stepped by the CPU backend. No exceptions.
2. `mujoco == 3.5.0`, `mujoco-mjx == 3.5.0` exact.
3. No JAX in the default import path. Config validation works with zero extras installed.
4. Any objective change lands in numpy **and** JAX **and** both `p` settings in the same
   commit, with parity extended to cover it.
5. Unrepresentable configs raise at load time. No silent fallbacks.
6. Every run writes its `ExperimentConfig` to the provenance record.
