# Handoff — Phases 0–4 done; next is PHASE 5 of `NEXT_STEPS.md`

*Written 2026-09-17 for whoever picks this up next. Read [`NEXT_STEPS.md`](NEXT_STEPS.md)
§PHASE 5 (lines 231–278) and this file's "Traps and invariants" section before touching a
diffmjx repo. `FPL_FINDINGS.md` §S12 is the primary result you are building on.*

**Nothing is committed.** Phases 0, 1, 2, 3, 4 are all in the working tree, awaiting review
and milestone tags (0/1/2/3/4). Do not commit or tag Phase 5 work without that landing first.

**Milestone 4 is the paper.** Phase 5 is a strict upgrade over an already-publishable
result. On any hard failure it is legitimate to drop Phase 5 and start drafting.

---

## Where we are

| phase | items | gates | verdict |
|---|---|---|---|
| Phase 0 — close the asymmetric confounds | S1–S4 | G1 PASS, G2 PASS, G3 FAIL (keep `terminal_value=False`) | done |
| Phase 1 — earn trust in the derivatives | S5–S8 | G4 PASS, G5 mitigated, G6 PASS | done |
| Phase 2 — FPL cost DDP-compatible | S9 | G7 PASS | done |
| Phase 3 — build iLQR | S10–S11 | G8 PASS, G9 2/3 (speed FAIL 9.8×), G10 FLAG | done |
| Phase 4 — the primary result (**MILESTONE 4**) | S12 | G11 resolved by construction | done |
| **Phase 5 — diffmjx** | **S13–S15** | **G12, G13, G14** | **← start here (optional)** |

Test suite: **315 passed, 5 skipped, 0 failed** (was 292/4 at the Phase 2 handoff).

---

## Phase 3 results (recap; details in `FPL_FINDINGS.md`)

Built `analytic_mppi/controllers/ilqr.py`: iLQR on the augmented (physics + accumulator)
state, registered at `("gradient","ilqg") → "ilqr"`. Backward pass is Gauss-Newton
**through physics only** — exact stage-cost curvature via cost-channel Hessians chained
through first-order step Jacobians, dropping only `F_xx`. Linearizing the augmented
dynamics without expanding the stage cost would zero out `V_xx` in `normal` mode; G8 fails
by construction.

- **GATE G8 PASS** — iLQR ↔ analytic Riccati LQR ≤ 1e-4 on pendulum + cart-pole. Full G8
  suite `tests/test_ilqr.py` (19 tests) and 3 dispatch tests in `tests/test_config.py`.
- **GATE G9** — 2/3 PASS on hopper (accept 0.94, λ trace bounded), **speed FAIL** at
  median 9.8× vs GradientMPC (criterion 20×). Recorded and continued per the runbook.
  Not a coding bug: `plan` time is steady across states (no per-step recompile) and
  linearization is one batched `vmap(jax.jacfwd)`. iLQR wins big where the objective is
  genuinely nonconvex (state 0 → 88.5×, state 60 → 19.2×); on mild states GradientMPC's
  first Adam step already reaches iLQR's final cost.
- **GATE G10 FLAG** — the knot basis is a headwind for FPL, not linear. Projection residual
  0.65 vs 0.51, `J_post − J_pre` 0.506 vs 0.070 — FPL loses ~7× more to the knot spline
  than linear. This is the "thesis is being understated" case: p<0 wants sharp switching
  near contact, 4 linear knots over 0.6 s cannot represent it.

S11 = `WarmStartedILQR`: composes any `SamplingController` with an `ILQRMPC` (mirrors
`cost_gd.py`). The sampler explores, iLQR polishes its mean, both plans are re-aligned and
shifted identically each MPC step.

## Phase 4 result — the paper

**S12 / GATE G11 — resolved by construction.** The 2×2 {FPL, linear} × {MPPI, iLQR},
hopper + walker, 10 seeds/cell, matched dynamics budget (256 rollout-equivalents/step in
both columns), atom floor 1e-3, published spec. The iLQR column uses no temperature
anywhere: warm start is `PredictiveSampling` (argmax over Gaussian cloud, mean included),
iLQR itself has no `-log(u)` bridge.

|  | mppi/fpl | mppi/linear | ilqr/fpl | ilqr/linear |
|---|---|---|---|---|
| hopper | 0.64 ± 0.29 (7/10) | 0.32 ± 0.15 (8/10) | **1.20 ± 0.52** (7/10) | 1.18 ± 0.51 (7/10) |
| walker | 1.17 ± 0.04 (10/10) | 0.47 ± 0.16 (10/10) | **1.57 ± 0.07** (10/10) | 0.86 ± 0.10 (10/10) |

FPL ≥ linear in every cell. Walker CIs are disjoint (1.50 vs 0.96 at the boundaries).
Combined with G10 (knot handicap on FPL), the "you found a better temperature" objection
is structurally eliminated on both tasks. `docs/mppi_math.md §6` addendum records the
resolution.

Second-order but paper-worthy: iLQR **dominates MPPI on every cell** (1.3× – 3.7× in
`prod`). The optimizer matters as much as the scalarization.

---

## Traps and invariants (do not re-learn these)

**Read `docs/jacobian_audit.md` before touching any autodiff-through-MJX code.** Phase 1's
whole point was to make these traps explicit.

1. **`jacfwd`, never `jacrev`, on `mjx.step`.** Reverse mode cannot differentiate the
   solver's `lax.while_loop`; it only surfaces on models big enough to reach it (pendulum
   works, hopper does not). `mjx_manifold.transition_jacobians` uses `jacfwd`; keep it.
2. **`jax.jacfwd` return structure is `[output_leaf][argnum]`, NOT `[argnum][output_leaf]`.**
   Cost me a session (pendulum's `nsensordata=0` surfaced this as a shape mismatch; on
   hopper it would have been a silent wrong Jacobian). The correct destructuring pattern:
   `((dxdq, dxdv, dxdu), (dsdq, dsdv, dsdu)) = jax.jacfwd(f, argnums=(0,1,2))(zq, zq, zu)`
   Pinned with a comment in `ilqr.py:_lin_one`.
3. **Linearize only via `mjx_manifold.linearization_model(model)`.** It disables
   warm-starting; `transition_jacobians` raises on a warm-start-enabled model or a CG
   solver. Do not bypass — a settled MjData takes the tangent-dead warm-start branch
   (relFro 48 → 1.55e-3). `jax.lax.stop_gradient` on `qacc_warmstart` does *not* help;
   it's already constant, that IS the bug.
4. **G5 is mitigated, not passed.** Multi-contact Jacobians carry a ~1.6e-3 residual, flat
   in iteration count. Fine for search directions validated by a line search on the true
   cost. **Not fine for deployed feedback gains** — Phase 7's `K_t` must be re-synthesized
   with one final backward pass on `mjd_transitionFD` (plumbing in `cost_gd.py`). Phase 5's
   S14 note on "two-mode gain synthesis" is the same lesson.
5. **f64 for derivative tests only**, via per-test `jax.experimental.enable_x64()` (pattern
   in `tests/test_jacobians.py`, `test_ilqr.py`). Production stays f32. Flipping the global
   x64 flag measurably shifts MJX rollouts.
6. **Quaternion tangent maps: use `mjx_manifold.integrate_pos` / `differentiate_pos`.**
   MJX's own `quat_integrate` / `quat_sub` have zero derivative exactly at the base point
   (guarded normalize). This is a linearization map, not an integration map.
7. **XLA Triton crashes on differentiated MJX solver graphs**
   (`gemm_fusion_autotuner.cc: Non-OK-status`, core dump — jax 0.6.2 / RTX 4070).
   `MJXBackend.__init__`, `dynamics/mjx_manifold.py`, and `tests/conftest.py` all pin
   `--xla_gpu_enable_triton_gemm=false` before any `import jax`. Whichever entry point
   initializes jax first decides the flag for the whole process, so all three must set it.
   **If Phase 5 introduces a fourth entry point** (a diffmjx-only module that imports jax
   before the others), it needs the same three-line guard.
8. **The closed loop is ALWAYS CPU-stepped** (invariant 11.2). MJX / diffmjx is
   planning-only. The physics of record has not moved since Phase 0.
9. **`mujoco == 3.5.0` exact.** Any diffmjx fork that upgrades core `mujoco` breaks
   comparability with every existing result — see G12's on-fail branch.
10. **`terminal_value=False` stays.** G3 in closed loop cost survival −36/−46% on the two
    hopper studies; the controller raises on the published-config combination. Do not "fix"
    that raise, it is invariant 11.5.

---

## Phase 3 propagation lesson worth institutionalizing

Phase 0 caught the `terminal_value` no-op only because of an explicit propagation check.
Phase 2 caught the `terminal_value` scope issue only because of an explicit propagation
check. Phase 3 caught a silent `jacfwd` index swap only on the pendulum-shape-mismatch —
on hopper it would have been a diffuse Jacobian error that G8 could have absorbed within
tolerance. **Every new autodiff path in Phase 5 gets a propagation check**:

- Assert the gradient the diffmjx path returns actually differs from stock MJX's
  (see `test_g5_disabling_warmstart_is_the_fix` for the pattern — measure a *ratio*
  against a known-broken baseline, so a no-op fix fails visibly).
- Assert the CFD/softjax/adaptive flags actually reach the compiled solver
  (`docs/jacobian_audit.md`'s pattern: instrument a synthetic scenario where the flag
  changes the answer by a known amount).

---

## PHASE 5 — S13, S14, S15, gates G12, G13, G14

Full runbook in [`NEXT_STEPS.md`](NEXT_STEPS.md) lines 231–278. The short version:

### S13 (feasibility, before any code) → G12

1. Clone `martius-lab/mujoco`, `martius-lab/mjx_diffrax`, `a-paulus/softjax` — README lists
   SSH access to private repos as a prerequisite. If any is unclonable, stop.
2. Install the mujoco-mjx fork **in a separate venv**, not `a-mppi/`. Confirm it does not
   drag core `mujoco` off 3.5.0 (`pip show mujoco` in both venvs).
3. Reproduce the Triton-gemm guard on the fork: import jax with the flag, compile one
   differentiated `mjx.step`, confirm no core dump.

**G12 PASS:** add a `[diffmjx]` extra to `pyproject.toml`, mutually exclusive with `[mjx]`
enforced at pip-resolve time (not by convention — a comment is not a check). Lazy-load
diffmjx like `dynamics/mjx_manifold.py` lazy-imports jax, so machines without the extra
still validate configs.

**G12 FAIL:** fork upgrades core `mujoco`. **Do not install it in `a-mppi/`.** Treat
diffmjx as an isolated study (a separate venv, a separate checkpoint dir, results reported
alongside 3.5.0 baseline explicitly). Or drop Phase 5 — Milestone 4 stands regardless.
Kill criterion, see NEXT_STEPS "Kill criteria" §.

### S14 (CFD) → G13

**CFD = Contact Force from a Distance.** Forward sim stays hard contact (straight-through
estimator); only the gradient sees the reaching-out forces. Two-mode gain synthesis
mandatory: CFD **on** for the search direction, CFD **off** (or `mjd_transitionFD`) for
the deployed `K_t`.

**G13** = the gradient sweep plot (point mass approaching a plane, parameterized by gap
φ). `∂(next state)/∂u` vs φ: stock MJX (flat zero pre-contact, jump at contact), FD at
several ε, CFD at several strengths, dense-sampled ground truth. Pass = state
*"CFD strength c gives anticipation range X mm"* as a designed quantity. Fail = the CFD
gradient is zero pre-contact too — CFD is not actually enabled.

**This is the propagation check for CFD** — do it before running any hopper study. The
`mjx_manifold` audit (`docs/jacobian_audit.md`) is the format.

### S15 (adaptive + softjax) → G14

`mjx_diffrax` for adaptive integration (Tsit5/Dopri5); `col_soft_enable` for softjax.
Geometry stays on primitives — mesh-mesh is unsupported. **Log compile time and run time
separately** in the provenance record.

**G14** = re-run G9's three criteria (`verification/s11_ilqr_g9.py`) with diffmjx on;
leave-one-out over {CFD, softjax, adaptive} for the ablation table. PASS = at least one
axis improves criterion (1) — remember G9's Phase 3 result was 9.8× median vs a 20× bar;
diffmjx is the runbook's expected way to close that gap. FAIL = report the negative;
"informative contact gradients did not change the FPL conclusion" is a legitimate
appendix result. Phase 4 stands regardless.

---

## What to keep, adapt, and rerun

- **`verification/s11_ilqr_g9.py` is the G14 driver template.** It measures exactly the
  three G9 criteria in the exactly the checkpointed form G14 needs; run it with the
  diffmjx-configured MJX backend and the leave-one-out flag matrix. Do NOT modify the
  Phase 3 file in place — copy it to `verification/s15_diffmjx_g14.py`.
- **`ILQRMPC` needs no changes for Phase 5.** All the diffmjx machinery lives at the
  backend / model layer; `make_ilqr_solver` consumes `mjx.step` through
  `mjx_manifold.linearization_model` and does not know about CFD/softjax/adaptive.
  If a new flag needs a controller-level knob, add it as a kwarg on `ILQRMPC.__init__`
  passed through to `make_ilqr_solver`; do not fork the module.
- **`WarmStartedILQR` stays as-is.** Phase 5 is optimizer-side; the sampler is unchanged.
- **`docs/jacobian_audit.md`** is the format for CFD/softjax/adaptive audits. Add sections
  §S14, §S15 rather than a new file — one derivative-audit doc is the invariant.
- **`FPL_FINDINGS.md`** is where G13, G14 verdicts go. The Phase 3 / Phase 4 entries
  established the pattern; keep it.

## What NOT to do

1. **Do not touch Phase 4 numbers.** S12 is committed to the paper. If diffmjx changes a
   number reported in the S12 table, it is a *new experiment* under diffmjx, not a
   re-run of S12. Phase 4 uses stock MJX 3.5.0; that is the record.
2. **Do not deploy CFD-biased gains on the real robot.** The two-mode synthesis rule is
   not a "when we get around to it" — Phase 7 / S19's stabilizability sweep will fail
   visibly if it is skipped, and it will fail *silently* on a physical system.
3. **Do not use diffmjx to "fix" G9-1.** The runbook allows it; the paper is written on
   Phase 4 numbers alone. Do not couple the Phase 4 story to Phase 5 success.
4. **Do not commit Phase 5 into the same PR/tags as Phase 0–4.** Milestones 0–4 want a
   clean review pass first.

---

## Files changed in this session (all uncommitted; nothing here has been committed since
before Phase 0)

**New**
```
analytic_mppi/controllers/ilqr.py            S10 core + ILQRMPC + WarmStartedILQR
tests/test_ilqr.py                           G8 + hopper controller + wrapper (19 tests)
verification/s11_ilqr_g9.py                  G9/G10 driver
verification/s12_the_2x2.py                  S12 driver
verification/checkpoints/s11_ilqr_g9/        G9/G10 results (JSONL)
verification/checkpoints/s12_the_2x2/        S12 results (JSONL)
scripts/live_demo.py                         MuJoCo viewer for mppi | ilqr | warm_ilqr
```

**Modified**
```
analytic_mppi/controllers/accumulator.py     terminal split into fold_terminal + readout
                                             (behavior-preserving; G7 re-gates it)
analytic_mppi/controllers/__init__.py        register ILQRMPC
analytic_mppi/config.py                      ("gradient","ilqg") dispatch + backend check
tests/test_config.py                         3 dispatch tests
tests/test_mppi_smoke.py                     skip ilqr like gradient_mpc
docs/mppi_math.md                            §3 box-bounds note; §6 G11 resolution
FPL_FINDINGS.md                              S10/G8, S11/G9/G10, S12/G11 entries
```

---

## Open items NOT in Phase 5 scope (do not silently absorb)

1. **Review + commits + tags** for all of Phases 0–4 — entire working tree is uncommitted;
   milestone tags 0/1/2/3/4 all pending user's review. Do not start Phase 5 commits
   without milestone tags landing first.
2. **`mismatch_robustness_sweep.py` never ran floor-on** — needs `run_study`-level wiring;
   4th headline study.
3. **Walker, quadruped and cube were never re-measured post-floor.** G1/G2 are hopper-only.
   G10's knot handicap finding also has not been measured beyond hopper.
4. **G3's A→B asymmetry** (discount-weighting moved the linear arm, not FPL) has no
   mechanism. Do not bank the gap widening from that.
5. **Quadruped `<hfield>` fix** landed in this session; quadruped is now runnable but has
   not been used since. Any Phase 5 CFD study benefits from a contact-heavy task and
   quadruped is the natural next environment beyond hopper.
6. **G9-1 speed failure is unresolved.** Phase 5's premise is that diffmjx closes it.
   Track this measurement — if G14's ablation shows diffmjx does NOT help speed, that is
   the informative negative and should be reported.

## Two standing rules

**Never skip a gate.** Every expensive failure in this project is silent, and gates are
where they surface.

**Verify every new flag / objective / gradient path actually reaches the controller
before spending compute.** The `terminal_value` no-op, the MjData-history Jacobian bug,
and the `jacfwd` index swap were all caught by explicit propagation checks — and would
have been silent otherwise.
