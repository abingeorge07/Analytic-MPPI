# Handoff — Phases 0 and 1 are done; next is PHASE 2 of `NEXT_STEPS.md`

*Written 2026-09-11 for whoever picks this up next. Read `NEXT_STEPS.md` §PHASE 2 and
`REPO_STATE.md` §9.0 first; this file is what changed and what to watch out for.*

**Nothing is committed.** Everything below is in the working tree, awaiting review.

---

## Where we are

| phase | items | gates | verdict |
|---|---|---|---|
| Phase 0 — close the asymmetric confounds | S1–S4 | G1, G2, G3 | done |
| Phase 1 — earn trust in the derivatives | S5–S8 | G4, G5, G6 | done |
| **Phase 2 — make the FPL cost DDP-compatible** | **S9** | **G7** | **← start here** |

Test suite: **225 passed**, 1 skipped, 8 failures — all 8 pre-existing and unrelated
(see "Known broken, not ours" below). Was 127 passed before this work.

---

## Phase 0 results

### S1 — the fulfillment-atom floor (WO-3.4)

`power_mean` had always clipped terms at `eps=1e-8` as a log/division guard. That clip was
never an objective decision and it *is* a confound: `∂M_p/∂x_i` diverges as an atom → 0 for
every `p < 1` but equals `w_i` at `p = 1`, so a floor is needed only by the FPL arm.

Now an explicit `objective.atom_floor`, applied **at the atom boundary** — not inside
`power_mean`, which receives per-term discount-*sums* under `fpl_discounted` and group
scores under `fpl_layered`, so clamping there would floor a different quantity per mode.
Applied identically in the numpy, JAX and cost-GD paths and to **both** `p` settings.

**Default stays `1e-8`** so existing configs and stored provenance records keep their
meaning; `1e-3` is opt-in per experiment.

Exposure was worse than `docs/mppi_math.md §8b` claimed: `atom_soft_floor` defaults to
`0.0`, so hopper's ramps are hard clips too and hopper is the **most** exposed robot
(velocity atom at the clip 95.7% of the time). Raising the floor moves FPL scores by mean
|ΔS| = 9.62 and the linear arm by 0.0004 — a ~24,000× asymmetry from an identical clamp.

### GATE G1 — **PASS**

Hopper, 16 seeds. The gated quantity is FPL minus the *best* linear weight:

| study | gap @ `1e-8` | gap @ `1e-3` | change |
|---|---|---|---|
| portability | +0.4097 (CI clears) | +0.3881 (still clears) | −5% |
| zero_tuning | +0.4131 (clears) | +0.3896 (CIs overlap) | −6% |

One flag across 33 per-config comparisons: `FPL·CMA` survival 0.70 → 0.90 (disjoint Wilson
CIs). That **qualifies the "nothing beats plain MPPI" sampler-race conclusion** — it was
measured on an ε-plateau where failed rollouts were indistinguishable, which is exactly the
landscape a covariance-fitting sampler cannot exploit. Re-measure before repeating it.

### S3 / GATE G2 — **PASS**

| ε | `1e-2` | `1e-3` | `1e-4` | `1e-6` | `1e-8` |
|---|---|---|---|---|---|
| portability | +0.4147 | +0.3881 | +0.4465 | +0.4876 | +0.4097 |
| zero_tuning | +0.3530 | +0.3896 | +0.3057 | +0.3430 | +0.4131 |

Non-monotone in both; spread sits inside a single measurement's CI. The decisive number is
direction: between ε=`1e-2` and ε=`1e-8` (floor-on vs effectively none) the gap moves
−0.005 and +0.060. *"FPL only wins because of the floor"* is answered with data.

### S4 / GATE G3 — **NEGATIVE. Keep `terminal_value=False`.**

Two separate findings:

1. **WO-3.3 does not reach the published objective.** The published spec is `fpl_cost` +
   `fpl_time_p=-2.0` + `fpl_time_discount=False`, which aggregates time as
   `power_mean(per_step, time_p, weights=None)` — an *unweighted* soft-min with no
   `(1-γ)/(1-γ^H)` anywhere. The thing WO-3.3 replaces is not present. The controller now
   **raises** on that inert combination rather than silently no-op (invariant 11.5).
2. **Where it does reach, it is harmful.** Three arms isolating the confounder:

   | arm | FPL prod | gap | FPL survival |
   |---|---|---|---|
   | A published | 0.777 | +0.410 | 0.81 |
   | B +discount-weighted soft-min | 0.778 | +0.541 | 0.88 |
   | C +WO-3.3 tail | 0.604 | +0.334 | **0.56** |

   B→C is the terminal-value question alone: survival −36% (portability), −46%
   (zero_tuning), where the FPL advantage collapses +0.427 → +0.049. At `H=31, γ=0.99` the
   tail weight is `γ³⁰ = 0.740` vs `≈0.010` per interior step — a 74× concentration on the
   final step that guts the weakest-link-over-time property. WO-3.3's aside *"better: a
   fitted terminal estimate"* is **not optional**.

Synthetic half: the tail does close the *truncation* cause of `§8a` (a rollout collapsing
at 40/50 drops 0.941 → 0.612, and `time_p=-1` becomes fall-time invariant). The
*order-of-operations* cause is Jensen and cannot be closed by any convex time-weighting —
`fpl_discounted ≥ fpl_cost` in 100% of 4000 random rollouts either way.

### Environment finding — accepted, affects all published numbers

**The August checkpoint campaign does not reproduce.** Same seed/config: cached
`vx = 1.0275`, today `0.6563`. Not the floor — pre-S1 code reproduces today's value exactly.
Cause is the pin: `mujoco>=3.2,<3.6` was added only in HEAD (`8a9957e`); the August runs
used the unbounded pin at 3.8.1 and the venv is now on 3.5.0 for mjx compatibility
(invariant 11.3's hazard, realized). **Decision taken: accept the 3.5.0 values as the
baseline of record.** Older absolute numbers elsewhere in `FPL_FINDINGS.md` predate the pin.

---

## Phase 1 results

### S5/S6 / GATE G4 — **PASS**, ≥3 orders of margin

`jax.jacfwd(mjx.step)` vs `mjd_transitionFD`, velocity-space layout, f64:

| system | nq | nv | nu | relFro `A` |
|---|---|---|---|---|
| pendulum | 1 | 1 | 1 | 5.59e-12 |
| double pendulum | 2 | 2 | 2 | 1.35e-11 |
| free body | 7 | 6 | 0 | 2.97e-11 |
| 7-dof arm | 7 | 7 | 7 | 1.61e-09 |

Two things that cost real time:

* **MJX's own quaternion maps are non-differentiable where a Jacobian needs them.**
  `quat_integrate`/`quat_sub` route through `normalize_with_norm`, evaluated at exactly
  zero argument, where the guarded division has zero derivative. Fixed in
  `analytic_mppi/dynamics/mjx_manifold.py` with first-order-exact local maps. These are
  **linearization maps, not integration maps.**
* **Use `jacfwd`, not `jax.jacobian`.** `jax.jacobian` is `jacrev`, which cannot
  differentiate the solver's `lax.while_loop` — and this only surfaces on models large
  enough to reach that path. WO-1's brief says `vmap(jax.jacobian(step_fn))`; that is wrong
  as written.

### S7 / GATE G5 — **FAILED, root-caused, now MITIGATED. Read this before WO-1.**

MJX's autodiff does not differentiate its own forward function once constraints are active.
Established by comparing AD against a finite difference of **`mjx.step` itself** (which
matches CPU FD to 6+ digits), not just against `mjd_transitionFD`.

**Root cause: solver warm-starting.** `solver.py:590`:

```python
qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
```

`d.qacc_warmstart` is an *input* carrying zero tangent. When the warm branch wins the
solver is seeded tangent-free, and every subsequent update is gated by `improved`
(a comparison → zero derivative), so the tangent never recovers.

| system | nefc | warm ON | warm OFF (Newton) | warm OFF (CG) |
|---|---|---|---|---|
| joint limit | 1 | 17.10 | **1.27e-07** | 1.27e-07 |
| sphere/plane | 4 | 10.33 | 6.89e-04 | 3.43e+00 |
| box/plane | 16 | 48.05 | **1.55e-03** | **1.38e+02** |

**Three traps to remember:**

1. `jax.lax.stop_gradient` on `qacc_warmstart` does **nothing** — it is already constant;
   that *is* the bug. You must stop *using* it (`DisableBit.WARMSTART`).
2. **CG must never be differentiated** — worse than leaving warm-start on.
3. **Correctness depends on the history of the `MjData`.** A state built by *settling* takes
   the broken branch; a state *set directly* on a fresh `MjData` takes the good one. Every
   realistic linearization point — a settled pose, an iLQR nominal trajectory — is the
   broken case. This is why it first looked irreproducible.

Enforced in code: `mjx_manifold.linearization_model(model)` disables warm-start, and
`transition_jacobians` **raises** on a warm-start-enabled model or a CG solver.

**Status: mitigated, not passed.** 1.55e-03 on multi-contact is ~1000× above G4's bar and
flat in iteration count (that is the `improved` gate, not convergence). Fine as an iLQR
search direction validated by a line search on true cost; **not** fine for synthesizing
deployed feedback gains `K_t` — WO-2.2's two-mode gain synthesis applies even without
diffmjx.

### S8 / GATE G6 — **PASS**

Two spheres swept through the contact boundary: exactly 0.0 error separated, 1.8e-11 to
2.1e-07 in contact, all finite, no padding artefacts, unchanged when loaded to 11 N.

---

## NEXT: PHASE 2 — S9 and GATE G7

From `NEXT_STEPS.md`:

> **S9 — Additive-accumulator decomposition `[M]`.** Implement the state augmentation from
> REPO_STATE WO-3.2. Every power mean is quasi-arithmetic (`g⁻¹(Σ g(x))`), so this is
> **exact**, not an approximation.
>
> | mode | accumulator | terminal |
> |---|---|---|
> | `time_p = p ≤ 0` | `z ← z + w_t x_t^p` | `−z^{1/p}` |
> | `p = 0` | `z ← z + w_t log x_t` | `−exp(z)` |
> | `fpl_discounted` | `z_j ← z_j + γᵗ c_j(t)` | `−M_p(z)` |
> | `fpl_layered` | one accumulator per group | nested |
>
> Done when: augmented dynamics are block-triangular (`∂x'/∂z = 0`) and that is **asserted
> in a test**, not assumed.
>
> **GATE G7** — 1e-6 parity against the numpy scorer, all five `objective.mode` values,
> hopper and walker. Extend `tests/test_jax_costs.py`. On fail, `fpl_layered` is the likely
> offender (nested means) — rederive, do not special-case.

### Scoping question that was open when this was handed over

S9 as written covers all five modes, but Phase 0 produced reasons to question two of them:

* **`fpl_discounted` is structurally optimistic** (G3: Jensen, unfixable by any
  time-weighting) and should not back a safety claim. Building exact DDP machinery for it
  may be wasted effort.
* **`fpl_layered` is the mode most likely to break the decomposition** (nested means) and is
  already CPU-only by scope (`REPO_STATE` §13).
* **The published configs use `fpl_cost` + `time_p=-2.0`**, which S9 covers via the
  `time_p ≤ 0` row. That row is the one that actually unblocks WO-1 for the results that
  matter.

A defensible reduced scope is `time_p ≤ 0`, `p = 0` and `normal`, leaving `fpl_discounted`
and `fpl_layered` explicitly out with a note. **This was not decided — ask before
committing to the full five.**

### One thing S9 must not repeat

Both Phase 0 objective changes were verified by checking that the flag *actually reaches the
controller* before spending compute. The `terminal_value` flag turned out to be a silent
no-op in the published configuration, and that was caught only because of an explicit
propagation check. **Do the same for the augmented state**: assert the accumulator changes
the score before running any study, or a no-op will read as "the decomposition is exact".

---

## Files changed (all uncommitted)

**Modified**
```
analytic_mppi/config.py                     objective.atom_floor, objective.terminal_value
analytic_mppi/tasks/base.py                 floor_atoms, discount_weights, ATOM_FLOOR_*
analytic_mppi/tasks/jax_costs/_base.py      jnp mirrors of both
analytic_mppi/tasks/__init__.py             re-exports
analytic_mppi/controllers/sampling_base.py  _floor, terminal-value aggregation, guards
analytic_mppi/controllers/jax_scoring.py    terminal_value plumbed
analytic_mppi/controllers/cost_gd.py        floor + terminal_value in the analytic gradients
analytic_mppi/controllers/gradient_mpc.py   both kwargs
analytic_mppi/controllers/{mppi_v2,cem,dial,mppi_cma}.py   explicit kwarg threading
verification/_experiment.py                 ATOM_FLOOR / TERMINAL_VALUE / TIME_DISCOUNT globals
docs/mppi_math.md                           §8a and §8b updated with G1/G3 outcomes
FPL_FINDINGS.md                             G1, G2, G3 sections
tests/test_jax_costs.py                     floor + terminal-value parity
```

**New**
```
analytic_mppi/dynamics/mjx_manifold.py      manifold maps + transition_jacobians (+ guards)
tests/test_atom_floor.py                    floor and terminal-value behaviour
tests/test_jacobians.py                     G4, G5, G6
docs/jacobian_audit.md                      the full derivative audit
verification/s2_atom_floor_ab.py            G1 + G2 driver
verification/s4_terminal_value_g3.py        G3 three-arm driver
verification/checkpoints/s2_atom_floor/     ~4400 trials
verification/checkpoints/s4_terminal_value/ ~1440 trials
```

`verification/checkpoints/*.jsonl` (the published campaign) is **untouched**.

## Open items

1. **Review.** Nothing committed. Two changes specifically want sign-off: the
   `fpl_terminal_value` guard that now *raises*, and the three module globals in
   `_experiment.py` (deliberately global so they cannot be set per-arm).
2. **Two trivial pre-existing fixes, not made** (they are in your files):
   `analytic_mppi/envs/barkour/barkour.xml:89` declares an `<hfield>` whose PNG was never
   committed and which no geom references — deleting the line unblocks 3 tests and every
   quadruped study; and `tests/test_mppi_smoke.py:89` `_algo_kwargs` lacks entries for
   `fpl_colored` / `fpl_shielded` / `fpl_tempered`, which is the other 3 failures. Together
   that is all 8 "pre-existing failures".
3. **`mismatch_robustness_sweep.py` never ran floor-on** — it uses `run_study` rather than
   `run_trial`, so it needs its own wiring. It is the 4th headline study.
4. **Walker, quadruped and cube were never re-measured.** G1/G2 are hopper-only by scope.
5. **Unexplained asymmetry.** G3's A→B arm: discount-weighting the soft-min left FPL
   unchanged (0.777 → 0.778) but degraded the linear arm (0.368 → 0.237). A change that
   moves one arm and not the other is exactly the pattern this work order exists to catch,
   and there is no mechanism for it yet. Do not bank the apparent gap widening.
