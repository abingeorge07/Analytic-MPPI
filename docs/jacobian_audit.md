# Jacobian audit — is the MJX derivative the real derivative?

*WO-0 / NEXT_STEPS Phase 1. Status: **all of S5-S8 run. G4 and G6 PASS. G5 FAILED, root
cause found (solver warm-starting), and is now MITIGATED to 1.6e-3 on 16-constraint
multi-contact — still short of the 1e-6 bar, so not a clean pass.** Milestone 1 reached
with a documented caveat; the gradient arm is viable on stock MJX for search directions
but not for deployed gains. See the G5 section.*

A wrong Jacobian does not announce itself. It produces plausible iLQR trajectories that
converge to the wrong thing, and every downstream result inherits the error silently. This
phase produces no features on purpose.

---

## S5 — the f64 harness

`jax.experimental.enable_x64()` is applied **per test**, via an autouse fixture in
`tests/test_jacobians.py`, not globally:

* production MJX rollouts stay float32 deliberately (`MJXBackend`'s docstring: f64 on a
  consumer GPU is ~64× slower and the sampler's noise dwarfs f32 error);
* the global flag leaks across test modules and measurably shifts MJX rollouts, which is
  why `test_jax_costs.py` already scopes it the same way.

Without f64 the comparison against `mjd_transitionFD` sits at ~1e-3 — pure round-off that
reads exactly like a real bug. `test_f64_is_actually_on` pins it, because every tolerance
below is meaningless if the flag silently fails to apply.

## S6 / GATE G4 — smooth-system Jacobian parity

`jax.jacfwd(mjx.step)` vs `mujoco.mjd_transitionFD` (centered, `eps=1e-6`), in MuJoCo's
velocity-space layout `[qpos_tangent (nv), qvel (nv)]`, at a randomized off-axis operating
point with nonzero velocity and control. All four systems are smooth and contact-free.

**Pass criterion: relative Frobenius error ≤ 1e-6. Result: PASS, with 3+ orders of margin.**

| system | nq | nv | nu | relFro `A` | relFro `B` |
|---|---|---|---|---|---|
| pendulum | 1 | 1 | 1 | 5.59e-12 | 4.11e-11 |
| double pendulum | 2 | 2 | 2 | 1.35e-11 | 4.10e-11 |
| **free body** | **7** | **6** | 0 | **2.97e-11** | — |
| 7-dof arm | 7 | 7 | 7 | 1.61e-09 | 2.11e-11 |

The arm is ~100× looser than the rest and still 600× inside the gate; that is the FD
reference degrading on a longer chain, not the AD side.

---

## Two findings that cost real time, recorded so nobody re-derives them

### 1. MJX's own quaternion maps are not differentiable where a Jacobian needs them

This is the failure WO-0.2 predicted ("this is where quaternion / tangent-space handling
with `nq ≠ nv` silently breaks — the error localizes to the rotational block"). It is not a
physics mismatch and not an `nq ≠ nv` bookkeeping slip. It is an **AD singularity inside
MJX**.

`mjx.math.quat_integrate` and `mjx.math.quat_sub` both route through

```python
normalize_with_norm(v):  n = norm(v);  x = v / (n + 1e-6 * (n == 0.0))
```

A Jacobian evaluates the exp map at `v = 0` (zero perturbation) and the log map at
`qa == qb` (zero residual) — i.e. **exactly on that guarded branch**, where the derivative
is `0`, not the true value. Measured directly:

```
d(quat_integrate)/dv at v=0   ->  all zeros   (true value: 0.5 * I on the vector part)
d(quat_sub)/dqa   at qa=qb    ->  all zeros   (true value: 2.0 * I)
```

Symptom: the rotational block of `A` comes back as exactly `0.0` where it should be the
identity, while every translational and hinge block is correct — `relFro = 0.5` on the free
body against `1e-11` on the pendulum. An exactly-zero AD block is the tell: that is gradient
not flowing, not physics disagreeing.

Fix in `analytic_mppi/dynamics/mjx_manifold.py`: maps that agree with the true ones **to
first order at the base point**, which is all a Jacobian is.

```
exp:  q ⊗ exp(v)      ≈ normalize(q ⊗ (1, v/2))     exact d/dv at v = 0
log:  log(qb⁻¹ ⊗ qa)  ≈ 2 · vec(qb⁻¹ ⊗ qa)          exact d/dqa at qa = qb
```

For a unit quaternion `(w, u)`, `log = 2·atan2(|u|, w)·u/|u| → 2u` as `u → 0`, so value and
first derivative both match and only second-order terms differ. **These are linearization
maps, not integration maps** — away from the base point they are approximations. Use
`mj_integratePos` / MJX's own maps to actually advance a state.

`test_mjx_native_quat_maps_are_singular_at_the_base_point` pins the upstream behaviour: if a
future MJX release fixes it, that test fails and the local maps can be deleted. That is the
intended signal, not a regression.

### 2. Use `jacfwd`, not `jax.jacobian`

`jax.jacobian` is `jacrev`. MJX's constraint solver uses `lax.while_loop`, which
reverse-mode cannot differentiate:

> `ValueError: Reverse-mode differentiation does not work for lax.while_loop`

Longer kinematic chains reach that code path where a 1–2 dof toy does not, so **jacrev
fails only on the bigger models** — pendulum and double pendulum passed while the 7-dof arm
raised. Forward mode has no such restriction and is the right tool anyway: the transition
map is `R^(2nv+nu) → R^(2nv)`, so forward costs about the same as reverse and composes with
`vmap` across the horizon.

**Consequence for WO-1.** The brief specifies "Jacobians via `vmap(jax.jacobian(step_fn))`".
Taken literally that raises on any model big enough to matter. It must be
`vmap(jacfwd(...))`. `mjx_manifold.transition_jacobians` is written to be exactly the
per-timestep `(A_t, B_t)` that iLQR's backward pass needs, so WO-1 should `vmap` it rather
than rebuild the tangent-space plumbing.

---

## S7 / GATE G5 — solver-unroll sensitivity: **FAIL**

G4 says nothing about constraints: its four systems are constraint-free, so the solver
barely runs. G5 asks the real question — MJX differentiates the constraint **solver
algorithm**, so does that algorithmic derivative approach the true one as the solve
converges?

**It does not. The error is flat from iteration 1 and grows with constraint count.**

Box resting on a plane (16 `efc` rows), `‖A_ad − A_ref‖ / ‖A_ref‖` against a converged FD
reference, `tolerance=0` so the solver cannot exit early:

| iters | AD error (Newton) | AD error (CG) | FD control (CG) |
|---|---|---|---|
| 1 | 48.05 | 48.05 | 2.70e-01 |
| 2 | 48.05 | 48.05 | 1.32e-09 |
| 4 | 48.05 | 48.05 | 9.21e-14 |
| 16 | 48.05 | 48.05 | 1.24e-14 |
| 128 | 48.05 | 48.05 | 0.0 |

The **FD control is the important column**: it behaves exactly as a converging solve
should — badly wrong at 1 iteration, converged by 4. So the sweep is working and the
solver really is iterating. The AD column does not move at all.

With warm-starting left on (the default), every constraint-active system is badly wrong:

| system | active constraints (`nefc`) | AD relFro, warm-start ON |
|---|---|---|
| joint limit at its stop | 1 | **17.10** |
| sphere resting on a plane | 4 | **10.33** |
| box resting on a plane | 16 | **48.05** |

**Why this looked irreproducible at first — worth knowing, it is a trap.** The same system
at the same `qpos` gives either `1.3e-07` or `17.1` depending on *how the state was
obtained*, because of this branch:

```python
qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
```

* State built by **settling** (`mj_step` in a loop): `qacc_warmstart` is a good guess, the
  **warm branch wins**, the solver is seeded tangent-free → derivative wrong.
* State **set directly** on a fresh `MjData`: `qacc_warmstart` is zero, a bad guess, the
  **smooth branch wins**, the seed carries a tangent → derivative correct.

So the correctness of an MJX contact Jacobian silently depends on the *history* of the
`MjData` it came from. Any benchmark that linearizes about a settled or rolled-out state —
i.e. every realistic one, including iLQR's nominal trajectory — takes the broken branch.

**Methodology note.** The number that makes this conclusive is not AD-vs-`mjd_transitionFD`
— that could be MJX and CPU MuJoCo simply disagreeing on contact physics. It is
AD-vs-**finite-difference-of-MJX's-own-`mjx.step`**. Those agree to 6+ digits with the CPU
FD, and with each other, at every `eps` tried. So MJX's forward function is correct and
matches CPU in a neighbourhood; **its autodiff does not differentiate its own forward
function** once constraints are active and the warm branch is taken.

### Why it fails — the mechanism

**1. The AD Jacobian is bit-identical under every solver knob.** `iterations` 1→128,
`ls_iterations` 1→50, `tolerance=0`, Newton vs CG, warm-started vs cold: `‖A_ad − A_ad(1)‖`
is `0.000e+00` in every case. The derivative does not reflect the solve *at all*, so this
was never a convergence question — extra iterations cannot fix what never varies.

**2. It is over-stiff by ~50×, not merely different.** Box on a plane:

| | Frobenius norm |
|---|---|
| `A_ad` (MJX autodiff) | **447.1** |
| `A_true` (contact FD) | 9.1 |
| `A_nocontact` (plane deleted) | 3.5 |

So AD is not *ignoring* contact — that would land near 3.5. It is massively
over-weighting it. That is the signature of differentiating the raw constraint-stiffness
response (`solref`/`solimp` penalty) **without** the iterative correction that makes the
converged constrained solution soft.

**3. The solver gates its own updates on primal-only boolean comparisons.**
`mjx/_src/solver.py:546`:

```python
improved = (lo.cost < p0.cost) | (hi.cost < p0.cost)   # a comparison -> derivative 0
qacc = ctx.qacc + improved * ctx.search * alpha
```

plus `alpha = jp.where(lo.cost < hi.cost, ...)`, the `jp.where` tree-selects that build the
bracketing interval, and `_while_loop_scan`'s `lax.cond` no-op branch. Every one of these
routes the *value* correctly while carrying no derivative information of its own, so the
Newton correction never reaches the tangent. There is no `stop_gradient` anywhere in MJX —
the gradient loss is entirely through these comparison gates.

**Confidence.** (1) and (2) are directly measured. (3) is the code, and its zero derivative
is measured — but the causal chain from (3) to the specific factor of 50 is the best
available explanation, not something isolated by patching MJX. What is certain is that the
derivative is wrong, insensitive to every solver setting, and wrong in the over-stiff
direction.

**Why the two-sphere case in G6 still passes:** there the contact carries almost no force
at the linearization point, so the first constraint evaluation and the converged solution
nearly coincide and the missing correction is negligible. The error appears when the
constraint is genuinely load-bearing and coupled — which is every footfall.

### The fix: disable solver warm-starting in the linearization path

`solver.py:590`:

```python
qacc = d.qacc_smooth
if not m.opt.disableflags & DisableBit.WARMSTART:
    warm = Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
    smth = Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
    qacc = jp.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
```

`d.qacc_warmstart` is an **input**, not a function of the perturbed state, so it carries
**zero tangent**. When the warm branch wins, the solver's iterate is seeded tangent-free —
and because every subsequent update is gated by the zero-derivative `improved` comparison,
the tangent never recovers. `d.qacc_smooth` *is* a differentiable function of the state, so
taking that branch seeds the tangent correctly.

Measured, box on a plane (truth `‖A_true‖ = 9.135`):

| variant | relFro vs truth | `‖A_ad‖` |
|---|---|---|
| baseline (`make_data`, warmstart = 0) | 4.81e+01 | 447.14 |
| warmstart = converged value | 4.81e+01 | 447.14 |
| warmstart = converged, `jax.lax.stop_gradient` | 4.81e+01 | 447.14 |
| **`DisableBit.WARMSTART`** | **1.55e-03** | **9.14** |
| `DisableBit.WARMSTART` + `iterations=200` | 1.55e-03 | 9.14 |

**A 31,000× improvement, and the norm lands on the true value.** Note `stop_gradient` on
the warm-start does nothing — it is already constant; that *is* the bug, so removing more
gradient cannot help. The intervention has to be to stop *using* it.

This does not touch the physics of record: the closed loop is always CPU-stepped
(invariant 11.2), and warm-starting should be disabled **only in the Jacobian/linearization
path**, not in production rollouts where it is a legitimate speedup.

**Residual: 1.55e-03, still above G4's 1e-6 gate**, and it does not shrink with more
iterations — consistent with the `improved` gate leaving the tangent slightly
under-converged. So this is a mitigation that moves the failure from *catastrophic* to
*small-but-real*; whether 1e-3 Jacobians are acceptable is a per-use decision (plausibly
fine for an iLQR search direction validated by a line search on true cost; **not** fine for
synthesizing deployed feedback gains `K_t`, cf. WO-2.2).

### The fix is Newton-only — CG must not be differentiated

Full sweep, warm-start off, relFro vs a converged FD reference:

| system | `nefc` | Newton | CG |
|---|---|---|---|
| joint limit at its stop | 1 | **1.27e-07** ✓ meets 1e-6 | 1.27e-07 ✓ |
| sphere on a plane | 4 | 6.89e-04 | 3.43e+00 |
| box on a plane | 16 | **1.55e-03** | **1.38e+02** |

CG is not merely worse — on the 16-constraint case it is worse *than leaving warm-starting
on* (138 vs 48). Its Polak-Ribière `beta` recursion carries the stranded tangent forward
across iterations, so the fix does not reach it.

**Both hazards are now enforced in code.** `transition_jacobians` raises if the model still
has warm-starting enabled, or if `opt.solver` is CG (`allow_cg` / `allow_warmstart` escape
hatches exist purely so the characterisation tests can measure the bad paths). Build the
model with `mjx_manifold.linearization_model(model)`; this is invariant 11.5 applied to a
silently-wrong number rather than an unrepresentable config.

### What this means for the plan — this is a kill-criterion trigger

Per NEXT_STEPS: *"G5 fails — MJX autodiff does not converge to the true Jacobian. Switch
the whole gradient arm to `mjd_transitionFD` (slower, but `cost_gd.py` already has the
plumbing) or invest in implicit differentiation. Decide before S10."*

Options, in the order I would weigh them:

1. **`mjd_transitionFD` for the iLQR Jacobians.** It is the only source demonstrated
   correct here, `cost_gd.py` already has the FD + tangent-space plumbing, and iLQR needs
   only *one-step* Jacobians — the exact thing `mjd_transitionFD` returns. Cost is CPU
   wall-clock and no `vmap`.
2. **diffmjx (WO-2), moved earlier.** Its `scan_loop` replaces the solver's `lax.while_loop`
   specifically so the iteration count becomes explicit and differentiable — i.e. it is
   plausibly a direct fix for this, not merely a contact-gradient upgrade. WO-2.0's
   feasibility check would need to come before WO-1 rather than after.
3. **Implicit differentiation** of the constraint solution. Correct, and the most work.

Note this does **not** invalidate `GradientMPC`: it already only ever claimed to descend
the objective it was given, and on contact-free tasks G4 says its Jacobians are exact.
It does mean any contact-task gradient result from the MJX path to date should be treated
as unverified.

## S8 / GATE G6 — contact-set changes: **PASS**

MJX pads the contact array, so it is easy to end up differentiating padding. Two spheres
swept through the contact boundary, AD compared against a finite difference of **MJX's own
step** (this gate asks whether AD differentiates the function MJX computes; whether that
matches MuJoCo is G5's question):

| gap (m) | `ncon` | AD vs FD-of-mjx | finite? |
|---|---|---|---|
| 0.300 / 0.220 / 0.2010 | 0 | 0.0 | yes |
| 0.1999 | 1 | 2.14e-07 | yes |
| 0.190 | 1 | 1.76e-11 | yes |
| 0.150 | 1 | 2.99e-11 | yes |

No NaNs, no garbage, no padding artefacts, and the separated regime is exactly zero rather
than noise. Loading the contact (pressing the spheres together up to 11 N of constraint
force) leaves the error at ~2e-07. So the contact-set *machinery* is sound; G5's failure is
about the solver's derivative, not about contact bookkeeping.

---

## Milestone 1 — reached, with a failed gate

All four gates have been run. G4 and G6 pass; **G5 fails and changes the plan.** The
honest summary:

* Derivatives are **trustworthy for smooth, contact-free dynamics** (G4: ≤1.6e-09).
* Contact-set changes are handled correctly (G6).
* **Derivatives through an active constraint solve are not trustworthy** (G5), and the
  error grows with constraint coupling to the point of being useless for multi-contact.

WO-1 must not be written against MJX autodiff for contact tasks until one of the three
options above is chosen.

## Reproduce

```bash
./a-mppi/bin/python -m pytest tests/test_jacobians.py -q
```
