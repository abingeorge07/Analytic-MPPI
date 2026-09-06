# Why optimizing the FPL conjunction harder reduced hopper survival

Diagnosis of the anomaly in which three independent ways of optimizing the FPL objective
harder — sharpening the cross-atom conjunction (`fpl_p`), sharpening the temporal
weakest-link (`fpl_time_p`), and spending more samples (`K`) — each *reduced* survival.

Reproduction scripts:
[`verification/fpl_binding_probe.py`](../verification/fpl_binding_probe.py) (instrumentation),
[`verification/fpl_monotonicity.py`](../verification/fpl_monotonicity.py) (the sweep),
[`verification/fpl_budget_scaling.py`](../verification/fpl_budget_scaling.py) (the K axis).
Raw data under `runs/diagnostics/`.

---

## 0. First, a measurement-noise correction

The original table was read at 10 seeds. It overstates the effect by roughly 3x.

The hopper closed loop is **chaotic at machine precision**. `fpl_conj_indices` with
`fpl_outer_p = -1` was constructed to be an *exact algebraic no-op* at `fpl_p = -1`
(verified: max score difference 3.6e-15 over the whole (K,H) cloud, byte-identical actions
for the first two control steps). Run for 400 steps it nevertheless produced a **different
survival rate on the same 30 seeds: 0.70 vs 0.53.**

So at n=30 the measurement floor on a hopper survival number is about ±0.17, and at n=10 it
is about ±0.30. The original spread of 0.30–0.70 is almost entirely within that floor. Every
number below is therefore at **n=100** (Wilson width ≈ ±0.07), and every claim is backed by
a two-proportion test.

At n=100 the effect is real, but modest — not the collapse the 10-seed table suggested — and
**only at the 0.9 s horizon**:

| axis | `orig` @ h=0.9 s | z | p | | `orig` @ h=0.6 s | p |
|---|---|---|---|---|---|---|
| `fpl_p` −1 → −8 | 0.94 → 0.85 | +2.08 | **0.038** | | 0.49 → 0.42 | 0.320 (ns) |
| `fpl_time_p` −2 → −16 | 0.94 → 0.75 | +3.71 | **0.0002** | | 0.49 → 0.41 | 0.256 (ns) |

The 0.6 s geometry — where the original table was measured — is a **coin flip at every cost
composition**. At n=100, none of the six conjunction-strength cells differs from the base
cell (all p ≥ 0.087; survival 0.37–0.49). Nothing about cost composition is measurable
there, because the controller is not competent there. The anomaly, as an anomaly, exists
only at the horizon where the controller actually works.

---

## 1. What the evidence supports

**The binding-atom inversion (your leading hypothesis) is confirmed — but there are two
distinct inversions, not one, and they live on different axes.**

`power_mean(f, p)` with `p < 0` is a *"raise whichever atom is currently smallest"*
operator. It is a safety mechanism only if the smallest atom is a safety atom. It usually
isn't. Instrumenting the softmax-weighted argmin over the entire (K × H) rollout cloud, at
every control step of every episode:

### 1a. Progress atom captures the conjunction (`fpl_p` axis)

At the healthy 0.9 s horizon the argmin distribution is stable all episode:

| window | height | orient | **veloc** | control |
|---|---|---|---|---|
| 0–2 s | 0.069 | 0.209 | **0.523** | 0.199 |
| 2–4 s | 0.101 | 0.248 | **0.376** | 0.276 |
| 4–6 s | 0.054 | 0.211 | **0.515** | 0.220 |
| 6–8 s | 0.054 | 0.194 | **0.534** | 0.218 |

Velocity is the binding atom about half the time. So "sharpen the conjunction" literally
means "go faster". The signature confirms it: as `fpl_p` goes −1 → −8, forward speed
**rises** 1.634 → 1.786 m/s while min-uprightness **falls** 0.67 → 0.59 and survival falls
0.94 → 0.85. The optimizer is doing exactly what it was asked; the ask was wrong.

Velocity is a *progress* objective — there is no floor it must hold. Height, orientation and
control are *constraints*. A min-conjunction over the union of the two kinds cannot
distinguish "the constraint closest to violation" from "the goal furthest from achieved".

### 1b. Unachievable constraint monopolizes the conjunction (`fpl_time_p` axis)

At the myopic 0.6 s horizon the height atom is not achievable — a hopper must compress its
leg, and the band's floor (0.85) sits inside the ordinary compression trough. So it clips to
zero constantly, and because `power_mean(p<0)` is pinned at ~eps as soon as *any* atom
clips, the composite goes blind. Measured over the (K × H) cloud: the height atom alone is
clipped to 0 in **28–58 %** of cells, *some* atom is clipped in **33–61 %**, and
**74–87 % of the 256 rollouts have their FPL reward sitting at the eps floor** — mutually
indistinguishable, so the effective sample budget is roughly a quarter of K.

`fpl_time_p` scores a rollout by its worst moment, and the worst moment is almost always the
clipped compression trough. Sharpening it therefore *concentrates the entire objective onto
the one atom that carries no information*. At `fpl_time_p = -8` the conjunction degenerates
to a single atom:

| window (surviving episodes) | height | orient | veloc | control |
|---|---|---|---|---|
| 0–2 s | 0.044 | 0.152 | 0.578 | 0.225 |
| 2–4 s | 0.488 | 0.218 | 0.094 | 0.199 |
| 4–6 s | **0.927** | 0.001 | 0.071 | 0.000 |
| 6–8 s | **0.999** | 0.001 | 0.000 | 0.000 |

Orientation, velocity and control have *zero* influence for the last four seconds. And the
controller still fails at the atom it collapsed onto: realized height fulfilment drops
0.784 → 0.471 and speed drops 1.68 → 1.26.

The corroborating diagnostic kills hypothesis 2 outright: as `fpl_time_p` sharpens, **ESS
RISES** (42.8 → 58.5) and log-reward IQR falls (8.99 → 5.57). This is not weight collapse —
it is the opposite, a loss of discrimination as rollouts become mutually indistinguishable.

### 1c. The safety atom never binds — including during the fall

Aligned on the fall instant (h=0.6 s, `fpl_p=-1`):

| window | height | **orient** | veloc | control |
|---|---|---|---|---|
| >3 s before fall | 0.140 | 0.155 | **0.530** | 0.175 |
| 3–1 s before | 0.324 | 0.174 | 0.284 | 0.218 |
| 1.0–0.5 s before | 0.528 | 0.153 | 0.238 | 0.081 |
| final 0.5 s | **0.631** | 0.237 | 0.024 | 0.107 |

Orientation — the atom that *defines* falling — peaks at 0.24 even in the final half-second.
Its share does not rise as the robot dies; at `fpl_p=-4` and `fpl_time_p=-8` it *falls*
(0.223 → 0.145 and 0.137 → 0.059). Hypothesis 1's zero-margin observation is correct and it
is systemic: `orientation_floor == fall_threshold` in **hopper (0.60/0.60), walker
(0.60/0.60) and quadruped (0.50/0.50)**.

### 1d. The third axis — sample budget — was never an anomaly

Sweeping K at the valid operating point, 100 seeds, same objective
([`fpl_budget_scaling.py`](../verification/fpl_budget_scaling.py)):

| K | `orig` survival | vx | ESS | | `both` survival | ESS |
|---|---|---|---|---|---|---|
| 256 | 0.94 [0.88,0.97] | 1.634 | 22.0 | | 0.88 [0.80,0.93] | 24.2 |
| 512 | 0.97 [0.92,0.99] | 1.669 | 45.6 | | 0.90 [0.83,0.94] | 50.1 |
| 1024 | 0.96 [0.90,0.98] | 1.721 | 87.2 | | 0.90 [0.83,0.94] | 98.7 |

**More samples never hurt.** Survival is flat-to-rising and speed rises monotonically. The
reported K=512 → 0.60 and K=1024 → 0.40 were 10-seed artifacts at the coin-flip geometry.
ESS also scales linearly with K (8.6 % of budget throughout), so there is no
budget-driven weight concentration either — a second, independent refutation of hypothesis 2.

So of the three "optimize harder, survive less" axes, **two are real and one is not.**

### Verdicts on the secondary hypotheses

| # | hypothesis | verdict |
|---|---|---|
| 1 | zero-margin safety atom | **Confirmed as a defect**, but fixing it alone does *not* restore monotonicity — see §2 |
| 2 | exploit-collapse / ESS collapse | **Refuted twice.** ESS *rises* as survival falls (42.8 → 58.5); and ESS stays a constant 8.6 % of K as K quadruples |
| 3 | horizon myopia | **Confirmed as the dominant lever.** h=0.6 s ≈ 0.5 survival; h=0.9 s ≈ 0.94 |
| 4 | velocity reward-hacking | **Confirmed on the `fpl_p` axis only** (speed ↑, uprightness ↓); cannot explain the `fpl_time_p` axis, where speed *falls* |

---

## 2. The repair

Two defects, two minimal changes, each opt-in and each a no-op at its default.

**`soft` — remove the absorbing zero.** [`tasks/base.soft_ramp`](../analytic_mppi/tasks/base.py)
replaces each atom's hard `clip(r, 0, 1)` with a strictly monotone exponential tail below the
floor. The satisfied band and the in-band ordering are untouched; the atom simply never stops
carrying information. Enabled by `atom_soft_floor=0.05`.

**`split` — keep progress objectives out of the conjunction.**
[`controllers/sampling_base._collapse_objectives`](../analytic_mppi/controllers/sampling_base.py)
runs the `p<0` conjunction over the constraint atoms only (`fpl_conj_indices=[0,1,3]`) and
composes the result with the progress atom at an outer power-mean. With the default
mass-preserving outer weights this is an *exact* no-op at `fpl_outer_p == fpl_p`, so it
changes only *where* a sharper `fpl_p` acts, not the operating point.

### Verification — 100 seeds, 8 s, K=256, horizon 0.9, Wilson 95 %

Each fix repairs exactly the axis its mechanism predicts, and only that axis:

| variant | `fpl_p` −1 → −8 | verdict | `fpl_time_p` −2 → −16 | verdict |
|---|---|---|---|---|
| `orig` | 0.94 → 0.85 | **HURTS** (p=0.038) | 0.94 → 0.75 | **HURTS** (p=0.0002) |
| `soft` | 0.93 → 0.79 | HURTS (p=0.004) | 0.93 → 0.91 | **fixed** (p=0.60) |
| `split` | 0.89 → 0.93 | **fixed** (p=0.32) | 0.89 → 0.80 | partial (p=0.079) |
| **`both`** | **0.88 → 0.95** | **fixed** (p=0.076) | **0.88 → 0.95** | **fixed** (p=0.076) |
| `margin` alone | 0.98 → 0.85 | HURTS (p=0.001) | 0.98 → 0.76 | HURTS (p<0.0001) |

At the sharpest settings `both` is significantly better than `orig`:

| cell | `orig` | `both` | z | p |
|---|---|---|---|---|
| `fpl_p = -8` | 0.85 [0.77,0.91] | **0.95 [0.89,0.98]** | +2.36 | 0.018 |
| `fpl_time_p = -16` | 0.75 [0.66,0.82] | **0.95 [0.89,0.98]** | +3.96 | 0.0001 |

And the physical quantity moves the right way. Min-uprightness vs conjunction strength:

| | −1/−2 | −2/−4 | −4/−8 | −8/−16 |
|---|---|---|---|---|
| `orig`, `fpl_p` | 0.67 | 0.62 | 0.56 | 0.59 |
| `orig`, `fpl_time_p` | 0.67 | 0.65 | 0.56 | **0.38** |
| `both`, `fpl_p` | 0.71 | 0.70 | 0.74 | **0.77** |
| `both`, `fpl_time_p` | 0.71 | 0.73 | 0.73 | **0.76** |

With the repair, tightening the conjunction *buys* uprightness monotonically — the FPL
thesis behaving as designed. Without it, tightening *spends* uprightness.

### Honest cost, and the scope condition on `soft`

`both` trades base-point performance for the guarantee: at the loose setting it is 0.88 vs
`orig`'s 0.94, and its best `prod` is 1.369 vs `orig`'s 1.549 (speed 1.42 vs 1.63 m/s).
The correct way to use it is at a *sharp* conjunction, which is precisely what it makes safe.

**`soft` is only valid where the atom is achievable.** At the 0.6 s horizon, where the height
atom is chronically clipped, softening the floor is severely harmful — survival 0.49 → 0.23
for `soft` (p=0.00013) and 0.49 → 0.22 for `both` (p=0.00007), while `split` alone is
harmless (0.49 → 0.57, p=0.26).

The reason is instructive. Where a constraint is chronically violated, the hard clip is doing
load-bearing work as a *barrier*: annihilating any rollout that sags is brutal, but it is
what keeps the hopper tall. Replacing it with a floor at f_min = 0.05 downgrades "sagging" from
fatal to merely bad, and the optimizer promptly accepts sagging. So the soft floor buys
information at the cost of barrier strength — a good trade only when the atom is not being
violated anyway.

This gives a clean applicability rule: **soften an atom's floor only where it is achievable
at the current operating point.** Where it isn't, the honest fix is the horizon or the band,
not the barrier. (And at h=0.6 s that is moot regardless — see §0, nothing is measurable there.)

### Negative result worth recording

Raising the orientation floor above the fall line (`orientation_floor=0.75`, hypothesis 1's
proposed fix) gives the **best single operating point of any variant — 0.98 [0.93,0.99]** —
but makes monotonicity *worse* on both axes (0.98 → 0.85 and 0.98 → 0.76). Margin and
monotonicity are separate problems: margin decides whether the safety atom can *ever* bind
in time, composition decides whether sharpening the conjunction *targets* it. Combined with
`split` it is non-decreasing on `fpl_p` (0.94 → 0.96) but still degrades on `fpl_time_p`
(0.94 → 0.70), because it does nothing about the clipping saturation.

---

## 3. Is this hopper-specific or general?

**General — it is a property of min-conjunction objectives, and the conditions are already
present in walker, quadruped and (for the first condition) LEAP-cube.**

The failure needs two ingredients, neither of which is about hoppers:

1. **A progress atom inside the conjunction.** `walker` and `quadruped` both put
   `velocity_fulfillment` in the same flat `power_mean`; quadruped adds `heading`. Any
   `p<0` composition over {constraints ∪ goals} will preferentially optimize the goal
   whenever the goal is the least-satisfied thing — which, for a hard goal, is most of
   the time.
2. **A clipped atom with a flat zero region.** Every atom in the suite is
   `np.clip(..., 0, 1)`. Any of them touching zero pins the whole conjunction at eps and
   destroys discrimination among rollouts — and `p<0` weights that dead region *most*
   heavily. This is the general statement: **a conjunctive power-mean is maximally
   sensitive to exactly the region where a clipped atom carries no information.**

A third, spec-level defect is also systemic: the orientation atom reaches 0 exactly at the
episode's fall threshold in all three locomotion tasks, so the only true safety atom has
zero margin and cannot bind before the failure it guards.

Practical implications for the other environments:

- **walker / quadruped**: expect the same inversion. Apply `fpl_conj_indices` excluding
  velocity (and heading, for quadruped), and give the atoms soft floors. Quadruped's
  `posture` and `com_bob` atoms are constraints and belong inside the conjunction.
- **LEAP-cube**: has no fall threshold and its progress atom (`angle`) is the *point* of the
  task, so 1b (clipping saturation) applies but 1a is less clear-cut — the drop constraint
  (`hold_xy`) is the atom that should be conjoined against.
- **The general prescription**: partition atoms into constraints and progress *before*
  choosing `p`, conjoin only the constraints, and never let a conjoined atom have a flat
  zero region.
