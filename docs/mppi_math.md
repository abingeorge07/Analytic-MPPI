# The MPPI formulation in this repo: what it derives, and where the derivation stops

Reference points in the code:
`analytic_mppi/controllers/sampling_base.py` (rollout + scoring),
`analytic_mppi/controllers/mppi_v2.py` (weighting + mean update),
`analytic_mppi/tasks/base.py::power_mean` (the FPL operator).

Every quantitative claim below is verified numerically against the actual `power_mean`
implementation; the check scripts are reproduced in §9.

---

## 1. The implemented update is not Williams' update — a term is missing

Williams' information-theoretic derivation targets

$$q^*(V) \;\propto\; p(V)\,e^{-S(V)/\lambda}$$

where **`p` is the base (uncontrolled) measure** $\mathcal N(0,\Sigma)$ and $q_U=\mathcal N(U,\Sigma)$
is the proposal. Importance sampling from $q_U$ with $V = U + E$ needs the likelihood ratio

$$\frac{p(V)}{q_U(V)} \;=\; \exp\!\Big(-U^\top\Sigma^{-1}E \;-\; \tfrac12 U^\top\Sigma^{-1}U\Big).$$

The second term is sample-independent and cancels. The **first term, $-U^\top\Sigma^{-1}E$, does not**,
and it is absent from both `mppi.py:71` and `mppi_v2.py:138`, which use the bare
$w_k\propto\exp(-(S_k-\min_j S_j)/\lambda)$.

So what the code implements is the moment-matched projection of a **different** target:

$$\tilde q^*(V)\;\propto\; q_U(V)\, e^{-S(V)/\lambda} \qquad\text{(tilt the \emph{current proposal}, not the prior)}$$

This is not a typo-level bug — it is what nearly every practical MPPI implementation does
(hydrax included) — but it changes the object being computed:

* **There is no control-effort anchor.** Williams' $\lambda U^\top\Sigma^{-1}E$ term is exactly the
  implicit quadratic control cost $\tfrac{\lambda}{2}U^\top\Sigma^{-1}U$ that ties the solution to
  the reference. Without it, the recursion $U \leftarrow \mathbb E_{\tilde q^*}[V]$ is a
  **soft-argmax / mean-shift hill climb** on $S$, whose fixed points are local minima of $S$ plus
  an $O(\Sigma)$ smoothing bias — not the free-energy-optimal control.
* **The free-energy lower bound no longer applies.** Any statement of the form "MPPI optimizes a
  bound on the optimal cost-to-go" is unavailable for this code as written. If a paper draft
  makes that claim, it needs to be dropped or the term restored.

**Consequence for your FPL-vs-linear thesis: none, directly.** The term is missing identically on
both sides of the comparison, so it does not confound the contrast. It only limits what you may
*claim* about the underlying algorithm.

---

## 2. The knot-spline parameterization is self-consistent (this part is fine)

Sampling happens in knot space (`mppi_v2.py:97`) and the mean update happens in knot space
(`mppi_v2.py:117/132/148`). With $U_{1:H} = A\Theta$ for a fixed interpolation matrix $A$, the
induced law on control sequences is $\mathcal N(A\bar\Theta,\ \sigma^2 AA^\top)$ — degenerate,
rank $\le M\cdot n_u \ll H\cdot n_u$. That degeneracy would break the derivation if the update
were computed on $U$; because it is computed on $\Theta$, the whole argument goes through with
$V:=\Theta$ and "dynamics" $:=$ rollout∘interpolate. **No issue.**

The real cost is representational, not mathematical: with `spline_type="zero"` and
`num_knots=4` over a 1 s horizon, you are optimizing over piecewise-constant controls with
0.25 s segments. That is the actual search space, and it is worth stating explicitly in any
write-up.

---

## 3. Clipping breaks the change of measure (a real, asymmetric confound)

`sampling_base.py:212` clips knots to the box **after** sampling. The samples are then no longer
$\mathcal N(\bar\Theta,\sigma^2 I)$: the law becomes a truncated Gaussian with **atoms on the box
faces**, while the weights are still computed as if it were Gaussian. Two consequences:

1. Saturated samples collapse onto identical points, so ESS **overstates** true sample diversity.
2. At saturation the noise is one-sided, so the weighted mean is biased inward and the effective
   step size along that coordinate is attenuated.

This is the one place where a shared piece of machinery can bias the comparison **asymmetrically**:
FPL's conjunction discourages extremes, whereas an aggressive linear weight drives the mean toward
the box boundary and therefore clips more. If the aggressive linear baselines are saturating and
FPL is not, part of the measured gap is clipping bias rather than scalarization.

> **Concrete check worth running:** log the fraction of knots at the box boundary per controller
> per step and report it alongside the Pareto plots. If it is ~0 for both, the confound is dead
> and you can say so in one sentence. If it differs, that number belongs in the paper.

**How the iLQR arm handles bounds (S10 decision).** `controllers/ilqr.py` does *not*
clip-after-update — that would re-introduce this same confound into the optimizer arm (a
clipped Newton step is biased inward exactly like a clipped sample cloud, and the FPL and
linear objectives saturate differently). Instead it solves the box-constrained
$Q_{uu}$ sub-problem by projected Newton per step: the feedforward $k_t$ is the minimizer of
$\tfrac12\,\delta u^\top \bar Q_{uu}\,\delta u + Q_u^\top \delta u$ subject to
$u_{\min}\le u_t+\delta u\le u_{\max}$ (fixed clamp-and-solve iterations), and the feedback
rows $K_t$ of clamped coordinates are zeroed, so the closed-loop update never *relies* on
control authority it does not have. The forward line search additionally clips at execution,
which is exact (the plan being evaluated is the plan being scored — there is no importance
weight to break). Residual caveat: the spline projection $\theta = W^+u^*$ *can* leave the
box, so $\theta$ is clipped after projection; with `spline_type="linear"` (the iLQR arm's
default) the overshoot is bounded by the projection residual, which is instrumented per act
(`last_proj_resid`, GATE G10).

---

## 4. The FPL score itself: correct as specified

$S_k=-\log u_k$ with the min-shift gives

$$w_k \;\propto\; \exp\!\Big(\tfrac{\log u_k - \log \max_j u_j}{\lambda}\Big)\;=\;\Big(\frac{u_k}{\max_j u_j}\Big)^{1/\lambda},$$

which is exactly HANDOFF §3 step 4. Sign correct, min-shift present, applied every step,
$u>0$ guaranteed by the $\varepsilon$-clip in `power_mean`. The target
$\tilde q^*\propto q_U\cdot u^{1/\lambda}$ is a proper posterior with bounded likelihood
$u^{1/\lambda}\in(0,1]$. **This part of the formulation is sound.**

---

## 5. The core result: FPL is a *second-order* correction to a uniform linear scalarization

This is the piece that explains both your wins and your nulls.

Write $F_i = 1-e_i$ where $e_i\ge 0$ is the **shortfall** on objective $i$. Expanding
$S=-\log\mu_p(F)$ about $F=\mathbf 1$:

$$\boxed{\;S \;=\; \overline{e} \;+\; \tfrac12\,\overline{e}^{\,2} \;+\; \frac{1-p}{2}\,\mathrm{Var}(e)\;+\;O(e^3)\;}$$

where $\overline e$ and $\mathrm{Var}(e)$ are the mean and variance **across objectives**.
(Verified: residual is $O(e^3)$ for $n\in\{3,5\}$, $p\in\{1,0,-1,-2\}$.)

Three things fall out immediately.

**(a) The first-order term is $p$-independent.** $\nabla_F S\big|_{F=\mathbf 1} = -\tfrac1n\mathbf 1$
for *every* $p$ — verified to 6 decimals for $p\in\{1,0,-1,-3\}$. Near full satisfaction, FPL is
**identical to a fixed uniform linear scalarization of the fulfillments**, no matter how negative
$p$ is. The $p$ knob is not doing anything at first order.

**(b) The entire FPL mechanism is the $\frac{1-p}{2}\mathrm{Var}(e)$ term.** It penalizes
*imbalance across objectives*. At $p=1$ it vanishes exactly (linear scalarization is indifferent to
imbalance — the paper's "linear behaves like a disjunction" critique, made quantitative). At $p<0$
its coefficient exceeds $\tfrac12$ and grows linearly in $|p|$. That is the whole story.

**(c) It predicts the size of the effect, and hence the nulls.** The FPL-vs-linear signal is
$O(\mathrm{Var}(e))$ — second order — against a first-order term $O(\overline e)$. With typical
shortfall $e\sim0.05$ and $p=-2$, the differentiating term is $\approx 0.004$ versus $0.05$: **~8%
of the score variation.** Under a finite-$K$ Monte Carlo estimate, that is easily inside the noise.

> **This is why "FPL adds nothing when a safe conservative operating point exists" is a theorem, not
> an observation.** In the satisfied regime $e\to0$, the mechanism is quadratically suppressed. The
> gap can only open where shortfalls are large — i.e. exactly the "dynamic competition" and
> "traction loss" regimes where you measured wins. Your empirical pattern is the expansion.

This is a strong thing to put in a paper: it converts "we found the gap opens under competition"
into "the gap *must* be $O(\mathrm{Var}(e))$ and therefore *cannot* open otherwise."

---

## 6. $\lambda$ and $p$ are confounded — and worst exactly where your results live

$\lambda$ acts on $S=-\log u$, and $p$ changes the scale of $\log u$. Measured on identical
rollouts ($K=1000$, $n=3$ atoms, $\lambda=0.1$), varying **only** $p$:

| shortfall scale | $p=+1$ | $p=0$ | $p=-1$ | $p=-2$ | $p=-5$ |
|---|---|---|---|---|---|
| $s=0.02$ | ESS 995 | 995 | 995 | 995 | 995 |
| $s=0.10$ | ESS 891 | 887 | 882 | 877 | 862 |
| $s=0.40$ | ESS **312** | 272 | 244 | 224 | **186** |

In the easy regime $p$ and $\lambda$ are effectively orthogonal. In the competitive regime a fixed
$\lambda$ makes the controller **~1.7× greedier at $p=-5$ than at $p=1$**. So a $p$-sweep at fixed
$\lambda$ is partly a greediness sweep — and the confound is concentrated in precisely the regime
your headline results come from.

> **Fix (cheap and defensible):** set $\lambda$ per timestep by bisection to hit a *target ESS*
> (e.g. ESS/K = 0.2), identically for both controllers. Then $p$ is a pure semantics knob and the
> comparison is genuinely apples-to-apples. Failing that, report ESS at every $(p,\lambda)$ cell.
> A reviewer who knows MPPI will ask this.

> **Structural fix, adopted (S12 / GATE G11): resolved by construction.** The Milestone 4
> comparison places FPL and linear on the same optimizer *twice*, once with MPPI's softmax
> update rule and once with iLQR, whose update rule has **no $\lambda$ anywhere** — no
> temperature, no softmax, no $-\log u$ bridge. To keep the iLQR column temperature-free
> end-to-end its warm-start proposer is `PredictiveSampling` (argmax over the Gaussian
> cloud with the current mean always included), not MPPI. If the FPL–linear gap were an
> artifact of $\lambda/p$ coupling in the sampling update, the gap should collapse (or
> flip) in the iLQR column. Measured, hopper + walker, 10 seeds/cell, matched dynamics
> budget (`verification/s12_the_2x2.py`): FPL is $\ge$ linear in every cell — decisively
> on walker (1.57 ± 0.07 vs 0.86 ± 0.10, CIs disjoint) and at parity on hopper
> (1.20 ± 0.52 vs 1.18 ± 0.51). The gap survives an optimizer with no temperature. GATE
> G10 additionally establishes that the knot basis is a **headwind** for FPL specifically
> in the iLQR column (projection residual 0.65 vs 0.51; $J_{\text{post}}-J_{\text{pre}}$
> gap 0.51 vs 0.07), so this is not a boost from the new optimizer either. See
> `FPL_FINDINGS.md#S12 — the {FPL, linear} x {MPPI, iLQR} 2x2` for the full table.

---

## 7. Weight collapse in the other direction: the "mushy" regime

$S=-\log u$ with $u\in[\varepsilon,1]$ has a **bounded** dynamic range, unlike an unbounded quadratic
cost. When the controller is doing well, all $K$ rollouts cluster near $u\approx1$ and the weights
go uniform — the update degenerates to the sample mean, i.e. the previous mean plus $O(\sigma/\sqrt K)$
noise. Measured at $p=-2,\ \lambda=0.1,\ K=1000$:

| shortfall $s$ | $u$ range | std$(S)$ | ESS |
|---|---|---|---|
| 0.005 | [0.990, 1.000] | 0.0017 | **999.7** |
| 0.02  | [0.958, 0.999] | 0.0073 | 994.9 |
| 0.05  | [0.895, 0.996] | 0.0179 | 971.5 |
| 0.20  | [0.434, 0.990] | 0.1114 | 591.9 |
| 0.50  | [0.000, 0.955] | 5.6352 | 147.2 |

**FPL-MPPI is structurally a safety-*triggered* controller**: it has meaningful selection pressure
only when some atom approaches its floor. That is a feature for the robustness story and a
liability for nominal-regime performance — and it is the same mechanism as §5(c), seen through the
weighting rather than the score.

The `proportional` mode (`mppi_v2.py:109`) is the pathological limit: $w_k\propto u_k$ with no
temperature at all. On the same rollouts it gives ESS 873/1000 where softmax gives 220 — it
**cannot be made selective**, since it has no knob. It is therefore not a valid apples-to-apples
ablation of the softmax path; it is a strictly weaker rule. The `exp` mode (ESS 290) is closer but
still limited by $u$'s $[0,1]$ range: the maximum achievable log-weight spread is $1/\lambda$.

---

## 8. Two bugs-in-waiting in the scoring pipeline

### 8a. `use_fpl_discounted` is systematically optimistic (order of operations)

`_score_fpl` has two modes that differ in whether you conjoin-then-average or average-then-conjoin
over time. **These do not commute.** $\mu_p$ is concave for $p\le1$, so by Jensen

$$\mu_p\big(\mathbb E_t[f]\big)\;\ge\;\mathbb E_t\big[\mu_p(f)\big],$$

i.e. **`use_fpl_discounted` $\ge$ `use_fpl_cost`, always.** Measured gap (mean over 5000 random
rollouts, $H=50$, $n=3$): $+0.07$ at $p=0$, $+0.12$ at $p=-1$, $+0.16$ at $p=-2$; zero at $p=1$ as
required. The discounted mode averages each atom over time *first*, so "upright early, fallen late"
becomes indistinguishable from "mediocre throughout" — it launders exactly the failure the
conjunction exists to catch.

Worse, with `fpl_time_p=None` even `use_fpl_cost` takes a discounted *arithmetic* mean over time,
so the conjunction over objectives is defeated by a **disjunction over time**. Concretely, an
upright atom that holds at 1.0 and then collapses to 0.02:

| falls at step | `use_fpl_discounted` | `use_fpl_cost` (time-mean) | `use_fpl_cost` (`time_p=-1`) |
|---|---|---|---|
| 10/50 | 0.510 | 0.286 | **0.071** |
| 25/50 | 0.800 | 0.588 | **0.109** |
| 40/50 | 0.941 | 0.847 | **0.234** |

A rollout that falls at step 25 of 50 still scores **0.80** under the discounted mode. Note also
that with $\gamma=0.99$ over $H=50$ the discount only spans $1\to0.605$ — it is nearly a uniform
average, so discounting is not saving you here. **`fpl_time_p` is not an optional refinement; it is
what makes the min-fulfillment floor a trajectory property rather than a per-step one.**

> **[2026-09-11 — GATE G3: PARTIAL. There are TWO causes here, and WO-3.3 closes only one.]**
>
> `objective.terminal_value=True` replaces the renormalization $\frac{1-\gamma}{1-\gamma^H}\sum\gamma^t r_t$
> with an explicit tail $(1-\gamma)\sum_{t<H}\gamma^t r_t + \gamma^H v(x_H)$, $v$ = hold $r_H$.
> Re-running the table above (atom collapses to 0.02 and stays collapsed, $H=50$, $\gamma=0.99$),
> as `OFF → ON`:
>
> | falls at | `fpl_discounted` | `fpl_cost` (time-mean) | `fpl_cost` (`time_p=-1`) |
> |---|---|---|---|
> | 10/50 | 0.510 → **0.278** | 0.286 → **0.148** | 0.075 → **0.063** |
> | 25/50 | 0.800 → **0.483** | 0.588 → **0.267** | 0.123 → **0.073** |
> | 40/50 | 0.941 → **0.612** | 0.847 → **0.370** | 0.274 → **0.084** |
>
> **Cause 1 — horizon truncation. Closed.** The renormalization assumed the post-horizon tail
> equals the in-horizon average, which is why a late fall scored 0.94. At $\gamma=0.99,H=50$ the
> tail is $\gamma^{49}=0.61$ of the total discount mass, so this was never a small correction.
> The best consequence is in the last column: with the tail term, `time_p=-1` becomes nearly
> **fall-time invariant** (0.063 / 0.073 / 0.084, a 1.3× spread) where it previously ranged
> 0.075 → 0.274, a 3.6× spread rewarding late falls. A fall is now a fall whenever it happens.
>
> **Cause 2 — order of operations. NOT closed, and not closable this way.** The Jensen gap is a
> property of conjoining *after* averaging, not of the weights: $\mu_p$ is concave for $p\le1$, so
> $\mu_p(\sum_t w_t f_t)\ge\sum_t w_t\,\mu_p(f_t)$ for **every** convex weight vector $w$. Measured
> over 4000 random rollouts the gap shrinks but never vanishes or flips: mean $+0.129\to+0.088$,
> min $+0.083\to+0.030$, strictly positive in **100%** of cases both ways. No time-weighting
> change can remove it.
>
> **So the standing advice is unchanged and is now the *only* fix for cause 2:** use
> `fpl_cost` with `fpl_time_p ≤ 0`, not `use_fpl_discounted`. The published hopper/walker configs
> already do (`time_p=-2.0`), so they sit on the safe side of this — but `fpl_discounted` remains
> structurally optimistic and should not be used for a safety claim.
>
> Caveat on the tail estimate: holding $r_H$ constant is WO-3.3's *cheapest* $v$, and at
> $\gamma=0.99,H\approx30$ it carries ~74% of the weight — the objective becomes largely
> "terminal fulfillment, in-horizon correction". That is defensible (it is the honest tail) but
> it is a big lever, and a fitted $v$ is the stated improvement. Verified by
> `tests/test_atom_floor.py`; flag defaults to `False`, so nothing pre-existing moved.
>
> **[Closed-loop follow-through — the cheap tail is NET HARMFUL. Do not enable it.]**
> Two further facts, from `verification/s4_terminal_value_g3.py` (hopper, 16 seeds):
>
> 1. **It does not reach the published objective at all.** The published spec is
>    `fpl_cost` + `fpl_time_p=-2` + `fpl_time_discount=False`, i.e.
>    `power_mean(per_step, time_p, weights=None)` — an UNWEIGHTED soft-min with no
>    $\frac{1-\gamma}{1-\gamma^H}$ anywhere in it. The renormalization this section
>    criticizes lives in the `time_p=None` and `use_fpl_discounted` paths only. The
>    controller now raises on `terminal_value=True` in the inert combination (invariant 11.5).
> 2. **Where it does reach, it degrades survival badly.** Holding time-discounting fixed and
>    toggling only the tail: FPL survival $0.88\to0.56$ (portability) and $0.81\to0.44$
>    (zero_tuning), with the FPL-vs-linear gap collapsing $+0.427\to+0.049$ on the latter.
>    The tail weight $\gamma^{30}=0.740$ versus $\approx0.010$ per interior step is a **74×**
>    concentration on the final step, which guts precisely the weakest-link-over-time property
>    this section argues is what makes the floor a *trajectory* property. The cheap $v$
>    reintroduces §8a's defect from the other side: a mid-horizon collapse that has partly
>    recovered by step $H$ is forgiven.
>
> So the honest status of §8a is: **truncation is real but the cheap fix is worse than the
> disease under $p<0$ time aggregation.** A fitted $v(x_H)$ is required before this is usable,
> and the operative advice remains "use `fpl_cost` with `fpl_time_p ≤ 0`".

### 8b. The $\varepsilon$-plateau is still live on walker, quadruped and cube

`soft_ramp` exists and fixes this, but `grep` shows it is used **only in `hopper.py`** (and only when
`atom_soft_floor > 0`). `walker.py:80,88,96,99`, `quadruped.py:172,178,184,265,354,362` and
`cube.py:226` all use hard `np.clip(·, 0.0, 1.0)`. With $p=-1$ and one atom clipped to 0:

| margin below floor | hard clip | `soft_ramp` |
|---|---|---|
| 0.0 | $u=3\!\times\!10^{-8}$, $S=17.32$ | $u=0.136$, $S=1.99$ |
| 0.5 | $u=3\!\times\!10^{-8}$, $S=17.32$ | $u=0.053$, $S=2.93$ |
| 2.0 | $u=3\!\times\!10^{-8}$, $S=17.32$ | $u=0.003$, $S=5.90$ |

**Every failing rollout receives the identical score $17.32$**, so among failing rollouts there is
no gradient at all — the controller cannot tell "just tipped" from "inverted," which is the
information it most needs for recovery. It also means the $S$ distribution is bimodal (a tight
cluster near 0 and a spike at 17.32), so a single $\lambda$ cannot be well-calibrated for both.
This also inflates the std$(S)=3.47$ figure at $s=0.4$ in §6 — that number is the plateau, not
genuine discrimination.

Three of your four headline robots are affected. Since the robustness experiments deliberately push
into the failure region, this is where it matters most.

> **[2026-09-11 — addressed by WO-3.4, and it was worse than described above.]**
>
> The plateau is not confined to walker/quadruped/cube. `soft_ramp` is gated behind
> `atom_soft_floor`, which **defaults to `0.0`** — so it is OFF on hopper too, and hopper's
> ramps are hard clips like everyone else's. Measured on hopper at $K=256$:
> `velocity_fulfillment` sits at the clip **95.7%** of the time and `orientation_fulfillment`
> 14.0%. Hopper was the *most* exposed headline robot, not the exception.
>
> `objective.atom_floor` now clamps every atom to $[\varepsilon,1]$ before any power mean, in
> the numpy, JAX and cost-GD paths and for **both** $p$ settings — the symmetry matters because
> a floor is needed only at $p<0$ (at $p=1$, $\partial M_p/\partial x_i=w_i$, no singularity),
> so a floor applied only where it is numerically required would itself be an FPL-vs-linear
> confound. Raising $1e{-8}\to1e{-3}$ moves FPL scores by mean $|\Delta S|=9.62$ and the linear
> arm by $0.0004$: a ~24,000× asymmetry from an identical clamp.
>
> Default stays at the incumbent $1e{-8}$ so pre-existing configs and provenance records keep
> their meaning. GATE G1 (hopper, 16 seeds): the FPL-minus-best-linear gap moves −5% to −6% and
> the separation holds — see `FPL_FINDINGS.md`. **Still open:** walker, quadruped and cube have
> not been re-measured, and the $S$-distribution bimodality argument above should be re-derived
> at $\varepsilon=1e{-3}$ ($S$ saturates at 6.9, not 17.32).

---

## 9. The Jensen gap on the mean update (a limit, not a fix)

The Minimum Fulfillment Bound applies **per sampled rollout**. The executed control is
$\sum_k w_k\Theta^k$, and $\Theta\mapsto u(\Theta)$ is neither concave nor unimodal, so

$$u\Big(\textstyle\sum_k w_k\Theta^k\Big) \;\;\text{can be} \ll\; \sum_k w_k\,u(\Theta^k).$$

Averaging "step left" and "step right" gives "fall over." The handoff already flags this as
"a strong bias, not an exact guarantee"; the sharper statement is that the loss is exactly the
**Jensen gap of $u$ under $\tilde q^*$**, and that FPL makes it *worse* than linear does: the $p<0$
conjunction saturates at $u\approx1$ over a wide set of $\Theta$ (§5a), so the tilted posterior is
broad and plateau-shaped rather than peaked, and its mean is correspondingly less representative.

`PredictiveSampling`'s greedy argmax has zero Jensen gap by construction — which makes the
argmax-vs-softmax ablation a **direct measurement** of this term rather than just a baseline.

---

## 10. The FPL objective decomposes exactly for DDP (S9 / WO-3.2, GATE G7: PASS)

None of the FPL scores is a sum of stage costs, which is the form iLQR/DDP needs. But every
power mean is **quasi-arithmetic** — $M_p(m_{0:H}) = g^{-1}\!\big(\sum_t \hat w_t\, g(m_t)\big)$
with generator $g(m)=m^p$ (or $\log m$ at $p{=}0$) — so carrying the running sum
$z_t = \sum_{s<t}\hat w_s\, g(m_s)$ as one extra state makes the accumulation **additive** and
pushes the entire nonlinearity into a terminal readout. This is an exact rewrite, not an
approximation. Implemented in `analytic_mppi/controllers/accumulator.py`; augmented dynamics
are block-triangular ($\partial x'/\partial z = 0$, asserted by autodiff through a real
`mjx.step` in `tests/test_accumulator.py`), so the iLQR backward pass on the physics block
is unchanged in size.

With the $-\log(\text{reward})$ cost bridge this codebase uses (NOT the $-z^{1/p}$
reward convention in NEXT_STEPS' S9 table — implementing that table literally scores a
different objective), the readouts are:

| mode | accumulator increment | terminal readout $J(z_H)$ |
|---|---|---|
| `fpl_cost`, `time_p=None` | $\hat w_t\, m_t$ | $-\log z_H$ |
| `fpl_cost`, `time_p=q\neq 0` | $\hat w_t\, m_t^{\,q}$ | $-\tfrac1q\log z_H$ |
| `fpl_cost`, `time_p=0` | $\hat w_t \log m_t$ | $-z_H$ **(exactly linear)** |
| `fpl_discounted` | $\hat w_t^{(j)} f_j(t)$, one slot per atom | $-\log M_p(z_H)$ |
| `fpl_layered` | same (per grouped atom) | $-\log M_p^{\text{outer}}(M^{\text{inner}}_{p_g}(z_H))$ |

($m_t$ is the per-step atom-axis composite; $\hat w_t$ the normalized time weights of
§WO-3.3, i.e. uniform for the published unweighted soft-min.) Two consequences:

- **At `time_p = 0` the FPL objective is already additive** — generator $\log$ and bridge
  $-\log$ cancel, leaving no terminal nonlinearity at all. That setting is the natural
  first target if iLQR's terminal handling ever becomes the suspect.
- **For `fpl_discounted`, $z_H$ *is* `traj.reward_terms`** — the per-objective vector the
  multi-objective controllers already consume. The augmentation exposes an existing
  quantity rather than inventing one.

GATE G7 (1e-6 parity vs the numpy scorers): **PASS** — `tests/test_jax_costs.py` gates
`normal`, all three `time_p` branches of `fpl_cost`, and `fpl_discounted` in JAX on real
hopper *and* walker atom surfaces; `tests/test_accumulator.py` additionally gates
`fpl_layered` and `hybrid` against the numpy scorer on `g1_standup` — the only tasks where
they are reachable (hopper/walker define neither `fpl_groups` nor `floor_term_indices`,
and `config.JAX_COST_MODES` already excludes both from the mjx path). The conjunction-split
collapse (`fpl_conj_indices`) rides through unchanged, since the decomposition touches only
the time axis.

---

## Summary

**Sound:** the knot-space parameterization (§2); the $-\log u$ score, its sign, min-shift and
positivity (§4).

**Sound but limits your claims:** the missing likelihood-ratio term means this is a mean-shift hill
climb, not Williams' free-energy optimizer — don't claim the bound (§1). It does not confound the
FPL-vs-linear contrast.

**Explains your results:** FPL $=$ uniform linear scalarization $+\ \frac{1-p}{2}\mathrm{Var}(e)$
(§5). The wins-under-competition / nulls-when-safe pattern is forced by this expansion, and the
same mechanism appears as ESS collapse to uniform in the easy regime (§7).

**Needs fixing before the next round of results:**
1. ~~`soft_ramp` on walker / quadruped / cube~~ — **addressed** by `objective.atom_floor`
   (WO-3.4), which floors every atom in every path and both arms; it applies to hopper too,
   which turned out to be the most exposed robot of the four (§8b). Remaining: re-measure
   walker / quadruped / cube, which G1 did not cover.
2. `fpl_time_p` set (not `None`), or move off `use_fpl_discounted` — **still required.**
   `objective.terminal_value` (WO-3.3) closes the *truncation* half of §8a and makes
   `time_p<0` fall-time invariant, but the *order-of-operations* half is Jensen and cannot be
   fixed by any time weighting, so `fpl_discounted` stays structurally optimistic (§8a).
3. ESS-targeted $\lambda$, or ESS reported per $(p,\lambda)$ — otherwise $p$-sweeps are confounded
   with greediness sweeps precisely in the competitive regime (§6).
4. Log knot-saturation fraction per controller to rule out clipping asymmetry (§3).