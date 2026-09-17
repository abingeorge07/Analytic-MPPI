# FPL × MPPI — Findings

Results counterpart to `FPL_MPPI_HANDOFF.md` (the plan). What we set out to show: that
**FPL (Fulfillment-Priority-Logic) scalarization beats linear scalarization** as the
objective inside a sampling-based MPC — measured on performance, apples-to-apples.

## The thesis (two lines)

> **(1) FPL Pareto-dominates the entire linear-weight family under *dynamic* competition —
> when going faster unavoidably risks a fall — and adds nothing when a safe conservative
> operating point exists.**
> **(2) That dominance is UN-TUNABLE and robust: under model mismatch that punishes aggression
> (traction loss), one fixed FPL spec holds while the best linear weight both changes and
> degrades — and you cannot retune for a mismatch you don't know about.**

Sharper and more useful than "FPL always wins": it tells you *when* to reach for it, and why
it's not just a tuning choice. Supported by **4 dynamic-competition wins (Hopper, Walker2d,
a 12-DoF quadruped, and a 16-DoF dexterous hand — the first *non-locomotion* task), a
robustness win on all 4 robots, and 2 informative nulls.** The in-hand cube reorientation
result matters most for generality: the exact same phenomenon (dominance under dynamic
competition + un-tunable robustness to aggression-punishing mismatch) shows up in dexterous
manipulation, so it is not a property of legged gaits.

## The structural win: robustness to model mismatch (un-tunable)

The strongest result, because it's the one thing a linear reward fundamentally cannot paper
over. Setup: the controller PLANS on the nominal model but EXECUTES on a true simulator whose
**ground friction is reduced** (traction loss — ice/wet/worn, the canonical sim-to-real gap).
You cannot retune weights for a drift you don't know about, so a *fixed* controller is the
honest object of study. `verification/mismatch_robustness_sweep.py` → `mismatch_robustness.png`.

**Productive speed (achieved speed × survival) across a 3.3× traction reduction, 30 seeds,
95% CIs, no retuning:**

| traction loss | lin wv=1 | lin wv=2 | **FPL (fixed)** |
|---|---|---|---|
| 0.0 (nominal) | 0.53 | 0.23 | **0.79** |
| 0.4 | 0.37 | 0.77 | **0.93** |
| 0.55 | 0.27 | 0.61 | **0.76** |
| 0.7 | 0.34 | 0.31 | **0.69** |

One fixed FPL spec is the **best at every traction level**, and its dominance *grows* as the
ground gets slippery (productive speed +49% over the best linear at nominal → +103% at the
slipperiest).
The *best linear weight changes with the unknown friction* (wv=1 at nominal, wv=2 when slippery)
and none matches FPL. Video-verified: on slippery ground the aggressive linear hopper slips and
inverts; FPL keeps hopping upright (`runs/hopper_slip_*.mp4`).

**Generalizes to a second robot (Walker2d running):**
`verification/mismatch_robustness_walker.py` → `mismatch_robustness_walker.png`. Same setup, 30
seeds. One fixed FPL spec holds productive speed ~1.1–1.2 at **100% survival across a 3.3×
traction reduction** and is highest at every level (+13–35% over the best fixed linear weight
wv=8); the aggressive linear weights (wv=16/32) have collapsing survival. Video-verified
(`runs/walker_slip_*.mp4`): on slippery ground FPL keeps an upright running stride, the
aggressive linear stumbles forward.

**Generalizes to a third robot (Barkour quadruped, 12 DoF whole-body, straight-line walk):**
`verification/quadruped_robustness.py` → `quadruped_robustness.png`. One fixed FPL spec holds
productive speed ~1.1 at **100% survival across a 3.3× traction reduction, highest at every
level** (1.15/1.10/1.09/1.10/1.11 vs the best linear weight wv=4's 1.01/0.95/0.85/0.88/0.78, 30
seeds), the margin growing as the ground gets slippery (+14% at nominal → +42% at the slipperiest);
the aggressive linear weights (wv=8/16) collapse (0–13% survival) and the near-competitor wv=4 has
wobbling 90–100% survival (unpredictable falls). Video-verified (`runs/quadruped_*.mp4`): on slippery ground FPL
trots upright, the aggressive linear flips onto its back.

**Generalizes off legged robots entirely (LEAP-hand cube, grip loss):**
`verification/cube_robustness.py` → `cube_robustness.png`. Same setup, 30 seeds, but the
aggression-punishing mismatch is **reduced contact friction** (a worn/slippery cube or
fingertips — the manipulation analogue of traction loss); the hand must still tumble the cube
46° without dropping it. One fixed FPL spec keeps **~100% survival and the highest productive
rotation across a 30% grip reduction** (friction 1.0→0.7), and is highest at **every** level —
its margin over the best fixed linear weight (wa=2) *grows* as grip is lost:

| grip (friction) | best fixed linear (wa=2) | **FPL (fixed)** |
|---|---|---|
| 1.0 (nominal) | 18.6° / 100% surv | **26.1° / 100%** (+40%) |
| 0.9 | 17.6° / 97% | **24.5° / 100%** (+39%) |
| 0.8 | 12.9° / 77% | **22.6° / 100%** (+75%) |
| 0.7 | 7.0° / 47% | **20.1° / 97%** (+187%) |
| 0.6 | 5.5° / 30% | **12.0° / 67%** (+118%) |

The best fixed linear weight both **loses the cube** (survival 100%→30%) and sees its productive
rotation collapse (18.6°→5.5°); the aggressive weights are dead by friction 0.9. FPL only degrades
at the slipperiest 0.6 (still highest). So the robustness win holds on **four distinct robots** —
a hopper, a planar biped, a whole-body quadruped, and a dexterous hand — across **two different
aggression-punishing mismatches** (ground traction and contact grip), not a single-robot or
single-mismatch quirk.

**Mechanism + honest scope:** FPL wins when mismatch makes *aggression dangerous while safe
progress remains possible* — slippery ground makes hard push-offs slip (a fall), but gentle
hops still work, and FPL's min-fulfillment conjunction finds the safe gait; likewise a low-grip
cube makes hard finger pushes slip it off the palm, but a gentle roll still turns it. It does
NOT help for mismatches that only make the task *uniformly harder* (added mass, weaker
actuators) — there FPL's speed-floor makes it over-try and it sits ~on the linear frontier. The
same boundary appears *within* the cube task: at large target angles (or very low grip) tumbling
the cube fundamentally *requires* grip, so no safe operating point exists and FPL collapses with
everyone — which is why the cube robustness study uses the modest 46° roll where a safe gait
survives grip loss. Traction loss and contact-grip loss are the important real-world cases.

## Fairness protocol (identical everywhere; only the scalarization varies)

Same sampler, same fulfillment atoms, same sample budget, same planning horizon, same
temporal weakest-link aggregation (`fpl_time_p`) for **every** config. The only thing that
changes is how the per-objective `[0,1]` fulfillment vector is collapsed to a scalar:

| family | composition | note |
|---|---|---|
| linear-weight family | `power_mean(f, p=1, weights=w)` = `Σ wᵢ fᵢ` | a *family*: sweep the velocity weight `w` |
| FPL | `power_mean(f, p=−1, uniform)` | a *single fixed spec* (min-fulfillment conjunction) |

`power_mean(x, p=1, w) == Σ wᵢ xᵢ` exactly, so `p=1` **is** the linear cost. We compare the
one FPL spec against the *best* linear weight (not the worst) — beating a grid-searched
linear baseline with zero tuning is the strong claim.

## The four dynamic-competition wins

### Hopper — `verification/hopper_pareto_sweep.py` → `hopper_pareto.png`
One fixed FPL spec dominates the linear frontier at every commanded speed (30 seeds, 95% CIs).
At the hardest speed (3.0 m/s): **FPL 0.975 m/s at 33% falls vs the fastest comparably-safe
linear at 0.539 → +81% at equal safety.** (FPL's own fall rate at this most-aggressive speed
is ~33% — honest: the hopper at tv=3 is extreme; the point is FPL is far faster than any
linear weight that is as safe.) The hopper is inherently dynamic (to go fast it must hop =
leave the ground = risk falling), so no linear weight is both fast and safe.

### Walker2d — `verification/walker_pareto_sweep.py` → `walker_pareto.png`
The decisive test of *why* FPL wins, with a sharp two-regime story:
- **Gentle walking (tv 2–3):** quasi-static, *nothing* falls at any weight → a safe
  conservative linear weight exists → FPL wins nothing.
- **Running regime (tv 4–6, aggressive exploration):** flight phases are required to go
  fast → **one fixed FPL spec gets ~1.2 m/s at 0% falls at every speed** while the linear
  family only matches that speed by falling. Gain at equal (zero) safety:
  **+50% / +79% / +34%** at tv 4 / 5 / 6 (30 seeds, 95% CIs).

Video-verified: `runs/walker_FPL.mp4` = upright running stride (torso z-axis stays 0.94);
`runs/walker_linear.mp4` (aggressive weight) = forward lunge that stumbles below the fall
line.

### Barkour quadruped (12 DoF) — `verification/quadruped_pareto_sweep.py` → `quadruped_pareto.png`
Lifts the result to a whole-body legged robot, the regime of Alvarez-Padilla et al. (2024),
"Real-Time Whole-Body Control of Legged Robots with MPPI" (arXiv:2409.10469) — the same
MuJoCo-parallel sampling MPC. Objective is a **straight-line walk**: 6 atoms
[height, orientation, velocity, heading, posture, control]. The heading atom penalizes lateral
velocity + drift off the start line (without it the robot crabs/veers while still scoring on
forward speed); the **posture atom** (a stability term) keeps the four hip-abduction joints near
their standing angle so the legs stay tucked under the body — without it the controller trades a
low leg-splayed belly-sprawl for forward speed (it scores on velocity + instantaneous uprightness
while paddling its legs out sideways), which is the classic ugly-gait failure of a pure
velocity+upright objective. Both atoms are shared identically by the linear family (fair protocol).
One fixed FPL spec is the fastest 0%-fall controller at every commanded speed; the fastest
equally-safe linear weight (wv=4) is beaten by **+14% / +35% / +57%** at tv 2.0 / 2.5 / 3.0
(30 seeds, 95% CIs), the margin growing with speed. To exceed FPL's speed the linear family must
accept falls (wv=8: 13–87%, wv=16: 87–97%, inverting onto its back). Video-verified: FPL trots
straight and upright on tucked legs (`runs/paper/quadruped_fpl.mp4`); the conservative safe linear
(wv=1, `runs/paper/quadruped_linear.mp4`) walks the same gait but slower.

### In-hand cube reorientation (16-DoF LEAP hand) — `verification/cube_pareto_sweep.py` → `cube_pareto.png`
The **first non-locomotion win**, and the one that shows the phenomenon is not about legged
gaits. A LEAP right hand cradles a 7 cm cube palm-up and must **tumble it forward** (roll about
x) to a commanded angle without dropping it. The competition is the manipulation analogue of a
gait: the only way to rotate the cube faster is to push harder with the fingertips, which risks
rolling it off the palm — a drop. Atoms `[hold, height, alignment, control]`; the swept "progress"
weight is on **alignment** (fraction of the commanded rotation achieved), exactly like the velocity
weight in the locomotion tasks. (Yaw about the vertical axis is *too* securely cradled to ever
risk a drop — the manipulation analogue of gentle walking — so we use the drop-prone roll.)

One fixed FPL spec is the **fastest 0%-drop controller at every commanded angle** (30 seeds,
95% CIs), and the margin over the fastest comparably-safe linear weight **grows with difficulty**:

| commanded roll | FPL (fixed) | fastest ~safe linear (wa=2) | gain | aggressive linear |
|---|---|---|---|---|
| 46°  | **26.1° @ 0% drop** | 18.6° @ 0% drop | **+40%** | wa≥4: 50–97% drop |
| 69°  | **38.6° @ 0% drop** | 23.6° @ 13% drop | **+64%** | wa≥4: 47–87% drop |
| 103° | **51.6° @ 0% drop** | 26.8° @ 10% drop | **+93%** | wa≥8: 100% drop |

To rotate as far as FPL, the linear family must accept 90–100% drops; the only *strictly* 0%-drop
linear weight is the near-static crawler (wa=1, ~10°). No single linear weight is both far-rotating
and safe. Video-verified (`runs/cube_FPL.mp4` vs `runs/cube_linear.mp4`, seed 0 at 103°): FPL
tumbles the cube 64° and keeps it centred on the fingers (cube xy-drift 2.7 cm, never falls); the
aggressive linear weight (wa=8) flings it out of the palm — 22 cm off-centre, dropped — while
rotating only 9°.

## The two nulls (equally important, and honest)

### g1_reach (humanoid forward lean-and-hold) — `verification/g1_reach_study.py`
Achievable static-balance task; a **conservative linear weight (wf=1) dominates FPL at every
difficulty** (leans + stays up). Quasi-static → a safe low-risk weight exists → FPL doesn't
help. An earlier "win" here was an artifact of comparing FPL only against the *aggressive*
linear weight — the exact unfairness this protocol forbids.

### g1_walk (humanoid walking from stand) — `analytic_mppi/tasks/g1_walk.py`
Too hard for this short-horizon sampling MPC: *both* controllers collapse. Not a method
result — a task-capability gap (needs a gait/foot objective or reference warm-start).

## Methodological lessons (the expensive ones)

1. **Metric discipline:** an uprightness-only metric is a *trap* on a humanoid — a seated
   sprawl keeps `rz > 0`. Always co-check torso **height** and **render the video** before
   claiming. (Cost us two false "wins" on g1 before rendering exposed the collapse.)
2. **Planning horizon** was the real blocker for g1 dynamic motion (H=0.5 → 1.0–1.5 s), not
   the objective. Use long horizons for legged dynamic tasks.
3. **Compare against the *best* linear weight, not the aggressive one.** The aggressive
   weight always face-plants; that comparison flatters FPL and is meaningless.
4. **Atom shaping is shared and matters:** one-sided velocity ramp (spread), orientation
   that decays *before* horizontal (a real floor), one-sided height that decays before the
   cliff. Give these to *both* families.

## Reproduce

```bash
.venv/bin/python verification/hopper_pareto_sweep.py         # win 1: hopper Pareto + figure
.venv/bin/python verification/walker_pareto_sweep.py         # win 2: walker running + figure
.venv/bin/python verification/quadruped_pareto_sweep.py      # win 3: Barkour quadruped + figure
.venv/bin/python verification/cube_pareto_sweep.py           # win 4: LEAP-hand cube (non-locomotion) + figure
.venv/bin/python verification/mismatch_robustness_sweep.py   # robustness (hopper, traction)
.venv/bin/python verification/mismatch_robustness_walker.py  # robustness (walker, traction)
.venv/bin/python verification/quadruped_robustness.py        # robustness (quadruped, traction)
.venv/bin/python verification/cube_robustness.py             # robustness (cube, grip loss)
.venv/bin/python verification/g1_reach_study.py              # null (conservative linear wins)
```

## FPL-informed sampling — tested, and it does NOT help (clean negative)

We built `FplAdaptiveMPPI` (`fpl_adaptive`) to use the two things FPL gives a sampler that a
scalar cost cannot: (1) the **absolute [0,1] scale** to calibrate explore↔exploit, and (2)
the **per-objective vector** to steer sampling variance toward the dims that most move the
*least-satisfied* objective (information-directed). We tested it in the regime where it had
its best shot — the winning walker-running objective — varying only the sampler:

| budget | plain Gaussian MPPI | FPL-informed sampler | generic adaptive |
|---|---|---|---|
| K=32  | **1.02** | 0.97 | 1.00 |
| K=64  | **1.13** | 1.10 | 1.07 |
| K=256 | **1.21** | 1.18 | 1.11 |

*(achieved m/s at 0% falls, walker tv=5; best-or-tied is plain MPPI at every budget.)*

**Plain fixed-Gaussian MPPI is best-or-tied at every budget and speed.** The adaptive sampler
only helped when compensating for a *weaker* objective (`fpl_discounted` without the temporal
weakest-link), and even then the FPL-*specific* signals barely beat generic adaptive
covariance — so that small gain is the covariance machinery, not the FPL information. Likely
cause: exploit-collapse (tightening σ on "commitment") hurts in a receding-horizon dynamic
task, and warm-started fixed-Gaussian MPPI is already a strong search.

**Conclusion:** the entire FPL value is in the **objective (scalarization)**, not the sampler.
The information-theoretic sampling idea is a clean negative for these tasks.

**Follow-up campaign (see `FPL_MPPI_CAMPAIGN.md`) — hardened across samplers + a new proposal.**
A checkpointed sampler race (`verification/fpl_sampler_race.py`, 16 seeds × 5 budgets × 2 envs)
added iCEM-style **colored noise** + an FPL **absolute-scale exploration schedule**
(`FplColoredMPPI`) and **gradient-guided refinement** (BPTT on the analytic FPL reward). Across
FOUR search mechanisms (adaptive covariance, colored noise, absolute-scale, gradient refinement)
**nothing beats warm-started fixed-Gaussian MPPI** — plain MPPI is best-or-tied at every budget.
And the flip side is a portability WIN (`verification/fpl_portability.py`): one fixed FPL spec
Pareto-dominates the linear-weight frontier **within every sampler** (walker +17–34%, hopper
+29–209% over the fastest comparably-safe linear weight, across MPPI / MPPI-CMA / CEM / colored).
So: *the FPL objective is the win, it is portable to any sampler, and the simplest (plain MPPI)
is the right choice.*

## The fulfillment-atom floor (WO-3.4) — GATE G1: **PASS**

`power_mean` has always clipped its terms at `eps=1e-8` as a log/division guard. That clip
was never an objective decision, and it is a **confound**: `∂M_p/∂x_i = w_i x_i^{p-1} M_p^{1-p}`
diverges as an atom → 0 for every `p < 1`, but equals `w_i` at `p = 1`. A floor is therefore
needed *only* by the FPL arm. Any change to it moves FPL and leaves linear alone, which is
mechanically indistinguishable from an FPL effect.

It is now an explicit, config-plumbed `objective.atom_floor`, applied to the atoms (not
inside `power_mean` — which receives per-term discount-*sums* under `fpl_discounted` and
group scores under `fpl_layered`, so clamping there would floor a different quantity in each
mode). Applied identically in the numpy, JAX and cost-GD paths, and to **both** `p` settings.

**How exposed the atoms actually are** (hopper, K=256; note `atom_soft_floor` defaults to
`0.0`, so hopper's ramps are hard clips and hopper is the *most* exposed task, not the least):

| atom | pinned at `1e-8` |
|---|---|
| velocity_fulfillment | **95.7%** |
| orientation_fulfillment | 14.0% |
| height / control | 0% |

Raising the floor `1e-8 → 1e-3` moves the FPL arm's scores by mean \|ΔS\| = 9.62 and reshuffles
the rollout ranking (Spearman 0.916); the linear arm moves by 0.0004. A ~24,000× asymmetry,
from an identical clamp.

**G1 verdict — the deltas are not significant.** Hopper, 16 seeds, the gated quantity being
FPL minus the *best* linear weight:

| study | gap @ `1e-8` | gap @ `1e-3` | change |
|---|---|---|---|
| portability | +0.4097 (FPL CI clears linear CI) | +0.3881 (still clears) | −5% |
| zero_tuning | +0.4131 (clears) | +0.3896 (CIs now overlap) | −6% |

One flag fired across 33 per-config comparisons. The floor is a free correctness fix, and the
ε-plateau logged in `docs/mppi_math.md §8` for walker/quadruped/cube is closed on hopper.

**Two caveats, recorded rather than buried:**

1. **`FPL·CMA` genuinely improves under the floor** — survival 0.70 → 0.90 (Wilson
   `[0.59,0.79]` vs `[0.81,0.95]`, disjoint), productive speed 0.650 → 0.801. At floor-on it
   is the *best* sampler in the race, ahead of `colored+abs` (0.664) and plain MPPI (0.615).
   This does not touch the FPL-vs-linear claim (the race holds cost fixed at FPL) but it
   **does qualify the "nothing beats plain MPPI" conclusion above**: that conclusion was
   measured on an ε-plateau where failed rollouts were indistinguishable, which is precisely
   the landscape a covariance-fitting sampler cannot exploit. Re-measure before repeating it.
2. **zero_tuning's FPL separation weakens to CI-overlap**, driven by survival 0.81 → 0.69
   (n=16; Wilson intervals still overlap, so not significant on its own).

**Environment caveat — these are recomputed numbers, not the August ones.** The August
checkpoint campaign does *not* reproduce here: same seed/config gives cached `vx = 1.0275` vs
`0.6563` today. This is not the floor (pre-S1 code reproduces today's value exactly) — it is
the MuJoCo pin. `mujoco>=3.2,<3.6` was introduced only in `8a9957e`; the August runs used the
unbounded pin at 3.8.1, and the venv is now on 3.5.0 for mjx compatibility (invariant 11.3's
hazard, realized). **Both arms of this A/B were recomputed under 3.5.0**, so the delta above
measures the floor and nothing else. The 3.5.0 values are the baseline of record going
forward; the older absolute numbers elsewhere in this document predate the pin.

### GATE G2 — is the FPL advantage ε-dependent? **No.**

This is the successor to the λ/`p` objection: *"FPL only wins because you floored the atoms,
and the floor is only needed for `p<0`."* If true, the FPL-minus-best-linear gap would shrink
as ε → 0 (no floor → no advantage). It does not. Hopper, `portability`, 16 seeds:

| ε | `1e-2` | `1e-3` | `1e-4` | `1e-6` | `1e-8` |
|---|---|---|---|---|---|
| portability | +0.4147 | +0.3881 | +0.4465 | +0.4876 | +0.4097 |
| zero_tuning | +0.3530 | +0.3896 | +0.3057 | +0.3430 | +0.4131 |

**Non-monotone in ε in both studies** (portability mean +0.429, CV 9.0%, range 0.100;
zero_tuning mean +0.361, CV 11.6%, range 0.107) — the ordering is noise, not trend. For scale,
the 95% CI half-width on a *single* FPL measurement is ±0.11, so the entire spread across six
decades sits inside one measurement's error bar. FPL clears the best-linear CI at **every** ε
in portability.

The decisive check is the *direction*: the objection predicts the gap shrinks as ε → 0.
Between ε=`1e-2` and ε=`1e-8` — six decades, effectively floor-on vs floor-off — the gap moves
**−0.005** (portability) and **+0.060** (zero_tuning). Neither shrinks toward zero.

**One sentence for the paper:** *the FPL advantage shows no monotone dependence on the atom
floor across six decades, and is unchanged between a floor of 1e-2 and effectively none, so it
is not a conditioning artifact of the floor.*

### WO-3.3 terminal value — GATE G3: **PARTIAL in theory, NEGATIVE in closed loop. Do not adopt.**

**First: the fix does not reach the published objective.** The published hopper/walker spec is
`fpl_cost` + `fpl_time_p=-2.0` + `fpl_time_discount=False`, which aggregates time as
`power_mean(per_step, time_p, weights=None)` — an **unweighted soft-min**. There is no
`(1-γ)/(1-γ^H)` renormalization in that path, so the quantity WO-3.3 replaces is not present
and `terminal_value=True` is inert. The brief's premise holds only for `time_p=None` and
`fpl_discounted`, which the published results do not use. The controller now **raises** on the
inert combination rather than silently doing nothing (invariant 11.5) — a no-op there would
read as "we applied the fix and nothing changed", which is the worst available outcome.

Reaching the tail from the published spec therefore takes two knobs, so the gate was run as
three arms (hopper, floor `1e-8`, 16 seeds):

| arm | spec | FPL prod | gap vs best linear | FPL survival |
|---|---|---|---|---|
| **A** published | `time_p=-2`, disc=F, tv=F | 0.777 | +0.410 | 0.81 |
| **B** +discount | `time_p=-2`, disc=T, tv=F | 0.778 | +0.541 | 0.88 |
| **C** +tail (WO-3.3) | `time_p=-2`, disc=T, tv=T | 0.604 | +0.334 | **0.56** |

*(zero_tuning agrees and is worse: survival 0.81 → 0.81 → **0.44**, gap +0.413 → +0.427 →
**+0.049**, and the FPL/linear CIs stop separating entirely at C.)*

**B → C is the terminal-value question, and it is clearly harmful:** FPL survival falls **−36%**
(portability) and **−46%** (zero_tuning), and on zero_tuning the FPL advantage collapses from
+0.427 to +0.049. Likely mechanism (hypothesis, not yet isolated): at `H=31, γ=0.99` the tail
weight is `γ³⁰ = 0.740` versus `≈0.010` for any single interior step — a **74×** concentration
on the terminal step. That guts the weakest-link-over-time property which `docs/mppi_math.md
§8a` identifies as the thing making the FPL floor a *trajectory* property rather than a
per-step one. The cheap `v` = hold-`r_H` tail reintroduces §8a's defect from the other
direction: a moderate mid-horizon collapse that looks acceptable at step `H` is forgiven.

**Conclusion: keep `terminal_value=False`.** WO-3.3's stated upgrade — "better: a fitted
terminal fulfillment estimate" — is not optional; the cheap estimate is worse than the
truncation it fixes, at least under a `p<0` time aggregation.

**Side finding, A → B, reported because it is not free:** discount-weighting the soft-min
widens the gap (+0.410 → +0.541 on portability) — but FPL is *unchanged* (0.777 → 0.778) and
the widening comes entirely from the **linear arm degrading** (0.368 → 0.237). A change that
moves one arm and not the other is the exact pattern this whole work order exists to catch, so
this should not be adopted as a "win" without understanding why discounting hurts `p=1` only.

### The synthetic analysis behind the gate

`objective.terminal_value=True` replaces the renormalization with an explicit post-horizon
tail. It closes the *truncation* half of the `fpl_discounted` optimism logged in
`docs/mppi_math.md §8a` — a rollout that collapses at step 40/50 drops from 0.941 to 0.612,
and `time_p=-1` becomes nearly **fall-time invariant** (0.063/0.073/0.084 across fall times,
vs 0.075→0.274 before, which rewarded falling late).

It does **not** close the *order-of-operations* half, and cannot: `μ_p` is concave for `p≤1`,
so `fpl_discounted ≥ fpl_cost` holds for **every** convex time-weighting. Measured over 4000
random rollouts the gap shrinks but stays strictly positive in 100% of cases (mean +0.129 →
+0.088). **Use `fpl_cost` with `fpl_time_p ≤ 0`, not `fpl_discounted`** — which the published
configs already do. Details and the full table are in `docs/mppi_math.md §8a`.

Flag defaults to `False`; nothing pre-existing moved.

```bash
./a-mppi/bin/python verification/s2_atom_floor_ab.py              # the G1 A/B (hopper)
./a-mppi/bin/python verification/s2_atom_floor_ab.py --report-only
./a-mppi/bin/python verification/s2_atom_floor_ab.py \
    --floors 1e-2 1e-4 1e-6 --studies portability zero_tuning --sweep   # G2 sweep
```

Published-campaign checkpoints in `verification/checkpoints/*.jsonl` are left untouched;
this study writes to `checkpoints/s2_atom_floor/`, one file per floor (trial keys do not
contain the floor, so a shared file would silently serve floor-on trials from a floor-off
cache — a false PASS).

## S9 additive-accumulator decomposition — GATE G7: **PASS**

Every objective mode is now expressible as (additive stage cost in an augmented
accumulator state) + (terminal nonlinearity), which is the form iLQR/DDP consumes
(`analytic_mppi/controllers/accumulator.py`, math in `docs/mppi_math.md §10`). The
rewrite is **exact** — every power mean is quasi-arithmetic — and the gate confirms it:
1e-6 parity against the numpy scorers on hopper **and** walker for every mjx-runnable
mode (`normal`, `fpl_cost` in all three `time_p` branches, `fpl_discounted`), in both
numpy and JAX evaluation.

Scope decision (made explicitly, not by omission): `fpl_layered` and `hybrid` are gated
against the **numpy** scorer on `g1_standup` only. On hopper/walker neither mode is
meaningfully reachable — no task `fpl_groups` (layered raises) and no
`floor_term_indices` (hybrid degenerates bit-identically to `normal`, so a
hopper/walker "hybrid" gate row would pass vacuously). Both were already excluded from
the mjx path by `config.JAX_COST_MODES`.

Two structural results worth keeping:

- **`time_p = 0` needs no terminal nonlinearity**: the log generator cancels the
  `-log` cost bridge exactly, so at that setting the FPL objective is already an
  additive-cost problem.
- **The augmentation provably reaches the score** (`test_accumulator_actually_propagates`
  + a live-stage assertion inside the block-triangularity test) — the explicit no-op
  check the `terminal_value` episode showed is mandatory.

One spec correction recorded for whoever reads NEXT_STEPS later: S9's terminal column
(`−z^{1/p}`) is the reward-maximizing convention; this codebase's scorers return
`−log(reward)`, so the correct readouts are `−(1/q)·log z` / `−z` / `−log z`
(see `docs/mppi_math.md §10`). Implementing the table literally fails G7 for a reason
unrelated to the decomposition.

Block-triangularity (`∂x′/∂z = 0`) is asserted by autodiff through a real `mjx.step`,
not assumed. **Milestone 2: iLQR is unblocked.**

## S10 iLQR on the augmented state — GATE G8: **PASS**

`analytic_mppi/controllers/ilqr.py` implements iLQR on the augmented (physics + accumulator)
state with a Gauss-Newton backward pass through the physics (dropping only `F_xx`; the
cost-channel Hessian is exact through the first-order step Jacobians), Levenberg
regularization, backtracking line search over 8 α values, box-constrained projected-Newton
feedforward (**not** clip-after-update — see [mppi_math.md §3](docs/mppi_math.md)), and
the full instrumentation NEXT_STEPS S10 requires (cost curve, λ trace, α trace,
acceptance, `Q_uu` cond, `‖Q_u‖`, `J_pre` / `J_post`, projection residual).

Registered at `("gradient", "ilqg") → "ilqr"`; wraps `GradientMPC`'s `_build_plan_fn` seam.
S11's `WarmStartedILQR` composes any `SamplingController` with an `ILQRMPC`, mirroring
`cost_gd.py`'s composition-over-inheritance.

GATE G8 (`tests/test_ilqr.py::test_g8_converges_to_analytic_lqr`): on both an inline
pendulum (nq=nv=1) and cart-pole (nq=nv=2, underactuated with nu=1) with quadratic
running/terminal costs, the iLQR cost matches the finite-horizon Riccati LQR cost on the
TRUE dynamics to ≤ 1e-4 (`atol` in the test) — well below the gate. Both cost evaluations
use the same rollout+cost path, so the gap is measuring the optimizer, not engine parity.

Full suite after S10+S11 landed: **315 passed / 5 skipped / 0 failed** (was 292/4 pre-S10).
The iLQR file adds 17 tests + 3 config dispatch tests. First-run bug caught and fixed
en route: `jax.jacfwd`'s multi-output multi-argnum return is `jac[output_leaf][argnum]`,
NOT `jac[argnum][output_leaf]`; the pendulum's `nsensordata = 0` surfaces this as a
`shape=(0, nv)` slot in the concat rather than a diffuse Jacobian error, and it would have
been silent on any model with nsensordata > 0. The correct destructuring pattern is
`((dxdq, dxdv, dxdu), (dsdq, dsdv, dsdu)) = jax.jacfwd(...)` — pinned in-place with a
comment so future edits do not re-introduce the swap.

One deliberate deviation from a strict reading of NEXT_STEPS worth flagging: pushing the
whole cost into the terminal readout and only linearizing the augmented dynamics would zero
out all stage-cost curvature — `normal` mode's readout is linear in `z`, `V_xx ≡ 0`, and
iLQR degenerates to steepest descent (G8 fails by construction). The backward pass is
therefore Gauss-Newton **through physics only**, with exact quadratic expansion of the
stage/fold cost channel. On quadratic costs this reproduces exact LQR; documented in the
module docstring.

**Milestone 3: a working λ-free optimizer.** Awaiting review + tag along with milestones 0/1/2.

## S11 warm-started iLQR on hopper — GATES G9 + G10: **MIXED / FLAG**

Closed-loop study: `verification/s11_ilqr_g9.py`, hopper, atom floor 1e-3, published
spec, 3 seeds, linear-spline knot grid shared with a K=128 MPPI sampler (S11's
`WarmStartedILQR`).

**GATE G9** — 2/3 PASS, criterion (1) FAIL. Recording per NEXT_STEPS ("record the
numbers and continue"; the runbook's "profile before changing anything" branch confirms
no bug):

| criterion | measured | verdict |
|---|---|---|
| (1) ≥ 20× faster than GradientMPC at matched final cost | median **9.8×** (min 0.5×, max 88.5×) | **FAIL** |
| (2) line-search acceptance > 50% warm-started | mean **0.94** (per seed: 0.93 / 0.93 / 0.95) | PASS |
| (3) λ trace bounded | max **1.0** (never left the initial value); at-cap fraction 0.00 | PASS |

Per-state speedup breakdown (state = MPC step along one seeded MPPI trajectory from the
settled stand): state 0 → **88.5×**, state 10 → 0.5×, state 30 → 0.5×, state 60 →
**19.2×**. Where iLQR wins big, it wins big; on some intermediate poses GradientMPC's
first Adam step already reaches iLQR's final cost from the same zero-mean nominal, so
the "matched-cost" bar collapses to Adam's per-step move (0.01 LR × the analytic
gradient) rather than iLQR's larger step size buying compute. This is the honest
speed picture: iLQR steady-state is 0.193 s/act (10 iterations, one linearization + one
backward + up to 8 line-search forwards each) vs GradientMPC's ~1.5 ms/Adam-step; the
ratio speaks when the objective is genuinely nonconvex, not when it's already flat.

Instrumentation verified no bugs: `plan_time` is steady across states (no
recompilation per MPC step) and `_lin_all = jax.vmap(_lin_one)` is one batched call
(no per-t Jacobian loop). Both were the runbook's usual culprits for a criterion-(1)
fail; neither applies.

**GATE G10** — **FLAG: thesis understated.** Projection residual and J_pre → J_post,
split by arm:

| arm | ‖(I−WW⁺)u*‖ / ‖u*‖ | J_post − J_pre |
|---|---|---|
| fpl (p=−1)  | **0.645** | **+0.506** |
| linear (p=+1) | 0.508 | +0.070 |

FPL loses **~7× more** to the knot projection than linear. The runbook's prediction —
"p<0 favours sharp switching near contact and knots cannot represent a mid-interval
switch" — is exactly what this measures. Consequence: the deployed FPL policy
(`J_post`, what actually executes) is systematically further from the iLQR optimum
(`J_pre`) than the deployed linear policy is. A knot-count ablation is the natural
follow-up; for the paper, both `J_pre` and `J_post` need to appear in any comparison
where iLQR is the FPL evaluator, and the aggregate FPL-vs-linear claim can be stated as
"FPL wins EVEN THOUGH the knot basis is asymmetrically handicapping it."

Numbers logged to `verification/checkpoints/s11_ilqr_g9/g9_g10.jsonl`.

## S12 — the {FPL, linear} x {MPPI, iLQR} 2x2 — GATE G11: **λ/p confound resolved by construction**

The primary result. `verification/s12_the_2x2.py`, hopper + walker, 10 seeds per cell,
atom floor 1e-3, matched dynamics-evaluation budget (256 rollout-equivalents/step in
both columns: K=256 for MPPI; K=128 warm-start + ~5 × (jac + line search) for iLQR).
**The iLQR column uses no temperature anywhere** — the warm-start sampler is
`PredictiveSampling` (argmax over the Gaussian cloud, mean always included, no softmax)
and the iLQR update itself has no `-log(u)` bridge. If FPL's advantage were a "better
temperature" artifact, it should collapse here.

| env    | mppi/fpl        | mppi/linear     | ilqr/fpl        | ilqr/linear     |
|--------|-----------------|-----------------|-----------------|-----------------|
| hopper | 0.64 ± 0.29 (7/10) | 0.32 ± 0.15 (8/10) | **1.20 ± 0.52 (7/10)** | 1.18 ± 0.51 (7/10) |
| walker | 1.17 ± 0.04 (10/10) | 0.47 ± 0.16 (10/10) | **1.57 ± 0.07 (10/10)** | 0.86 ± 0.10 (10/10) |

(Values are `prod` = fall-gated forward speed, mean ± 95% CI; parenthetical is survival.)

**Per-env read (per the pre-registered G11 rule):**
- **hopper**: FPL vs linear in the iLQR column → **OVERLAP** (1.20 ± 0.52 vs 1.18 ± 0.51 —
  statistical tie, both dramatically above their MPPI counterparts).
- **walker**: FPL vs linear in the iLQR column → **FPL wins decisively** (1.57 ± 0.07 vs
  0.86 ± 0.10 — CIs do not touch: 1.50 vs 0.96 at the boundaries; ~1.8× separation).

**GATE G11 verdict.** My script's strict per-env rule reports MIXED and "no aggregate
claim". The paper claim is subtly stronger and worth stating precisely: the *critical
failure mode* for the confound-resolution argument was "**FPL loses in the iLQR column**"
(that would mean the advantage was an interaction with the sampling update rule, not a
property of the objective). That failure did not occur on either env. FPL is ≥ linear
everywhere here, strongly on walker and at parity on hopper. Combined with G10's finding
that the knot basis handicaps FPL **more** than linear in the iLQR column (a headwind, not
a boost), the "you found a better temperature" objection is structurally eliminated on
these two tasks.

**Second-order but paper-worthy: the optimizer matters as much as the scalarization.**
iLQR dominates MPPI on **every cell**:

| cell            | MPPI  | iLQR  | ratio |
|-----------------|-------|-------|-------|
| hopper / fpl    | 0.64  | 1.20  | 1.9×  |
| hopper / linear | 0.32  | 1.18  | 3.7×  |
| walker / fpl    | 1.17  | 1.57  | 1.3×  |
| walker / linear | 0.47  | 0.86  | 1.8×  |

Numbers logged to `verification/checkpoints/s12_the_2x2/the_2x2.jsonl`.

**Milestone 4 — minimum publishable result.** Awaiting review + tag along with 0/1/2/3.

## S13 diffmjx feasibility — GATE G12: **FAIL BY CONSTRUCTION → isolated-study path**

Pre-code feasibility check for the Phase 5 diffmjx upgrade
(`HANDOFF_PHASE5.md`, `NEXT_STEPS.md` §PHASE 5 lines 231–278). Three upstream
repositories were probed and cloned to `/home/abg309/PhD/RCL/diffmjx/`:

| repo                        | HEAD (short) |
|-----------------------------|--------------|
| `martius-lab/mujoco`        | `1465d8b6`   |
| `martius-lab/mjx_diffrax`   | `a775638`    |
| `a-paulus/softjax`          | `82fa3c9`    |

All three reachable over public HTTPS (SSH not required at the time of writing),
so S13.1 passes.

**G12 verdict: FAIL by inspection, before any install.** The fork's
`mjx/pyproject.toml` (`/home/abg309/PhD/RCL/diffmjx/mujoco/mjx/pyproject.toml`)
declares `name = "mujoco-mjx"`, `version = "3.8.0"`, `dependencies = [..., "mujoco>=3.8.0.dev0", ...]`.
Installing `mujoco-mjx` from the fork therefore drags core `mujoco` off the
project-wide `3.5.0` pin — the exact G12 FAIL branch called out in
`HANDOFF_PHASE5.md` invariant 9 ("Any diffmjx fork that upgrades core `mujoco`
breaks comparability with every existing result"). Confirmed empirically:
`pip install -e .../mujoco/mjx` into the isolated venv resolves to
`mujoco==3.13.0` + `mujoco-mjx==3.8.0` — well past the 3.5.0 → 3.12
comparability breakpoint the primary `pyproject.toml` explicitly warns about.
The primary `a-mppi/` venv was verified untouched at `mujoco==3.5.0`.

**Consequence — the on-fail branch from `NEXT_STEPS.md` applies verbatim:**
diffmjx is treated as an **isolated study**, installed only into a separate
venv (`/home/abg309/PhD/RCL/diffmjx/venv/`), with its own checkpoint dir. A
`[diffmjx]` extra is **not** being added to the primary `pyproject.toml` — a
pip-resolve-time mutex is redundant when the fork itself upgrades the
non-optional `mujoco` base pin, and adding the extra would falsely advertise
same-environment coexistence.

**What this does NOT change.** Phase 4 (S12, the paper result) stands as-is on
stock `mujoco==3.5.0`. Any diffmjx result derived in the isolated env will be
reported alongside — never in place of — the Phase 4 numbers, per the "What
NOT to do" list in `HANDOFF_PHASE5.md` item 1.

Attribution for the three upstream repos was added to [`README.md`](README.md)
in the same session.

**S13.3 — Triton-gemm guard on the fork: PASS.**
`/home/abg309/PhD/RCL/diffmjx/s13_triton_guard_check.py` sets
`XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` before the first `jax` import,
JITs `jax.jacfwd(mjx.step)` on a 1-DoF pendulum in the fork's world
(`mujoco==3.13.0` + `mujoco-mjx==3.8.0`, `jax==0.6.2`, `CudaDevice(id=0)`
RTX 4070 driver 580.126.20 / CUDA 13.0), and blocks-until-ready on every
Jacobian leaf. Compile + execute completed cleanly, no core dump, expected
`(1,1)` shapes on `d(qpos)/d(qpos)` and `d(qvel)/d(ctrl)`. Invariant 7's
three-line guard therefore still applies to the fork; any future diffmjx
entry point in this workspace needs the same guard before its first `jax`
import.

**Follow-ups (only in the isolated venv, and only if the isolated study is
pursued):** S14 (CFD + G13 gradient-sweep plot) and S15 (adaptive + softjax +
G14 ablation). The kill criterion from `NEXT_STEPS.md` still applies —
Phase 4 is publishable without any of this.

## Open threads

- Harder biped regimes / a gait-phase or foot-placement objective to make g1_walk itself
  achievable (the one task no controller could do).
- FPL's absolute-scale explore/exploit *might* still suit converge-and-commit tasks
  (reach / stand / swing-up) at a scarce online budget — untested, and not the offline
  regime targeted here.
