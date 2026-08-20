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

## Open threads

- Harder biped regimes / a gait-phase or foot-placement objective to make g1_walk itself
  achievable (the one task no controller could do).
- FPL's absolute-scale explore/exploit *might* still suit converge-and-commit tasks
  (reach / stand / swing-up) at a scarce online budget — untested, and not the offline
  regime targeted here.
