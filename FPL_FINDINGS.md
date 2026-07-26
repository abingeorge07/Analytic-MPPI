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
it's not just a tuning choice. Supported by **2 dynamic-competition wins, 1 robustness win,
and 2 informative nulls.**

## The structural win: robustness to model mismatch (un-tunable)

The strongest result, because it's the one thing a linear reward fundamentally cannot paper
over. Setup: the controller PLANS on the nominal model but EXECUTES on a true simulator whose
**ground friction is reduced** (traction loss — ice/wet/worn, the canonical sim-to-real gap).
You cannot retune weights for a drift you don't know about, so a *fixed* controller is the
honest object of study. `verification/mismatch_robustness_sweep.py` → `mismatch_robustness.png`.

**Productive speed (achieved speed × survival) across a 3.3× traction reduction, 20 seeds, no
retuning:**

| traction loss | lin wv=1 | lin wv=2 | **FPL (fixed)** |
|---|---|---|---|
| 0.0 (nominal) | 0.62 | 0.35 | **0.73** |
| 0.4 | 0.37 | 0.81 | **0.94** |
| 0.55 | 0.15 | 0.62 | **0.80** |
| 0.7 | 0.33 | 0.46 | **0.62** |

One fixed FPL spec is the **best at every traction level**, and its dominance *grows* as the
ground gets slippery (fastest equally-safe linear: +18% at nominal → +126% at friction 0.3).
The *best linear weight changes with the unknown friction* (wv=1 at nominal, wv=2 when slippery)
and none matches FPL. Video-verified: on slippery ground the aggressive linear hopper slips and
inverts; FPL keeps hopping upright (`runs/hopper_slip_*.mp4`).

**Generalizes to a second robot (Walker2d running):**
`verification/mismatch_robustness_walker.py` → `mismatch_robustness_walker.png`. Same setup, 16
seeds. One fixed FPL spec holds productive speed ~1.1–1.2 at **100% survival across a 3.3×
traction reduction** and is highest at every level (+12–24% over the best fixed linear weight
wv=8); the aggressive linear weights (wv=16/32) have collapsing survival. Video-verified
(`runs/walker_slip_*.mp4`): on slippery ground FPL keeps an upright running stride, the
aggressive linear stumbles forward. So the robustness win is not a hopper quirk — it holds on
two distinct legged robots.

**Mechanism + honest scope:** FPL wins when mismatch makes *aggression dangerous while safe
progress remains possible* — slippery ground makes hard push-offs slip (a fall), but gentle
hops still work, and FPL's min-fulfillment conjunction finds the safe gait. It does NOT help
for mismatches that only make the task *uniformly harder* (added mass, weaker actuators) —
there FPL's speed-floor makes it over-try and it sits ~on the linear frontier. Traction loss
is the important real-world case.

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

## The two wins

### Hopper — `verification/hopper_pareto_sweep.py` → `hopper_pareto.png`
One fixed FPL spec dominates the linear frontier at every commanded speed. At the hardest
speed (3.0 m/s): **FPL 0.925 m/s at 19% falls vs the fastest comparably-safe linear at
0.554 → +67% at equal safety.** The hopper is inherently dynamic (to go fast it must hop =
leave the ground = risk falling), so no linear weight is both fast and safe.

### Walker2d — `verification/walker_pareto_sweep.py` → `walker_pareto.png`
The decisive test of *why* FPL wins, with a sharp two-regime story:
- **Gentle walking (tv 2–3):** quasi-static, *nothing* falls at any weight → a safe
  conservative linear weight exists → FPL wins nothing.
- **Running regime (tv 4–6, aggressive exploration):** flight phases are required to go
  fast → **one fixed FPL spec gets ~1.2 m/s at 0% falls at every speed** while the linear
  family only matches that speed by falling 25–88%. Gain at equal (zero) safety:
  **+52% / +22% / +30%** at tv 4 / 5 / 6 (16 seeds).

Video-verified: `runs/walker_FPL.mp4` = upright running stride (torso z-axis stays 0.94);
`runs/walker_linear.mp4` (aggressive weight) = forward lunge that stumbles below the fall
line.

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
.venv/bin/python verification/hopper_pareto_sweep.py       # win 1 + figure
.venv/bin/python verification/walker_pareto_sweep.py       # win 2 + figure (running regime)
.venv/bin/python verification/mismatch_robustness_sweep.py   # robustness win (hopper, traction)
.venv/bin/python verification/mismatch_robustness_walker.py  # robustness win (walker, traction)
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

## Open threads

- Harder biped regimes / a gait-phase or foot-placement objective to make g1_walk itself
  achievable (the one task no controller could do).
- FPL's absolute-scale explore/exploit *might* still suit converge-and-commit tasks
  (reach / stand / swing-up) at a scarce online budget — untested, and not the offline
  regime targeted here.
