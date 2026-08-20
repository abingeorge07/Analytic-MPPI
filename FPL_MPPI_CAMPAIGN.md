# FPL-MPPI campaign — sampler comparison, portability, zero-tuning, videos

Second results doc (companion to `FPL_FINDINGS.md`), for the paper's **part 1**: *FPL-MPPI is a
robust, low-tuning controller you can drop on a new robot morphology to get an honest read on
whether it can do a task.* This campaign answers the reviewer questions the first doc didn't:

1. **Is the FPL win a property of the objective or the MPPI sampler?** → the objective. It
   transfers to every sampler we tried (MPPI, MPPI-CMA, CEM, and a new colored-noise sampler).
2. **Can a smarter / FPL-native sampler beat plain MPPI?** → No, across 5 samplers × 5 budgets
   × 16 seeds. Rigorous negative; the value is entirely in the objective.
3. **Does the FPL objective work off-the-shelf across robots with no per-task tuning?** → Yes:
   one global FPL spec is the most productive objective on walker+hopper+cube; no single linear
   weight transfers (Result 3).
4. **Video, not just numbers.** → side-by-side FPL-vs-linear clips (walker, hopper done; cube next).

Everything is **checkpointed**: every (experiment, env, sampler, cost, K, seed) trial is one
durable line in `verification/checkpoints/*.jsonl` (flush+fsync). A crash resumes with zero
recomputation. `verification/_checkpoint.py` (store) + `verification/_experiment.py` (per-env
matched specs + `run_trial`, which enforces the fairness protocol in one place).

## Fairness protocol (unchanged from FPL_FINDINGS)
Within any comparison, EVERYTHING is identical except the axis under study. Samplers are compared
with the FPL cost held fixed (p=−1 min-fulfillment). Costs are compared with the sampler held
fixed; `power_mean(f, p=1, w) == Σ wᵢ fᵢ` exactly, so the linear family is `p=1` + swept weight
and FPL is `p=−1` uniform — only the objective-axis composition differs. `MppiCma`/`CEM` were
patched to forward `fpl_time_p`/`fpl_weights` so the identical cost really does run on each.

---

## Result 1 — FPL-native SAMPLER race: a clean negative (`verification/fpl_sampler_race.py`)

The prior negative (`fpl_adaptive.py`: adaptive diagonal covariance + binding-objective steering)
tested one idea. This retries with the technique the sampling-MPC literature says matters —
**colored / temporally-correlated noise** (iCEM; Pinneri et al. 2021) — plus an **FPL-native
absolute-scale exploration schedule** (widen σ when the best composite fulfillment in [0,1] is
low, commit when high; only a bounded reward makes "is anything good yet?" well-posed). New
controller `FplColoredMPPI` (`fpl_colored`). Cost fixed to the winning FPL spec; only the proposal
varies; sweep the budget K because sampler quality matters most when samples are scarce.

**Productive speed (achieved × survival), 16 seeds, 95% CI — `fpl_sampler_race.png`:**

| walker (running) | K=16 | K=32 | K=64 | K=128 | K=256 |
|---|---|---|---|---|---|
| FPL·MPPI (white)        | **0.92** | 1.03 | **1.11** | **1.15** | **1.20** |
| FPL·CMA                 | 0.89 | **1.04** | 1.07 | 1.12 | 1.12 |
| FPL·CEM                 | 0.85 | 0.98 | **1.11** | 1.13 | **1.20** |
| FPL·colored             | 0.85 | **1.04** | **1.11** | 1.13 | 1.13 |
| FPL·colored+abs (ours)  | 0.77 | 0.99 | 1.10 | 1.13 | 1.14 |

Plain white MPPI is **best-or-tied at every walker budget**; on hopper (noisier) CMA is
marginally top at high K and everything overlaps within CIs. The FPL-native colored+abs variant
is tied-or-slightly-worse, notably at K=16 (inflating exploration wastes a scarce budget).

**Verdict:** neither colored noise nor the absolute-scale schedule beats plain MPPI. A third,
categorically-different mechanism — **gradient-guided refinement** (BPTT through MuJoCo FD
Jacobians on the analytic FPL reward, `fpl_gradient_probe.py`) — also fails: at K=16 it is
~100–300× slower per step (37.6 s/episode on walker vs 0.1 s) and scores *worse* than plain
MPPI at the same K (walker 0.88 vs 0.95), nowhere near plain MPPI at K=256 (1.20). So across
FOUR search mechanisms (adaptive covariance, colored noise, absolute-scale schedule, gradient
refinement) nothing beats warm-started fixed-Gaussian MPPI. The value is entirely in the FPL
**objective**. This is the evidence for the "FPL-portability" framing the user chose: *use FPL
cost with any sampler; pick the simplest (plain MPPI).*

## Result 2 — FPL PORTABILITY across samplers (`verification/fpl_portability.py`) — WIN

The backbone. For each sampler, one fixed FPL spec (p=−1) vs the whole linear-weight family;
report FPL's productive speed vs the **fastest comparably-safe** linear weight (fall ≤ FPL's).
16 seeds, 95% CIs. `fpl_portability.png`: in every (env, sampler) panel the FPL star sits
above-and-right of the linear frontier — faster AND at least as safe.

| env | MPPI | MPPI-CMA | CEM | FPL-colored |
|---|---|---|---|---|
| **walker** (speed m/s @ surv) | FPL 1.20 vs lin 0.98 → **+22%** | 1.12 vs 0.94 → **+20%** | 1.20 vs 1.03 → **+17%** | 1.14 vs 0.85 → **+34%** |
| **hopper** | 0.93 vs 0.30 → **+209%** | 0.90 vs 0.35 → **+156%** | 1.07 vs 0.83 → **+29%** | FPL 0.72 @ 100% surv; **no linear weight is as safe** |

FPL Pareto-dominates the linear-weight frontier **within every sampler** on walker+hopper (all
FPL points at 88–100% survival). The advantage is a property of the OBJECTIVE — it does not
depend on which sampler draws the rollouts. Combined with Result 1 (no sampler beats plain MPPI),
the paper's recommendation is unambiguous: **FPL cost + the simplest sampler (plain MPPI).**

**Cube extension (`fpl_portability_cube.png`), honest resolution.** At K=256/12 seeds FPL leads
strongly on **three of four** samplers — MPPI **+155%**, CEM **+190%**, FPL-colored **+229%** over
the fastest comparably-safe linear weight (all FPL at 90–100% survival). The higher budget resolved
CEM (a −0% tie at K=128 → a clean +190% win) but confirmed **MPPI-CMA is a genuine tie (−4%)**, not
a budget artifact: on the contact-rich cube, CMA's covariance adaptation lets even the conservative
linear weight rotate ~59° at 100% survival. So cross-sampler portability is *clean on the two
locomotion robots* and *holds for MPPI/CEM/colored on the cube, with CMA a tie* — the honest one
exception in the matrix. (Quadruped cross-sampler was not run — 343 s/episode makes a 4-sampler
matrix impractical; its MPPI objective win stands at 30 seeds, `quadruped_pareto.png`.)

## Result 3 — ZERO-TUNING / off-the-shelf (`verification/fpl_zero_tuning.py`) — WIN

One GLOBAL objective spec dropped on every robot, no per-task retuning; only the objective is
global (each env keeps its structural sampler). 16 seeds. Cell = progress (% of FPL) × survival.
`fpl_zero_tuning.png` — FPL is boxed (most productive) in **every** row; every linear weight has
a collapse:

| objective | walker (of FPL) | hopper | cube |
|---|---|---|---|
| **FPL (one spec)** | **100% @100% surv** | **100% @88% surv (best)** | **100% @100% surv** |
| linear wv=0.5 | 23% (timid) | 32% | 2% (barely turns) |
| linear wv=1   | 29% (timid) | 81% @62% surv | 37% |
| linear wv=2   | 36% (timid) | 138% @**25%** surv | 90% @88% |
| linear wv=4   | 57% | 175% @**0%** surv (falls) | 98% @69% (drops) |
| linear wv=8   | 82% | 179% @**0%** surv (falls) | 92% @50% (drops) |

The linear weight you'd pick differs per robot and none transfers: wv=4/8 run the walker but
**fall on the hopper (0% survival)** and **drop the cube**; wv≤1 is too timid to move the walker
or turn the cube. The **same** FPL spec is the most productive objective on all three — the
"drop it on a new morphology and get an honest read, no hyperparameter hunt" claim, demonstrated.

## Result 4 — Mismatch robustness is PORTABLE across samplers — WIN
(`verification/fpl_robustness_samplers.py` → `fpl_robustness_samplers.png`)

The cost-axis robustness win (one fixed FPL spec holds speed+survival under unretuned traction/grip
loss) is established in `FPL_FINDINGS.md` for MPPI. Re-running the hopper traction sweep (tv=2.5,
plan nominal / execute slippery, 12 seeds, no retuning) inside **every** sampler confirms it is a
property of the OBJECTIVE: FPL rides above the linear-weight family across the whole traction sweep
in all four panels. Survival across friction 1.0→0.35:

| sampler | FPL survival | best linear weight |
|---|---|---|
| MPPI | 92/92/75/75% | noisy, ≤92%, weight-dependent |
| MPPI-CMA | **100/100/92/92%** | 33–83% |
| CEM | 92/83/92/83% | 42–83% |
| FPL-colored | 92/83/83/67% | ≤75% |

FPL holds the highest productive speed AND survival at essentially every traction level within
every sampler — the un-tunable robustness does not depend on MPPI.

## Result 5 — Ease of cost specification (concrete, from the task code)

The linear cost couples objectives through **cross-objective trade-off weights**: unbounded
numbers whose *ratios* encode priority, with no natural scale, that must be co-tuned (and, per
the mismatch study, re-tuned when the robot/difficulty changes). The FPL cost couples objectives
through a parameter-free rule (`min`-fulfillment, `power_mean p=−1, uniform`); each objective is
specified alone as a **bounded [0,1] fulfillment with a satisfied-band in physical units**, read
directly from task intent — no ratios, no cross-objective weights.

**Hopper, term by term (`analytic_mppi/tasks/hopper.py`):**

| objective | linear cost term | FPL atom |
|---|---|---|
| height   | `10.0·(h−h*)²`  (weight **10**) | full≥**1.0 m**, floor **0.5 m** (a band) |
| upright  | `50.0·(1−zₐ)²`  (weight **50**) | full≥**cos18°**, floor **cos53°** |
| velocity | `5.0·(v−v*)²`   (weight **5**)  | ramp v/v* (fraction of target) |
| control  | `0.3·Σu²`       (weight **0.3**) | `1−¼·mean(u²)` |
| **compose** | Σ (4 hand-tuned weights **10/50/5/0.3** + the swept family weight) | `min`-fulfillment (**0** free weights) |

To specify the linear cost you must answer "how many times more important is uprightness than
speed?" (why 50 vs 5? — a grid search), and the answer does not transfer across robots. To
specify the FPL cost you answer "what torso height / tilt / speed counts as *doing the task*?"
— physically meaningful thresholds you can read off the robot, and the min-conjunction handles
the trade-off. Count: **linear = 4 arbitrary trade-off weights (+ per-difficulty retuning); FPL
= 0**. The same pattern holds for walker (weights 10/10/1/0.001) and cube (10/50/5/1 + a
100·hinge barrier). This is the "you can structure it because you know the intent" claim, made
concrete: FPL's parameters are *interpretable and auditable* (a bound where an objective is
met), the linear cost's are *arbitrary and entangled* (a ratio with no units).

## Result 6 — a SECOND FPL-native layer: the WEIGHTING / temperature stage

The sampler race closed the *proposal* stage. This opens the other half of MPPI-as-inference —
the importance **weights** (the stage SV-MPC / adaptive-importance-sampling / "adaptive cooling
via entropy feedback" all target). Standard MPPI weights rollouts by `w_k ∝ (u_k/max u)^{1/λ}`;
the temperature λ is a per-task knob because the right sharpness depends on the *spread* of the
rewards, and a scalar cost has no absolute reference for "how good is good." FPL's bounded reward
supplies exactly that, so we can set the weights by a **target effective-sample-size** instead of
a temperature (`FplTemperedMPPI`, `weight_mode="adaptive_ess"`: solve λ per step to hold ESS).

**Temperature-transfer study (`verification/fpl_weighting_study.py` → `fpl_weighting.png`), 12
seeds.** The *same* fixed λ=0.05 produces ESS≈14 on walker but ESS≈56 on hopper (narrower reward
spread), and warm λ on hopper collapses to ESS≈123–159 (near-uniform — no selection, the update
just averages). Targeting a fixed ESS fixes this:

| env | best fixed-λ | adaptive-ESS (target-fraction) |
|---|---|---|
| walker | 1.38 @100% surv (λ=0.05) | 1.33 @100% (ties) |
| hopper | 1.07 @**92%** surv, CI ±0.24 (λ=0.05) | **1.21 @100%** surv, CI ±0.18 |

On the noisy hopper, holding a fixed ESS **beats the best fixed temperature: +13% productive
speed, 100% vs 92% survival, ~25% tighter seed variance** — a fixed λ can't hold selectivity as
the per-step reward spread swings. (ESS-targeting itself is generic; FPL's bounded reward is what
makes the ESS target *meaningful and consistent* across robots.)

**FPL-native attribution under mismatch (`verification/fpl_weighting_robustness.py`
→ `fpl_weighting_robustness.png`), 16 seeds — honest mixed/marginal.** Under hopper traction loss,
the four weightings cross with heavily overlapping CIs; no clean winner. The FPL-calibrated
adaptive-ESS shows the *hint* it was designed for — best survival at the slipperiest level
(1.15 m/s @100% vs fixed-λ's 0.98 @81%), i.e. it hedges when the ground is worst — but it is worse
at nominal and within-noise elsewhere; the feasibility gate is erratic.

**Honest verdict on the weighting layer.** The importance-weighting stage is *more amenable to FPL
than sampling was*: targeting a fixed effective-sample-size — which FPL's bounded reward makes
well-calibrated — genuinely prevents the weight-collapse a fixed temperature suffers when the
reward spread is narrow (the clean +13% hopper result at tv=2.0). But the effect is **modest and
regime-dependent** (it flips to a fixed-λ advantage at tv=2.5), the *main* driver is generic
ESS-targeting rather than an FPL-exclusive mechanism, and the FPL-specific calibration/gate is
within-noise under mismatch. So this is a real, publishable *secondary* finding — "FPL's absolute
scale lets the weights self-calibrate, useful where a fixed temperature collapses" — **not** a
clean, large win on the order of the objective-level portability / zero-tuning / robustness
results. The headline stands: **the dominant, reliable value of FPL is in the OBJECTIVE.**

## Result 7 — a THIRD FPL-native layer: hard-safety SHIELD / lexicographic constraint

The most distinctly-FPL idea: a scalar weighted cost can NEVER encode strict priority (any finite
weight lets you trade safety for enough reward), but FPL's per-objective bounded fulfillment can —
treat the SAFETY atoms as a hard constraint (min over-time safety ≥ τ, optional terminal safety ≥
τ_T for recursive feasibility) and optimize PERFORMANCE only among the certified-safe rollouts
(`FplShieldedMPPI`, `fpl_shielded`). This decouples "how hard you try" from "how safe you are",
so in principle you could crank performance without falling.

**Result — honest negative for these short-horizon dynamic tasks (`verification/fpl_shield_study.py`
→ `fpl_shield.png`, hopper, 16 seeds).** The hard shield does NOT Pareto-beat the soft
min-fulfillment cost:
  * Hard constraints are BOUNDARY-SEEKING under a short horizon: they drive the plan to the edge
    of the horizon-feasible region, which is not safe *beyond* the horizon — a recursive-
    feasibility failure that makes the controller *fall more* in closed loop. The balanced shield
    (τ=0.3) collapses at the nominal tv=2.5 (**0.29 m/s @ 31% survival** vs plain FPL's
    **0.79 @ 88%**), and every shield variant is below plain FPL across the whole traction-loss
    sweep (16 seeds).
  * Adding a terminal-safety constraint restores some safety but only moves the controller along
    the SAME (speed, survival) frontier, more conservatively: the greedy shield buys survival at
    extreme aggression (tv=3.5: **94% vs plain's 69%**) but at lower speed, so productive speed is
    no better (**0.56 vs 0.62 m/s × survival**). No shield variant Pareto-beats the soft cost at
    any operating point tested.

**Why this matters (the insight):** the soft min-fulfillment conjunction isn't just convenient —
it is the *right* FPL design for short-horizon sampling MPC, because it balances safety and
performance continuously and stays centred in the safe region, whereas a hard lexicographic
constraint + short horizon is boundary-seeking. A hard shield/terminal-value would pay off with a
LONG horizon or a learned terminal value that certifies safety beyond the plan — i.e. exactly the
value-function regime the paper's **part 2 (FPL-MPPI → RL teacher-student)** provides. So this
"failure" cleanly motivates part 2.

## Videos (`verification/fpl_vs_linear_video.py`) — DONE (walker, hopper)

Synchronized side-by-side [ FPL | best-safe linear | aggressive linear ] with a live per-panel
readout. `runs/paper/{walker,hopper}_sidebyside.mp4`:
- **hopper**: FPL hops upright and forward; best-safe linear (wv=0.5) stands nearly still
  (vx≈0); aggressive linear (wv=2) pitches over into a faceplant (up→0.68, then falls).
- **walker**: FPL holds an upright running stride; aggressive linear (wv=16) lunges faster for a
  moment (vx≈2.8) but is toppling (up=0.68); best-safe linear (wv=4) stays up but slower on
  average. Exactly the Pareto story the numbers report, now visible.
- **cube**: FPL tumbles the cube 58° while holding it centred (drift 0.01); best-safe linear
  (wv=1) barely turns it (12°); aggressive linear (wv=8) **drops it** (drift 0.19 ≫ 0.06). The
  non-locomotion analogue of the faceplant.

- **quadruped**: FPL trots upright (up=0.97, 1.36 m/s); best-safe linear trots slower (1.02 m/s);
  aggressive linear (wv=8) **flips onto its back** (up=−0.82). All four robots now have the clip.

## Humanoid g1 (timeboxed) — honest NULL (`verification/g1_reach_aggressive.py`)

Attempt: push `g1_reach` into an AGGRESSIVE forward-target regime (0.3/0.5/0.7 m) hoping no safe
static lean can reach it → dynamic competition → an FPL win (the hopper structure applied to
posture). 6 seeds, H=1.0 s, K=128, iters=3. Metric = forward reach while staying tall
(min torso height > 0.5 m) AND upright (rz > 0.5).

Result — the conservative linear weight (wf=1) has the highest SAFE reach at **every** target
(safe% 83/33/50); FPL reaches farther (0.66–0.75 m) but **pitches the torso down below the
standing floor** (min_h ≈ 0.40), so its safe rate is 0–17%; aggressive linear topples (0%).

Why it's a null (and why that's consistent): on the hopper/walker/cube the competition has a
*safe-and-productive* operating point (a gentle gait / a centred roll) that FPL's min-fulfillment
floor finds. On g1 reach, going farther forward *inherently* lowers the torso — there is no
"reach far AND stay tall" operating point for the floor to protect, so FPL trades height for reach
and misses the standing threshold, while the conservative weight just reaches less. This is the
third g1 null (with g1_reach@0.2 and g1_walk) and it sharpens the thesis boundary: **FPL needs
dynamic competition in which a safe productive operating point exists.** The humanoid is a
capability/structure gap for this short-horizon sampling MPC — exactly the regime the paper's
part-2 (FPL-MPPI → RL teacher-student) is meant to unlock.

## Reproduce
All scripts are checkpointed (resume on re-run; delete the matching
`verification/checkpoints/*.jsonl` to force a fresh run).
```bash
.venv/bin/python verification/fpl_sampler_race.py                      # R1  sampler race (negative)
.venv/bin/python verification/fpl_gradient_probe.py                    # R1  gradient attempt (negative)
.venv/bin/python verification/fpl_portability.py                       # R2  portability (walker+hopper)
.venv/bin/python verification/fpl_portability.py --envs cube --seeds 12 --K 256  # R2  cube extension
.venv/bin/python verification/fpl_zero_tuning.py                       # R3  zero-tuning
.venv/bin/python verification/fpl_robustness_samplers.py               # R4  robustness across samplers
.venv/bin/python verification/fpl_weighting_study.py                   # R6  weighting: temperature transfer
.venv/bin/python verification/fpl_weighting_robustness.py              # R6  weighting under mismatch
.venv/bin/python verification/fpl_shield_study.py                      # R7  hard-safety shield (negative)
.venv/bin/python verification/fpl_vs_linear_video.py walker hopper cube quadruped   # videos
.venv/bin/python verification/g1_reach_aggressive.py                   # humanoid timebox (null)
```
Figures land in `verification/*.png`; videos in `runs/paper/*_sidebyside.mp4`. New controllers:
`fpl_colored`, `fpl_tempered`, `fpl_shielded` (all in `analytic_mppi/controllers/`).
