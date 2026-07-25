# FPL × MPPI — Working Handoff

> Drop this file in the repo root. In a fresh Claude Code session, start with:
> **"Read FPL_MPPI_HANDOFF.md, then look at my MPPI loop and FPL code."**
> This briefing carries over context from a planning conversation; the code itself was NOT reviewed there (it's on the local machine), so a full read of the actual repo is step one.

---

## 1. Goal

Combine two ideas into a **sampling-based controller** and look for **large, structural improvements over vanilla MPPI**, starting on the **Walker2d** (and possibly Hopper) MuJoCo env. No real-time/latency constraint — this runs **offline**. Downstream intent: use the resulting controller as an expert/**teacher to distill a fast neural policy (student)** for RL, ideally sharing one objective spec across teacher and student.

## 2. The two papers

**Paper A — Information-Theoretic MPC / MPPI** (Williams et al., 2017, arXiv:1707.02342).
- Sample K control sequences `V^k = U + E^k`, `E^k_t ~ N(0, Σ)`; roll out dynamics; compute scalar trajectory cost `S_k`.
- Weight: `w_k ∝ exp(-(1/λ)(S_k - ρ))`, `ρ = min_k S_k` (min-shift for numerical stability).
- Update: `U ← U + SGF * Σ_k w_k E^k` (SGF = Savitzky–Golay smoothing filter).
- Key knobs: λ (inverse temperature / greediness), Σ (sampling covariance — **held constant** in the paper), γ (control cost).
- The cost `S_k` in the paper is a **hand-weighted linear sum** of objective terms.
- Notable: they *tried* adaptive covariance and rejected it because it shrinks too fast for receding-horizon (online) control.

**Paper B — Fulfillment Priority Logic** (El Mabsout, Abdelgawad, Mancuso, 2025, arXiv:2503.05818).
- Each objective is a **fulfillment** `f ∈ [0,1]` (0 = failed, 1 = fulfilled).
- Compose with **power mean** `μ_p(x) = ((1/n)Σ x_i^p)^(1/p)`:
  - p→−∞ min · p=−1 harmonic · p=0 geometric · p=1 arithmetic (linear) · p→∞ max.
  - Conjunction `∧_p` uses p ≤ 0 (weights the least-fulfilled objective more).
  - Disjunction via De Morgan; priority offset `[φ]_δ = (u(φ)+max(δ,0))/(1+δ)` raises a baseline → curriculum/lexicographic ordering.
- Applied to **FQ-values** = normalized discounted returns in [0,1]: `FQ = (1-γ)Q`.
- Central critique: **linear scalarization under competing objectives behaves like a disjunction** (maximizes one axis, abandons another), not a convex combination.
- **Minimum Fulfillment Bound**: if `μ_p(f) = y` then every `f_i ≥ ᵖ√(n(y^p − 1) + 1)`. (e.g. n=2, p=−2, y=0.9 ⇒ each f_i ≥ 0.38.) High composite ⇒ guaranteed floor on every objective.

## 3. The core combination (the "right" way, not a bolt-on)

Replace MPPI's **linear scalarization** with FPL, keeping objectives as a vector until the last moment:

1. Roll out K trajectories.
2. Per rollout k, per objective i, compute discounted fulfillment
   `F_i^k = (1-γ) Σ_t γ^t f_i(x_t^k)  ∈ [0,1]`  (Monte-Carlo FQ-value analog).
3. Compose: `u_k = u_FPL(F_1^k, ..., F_n^k)` (power-mean conjunction + offsets).
4. Cost: `S_k = -log u_k`  ⇒  `w_k ∝ u_k^(1/λ)`. (Apply min-shift: `w_k ∝ (u_k / max_k u_k)^(1/λ)`.)
5. Update `U` as usual (weighted avg of E^k, then SGF smooth).

**Why it should beat linear-cost MPPI on legged robots:**
- *Min-fulfillment floor* ⇒ high-weight rollouts can't be secretly catastrophic on any axis (e.g. "about to fall" can't be bought back by speed reward). Caveat: the floor is per-sampled-rollout; the executed control is a weighted average in control space, so it's a strong bias, not an exact guarantee.
- *State-adaptive trade-offs for free*: `∂μ_p/∂f_j = (1/n)(u/f_j)^(1-p)` blows up as `f_j → 0`, so feeding `-log μ_p` into MPPI's exponential automatically concentrates update pressure on the **currently-binding** objective — no per-speed αᵢ retuning.

## 4. Offline unlocks (techniques the online constraint had forbidden)

Because this is offline, several things Williams rejected are back on the table — **but if you add any of them, add them to the linear baseline too, or you're measuring the machinery, not FPL.**
- **Adaptive covariance** (CEM/CMA-ES style): `Σ_t ← Σ_k w_k (E_t^k)(E_t^k)^T`. Shrinkage is *good* offline. FPL twist: inflate anisotropically along control dims that most affect the least-fulfilled objective.
- **Iterated inner loop per timestep**: run sample→weight→update to convergence before committing an action (online MPPI does exactly one).
- **λ annealing** across inner iterations: high→low (explore→greedy). `u_k^(1/λ)` sharpening is well-behaved on [0,1].
- **Priority-offset curriculum over iterations**: early passes weight only stability/safety fulfillments; later passes switch on speed/gait/energy. Same FPL formula reused for the eventual RL reward → teacher/student spec consistency.

## 5. Numerical traps (check these in the existing code FIRST)

- **Zero-fulfillment underflow**: if any `f_i = 0` and p=0 (geometric), `u_k=0 ⇒ S_k=+∞ ⇒ exp underflows`. Fix: min-shift is mandatory; floor fulfillments at ε; use finite negative p instead of p=0; offsets raise baselines (double as stabilizer).
- **Two temperatures fight**: FPL's p (how conjunctive) vs MPPI's λ (how greedy). Strong conjunction already compresses the spread of u_k, shrinking the range λ acts on. **Tune p first (semantics), then λ (statistics).**
- **Monitor ESS** = `(Σ w_k)^2 / Σ w_k^2`. If it collapses toward 1, the update is riding a single trajectory — back off p (less negative) or raise offset floors.
- Sign check: confirm `S_k = -log u_k` (lower cost = higher fulfillment), not a flipped sign.
- Confirm the min-shift is on the **cost** (or equivalently dividing u by max u), applied every step.

## 6. Flagship experiment — "Pareto-domination over the linear family"

**Claim to target:** not "FPL beats one tuned linear MPPI by X%," but **"one FPL spec dominates the lower envelope of the entire linear-weight family across a speed sweep, with no retuning."** That's the un-tunable, structural win.

Setup (Walker2d; start with the true MuJoCo model as the MPPI model to remove model-error confound):
- **Shared objective atoms** for both controllers:
  - `f_upright` — smooth ~1 in safe torso angle/height interior, **decaying to 0 BEFORE the termination cliff** (analog of Williams' time-decaying impulse penalty; decaying too late is the #1 reason the floor fails to help).
  - `f_speed` — one-sided saturating tracking of commanded forward velocity (no reward for exceeding → no incentive to sprint into a fall).
  - `f_energy` — `1 − normalized(Σ τ²)`.
- **Linear MPPI**: `S = weighted sum`, weights **FIXED across the whole speed sweep** (retuning per speed disproves the FPL thesis).
- **FPL MPPI**: `S = -log μ_p(f_upright, [f_speed]_δs, [f_energy]_δe)`, upright un-offset (top priority).
- **Sweep** commanded speed gentle → aggressive. Over N seeds, at each speed measure: achieved speed, fall rate (or time-to-fall), energy.
- **Plot**: linear family = one curve per weight setting on the speed-vs-fall-rate plane; overlay the single FPL controller. Expected: no single linear weighting is both fast and safe across the sweep; FPL walks the whole sweep near-zero fall rate.
- **Headline number**: fall rate at the most aggressive target.

**Fairness rules (non-negotiable):**
1. Everything except the scalarization identical between the two controllers.
2. Any offline machinery (adaptive Σ, inner loops, annealing) goes to BOTH.
3. Be generous to the baseline: grid-search linear weights, report the BEST linear controller per metric. Beating a weight-grid-searched linear baseline with zero search is the strongest claim.

**Operating point is the experiment:** below the actuation/friction limit, upright & speed don't compete and linear ≈ FPL (you'll see nothing). The gap only opens in the competitive regime — push the target hard. (Williams saw method differences appear only past the friction limit.)

## 7. De-risking ladder (do before/with the flagship)

- **Toy diagnostic** (pure numpy, no sim): two-objective competitive system `f0=b0(1−α b1)`, `f1=b1(1−α b0)`. Confirm FPL-in-MPPI weighted avg lands balanced while a linear cost collapses to the disjunction corner. If this fails, the plumbing is buggy — don't touch MuJoCo yet.
- **Hopper standing-still refusal** (fast, visually obvious): the FPL paper documents agents scoring ~1000 by standing still; conjunction over per-limb speed fulfillments assigns ~0 and forces motion. Linear-cost MPPI can find the same stand-still local optimum; FPL-MPPI structurally can't. Simpler than Walker (3 joints), fastest convincing contrast.

## 8. Suggested starting knobs (Walker)

- Horizon 30–50 steps (~1 s); K = 500–2000 (be generous, offline).
- FPL: p ≈ −1 to −2 (more negative = harder uprightness floor = safer but harder to optimize — main safety/optimizability dial). Small δ_s (speed on early), large δ_e (energy only once upright+speed satisfied).
- `f_upright` must decay before the healthy-range boundary.

## 9. Contingencies if the gap doesn't open

- No gap → operating point too easy: push speed target, make p more negative.
- `f_upright` decays only at the cliff → useless floor: make it decay earlier.
- Linear baseline genuinely fine (true-model MPPI is strong) → pivot the story to **robustness-across-speeds / no-retuning** (matches FPL's real thesis), OR move to a **learned/imperfect dynamics model**, where FQ-value/estimation error distorts prioritization and FPL's normalization + floor earn their keep (the paper flags this regime explicitly).
- Weighting degenerates to one trajectory → ESS collapse from p×λ interaction: back off p or raise offset floors.

## 10. First tasks for the Claude Code session

1. Read the actual MPPI loop + FPL operator + cost/fulfillment defs + config.
2. Verify §5 numerics in the real code (sign of `-log u`, min-shift present every step, ε-floor / offset, ESS logging).
3. Add ESS logging if missing; add a toggle to switch cost between `linear-sum` and `-log μ_p` with identical atoms (this is what makes the fair comparison possible).
4. Wire up the §7 toy diagnostic as a standalone script to validate the FPL+weighting path with no sim dependency.
5. Then build the §6 speed sweep + plotting.