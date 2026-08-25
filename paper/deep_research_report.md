# Bounded-Fulfillment Objectives in Sampling MPC: A Deep-Research Synthesis on Cost Representation, Robustness, and Sampling for CPU MPPI

*Deep-research synthesis prepared for the Analytic-MPPI / FPL-in-MPPI program. AI-assisted literature synthesis; see §2 and §9.*

---

## Abstract

This report assesses two program hypotheses against a verified corpus of 214 sources across seven search clusters, of which 51 distinct sources were deep-read (yielding 84 per-claim evidence records) and 87 are cited here, plus two adversarially surfaced items outside the corpus marked † (§3). H-A holds that MPPI driven by a Fulfillment Priority Logic (FPL) objective, bounded [0,1] atoms composed by a weighted power mean, yields a good-enough, mismatch-robust, morphology-portable controller on CPU MuJoCo at K ≤ 256. H-B holds that sharing that spec between an MPPI teacher and an RL student accelerates convergence. Both receive the verdict *supported with caveats*, for asymmetric reasons. H-A's enabling premises are strongly evidenced: the path-integral derivation is provably cost-agnostic, CPU sampling MPC at K = 32–256 is replicated across quadrupeds, humanoids and hands by four independent groups, and bounded mean-family composition beating tuned linear weights replicates in adjacent settings. H-A's *attribution* is not: nothing published or in-repo separates power-mean composition from mere boundedness, and bounded saturating costs improving robustness under model uncertainty is long-standing prior art. H-B's generic half, that an MPC teacher roughly halves RL iterations, is settled art with a ~2× anchor; its differentiating half is untested by anyone and has theoretical reasons to come out null. The in-repo evidence base is four platforms where FPL wins plus three Unitree G1 nulls, treated as the informative boundary condition rather than a failure to hide. We give a delta table against the weighted-STL line, a mandatory-ablation list, nine CPU-cheap sampling candidates with falsification tests, and a schedule making IEEE RA-L the primary venue and ICRA 2027 a conditional stretch.

---

## 1. Executive summary

**H-A: supported with caveats.** Everything the hypothesis needs to be *possible* is established. The information-theoretic derivation places no smoothness, convexity or boundedness requirement on the state cost, so S = −log(u) drops into the softmax without disturbing optimality (Williams et al., 2017; Eth). Real-time CPU sampling MPC at tiny sample counts is replicated across morphology classes by four groups (§3.2). And the slot is empty: the field's survey of MPC-as-inference taxonomizes priors, posterior families, temperature and constraints while assuming costs are given, with no multi-objective composition and no treatment of mismatch robustness as a design axis (Honda, 2025).

What is not established is attribution. Bounded costs compress the rollout-cost tail before the exponential weighting, the same effect CVaR buys at 3×10⁵ rollouts per step (Yin et al., 2022) and the Tsallis deformed exponential buys as a pure weight transform (Wang et al., 2021). MPPI-with-FPL computes exp(−S/λ) = u^(1/λ), making FPL *formally a member of the weight-transform family*. Until a clipped-linear control arm runs, with identical bounded atoms, p = 1 and no priorities, the program may attribute nothing to composition or priorities, only to boundedness.

**H-B: supported with caveats, and framed wrongly by default.** "MPC teacher accelerates RL versus scratch" is settled: ~2× fewer PPO iterations on Mini Cheetah (Youm et al., 2023), 5–500× compute savings with a value function in the planner loop (Lowrey et al., 2018), 3–4× data efficiency for a mutual teacher-student loop (Wang et al., 2025). "Sharing one hand-written spec is what helps" is untested by anyone, and the anchor paper gets its speedup with a *deliberately different* fine-tuning reward, crediting basin selection rather than objective content. H-B must be powered against generic MPC teaching (arm C), not RL from scratch, with a pre-registered decision rule and a planned publishable null.

**Confirmed threats.** Section 6 states seven labelled threats with responses and is their only full statement here. Three gate submission: the morphology count as propagated upward, the boundedness confound, and the strawman linear baseline.

**Top sampling recommendations.** The in-repo clean negative covered six mechanisms that either adapt the statistics of the Gaussian noise or replace the estimator outright; the literature's winners are elsewhere. Priority order: the linear-cost weight-transform ablation, at once the mandatory attribution experiment and a weight-side idea; spline/knot action parameterization, with a documented 100%-vs-0% effect on stairs under identical costs (Tao et al., 2026); Butterworth uniform-passband perturbation filtering on a single CPU core (Kicki, 2025); a fulfillment-keyed temperature schedule; and Biased-MPPI structural proposal injection (Trevisan & Alonso-Mora, 2024). Four of the five cost zero extra rollouts; only the spline refactor is a code change. All five appear as ranked candidates in §7.3.

**Positioning call.** IEEE RA-L is the primary target on its rolling deadline, carrying the full A1–A7 program. ICRA 2027 (15 September 2026) is a stretch conditional on A1, A2 and A5 returning unambiguous results by roughly 5 September, and going there means submitting without A3, which §6 marks blocking (§8). Headline the *empirical and mechanistic* claim: a bounded, min-like fulfillment cost Pareto-dominates a bounded linear family under dynamic competition and friction-type mismatch on four MuJoCo platforms, humanoid boundary disclosed as a first-class result. Not a new cost formalism, and not "one spec, any morphology." H-B pilot: **no**; future work.

---

## 2. Method note

Searches ran 19–20 August 2026 against the arXiv API (keyword queries plus `id_list` metadata batches for author, year and venue verification) and general web search, with forward-citation checks through Semantic Scholar; 139 queries were issued across seven clusters. Inclusion required a resolvable arXiv ID or DOI with a landing page that fetched, bearing on H-A, H-B, novelty or idea generation, and publication in 2019–2026, with canonical theory admitted regardless of age (Kappen, 2005; Todorov, 2009; Theodorou et al., 2010; Ng et al., 1999). Excluded: anything unresolvable, plus morphology search and co-design.

The funnel: 216 cluster-level candidate entries reached verification, two were dropped as unresolvable, 214 verified sources survived, 51 distinct sources were deep-read yielding 84 per-claim evidence records, and 87 are cited here — every bibliography entry is cited in the body — alongside two adversarially surfaced items outside the corpus that carry † wherever they appear. Both drops are one paper, *Accelerating Lyapunov-Stable Neural Control using Fulfillment Priority Logic*, surfaced independently in C4 and C5 with no arXiv ID and no findable landing page; it recurs in §4 and §9 as the FPL-citing work that could not be retrieved. The gap between 214 and the visible bibliography is screened-but-uncited material; the full verified list is the supplementary bibliography (`wf3/bib_compact.md`, 214 entries) and the per-claim evidence records for the deep-read set are in `wf3/evidence.json` (84 records covering 51 distinct sources).

| Cluster | Scope | Queries | Verified | Deep-read |
|---|---|---|---|---|
| C1 | MPPI variants and robustness | 13 | 40 | 12 |
| C2 | Sampling MPC for legged/dexterous robots; morphology-generality | 10 | 40 | 12 |
| C3 | Path-integral theory | 22 | 40 | 12 |
| C4 | Objective and cost representations, incl. signal temporal logic | 24 | 37 | 12 |
| C5 | Reward-design theory | 23 | 36 | 12 |
| C6 | MPC-as-teacher RL | 18 | 40 | 12 |
| C7 | Sampling-idea mining | 29 | 40 | 12 |

The Verified column sums to 273 rather than 214, and the Deep-read column to 84 rather than 51, because multi-cluster sources are counted once in each cluster they appear in. The saturation check was operational rather than statistical: a cluster closed when its final queries returned only sources already held or below the inclusion bar. No recall estimate was computed, so saturation means "new queries stopped producing new admissible sources," not a coverage guarantee. Roughly a fifth of the load-bearing sources carry 2025–2026 arXiv IDs with no accepted venue.

*Citation-year convention.* In-text years are arXiv submission years throughout; the accepted venue, where one exists, appears in the bibliography entry. So Kicki (2025) is arXiv:2503.11717, *ICRA 2026*, on the same convention as Schramm et al. (2025), Zhang et al. (2025) and Kurtz et al. (2025).

*Evidence grades.* **E1** an empirical finding replicated across independent groups. **E2** a single empirical result in a peer-reviewed or accepted venue. **E3** toy-scale or single-system simulation evidence. **E4** no accepted venue as of the 20 August 2026 search date. **Eth** a proved theorem; the grade attests the proof, and the separate question of whether its premises hold in this program is argued in the text, never inherited from the grade. The decision rule is acceptance status, not arXiv year: a paper accepted to a reviewed venue is E2 (E1 if the finding is independently replicated) regardless of when it was posted, and an unaccepted preprint is E4 regardless of posting year — including the FPL source paper, and including pre-2025 preprints such as Howell et al. (2022) and Alvarez-Padilla et al. (2024). Where a claim rests only on E4 sources, or only on one group, the text says so.

Two limitations are structural. This is an AI-assisted synthesis, disclosed in full in §9. And adversarial review was internal: two devil's-advocate passes surfaced three items outside the verified corpus — weighted STL (arXiv:2010.00752), PILCO's saturating cost (arXiv:1502.02860) and one concurrent MPC-guided RL work that the pass named but did not resolve to an identifier. The first two are verified-by-fetch and load-bearing for novelty; they carry † wherever they appear and must be added to the corpus before submission. The third is unresolved and therefore not load-bearing anywhere in this report; it is listed only so the gap is visible, on the same footing as the unlocatable FPL-citing paper below. In-repo results are unpublished, single-group, and labelled as such.

---

## 3. Literature landscape

### 3.1 MPPI variants and robustness

The architectural line runs from Tube-MPPI (Williams et al., 2018) through Robust MPPI (Gandhi et al., 2021) to residual-conservative adaptation (Yoon & Kim, 2026a), and its shared message is that warm-started MPPI is brittle under disturbance and the remedy sits *outside* the cost. Tube-MPPI matters disproportionately because its motivation is a cost-representation story told in reverse. Williams et al. wanted sparse weighted-indicator costs, philosophically close to FPL atoms; their zero-gradient plateaus left warm-started MPPI unable to recover from disturbance, so a tube was added, and the indicator-plus-tube controller reached 9 m/s where a hand-tuned dense cost reached 8 m/s and naive MPPI collapsed to 5 m/s. That is the in-repo Pareto claim with robustness bought architecturally. Clipped ramps with satisfied bands are the midpoint Tube-MPPI skipped: bounded and saturating like indicators, so bad rollouts cannot dominate the softmax, yet sloped inside the band, so the signal whose absence breaks indicator-MPPI survives. Atoms saturated outside their ramps reproduce that failure geometry exactly, so H-A holds only while disturbances keep the state inside the sloped region of at least some atoms.

Theory caps sensitivity from two directions. RMPPI's free-energy growth bound scales with the Lipschitz constants of the running and terminal costs, which bounded atoms shrink by construction (Gandhi et al., 2021; E3 for the FPL implication); RC-MPPI's mismatch-induced ranking distortion scales as 2·C_Δ·s̄/β, again capped by bounded atoms (Yoon & Kim, 2026a; E4). Neither bound has been instantiated for a saturating cost family, which is the cheapest route to a *mechanistic* rather than correlational claim. The risk-aware line is the closest rival explanation and is defused on compute grounds: RA-MPPI reaches the FPL-shaped outcome, 55.7–80% fewer collisions at unchanged lap times, by CVaR-reshaping the rollout-cost distribution at 307,000 rollouts per step on an RTX 3090 (Yin et al., 2022; E2), and unscented, chance-constrained and crowd-risk variants similarly multiply rollout counts (Mohamed et al., 2024, 2025; Trevisan et al., 2025). At K ≤ 256 on CPU nothing of that sort is affordable, so bounded atoms delivering tail-compression-like robustness for free would be a distinct mechanism; the missing comparison is matched-budget CVaR over the K rollouts already in hand.

### 3.2 Sampling MPC for legged and dexterous robots, and morphology-generality

The CPU-budget evidence is the strongest single block in this review: E1 by replication across four independent groups, though two of the four (Howell et al., 2022; Alvarez-Padilla et al., 2024) are themselves unaccepted preprints, so the replication rather than any single source carries the grade. MJPC gives 1–20 ms updates across a humanoid, a quadruped and a Shadow Hand, with predictive sampling competitive at N = 10 spline-perturbed rollouts (Howell et al., 2022); whole-body MPPI runs a real Go1 at 30–50 rollouts and 100 Hz on a desktop CPU (Alvarez-Padilla et al., 2024); reference-free sampling MPC produces gaits, jumps, backflips and handstands on a real Go2 at 30 samples and a 37-DoF G1 in simulation at 60–70, against competitors' 2048–4096 GPU samples (Schramm et al., 2025; E2); RGB reaches ~280 Hz on a G1 at K = 128 (Seo et al., 2026; E4); and LP-MPPI spans Gymnasium locomotion, two quadrupeds and a real F1TENTH car on a *single* CPU core at N = 100–256 (Kicki, 2025; E2). Whole-body iLQR with finite-difference MuJoCo derivatives runs in the same budget (Zhang et al., 2025; E2), the non-sampling contrast. GPU methods buy real capability, yet LP-MPPI still beat a DIAL-MPC baseline by 24–41% on quadruped trot and Schramm recovers annealing's benefits at 30 samples: sample count is not the frontier, sampling structure is. Four independent groups converge on cubic or Hermite splines with roughly four knots (Howell et al., 2022; Alvarez-Padilla et al., 2024; Schramm et al., 2025; Tao et al., 2026), and Tao's within-paper ablation holds the cost fixed while swapping only the parameterization; §7.1 gives the numbers and the consequence for H-A.

Morphology-generality is sharply delimited. Reference-free MPC claims quadruped-to-humanoid transfer with no algorithmic or sampler retuning (Schramm et al., 2025; E2), MJPC spans three morphology classes with one framework, TD-MPC2 runs one agent across 80 tasks and embodiments at 33 GPU-days (Hansen et al., 2023), and the cross-embodiment RL line trains per-distribution at GPU scale (Bohlinger et al., 2024; Feng et al., 2022; Shafiee et al., 2023; Doshi et al., 2024; Xi et al., 2025; Ai et al., 2025). But *every* sampling-MPC paper here hand-tunes weighted-norm objectives per task; nobody benchmarks cost-representation portability or mismatch robustness with seed-and-CI rigor. The gap is real, though the strongest rival points elsewhere: Schramm et al. attribute portability to spline parameterization under plain linear costs, and never test mismatch. Morphology search and co-design are out of scope. One boundary caps the "any morphology" rhetoric: every humanoid sampling-MPC result reviewed is simulation-only, and the in-repo G1 nulls (§6) are consistent with that shared unproven step.

### 3.3 Path-integral theory: why warm-started fixed-Gaussian MPPI is hard to beat

*Estimator optimality.* The tilted distribution q*(V) ∝ exp(−S/λ)p(V) exactly minimizes the free-energy/KL bound, and the MPPI iterate is its KL projection onto the Gaussian family; the derivation is agnostic to the state cost (Williams et al., 2017; Eth). Tsallis VI-MPC shows the cost-to-weight map is itself a free design axis: the deformed exponential interpolates continuously from MPPI to CEM's top-elite indicator, and modest r > 1 (the deformation exponent; see W1 in §7.2) cuts cost variance 60–85% across five morphologies from point mass to a 56-D-state humanoid, with the headline variance reduction at N = 64 (Wang et al., 2021; E2). That is the closest existing theory to what FPL's log-of-power-mean does — reshaping weight concentration rather than the sampler — and the program's pattern of cost-side wins and sampler-side losses is what the decomposition predicts.

*Why the sampler side failed.* Changing the sampling covariance is legitimate only with a likelihood-ratio correction, which enters as an extra running cost plus Girsanov terms; skip it and the estimator is biased, include it and weight variance inflates (Williams et al., 2015; E3). The convergence-optimal covariance yields 43–54% cost improvement but needs the cost Hessian, which a derivative-free CPU pipeline does not have (Yi et al., 2024; E2). On the locally quadratic landscape a warm start reaches, an isotropic Gaussian is already a serviceable preconditioner, as the preconditioned-gradient and expectation-maximization readings of MPPI make explicit (Fazlyab et al., 2026; Wang et al., 2026b).

*Implicit annealing via warm start.* Receding-horizon shifting starts every solve near the previous optimum, so the closed loop anneals implicitly and needs one cheap local correction per step. Every method that decisively beats fixed-Gaussian MPPI makes annealing explicit and iteration-rich, at 2048 samples on an RTX 4090 (Xue et al., 2024), up to 81,650 samples over dozens of iterations (Halder et al., 2025), or through entropy-feedback cooling (Wang et al., 2026a); §7.1 draws the consequence for the in-repo negative. Warm-starting itself remains almost untheorized, with finite-sample closed-loop stability available only for LTI systems (Yoon & Kim, 2026c). The one counterexample to "all guidance fails" is SVG-MPPI, which beats vanilla MPPI 3–4× on collision rate at 11 ms on CPU with a *single* Stein guide particle by preserving the closed-form update and moving only the nominal (Honda et al., 2023; E2); the in-repo GMM-plus-SUS failure replaced the estimator outright.

### 3.4 Objective representations, and the delta versus the STL line

STL robustness semantics and structured-cost sampling MPC have merged. The first line spent a decade repairing min/max robustness through smooth approximations (Pant et al., 2017; Gilpin et al., 2020; Welikala et al., 2023), average-based semantics (Lindemann & Dimarogonas, 2019; Mehdipour et al., 2019) and generalized means (Uzun et al., 2024); the second brought CVaR, CBF shields (Yin et al., 2023; Parwana et al., 2025) and rule hierarchies (Censi et al., 2019; Veer et al., 2022). STL costs inside path-integral optimizers now exist in at least four independent forms: a deterministic laptop-CPU solver (Halder et al., 2025), its lexicographic minimum-violation extension (Halder et al., 2026), priority-ordered STL under standard MPPI with scenario-tree CVaR at 220–240 rollouts and 0.77–2.49 ms/step on an RTX 5060 over kinematic-bicycle dynamics (Bouzid et al., 2026), and Stein-variational STL (Zheng et al., 2026). **"Logic-structured costs in sampling MPC" is not by itself a defensible novelty claim.** What survives must be stated as a table.

| FPL element | wSTL† (2020, arXiv:2010.00752) | AGM (Mehdipour et al., 2019) | D-GMSR (Uzun et al., 2024) | STL path-integral (Halder et al., 2025) | Priority-ordered STL-MPPI (Bouzid et al., 2026) | Delta verdict |
|---|---|---|---|---|---|---|
| Bounded output range | Weighted semantics over compatible robustness functionals; normalization not the goal | Normalized [−1,1], bounded, sound | Explicitly applies **no** normalization | Raw min/max, unbounded | 2^N rank rewards, unbounded | **Conceded.** AGM already bounds; [0,1] vs [−1,1] is cosmetic. Do not claim boundedness as novel. |
| Weighted power-mean composition | Weighted generalizations incl. mean-family | Arithmetic-geometric mean | Power means p ≥ 1, sound **and** complete | min/max, unsmoothed | Lexicographic scalarization | **Conceded.** Narrow to *tunable p including p < 0 on bounded atoms inside the MPPI softmax*. |
| Explicit sub-task priorities | **Yes**: weights encode priority among conflicting sub-tasks, used for synthesis | No | No | Bit-shifted violation levels (hard) | Rank rewards per rollout (hard) | **Conceded as capability.** Surviving difference is *soft* vs *hard*: no exponential compression of low-priority signal, no quantization brittleness. Veer's hierarchy needs exponential weight separation a^(N−i+1); on discretized lexicographic scalars we expect priority inversion until granularity is refined (**this report's inference, ungraded**; not asserted by Halder). |
| Satisfied-band clipped ramps (deadband, one-sided) | No counterpart | No; signed margin, no satisfied band | No | No | No | **No STL counterpart.** Unclaimed cost geometry, and the element Tube-MPPI's zero-gradient warning bears on. |
| Soft priority offsets | Closest analogue (weights), not offsets on bounded fulfillments | No | No | No (hard) | No (hard) | **Partially new.** Argue empirically, not representationally. |
| Discounted FQ-values as temporal aggregator | No | No | No | Time-robustness not discounted | No | **No counterpart anywhere.** Also the weakest theoretically (§3.5). |
| S = −log(u) coupling into the Boltzmann weight | No | No | No | Robustness used directly as cost | Robustness used directly as cost | **No counterpart.** Makes MPPI-with-FPL a power-law weight u^(1/λ) on a bounded score: a Tsallis-family transform, and the mechanistic hinge of the argument. |

†Verified by fetch during the adversarial pass but **not** in this report's 214-source verified corpus (§2). Must be added; any reviewer in STL-based control finds it in one search.

AGM-optimized policies survived noise at ~59% versus ~42% for smooth min/max approximations at equal nominal scores (Mehdipour et al., 2019; E2). That is bounded mean-based aggregation buying robustness per se, exactly the shape of the in-repo result. LSE/softmin at κ = 25 meanwhile fails a simple reach spec outright through masking, while mean-based measures converge (Uzun et al., 2024; E3).

Three findings bound the claim, and the first is a live failure mode. STL-SVPIO documents sparse, long-horizon, temporally nested costs collapsing MPPI's importance weights, with robustness 0.005 versus 0.108 and total failure on coordination tasks (Zheng et al., 2026; E4). Reconciled with Bouzid's success, the divide looks like dense-per-timestep versus sparse-terminal: FPL's clipped ramps are dense by construction, so the program sits on the safe side, but deeply nested temporal specs at K ≤ 256 are out of reach. That reconciliation is **this report's inference from two datapoints, ungraded**, and it is doing real work — it is the reason the program is told it can ignore a documented MPPI failure mode. Its falsification instrument is ablation A8's per-solve ESS logging (§6) together with candidate W4: if FPL-side weights collapse toward a handful of rollouts as the temporal weakest-link bites, the dense/sparse divide is the wrong cut and the failure mode applies here after all. Next, raw non-smooth min/max semantics are usable inside sampling optimizers, so the STL literature's smoothness rationale is irrelevant here; FPL's differentiation must rest on boundedness, dense signal and soft priorities, never on differentiability. Last, RRT-η already adopts FPL wholesale in an offline planner (Ahmad et al., 2026; E4), so the novelty sentence must read "first bounded-fulfillment logic objective inside *receding-horizon closed-loop* sampling MPC."

### 3.5 Reward-design theory and where FPL sits

Bounded, normalized, mean-like composition produces better-conditioned landscapes and more reproducible policies than either linear weighted sums or hard min/max logic, and four independent lines say so. FPL descends from geometric-mean multiplicative composition acting as a logical AND, which took quadrotor sim-to-real from roughly one-in-dozens flight-worthy agents to 100%, cut motor current from 22.87 A to 5.86 A, and reduced training variance in 19 of 24 Gym comparisons (Mysore et al., 2021; E2). Its independent ancestor is logic-derived reward: TLTL reached 100% success on a physical Baxter where a hand-tuned shaped reward failed to learn at all, the failure attributed to the coefficient-magnitude brittleness that bounded atoms remove (Li et al., 2016; E2). FPL itself reports one spec generalizing across HalfCheetah, Walker2d and Ant without retuning, insensitivity to p ∈ {0, −1} and to seeds, and up to ~5× sample-efficiency gains over SAC-class baselines — but only with a policy-gradient learner, never in MPC or planning (El Mabsout et al., 2025; E4, author group).

Gödel and min conjunctions are *single-passing*: at most one subobjective receives signal, while product, geometric and generalized-mean aggregators are non-vanishing everywhere (van Krieken et al., 2022; E3). Extending that from gradient descent to rank-based importance sampling is **this report's inference, ungraded**. A min-like cost should make K = 256 rollouts nearly indistinguishable except along one active constraint, whereas power means near p = 0 or −1 should preserve ranking information across all atoms; ablation A8's ESS logging and candidate W3's rank-on-linear cell are designed to settle it, and since the inference is load-bearing for novelty item 3 in §4, it should not be leaned on before they run. Scale is the second mechanism: linear scalarization is provably not scale-invariant, and hypervolume collapses from 6.9×10⁷ to 2.6×10⁷ under 10× penalty scaling (Abdolmaleki et al., 2020; E2). A degraded model rescales error magnitudes, and bounded fulfillments absorb that rescaling where linear weights do not — the cleanest published motivation for the mismatch-robustness story.

Three theoretical objections should be pre-empted rather than discovered. Nonlinear utilities break Bellman additivity, so MPPI's per-rollout scalarization and a critic's utility-of-expectations optimize different policies (Hayes et al., 2022; E3). Consistent aggregation across diverse time preferences provably requires non-Markovian rewards (Pitis, 2023; E3), making discounted FQ-values an approximation without an invariance guarantee. And only potential-based transformations preserve optima (Ng et al., 1999; Eth), so FPL *changes* the optimal policy rather than shaping toward it; position it as intent specification, never as shaping. Recorded contradictions: SAC fails to train at all on multiplicatively composed rewards on Ant (Mysore et al., 2021); reward-machine-derived shaping *decreased* performance for all methods in Water World (Toro Icarte et al., 2022); and hard lexicographic ordering needs a slack below an unknowable value gap (Skalse et al., 2022), consonant with the in-repo finding that hard shields lose to soft offsets.

### 3.6 MPC-as-teacher RL, with a quantified speed-up table

Teacher and student couple in four places: action cloning with distribution correction (Ross et al., 2010; Kahn et al., 2016), objective-bearing losses that embed the teacher's cost in the student loss (Carius et al., 2020; Reske et al., 2021), cost sharing by construction (Levine et al., 2015; Lowrey et al., 2018; Hansen et al., 2022, 2023; Wang et al., 2025), and pure data-side coupling with the reward untouched (Xing et al., 2026; Ball et al., 2023; Uchendu et al., 2022).

| Work | Coupling regime | Quantified effect | Baseline | Grade |
|---|---|---|---|---|
| Youm et al. (2023), IFM | DAgger then PPO, **deliberately different** reward | ~2× fewer iterations (~2000 vs ~4000); student 90% vs teacher 0% on 7.5 cm steps; 15× amortization | RL from scratch | E2 |
| Lowrey et al. (2018), POLO | Value trained on the planner's own reward; 80–120 CPU rollouts | Humanoid getup 24 vs 128 core-hours; in-hand 1 vs 500 CPU-hours; 8× horizon reduction | Model-free RL / trajopt | E2 |
| Wang et al. (2025), BMPC | Policy imitates a policy-seeded MPC expert; shared value | 3–4× data efficiency (90k vs 360k steps) | TD-MPC2 | E2 |
| Hansen et al. (2022), TD-MPC | Planner and policy share one learned value; policy seeds ~5% of 512 samples | 38-DoF Dog in ~1M steps; 16× wall-clock | LOOP | E1 (design replicated) |
| Levine et al. (2015), GPS | BADMM: teacher and student provably share one cost | Complete visuomotor skills from 156–288 real trials | Perception pipelines | E2 |
| Carius et al. (2020), MPC-Net | Hamiltonian loss carries teacher cost and value gradient | Stable ANYmal gaits from <10 min data; 0.125 ms vs 38 ms online | Behavioral cloning | E2 |
| Reske et al. (2021) | Same, multi-gait | 4× longer rough-terrain survival (10.6 s vs 2.6 s) on **identical data** | Behavioral cloning | E2 |
| Kurtz et al. (2025), GPC | Sampling-MPC teacher as score ascent, no demos | Matches/beats PPO at equal data on 7 systems to 29 DoF; **pure distillation fails** on humanoid standup | PPO | E2 (*ICRA 2026*) |
| Xing et al. (2026), MPC-Injection | 25% MPC transitions in replay, reward unchanged | Matches a ~21-term shaped reward in gait quality; **no steps-to-threshold speedup reported** | Shaped-reward RL | E4 |
| Li et al. (2026), MPC-guided RL | MPC trajectories converted into the PPO student's **reward** | Humanoid locomotion and loco-manipulation at scale | — | E4 |

Pure distillation is capped by the teacher — MPC-Net states outright that the student cannot outperform the MPC, and RL fine-tuning is what lifts it above — while amortization pays a replicated 15–300× online-compute dividend. But **no paper isolates the shared-spec ingredient.** GPS shares cost by construction and works; PLATO's student never sees the cost and works; IFM's fine-tuning reward is deliberately different and works; MPC-Injection changes no reward and still matches a twenty-one-term shaped reward. At scale, amortized planners often do no better than well-tuned model-free RL (Byravan et al., 2021). The theoretical bridge is bent too: policy gradient reduces to MPPI only under exp-transformed returns (Li & Chen, 2025; E4), so a student on raw fulfillments and a teacher softmaxing exp(−S) with S = −log(u) are not optimizing identical objectives (see also Bhole et al., 2025).

---

## 4. Novelty and gap analysis

**What remains new for H-A after the STL line.** Not the representation. wSTL† supplies weights and priorities over mean-family robustness functionals for control synthesis, AGM supplies bounded mean aggregation, D-GMSR supplies sound-and-complete power-mean composition, and Halder and Bouzid already run logic-composed and priority-ordered costs inside path-integral samplers, the latter in real-time MPPI at 220–240 rollouts, CUDA-parallelized on an RTX 5060 over kinematic-bicycle dynamics. The rollout count is comparable to the program's budget; the per-rollout cost is not, since full-physics CPU MuJoCo rollouts are orders of magnitude more expensive, so Bouzid is not a like-for-like occupant of the K ≤ 256 CPU slot. A related-work section claiming a new cost formalism will still be torn apart. What remains new is three empirical facts plus one theoretical opening:

1. **Satisfied-band clipped-ramp atoms** rather than signed robustness margins: a cost geometry with no STL counterpart, sitting between Tube-MPPI's brittle indicators and dense tuned norms.
2. **A single fixed spec Pareto-dominating a tuned weight family under model mismatch** at K ≤ 256 on CPU with the sampler held fixed, a comparison no paper in any cluster has run, in a regime the field's own survey says is not even a design axis.
3. **Soft priority offsets outperforming hard lexicographic shields**, against a literature that has only ever deployed hard scalarization and has documented its fragility analytically.
4. **Instantiating RMPPI's free-energy growth bound or RC-MPPI's distortion bound for a bounded saturating cost family.** Bounded costs trivially bound free energy, and no instantiation for a saturating cost family appears in the 214-source corpus (§2). This is the only route to a mechanism claim rather than a correlation.

Three concessions belong in the same breath. Cross-morphology objective generality is already demonstrated in sampling MPC by reference-free MPC with plain linear costs, attributed to the sampler (Schramm et al., 2025), and in RL at large procedurally generated embodiment counts with a general reward formulation (Ai et al., 2025; screened but not deep-read, so its exact embodiment count and any concession about reward-scale drift are not verified here). And a forward-citation check of the FPL paper, run 20 August 2026, found no case of FPL inside any MPC (four of five citing works retrieved; see §9): a benchmarking platform (Bandyopadhyay et al., 2026), the offline RRT-η adopter (Ahmad et al., 2026), a MORL value-interference paper (Vamplew et al., 2024) and a sim-adaptation paper (El Mabsout et al., 2023). The fifth, *Accelerating Lyapunov-Stable Neural Control using Fulfillment Priority Logic*, has no arXiv ID and no retrievable landing page, and on its title is a learning rather than an MPC use. The slot appears open; the window is closing from the STL side.

**H-B shared-spec differentiation matrix.** The question is not whether shared objectives exist, since they are a settled design pattern, but whether a *hand-written, interpretable, bounded, composed* spec shared between a sampling-MPC cost and an RL reward is occupied.

| Work | Teacher type | Student reward source | Spec shared? | Hand-interpretable? | Bounded / composed? |
|---|---|---|---|---|---|
| GPS (Levine et al., 2015) | iLQG local trajectory optimizers | Same task cost via BADMM + KL | Yes, by construction | Yes | No — unbounded quadratic, linear composition |
| POLO (Lowrey et al., 2018) | MPPI, 80–120 CPU rollouts | Env reward; value fitted to it | Yes, by construction | Yes | No — scalar env reward |
| MPC-Net (Carius et al., 2020) | Whole-body NMPC | None — Hamiltonian loss from cost + value gradient | Implicitly, via the loss | Yes | No |
| TD-MPC / TD-MPC2 (Hansen et al., 2022, 2023) | MPPI over a learned latent model | Learned Q shared with planner | Yes, but the shared object is **learned** | **No** — latent value | Return-normalized, not composed |
| BMPC (Wang et al., 2025) | Policy-seeded MPC expert | Shared learned value | Yes, learned | No | No |
| MPC-guided RL (Li et al., 2026) | Batched GPU centroidal MPC | **MPC trajectories converted into PPO reward** | Partly — guidance enters the reward channel | Partly | No |
| Jump-start / RLPD (Uchendu et al., 2022; Ball et al., 2023) | Any prior policy or dataset | Unchanged env reward | No | n/a | No |
| MPC-Injection (Xing et al., 2026) | Locomotion MPC, 25% replay | Unchanged env reward | No | n/a | No |
| **H-B as proposed** | FPL-MPPI, K ≤ 256 CPU | The same FPL spec as reward | Yes, and the shared object is a **written formula** | **Yes** | **Yes — bounded atoms, power-mean composition** |

The bottom row is genuinely empty. Its expected value is nonetheless undermined from both sides: the generic speedup is known art with a ~2× anchor, and the spec-sharing increment has a theoretical reason to be nil. "B beats RL from scratch" is unpublishable as novelty. Only "B beats C," same teacher with a mismatched fine-tuning reward, carries new content, and that is the comparison theory predicts may come out null.

---

## 5. Evidence-graded hypothesis assessment

### H-A — verdict: supported with caveats

**Supports.** Two things are undisputed, a theorem and one replicated empirical block. The theorem is the cost-agnosticism of the path-integral derivation (Williams et al., 2017; Eth). The replicated block is real-time CPU sampling MPC across humanoid, quadruped and hand at tiny K (Howell et al., 2022; Alvarez-Padilla et al., 2024; Schramm et al., 2025; Kicki, 2025; E1). Everything else in the support column is adjacent-setting analogy at E2, worth inventorying item by item because that is the layer a reviewer will probe.

| E2 source | Finding | Bearing on H-A |
|---|---|---|
| Wang et al. (2021) | Transform design is a free axis; r > 1 cuts cost variance 60–85% at N = 64 | Places FPL in the weight-transform family |
| Yin et al. (2022) | CVaR tail reshaping cuts collisions 55.7–80% at equal lap time | Tail compression is the rival mechanism |
| Mehdipour et al. (2019) | Bounded mean aggregation survives noise at ~59% vs ~42% | Bounded composition buys robustness per se |
| Li et al. (2016) | Logic reward 100% on hardware where tuned shaping failed | Coefficient-magnitude brittleness is real |
| Mysore et al. (2021) | Bounded conjunctive composition; current 22.87 → 5.86 A | Multiplicative AND transfers to sim-to-real |
| Abdolmaleki et al. (2020) | Linear scalarization not scale-invariant; hypervolume 6.9→2.6×10⁷ | Motivates static normalization under mismatch |
| Trevisan & Alonso-Mora (2024) | 0 collisions at K = 50 against collisions at K = 2000 | Proposal quality beats sample count (§7) |
| Lowrey et al. (2018) | Hard tasks solved at 80–120 rollouts | CPU budget suffices for contact-rich control |
| Kicki (2025) | Single-core MPPI at N = 100–256 across morphologies | CPU budget suffices across embodiments |

E3 supplies mechanism only: RMPPI's Lipschitz-scaled bound, the single-passing pathology of min-like operators (van Krieken et al., 2022), the survey's silence on cost-representation design (Honda, 2025). Every support that is *specific to bounded-fulfillment costs* is E4, and §6 states the selection effect that follows.

**Contradicts.** Sampling parameterization dominates cost design on the target morphology class (Tao et al., 2026; E4; numbers in §7.1). Fixed-covariance MPPI largely fails torque-level agile quadruped tasks at 0.2–0.3 success while annealing buys 92–126% under 2 kg mass mismatch (Xue et al., 2024; E2). Bounded costs with zero-gradient regions make warm-started MPPI brittle (Williams et al., 2018; E2). Plain MPPI breaks on sparse nested logic costs (Zheng et al., 2026; E4). Nominal-model MPC loses 0%-to-90% against an RL-fine-tuned student under unmodeled contact (Youm et al., 2023; E2). And a single Stein guide particle beats vanilla MPPI 3–4× on collisions at 11 ms on CPU (Honda et al., 2023; E2).

**Boundary conditions.** *Holds:* CPU feasibility at K = 30–256 across morphologies, though every such result uses spline or low-pass parameterization rather than per-timestep Gaussian noise. *Breaks:* contact-rich terrain under per-timestep Gaussian sampling; when all atoms clip and the sloped region vanishes; on sparse temporally nested specs; on torque-level agility, which has no published existence proof at fixed covariance and K ≤ 256. *Contested:* mismatch, where published evidence covers moderate *parametric* mismatch only and the in-repo protocol varies one parameter class (§6). *Open:* attribution on both axes, composition versus boundedness and cost versus sampler.

**In-repo evidence base.** Four platforms where one FPL spec wins on both dynamic competition and friction-type robustness: hopper, walker, Barkour-class quadruped (Caluwaerts et al., 2023), LEAP-hand cube (Shaw et al., 2023). Three Unitree G1 nulls: g1_reach, g1_reach_aggressive, g1_walk. The boundary condition they yield, that FPL needs dynamic competition with a safe and productive operating point, is the most informative sentence in the empirical program and should headline rather than hide. No pendulum FPL result exists: the only pendulum artifacts in the repo (`analytic_mppi/tasks/pendulum.py`, `verification/com_study/pendulum_com_*.py`) are a single-link centre-of-mass reference feasibility demo with no FPL objective and no win/null verdict, and they must not be counted as a morphology. Seed counts are heterogeneous (30 / 16 / 12 / 6) and at least one verdict flipped with budget, so every cell must carry its own n and CI. Section 6 gives the count, the three nulls in detail and the protocol.

**Minimal settling experiment.** Ablations A1–A5 of §6, run as one five-arm cost-representation comparison with the sampler held fixed, on the existing suite plus one contact-rich task, under degraded friction *and* +10–20% mass *and* actuator lag, reporting success with Wilson 95% CIs, tracking error and catastrophic-failure count at ≥30 seeds per cell. If A1 or A2 matches FPL, H-A deflates to "bounded costs help MPPI," a real but much weaker claim with long-standing prior art.

### H-B — verdict: supported with caveats; it is a compound claim

**Half one — "an MPC/MPPI teacher accelerates RL versus scratch" — is E1/E2 settled**, with the ~2× anchor and the multi-group replication of §3.6. Compute is not a barrier: teacher-in-loop value learning at 80–120 rollouts, policy-plus-MPPI coupling at K = 128 on CPU at ~280 Hz.

**Half two — "sharing the same FPL spec is what makes it work" — has zero supports** and four headwinds. The anchor speedup arrives with a deliberately different fine-tuning reward, credited to basin selection (E2). MPC-Injection reproduces a twenty-one-term shaped reward's gait quality with the reward untouched (E4). Under nonlinear utilities the Bellman equation does not decompose, so per-rollout utility and utility-of-expectation give different optima, the gap maximal at min-like p = −1 (Hayes et al., 2022; E3). And MPPI/PG equivalence holds only under exp-transformed returns (Li & Chen, 2025; E4). "Same spec" means same intent, not provably same optimum.

**A poison pill specific to the in-repo spec.** The teacher composition is min-fulfillment over atoms times weakest-link over time, so the return is dominated by the worst atom at the worst timestep. Min-like operators are single-passing (van Krieken et al., 2022), SAC fails to train at all on a conjunctive multiplicative reward on Ant (Mysore et al., 2021), and satisfied bands give zero reward gradient once inside the band, which is fine for ranking sampled rollouts and hostile to policy-gradient variance near the optimum. Meanwhile the FPL paper's own RL success used *discounted FQ-values*, not weakest-link. "The student shares the same spec" is therefore unimplementable as stated: either the student gets the weakest-link return, predicted untrainable and non-Bellman-decomposable, or a discounted surrogate, which is a different objective and collapses the differentiator into the generic basin effect.

**Minimal settling experiment.** Hopper first, then the quadruped, on the in-repo CPU stack. Arms: (A) RL from scratch on the FPL reward, one on-policy (PPO) and one off-policy learner given the SAC precedent; (B) DAgger-imitate the FPL-MPPI teacher, then fine-tune on the *same* FPL reward; (C) same teacher, fine-tune on a mismatched simple tracking reward; optionally (D) student injected as biased proposals during data collection (Trevisan & Alonso-Mora, 2024). Metric: environment steps and gradient iterations to a fixed fulfillment threshold on the spec's own scale, which sidesteps the Ng non-invariance confound. Thirty seeds, median steps-to-threshold with bootstrap CIs. **Pre-registered decision rule: the differentiating claim is supported only if B is significantly faster or better than C.** B ≤ 0.5·A alone reproduces known art. If B ≈ C, report the deflation, basin selection rather than spec sharing, as the finding. Gate the campaign on a pilot showing the FPL reward is trainable from scratch at all.

---

## 6. Threats to claims

*Every numeral in this section is an in-repo, unpublished, single-group result unless it carries a citation.*

**Morphology-count inflation (CRITICAL, confirmed).** The claim as propagated upward lists six morphologies including the G1 and a pendulum. The repo records four wins (hopper, walker, Barkour-class quadruped, LEAP-hand cube) and three Unitree G1 nulls, and no pendulum FPL result exists — the only pendulum artifacts in the repo (`analytic_mppi/tasks/pendulum.py`, `verification/com_study/pendulum_com_*.py`) are a single-link centre-of-mass reference feasibility demo with no FPL objective and no win/null verdict. The nulls are distinct failures, not one repeated: in g1_reach a conservative linear weight dominates FPL at every difficulty; in g1_reach_aggressive FPL pitches below the standing floor at a 0–17% safe rate; in g1_walk both controllers collapse, which is a task-capability gap rather than a method result. A reviewer who runs the repro scripts finds the inflated count in an afternoon, and it poisons every other number. *Response:* state four wins and three humanoid nulls, frame the G1 as the boundary condition (FPL needs dynamic competition with a safe, productive operating point) and never list it as supported. Every "one spec, N morphologies" sentence uses N = 4 and cites the null in the same paragraph. The pendulum demo, if mentioned at all, is named as a CoM-reference feasibility check and excluded from the count.

**Boundedness confound (CRITICAL, partially mitigated).** Bounded saturating costs improving robustness under model uncertainty is long-standing prior art. PILCO's saturating cost 1 − exp(−d²)† is the canonical instance; the † entry resolves to the 2015 journal version, so the ICML 2011 original must be retrieved with the other † items before any priority claim goes into print. MJPC ships an exponential risk transform (e^(Rl) − 1)/R in the very framework FPL competes against (Howell et al., 2022), and Tsallis and CVaR occupy the same slot inside MPPI. The mitigation is real: those transforms act on the *rollout cost distribution* whereas FPL's power mean acts *across objectives before temporal aggregation*, and none has satisfied bands or priority structure. But only the ablation shows the cross-objective axis carries weight beyond boundedness. *Response:* the clipped-linear ablation (A1) is mandatory before submission and is the single most important open question in the program; until it runs, attribute nothing to composition or priorities.

**Strawman baseline (CRITICAL, confirmed).** The in-repo linear family is power_mean(f, p = 1, w) over the *same* clipped atoms as FPL, while deployed linear costs are unbounded weighted norms whose magnitude near catastrophe is what encodes "falling is very bad." Capping every atom at 1 means a rollout that falls mid-horizon costs the p = 1 composite only marginally more than one that wobbles, and min-composition trivially restores that expressiveness. The cheapest rival explanation for every headline is therefore "FPL ≈ bounded progress reward plus a fall detector," since weakest-link-over-time times atom-min is functionally a did-anything-go-wrong indicator. The linear family also sweeps *one* weight with three to five fixed, a 1-D slice of a 4–6-D simplex, against an iteratively designed spec. Honest flip side: both families share bounded atoms, so the boundedness-versus-composition worry is *partially* controlled already. *Response:* add the unbounded MJPC-style arm with a logged multi-weight budget, the bounded-p=1-plus-fall-penalty arm, and one risk-transform rival; sweep ≥2 linear weights jointly; report the spec's design-iteration count next to the linear tuning budget.

**Friction-only mismatch protocol (MAJOR, confirmed).** Both robustness axes are the same physical parameter, contact friction, appearing as ground traction and as fingertip grip. Friction reduction punishes aggression while leaving a safe operating point, which is definitionally where a min-fulfillment floor must help, so protocol and mechanism are circularly matched. The repo already concedes the claim fails for mass and actuator mismatch, where FPL sits on the linear frontier because the mismatch makes the task uniformly harder rather than punishing aggression, and the cube operating point (46° roll) was chosen because FPL collapses with everyone at larger angles. *Response:* narrow the claim to aggression-punishing, friction-type mismatch; publish the mass and actuator null rows as first-class results with matched seeds and CIs; state that traction and grip are one parameter class; justify 46° and show the collapse beyond it.

**"Zero tuning" launders per-morphology engineering (MAJOR, partially mitigated).** What is shared is the composition rule. Thresholds in physical units, ramp shapes and the atom *sets* are per-robot: the quadruped required heading and posture atoms added specifically to kill crabbing and belly-sprawl. Each atom contributes roughly two thresholds, a shape choice and an include/exclude decision, plausibly as many degrees of freedom as the four weights it replaces, and the global spec was finalized after seeing all robots, so the evaluation is in-sample. *Response:* say "zero cross-objective trade-off weights; per-robot atoms specified once from physical intent"; publish a per-robot table counting every designed quantity next to the linear weight count; pre-register one new morphology with the atoms frozen beforehand.

**Metric construction (MAJOR, partially mitigated).** "Productive speed" = speed × survival is an in-house composite that maximally rewards balanced controllers, and conditioning the comparator on "fall rate ≤ FPL's" draws it from the timid end of the family; the +209% hopper headline is against a 0.30 m/s near-static baseline, and one cell returns no comparator at all. *Response:* make the (speed, survival) Pareto frontier the primary result and the composite a summary statistic; show one alternative pricing of survival and confirm the ordering is invariant; name every headline percentage's comparator absolute speed and survival; state the CMA-cube tie (−4% at K = 256) and the missing quadruped cross-sampler cell in the main text.

**Evidence-table selection effects (MAJOR, partially mitigated).** Every support specific to bounded-fulfillment costs is E4: FPL itself (El Mabsout et al., 2025), RRT-η's adoption (Ahmad et al., 2026), RC-MPPI's distortion bound (Yoon & Kim, 2026a) and Bouzid's real-time logic MPPI (Bouzid et al., 2026) — four preprints with no accepted venue as of the search date, one of them from the originating group. The morphology and CPU-budget premises are graded higher because their sources are accepted: Schramm et al. (2025) and Zhang et al. (2025) are E2 at *ICRA 2026*, Kicki (2025) is E2, while Seo et al. (2026) remains E4. That asymmetry is the problem in one sentence: the evidence is strongest exactly where it is least specific to the claim, since the E1s cover only generic admissibility and CPU feasibility and the E2s are adjacent-setting analogies. *Response:* state in related work that no peer-reviewed evaluation of FPL, or of any bounded-fulfillment power-mean cost, inside sampling MPC exists; that the FPL literature is two 2025–26 preprints, one from the originating group; and that adjacent evidence is motivation, never validation.

### Mandatory ablations before submission

| # | Ablation | What it decides | Cost | Blocking? |
|---|---|---|---|---|
| A1 | Clipped atoms, p = 1, no priorities | Whether anything attributes to composition rather than boundedness | Low — reuses the sweep harness | **Yes** |
| A2 | Bounded atoms, p = 1, **plus explicit fall/termination penalty** | Whether FPL reduces to "bounded progress reward + fall detector" | Low | **Yes** |
| A3 | Unbounded MJPC-style norms, ≥2 weights swept jointly, budget logged | Whether the win survives the baseline practitioners deploy | Medium | **Yes** |
| A4 | Mass and actuator-lag mismatch rows at full seed count | Converts a known in-repo null into a scoped first-class result | Low | **Yes** |
| A5 | One risk-transform rival (Tsallis exp_q, or CVaR over the existing K) | Whether the effect is a weight transform | Low — O(K), zero extra rollouts | Yes |
| A6 | Spline/knot parameterization held fixed across all cost arms | Removes the sampler-attribution confound; also a capability upgrade | Medium | Strongly advised |
| A7 | wSTL/AGM-style weighted generalized-mean STL cost, ≥2 morphologies | If it matches, the claim becomes "bounded mean-composed logic costs beat tuned linear weights," FPL one instantiation | Medium | Strongly advised |
| A8 | Per-solve ESS and free-energy logging across traction levels | Direct evidence for the bounded-weight-ratio mechanism | Negligible | Advised |
| A9 | One prospective, pre-registered held-out morphology | The only thing converting portability from in-sample to a claim | Medium | Advised |

---

## 7. Sampling: what the literature says and what to try next

### 7.1 Why the in-repo clean negative does not generalize

The six in-repo failures (adaptive covariance, colored noise, GMM plus stochastic-universal-sampling resampling, evolution-strategy gradients, backpropagation through time, hard shields) share one property: they adapt the *statistics* of the Gaussian noise, or replace the estimator. One caveat on the ledger before it is used. Hard shields are a constraint mechanism, and the finding that soft priority offsets beat them is a cost-side result in §5 and §6; the paper counts shields on the cost side, and nothing in the sampler argument below rests on them. Sorting the remaining literature by where each mechanism intervenes dissolves the paradox, because the field's winners almost never live where the in-repo negatives do. They concentrate on the weight transform, the schedule, and *structural injection* into the proposal, plus one proposal-side subfamily untried in-repo: changing what the noise is applied to.

**Action parameterization is the most replicated empirical finding in sampling MPC and is untried in-repo.** Four independent groups converge on cubic or Hermite splines with roughly four knots (Howell et al., 2022; Alvarez-Padilla et al., 2024; Schramm et al., 2025; Tao et al., 2026). The controlled number: under *identical costs*, per-timestep Gaussian noise scores 60% / 0% / 0% on flat / stairs / box traversal where four-knot cubic splines score 100%, and eight knots degrades robustness (Tao et al., 2026; E4, but a within-paper ablation). If the in-repo sampler is per-timestep Gaussian, then on contact-rich terrain no cost representation can rescue it, and every portability claim is hostage to the sampler rather than the cost. This is at once the largest threat to H-A's generality and the cheapest capability upgrade available. Judo supplies a Python-plus-MuJoCo CPU sampling-MPC harness with standardized tasks and controllers (Li et al., 2025, *Judo* — not to be confused with Li & Chen, 2025, the MPPI/policy-gradient equivalence paper of §3.6), so the A6 refactor need not start from a blank file.

**Uniform-passband low-pass filtering is a distinct member of the frequency family from the colored noise that failed.** LP-MPPI filters the perturbations through a Butterworth low-pass and criticizes colored noise explicitly for over-damping *within* the band, reporting single-CPU-core improvements across Gymnasium morphologies, two quadrupeds and a real race car (Kicki, 2025; E2; numbers under P2 in §7.2). The repo's colored-noise negative may therefore have closed the frequency axis by testing the wrong member of the family, and candidate P2 is the falsifiable version of that suspicion.

**Informed proposals beat raw sample count outright.** Biased-MPPI injects ancillary-controller rollouts (LQR, energy-based, braking, learned predictors) as *designated samples* rather than adapting noise, re-deriving the update for arbitrary mixtures so the cross-terms cancel and weights simplify to pure exp(−S/λ). At K = 50 it achieved zero collisions where standard MPPI still had collisions at K = 2000, and 0 versus 6 on a real Jackal at K = 300 (Trevisan & Alonso-Mora, 2024; E2). Informed proposals are *cheaper*, not costlier, than the standard derivation suggests; heavier-tailed mixtures point the same way (Mohamed et al., 2022). And one guide particle is enough: SVG-MPPI cuts collision rate 13.6% → 4.0% in simulation and 55.6% → 15.4% on hardware at 11 ms on CPU (Honda et al., 2023; E2). The lesson is not "mode-seeking is useless" but "do not replace the estimator."

The signature across all four is identical: **restrict or reshape the noise support toward physically plausible, low-frequency, task-informed controls, rather than adapting covariance online or resampling elites.** That is precisely the family that failed in-repo. Add the annealing evidence of §3.3 — every method that decisively beats fixed-Gaussian MPPI pays for it with samples or iterations the CPU budget does not have — and the in-repo clean negative reads as a **budget theorem about one subfamily, not a mechanism theorem about sampling**. Reporting it that way is stronger and more defensible than "sampling doesn't matter."

### 7.2 Nine FPL-aware candidates, by side

**Weight side.** *W1 — Tsallis deformed-exponential weighting on bounded fulfillment, with the linear-cost confound arm.* Replace exp(−S_k/λ) with w_k = (1 − (1 − u_k)/γ)₊^(1/(r−1)) applied directly to the rollout fulfillment u_k. Here r is the deformation exponent of the Tsallis deformed exponential (Wang et al., 2021): r → 1 recovers standard MPPI and larger r moves toward CEM's top-elite indicator, while γ plays λ's role, setting the fulfillment deficit beyond which a rollout gets zero weight. Because FPL atoms give an *absolute* scale, γ can be fixed globally rather than per task: pick one value on the hopper, freeze it, reuse it unchanged everywhere, and name it in the paper. That fixed-γ transfer is the whole claim; if γ must be re-picked per morphology, W1 has failed on its own terms. Anchor: modest r > 1 cut cost variance 60–85% at N = 64 (Wang et al., 2021). Outside the failed set: proposal, warm start and covariance untouched. Cost: O(K) scalars replacing the exp. Predicted failure: u^(1/λ) is already a power-law weight on a bounded score, so Tsallis may be redundant on top of FPL, and large r causes elite collapse. Test: hopper plus walker mismatch sweep, 30 seeds, four arms (FPL, FPL+Tsallis r = 2, linear-best+Tsallis, linear-best). **Doubles as mandatory ablation A5.**

*W2 — sample-free CVaR on the min atom over the existing K rollouts.* Compute m_k = min over atoms and horizon (already available from the power-mean decomposition), take the empirical α-quantile at α ≈ 0.2, and multiply each weight by sigmoid((m_k − q_α)/τ). This is the matched-budget version of RA-MPPI's 55–80% collision reduction, which cost 307k rollouts per step. Outside the failed set: soft, continuous and population-relative, unlike hard shields that zeroed rollouts. Cost: one O(K log K) sort. Predicted failure: p < 0 is already min-like, so the discount may double-count; small τ degenerates toward a shield and reproduces exploit-collapse during hopper flight phases. Test: hopper plus quadruped, 30 seeds, FPL vs FPL+CVaR vs linear-best+CVaR. **A null is a headline positive: matched-budget evidence that FPL already delivers implicit tail compression.**

*W3 — rank-based weighting (mechanism probe).* Replace magnitude weights with w_k = exp(−rank_k/h), h ≈ K/8, making the update invariant to any monotone distortion of the cost scale — the limiting case of the hypothesized FPL mechanism (anchors: Yoon & Kim, 2026a; Gandhi et al., 2021). Cost: one sort. Predicted failure: discarding magnitude should lose on nominal dynamics. The key cell is rank-on-*linear*: if it recovers most of FPL's robustness margin over linear-softmax, magnitude-distortion immunity is the mechanism.

*W4 — ESS-targeted per-solve temperature by bisection (mechanism probe).* Choose λ each solve so ESS = (Σw)²/Σw² hits ρ·K. With FPL the weights are u_k^(1/λ) on a bounded score, so ESS is monotone in λ and bisection has guaranteed brackets, a stability property unbounded quadratic costs do not give (anchor: Wang et al., 2026a). Cost: ~10 extra O(K) evaluations. Predicted failure: a well-tuned fixed λ plus warm start may already sit near target, so gains appear only in transients and a wrong ρ reintroduces one tuned knob under another name. **The logging half is the valuable half** (ablation A8).

**Schedule side.** *S1 — across-solve temperature keyed to executed absolute fulfillment.* Track u_exec, the fulfillment the executed state actually achieved, and set λ_t = λ_min + (λ_max − λ_min)(1 − u_exec)^κ: cool toward exploitation when satisfied, heat to flatten the softmax and let the warm start escape a now-wrong plan when mismatch bites. This is RC-MPPI's residual-driven relaxation keyed to the spec's own absolute signal instead of a dynamics-residual estimator, exploiting the one thing FPL uniquely provides — a task-meaningful absolute performance level needing no per-morphology normalization (anchor: Yoon & Kim, 2026a, where residual-driven adaptation lifted success 0.64 → 0.94 on a point mass with 0.9 s actuator lag and 0.56 → 0.96 on a 2R manipulator with inertial error, both in simulation at K = 1024–2048). Cost: one scalar EMA per control step, the cheapest idea here. Predicted failure: u_exec drops only *after* mismatch has hurt, so heating may arrive one recovery window too late on fast falls. Test: hopper plus walker, 30 seeds, fixed versus scheduled λ at two settings.

*S2 — horizon-graded covariance schedule (single-iteration DIAL-style annealing).* Fix σ(h) = σ₀(1 + c·h/H): small perturbations near t = 0 where the warm start is trustworthy, larger along the horizon where the shifted plan is stale. This transplants one of DIAL-MPC's two annealing axes into a budget where the across-iteration axis is unaffordable, fixed offline rather than adapted online — and DIAL-MPC gives *no schedule ablation*, so the along-horizon axis alone is untested even there (Xue et al., 2024). Cost: one precomputed length-H scale vector. Predicted failure: with H ≈ 20–40 the grading has little room, and DIAL's gains may be inseparable from its iteration-rich outer loop. Test: c ∈ {0, 1, 3} on hopper plus quadruped.

**Interaction side.** *I1 — fulfillment-space diversity reweighting.* Each rollout yields f_k ∈ [0,1]^A with A ≈ 3–6, a cheap bounded morphology-independent behavioral descriptor. After standard weighting, apply w_k ← w_k / (Σ_j w_j k(f_k, f_j))^β with β ≈ 0.3, preserving weight mass across distinct atom-tradeoff modes instead of collapsing onto one. This imports SVG-MPPI's lesson — preserve the closed-form update, add only a light interaction term — in fulfillment space rather than control space (Honda et al., 2023). Cost: O(K²A) ≈ 3×10⁵ multiply-adds at K = 256, sub-millisecond in numpy. Predicted failure: warm-started locomotion is mostly unimodal, so repulsion buys nothing and slightly biases the estimator. Test: 30 seeds on quadruped plus hopper, logging fulfillment-space clusters per solve; if the baseline is unimodal on >95% of steps, the negative is explained in one plot.

**Proposal side (structural injection and support reshaping, not noise-statistics adaptation).** *P1 — per-atom greedy ancillary proposals as designated Biased-MPPI samples.* Reserve ~5 of the K rollouts for designated non-Gaussian samples carrying spec structure: the zero-perturbation warm start, a braking sequence, and one greedy proposal per priority atom (a posture-recovery PD sequence for uprightness, a deceleration ramp for survival). Biased-MPPI's re-derivation makes the cross-terms cancel so weights stay pure exp(−S/λ) (Trevisan & Alonso-Mora, 2024). These matter exactly when mismatch makes the Gaussian bundle around a stale warm start uniformly bad. Cost: zero net extra rollouts — designated samples replace 5 Gaussian ones. Predicted failure: hand-coded atom-greedy controllers for underactuated hoppers are hard to make even locally competent, so their rollouts score worst and get ~zero weight; a braking proposal that occasionally wins could cause conservative freezing. Test: log the softmax weight captured by designated samples, binned by traction level; below 1% even at the slipperiest level, the mechanism is dead and the log says why.

*P2 — Butterworth uniform-passband perturbation filtering.* Pass each sampled perturbation sequence through a low-order Butterworth low-pass filter before adding it to the warm-started nominal. The mechanism is support reshaping, not statistics adaptation: covariance, weights and warm start are untouched, and the passband is flat, so components below the cutoff are not attenuated — precisely the criticism LP-MPPI levels at colored noise, whose 1/f^β envelope over-damps *within* the band. Outside the failed set: the in-repo negative tested colored noise, a different member of the frequency family, so the axis was closed by one instance rather than by the family. Anchor: Kicki (2025; E2) — a single CPU core at N = 100–256 across Gymnasium locomotion morphologies, two quadrupeds and a real F1TENTH car, roughly 2× better than ColoredMPPI on quadrupeds and 24–41% over a DIAL-MPC baseline on quadruped trot; low-frequency sampling has independent support (Vlahov et al., 2024; E2). Cost: one length-H filter pass per sampled sequence, O(K·H) scalar operations, zero extra rollouts, vectorizable alongside the existing noise draw. Predicted failure: at H ≈ 20–40 there are few resolvable frequencies, so passband shape may not matter, and a hopper's flight-to-stance transition needs high-frequency content the filter removes, so P2 may help the walker and quadruped while hurting the hopper. It also overlaps A6, since splines restrict the support in the same direction, so run the two against each other rather than in sequence. Test: hopper, walker and the Barkour-class quadruped, 30 seeds per cell, four sampler arms (per-timestep Gaussian, colored noise as tested in-repo, Butterworth at two cutoffs) crossed with two cost arms (FPL, linear-best), plus the spline arm on the walker for the overlap check. A null across all three platforms closes the frequency axis by family rather than by instance, which the in-repo negative does not.

Deliberately excluded: Stein particle transport proper (needs differentiable-simulator gradients and many iterations), explicit CVaR with auxiliary rollouts (infeasible at K ≤ 256), and covariance adaptation in any feedback form. That last exclusion should be scoped rather than asserted: online disturbance-covariance estimation has a 2026 instantiation with a stability argument attached (Yoon & Kim, 2026b; E4), so the in-repo negative is a result about the tested implementation at K ≤ 256, not a refutation of the family, and the paper should say so.

### 7.3 Ranked candidate table (the nine §7.2 candidates plus ablation A6)

| Rank | Idea | Side | Cost per solve | A positive result proves | A negative result proves |
|---|---|---|---|---|---|
| 1 | **W1** Tsallis transform + linear-cost confound arm | Weight | O(K) scalars, 0 extra rollouts | Weight-transform design is a live axis on bounded fulfillments; quantifies transform vs composition | The attribution ablation is settled: the win is not a generic weight-transform effect — the most valuable negative available |
| 2 | **Spline/knot parameterization** — ablation A6 from §7.1, not one of the nine §7.2 candidates | Proposal (structure) | Medium refactor; same K | Removes the sampler-attribution confound and likely unlocks contact-rich tasks (0%→100% precedent) | Bounds the spline finding to non-warm-started or torque-level regimes — a scoping of the field's most replicated result |
| 3 | **P2** Butterworth uniform-passband filtering | Proposal (support) | O(K·H) filter, 0 extra rollouts | The frequency axis is live and the in-repo colored-noise null was about one member, not the family | Closes the frequency axis by family — the negative the in-repo result currently only gestures at |
| 4 | **S1** fulfillment-keyed temperature schedule | Schedule | 1 scalar EMA, 0 extra rollouts | FPL's absolute scale enables a morphology-free adaptive temperature no unbounded cost can have | Bounded costs already stabilize the weights RC-MPPI's relaxation compensates for — direct mechanistic support for H-A |
| 5 | **W2** sample-free CVaR on the min atom | Weight | O(K log K), 0 extra rollouts | Cheap explicit tail reshaping composes with implicit tail compression | FPL already delivers implicit tail compression at matched budget — the headline mechanism claim |
| 6 | **W3** rank-based weighting *(probe)* | Weight | O(K log K) | If rank-on-linear recovers FPL's margin: magnitude-distortion immunity is the mechanism | The distortion-immunity hypothesis is wrong and the composition story strengthens |
| 7 | **P1** per-atom ancillary proposals | Proposal (structure) | 0 net extra rollouts | Spec structure transfers into the sampler; the untried proposal subfamily is live | Structural injection needs competent ancillary controllers, absent for underactuated systems — scopes Biased-MPPI |
| 8 | **W4** ESS-targeted temperature *(probe)* | Weight | ~10 O(K) evals | One dimensionless knob replaces per-task λ tuning across morphologies | Warm start already holds ESS near target — and the ESS logs publish regardless |
| 9 | **I1** fulfillment-space diversity reweighting | Interaction | O(K²A), sub-ms | Multimodality matters even in warm-started locomotion | Warm-started locomotion is unimodal, scoping the Stein/mode-seeking family out of this regime |
| 10 | **S2** horizon-graded covariance schedule | Schedule | 0 extra rollouts | One DIAL axis survives without the iteration-rich outer loop | Along-horizon annealing needs the iteration-rich regime — a boundary on the field's largest headline number |

Each negative closes a gap this review independently identified as open: the transform-versus-composition confound (1), the sampler-versus-cost confound (2), the frequency-family scoping the in-repo negative left undone (3), residual-driven temperature adaptation beyond toys (4), matched-budget risk baselines at K ≤ 256 (5), the bounded-cost weight-distortion mechanism (6, 8), structural proposal injection with logic-structured costs (7), and DIAL-MPC's missing schedule ablation (10). Not every row is a paper. Rows 1–5 carry standalone venue value either way and are the material for the CoRL-workshop fallback in §8; rows 6–10 are internal scoping results that belong in an appendix, not a title. Run them in rank order and report the nulls with the same seed counts and CIs as the wins.

---

## 8. Positioning and timeline

**Venues.** IEEE RA-L is rolling, with roughly six-month decisions and optional presentation transfer within 270 days of acceptance. ICRA 2027 (Seoul) takes contributed papers via PaperPlaza with a deadline the official page gave as **15 September 2026, 23:59 PST** when checked on 20 August 2026, a hard local-time deadline rather than AoE; since US Pacific observes PDT on that date, the literal PST reading is 16 September 07:59 UTC and the PDT reading 06:59 UTC, so re-verify in the final week and plan to the earlier. CoRL 2026 workshops run 12 November 2026 in Austin with deadlines still TBD, likely late September to mid October. L4DC 2027 has no site or deadline; based on prior years, expect October–November 2026.

**Recommended headline claim.** *A bounded-fulfillment, min-like objective inside standard warm-started MPPI Pareto-dominates a bounded linear-weight family under dynamic competition and friction-type model mismatch on four MuJoCo platforms at K ≤ 256 on CPU, with a disclosed humanoid boundary; the effect survives a clipped-linear control, and bounded atoms tighten the free-energy growth bound.* Everything there is measured or provable, the last clause contingent on the algebra closing.

**What to cut.** Any pendulum morphology claim (the repo's pendulum files are a CoM-reference demo, not an FPL result); the number six; "zero tuning"; "robust to model mismatch" unqualified; any claim of a new cost formalism, of first use of FPL outside RL, or of generality beyond the four tested platforms; any headline percentage that does not name its comparator's absolute speed and survival; and any H-B content beyond a labelled future-work paragraph.

**Venue logic, stated as arithmetic.** This report is written on 22 August 2026, so the ICRA deadline is 24 days away, which is 3.4 weeks. The full A1–A7 program does not fit in 24 days and no rescheduling makes it fit, so **RA-L is the primary target and ICRA 2027 is the stretch case**, not the other way round. Section 6 forces the same conclusion independently of the calendar: A3, the unbounded-baseline ablation, is marked blocking and is a multi-week joint weight sweep, so an ICRA submission is a submission with a baseline this report has already labelled a strawman.

**The 24-day ICRA attempt, with dates.** Only the low-cost items sharing the existing sweep harness are in scope.

| Dates | Days | Work |
|---|---|---|
| 22–31 Aug | 10 | A1 (clipped-linear, p = 1), A2 (bounded p = 1 plus fall penalty), A5 (risk-transform rival; A5 *is* idea W1), A4's mass and actuator-lag rows. One harness, four arms, ≥30 seeds per cell. |
| 1–5 Sep | 5 | Delta table against the STL line; wSTL† and PILCO† retrieved and written into related work; G1 boundary section drafted. **Gate on 5 Sep.** |
| 6–11 Sep | 6 | Regenerate figures with per-cell n and Wilson CIs; promote the (speed, survival) frontier over the composite; add the alternative pricing of survival. |
| 12–15 Sep | 4 | Internal review against §6's threat list, reproducibility pass, submit. |

**The 5 September gate.** Go to ICRA only if A1, A2 and A5 have all returned unambiguous results by then, meaning the clipped-linear and fall-penalty arms are clearly separated from FPL with non-overlapping CIs on at least three of the four platforms. Ambiguity on any of the three means no ICRA submission, since the central claim would then rest on an ablation that failed to settle. Explicitly outside the 24 days, not merely deferred within them: the theory block instantiating the RMPPI and RC-MPPI bounds for bounded saturating costs, A3's joint multi-weight sweep, A6's spline refactor with its contact-rich task, and A7.

**The RA-L program.** Everything above plus the cut items, on the rolling deadline: A3 through mid October, A6 and the contact-rich task through late October, the theory contribution with a five-day box and a proof-sketch fallback, then A7 and ideally A9's pre-registered held-out morphology. A CoRL workshop paper on the sampling negative plus the ESS logs (A8) timestamps the sampling result cheaply while RA-L is assembled.

**Go / no-go on an H-B pilot: no.** The differentiating comparison is arm C, which needs three trained arms at 30 seeds and does not fit on either schedule. The spec to be shared is not yet defined, weakest-link versus discounted-FQ being a real fork with the literature predicting the former may be untrainable, and a partial H-B result invites the reviewer to evaluate the paper on the weaker hypothesis. Frame H-B as future work with the pre-registered design from §5, including the planned publishable null.

---

## 9. Acknowledged limitations

Direct evidence for the central claim is thin and single-source. No peer-reviewed evaluation of FPL, or of any bounded-fulfillment power-mean cost, inside sampling MPC exists anywhere; the FPL literature is two 2025–2026 preprints, one from the originating author group and RL-only, plus one offline-planner adopter, and everything else in the support column is adjacent-setting analogy. The in-repo benchmark is one codebase, one researcher, no external replication, with heterogeneous seed counts and at least one verdict that flipped with sample budget.

The mismatch protocol is narrow, both robustness axes being contact friction in two guises; §6 states that threat and its response in full.

Several load-bearing sources have no accepted venue as of the search date. Listing them with their grades, so that the bibliography and the grades cannot drift apart:

| Load-bearing source | Status as of 20 Aug 2026 | Grade |
|---|---|---|
| El Mabsout et al. (2025), FPL | preprint, originating group | E4 |
| Ahmad et al. (2026), RRT-η | preprint | E4 |
| Yoon & Kim (2026a), RC-MPPI | preprint | E4 |
| Bouzid et al. (2026), priority-ordered STL-MPPI | preprint | E4 |
| Tao et al. (2026), spline-sampling comparison | preprint | E4 |
| Seo et al. (2026), RGB (G1 compute datapoint) | preprint | E4 |
| Zheng et al. (2026), STL-SVPIO | preprint | E4 |
| Schramm et al. (2025), reference-free morphology transfer | accepted, *ICRA 2026* | E2 |
| Zhang et al. (2025), whole-body MuJoCo MPC | accepted, *ICRA 2026* | E2 |
| Kicki (2025), LP-MPPI | accepted, *ICRA 2026* | E2 |

Three adversarially surfaced items (weighted STL, PILCO's saturating cost, and one concurrent MPC-guided RL work that was never resolved to an identifier and is therefore load-bearing nowhere; §2) sit outside the verified corpus and must be resolved before submission; venue and DOI resolution remains open for several 2026 preprints; and one FPL-citing paper, *Accelerating Lyapunov-Stable Neural Control using Fulfillment Priority Logic*, could not be located at all.

Finally, the AI-assistance disclosure. Search, screening, deep reading, evidence grading, adversarial review and the drafting of this report were performed by language-model agents. Verification was limited to landing-page fetches and cross-cluster consistency checks; no experiment in any cited paper was reproduced and no number was independently re-derived. Quantitative claims about cited works were copied from those works' text rather than re-derived, and are traceable through the per-claim records in `wf3/evidence.json`; the corpus statistics in §2 are self-reported by the search pipeline and are auditable against `wf3/bib_compact.md`; in-repo numbers are labelled as unpublished single-group results wherever they appear. Treat the synthesis as a well-sourced map, not an audited one.

---

## Bibliography

Abdolmaleki, A., et al. (2020). A distributional view on multi-objective policy optimization. *ICML 2020*. arXiv:2005.07513

Ahmad, et al. (2026). RRT^η: Sampling-based motion planning and control from STL specifications using arithmetic-geometric mean robustness. arXiv:2602.16825

Ai, et al. (2025). Towards embodiment scaling laws in robot locomotion. *CoRL 2025*. arXiv:2505.05753

Alvarez-Padilla, et al. (2024). Real-time whole-body control of legged robots with model-predictive path integral control. arXiv:2409.10469

Ball, P., et al. (2023). Efficient online reinforcement learning with offline data. *ICML 2023*. arXiv:2302.02948

Bandyopadhyay, et al. (2026). NeoRacer: An open, standardized 1:12 scale autonomous race car for benchmarking and education. arXiv:2607.26855

Bhole, et al. (2025). Unifying entropy regularization in optimal control. arXiv:2512.06109

Bohlinger, et al. (2024). One policy to run them all: An end-to-end learning approach to multi-embodiment locomotion. arXiv:2409.06366

Bouzid, et al. (2026). Autonomous driving with priority-ordered STL specifications under multimodal uncertainty. arXiv:2606.20336

Byravan, A., et al. (2021). Evaluating model-based planning and planner amortization for continuous control. arXiv:2110.03363

Caluwaerts, K., et al. (2023). Barkour: Benchmarking animal-level agility with quadruped robots. arXiv:2305.14654

Carius, J., et al. (2020). MPC-Net: A first principles guided policy search. *IEEE RA-L*. arXiv:1909.05197

Censi, A., et al. (2019). Liability, ethics, and culture-aware behavior specification using rulebooks. *ICRA 2019*. arXiv:1902.09355

Doshi, et al. (2024). Scaling cross-embodied learning. *CoRL 2024*. arXiv:2408.11812

El Mabsout, B., et al. (2023). Sim-anchored learning for on-the-fly adaptation. arXiv:2301.06987

El Mabsout, B., et al. (2025). Closing the intent-to-behavior gap via fulfillment priority logic. arXiv:2503.05818

Fazlyab, et al. (2026). Model predictive path integral control as preconditioned gradient descent. arXiv:2603.24489

Feng, G., et al. (2022). GenLoco: Generalized locomotion controllers for quadrupedal robots. *CoRL 2022*. arXiv:2209.05309

Gandhi, M., et al. (2021). Robust model predictive path integral control: Analysis and performance guarantees. *IEEE RA-L*. arXiv:2102.09027

Gilpin, Y., et al. (2020). A smooth robustness measure of signal temporal logic for symbolic control. *IEEE L-CSS*. arXiv:2006.05239

Halder, et al. (2025). Trajectory planning with signal temporal logic costs using deterministic path integral optimization. *ICRA 2025*. arXiv:2503.01476

Halder, et al. (2026). Lexicographic minimum-violation motion planning using signal temporal logic. *IEEE OJ-ITS*. arXiv:2604.20428

Hansen, N., et al. (2022). Temporal difference learning for model predictive control. *ICML 2022*. arXiv:2203.04955

Hansen, N., et al. (2023). TD-MPC2: Scalable, robust world models for continuous control. *ICLR 2024*. arXiv:2310.16828

Hayes, C. F., et al. (2022). A practical guide to multi-objective reinforcement learning and planning. *AAMAS Journal*, 36:26. arXiv:2103.09568

Honda (2025). Model predictive control via probabilistic inference: A tutorial and survey. arXiv:2511.08019

Honda, K., et al. (2023). Stein variational guided model predictive path integral control. arXiv:2309.11040

Howell, T., et al. (2022). Predictive sampling: Real-time behaviour synthesis with MuJoCo. arXiv:2212.00541

Kahn, G., et al. (2016). PLATO: Policy learning using adaptive trajectory optimization. arXiv:1603.00622

Kappen, H. J. (2005). Path integrals and symmetry breaking for optimal control theory. *J. Stat. Mech.* arXiv:physics/0505066

Kicki, P. (2025). LP-MPPI: Low-pass filtering for efficient model predictive path integral control. *ICRA 2026*. arXiv:2503.11717

Kurtz, V., et al. (2025). Generative predictive control: Flow matching policies for dynamic and difficult-to-demonstrate tasks. *ICRA 2026*. arXiv:2502.13406

Levine, S., et al. (2015). End-to-end training of deep visuomotor policies. *JMLR*. arXiv:1504.00702

Li, X., et al. (2016). Reinforcement learning with temporal logic rewards. arXiv:1612.03471

Li, A., & Chen, Y. (2025). Unifying model predictive path integral control, reinforcement learning, and diffusion models. arXiv:2502.20476

Li, et al. (2025). Judo: A user-friendly open-source package for sampling-based model predictive control. *RSS 2025 Workshop*. arXiv:2506.17184

Li, et al. (2026). Accelerating and scaling MPC-guided reinforcement learning for humanoid locomotion and manipulation. arXiv:2606.05687

Lindemann, L., & Dimarogonas, D. V. (2019). Robust control for signal temporal logic specifications using average space robustness. *Automatica*, 101, 377–387. arXiv:1607.07019

Lowrey, K., et al. (2018). Plan online, learn offline: Efficient learning and exploration via model-based control. *ICLR 2019*. arXiv:1811.01848

Mehdipour, N., et al. (2019). Arithmetic-geometric mean robustness for control from signal temporal logic specifications. *ACC 2019*. arXiv:1903.05186

Mohamed, I. S., et al. (2022). Autonomous navigation of AGVs in unknown cluttered environments: log-MPPI control strategy. *IEEE RA-L*. arXiv:2203.16599

Mohamed, I. S., et al. (2024). Towards efficient MPPI trajectory generation with unscented guidance: U-MPPI. *IEEE T-RO*. arXiv:2306.12369

Mohamed, I. S., et al. (2025). Chance-constrained sampling-based MPC for collision avoidance in uncertain dynamic environments. *IEEE RA-L*. arXiv:2501.08520

Mysore, S., et al. (2021). How to train your quadrotor: A framework for consistently smooth and responsive flight control via reinforcement learning. *ACM TCPS*, 5(4). arXiv:2012.06656

Ng, A. Y., et al. (1999). Policy invariance under reward transformations. *ICML 1999*, 278–287. doi:10.5555/645528.657613

Pant, Y. V., et al. (2017). Smooth operator: Control using the smooth robustness of temporal logic. *IEEE CCTA 2017*.

Parwana, H., et al. (2025). BR-MPPI: Barrier rate guided MPPI. arXiv:2506.07325

Pitis, S. (2023). Consistent aggregation of objectives with diverse time preferences requires non-Markovian rewards. *NeurIPS 2023*. arXiv:2310.00435

Reske, A., et al. (2021). Imitation learning from MPC for quadrupedal multi-gait control. *ICRA 2021*. arXiv:2103.14331

Ross, S., et al. (2010). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS 2011*. arXiv:1011.0686

Schramm, et al. (2025). Reference-free sampling-based model predictive control. *ICRA 2026*. arXiv:2511.19204

Seo, et al. (2026). RGB: RL guided whole-body MPPI for humanoid control. arXiv:2606.25123

Shafiee, M., et al. (2023). ManyQuadrupeds: Learning a single locomotion policy for diverse quadruped robots. *ICRA 2024*. arXiv:2310.10486

Shaw, K., et al. (2023). LEAP Hand: Low-cost, efficient, and anthropomorphic hand for robot learning. arXiv:2309.06440

Skalse, J., et al. (2022). Lexicographic multi-objective reinforcement learning. *IJCAI 2022*, 3430–3436. arXiv:2212.13769

Tao, et al. (2026). Sampling strategy design for model predictive path integral control on legged robot locomotion. arXiv:2601.01409

Theodorou, E., et al. (2010). A generalized path integral control approach to reinforcement learning. *JMLR*, 11(104), 3137–3181.

Todorov, E. (2009). Efficient computation of optimal actions. *PNAS*, 106(28), 11478–11483. doi:10.1073/pnas.0710743106

Toro Icarte, R., et al. (2022). Reward machines: Exploiting reward function structure in reinforcement learning. *JAIR*, 73, 173–208. arXiv:2010.03950

Trevisan, E., & Alonso-Mora, J. (2024). Biased-MPPI: Informing sampling-based model predictive control by fusing ancillary controllers. *IEEE RA-L*. arXiv:2401.09241

Trevisan, E., et al. (2025). Dynamic risk-aware MPPI for mobile robots in crowds. *IROS 2025*. arXiv:2506.21205

Uchendu, I., et al. (2022). Jump-start reinforcement learning. *ICML 2022*. arXiv:2204.02372

Uzun, et al. (2024). Optimization with temporal and logical specifications via generalized mean-based smooth robustness measures. arXiv:2405.10996

Vamplew, P., et al. (2024). Issues with value-based multi-objective reinforcement learning: Value function interference and overestimation sensitivity. arXiv:2402.06266

van Krieken, E., et al. (2022). Analyzing differentiable fuzzy logic operators. *Artificial Intelligence*. arXiv:2002.06100

Veer, S., et al. (2022). Receding horizon planning with rule hierarchies for autonomous vehicles. arXiv:2212.03323

Vlahov, B., et al. (2024). Low frequency sampling in model predictive path integral control. *IEEE RA-L*, 9(5), 4543–4550. arXiv:2404.03094

Wang, Z., et al. (2021). Variational inference MPC using Tsallis divergence. *RSS 2021*. arXiv:2104.00241

Wang, et al. (2025). Bootstrapped model predictive control. *ICLR 2025*. arXiv:2503.18871

Wang, et al. (2026a). Information-theoretic adaptive cooling for deterministic MPPI via entropy feedback. arXiv:2607.14245

Wang, et al. (2026b). Generalized model predictive path integral control as expectation-maximization. arXiv:2606.00317

Welikala, S., et al. (2023). Smooth robustness measures for symbolic control via signal temporal logic. arXiv:2305.09116

Williams, G., et al. (2015). Model predictive path integral control using covariance variable importance sampling. arXiv:1509.01149

Williams, G., et al. (2017). Information theoretic model predictive control: Theory and applications to autonomous driving. *IEEE T-RO*. arXiv:1707.02342

Williams, G., et al. (2018). Robust sampling based model predictive control with sparse objective information. *RSS XIV*. doi:10.15607/RSS.2018.XIV.042

Xi, et al. (2025). UniLegs: Universal multi-legged robot control through morphology-agnostic policy distillation. *IROS 2025*. arXiv:2507.22653

Xing, et al. (2026). MPC-Injection: Biasing off-policy locomotion RL toward controller-induced behavior basins. arXiv:2606.26392

Xue, H., et al. (2024). Full-order sampling-based MPC for torque-level locomotion control via diffusion-style annealing (DIAL-MPC). arXiv:2409.15610

Yi, et al. (2024). CoVO-MPC: Theoretical analysis of sampling-based MPC and optimal covariance design. arXiv:2401.07369

Yin, J., et al. (2022). Risk-aware model predictive path integral control using conditional value-at-risk. arXiv:2209.12842

Yin, J., et al. (2023). Shield model predictive path integral. *IEEE RA-L*. arXiv:2302.11719

Yoon, & Kim. (2026a). Residual-conservative model predictive path integral control. arXiv:2607.06950

Yoon, & Kim. (2026b). Adaptive MPPI with online disturbance covariance estimation: Provable stability tightening via spatial smoothing. arXiv:2607.08942

Yoon, & Kim. (2026c). Finite-sample closed-loop stability of model predictive path integral control for LTI systems. arXiv:2607.04006

Youm, D., et al. (2023). Imitating and finetuning model predictive control for robust and symmetric quadrupedal locomotion. *IEEE RA-L*. arXiv:2311.02304

Zhang, et al. (2025). Whole-body model-predictive control of legged robots with MuJoCo. *ICRA 2026*. arXiv:2503.04613

Zheng, et al. (2026). STL-SVPIO: Signal temporal logic guided Stein variational path integral optimization. arXiv:2603.13333

**Cited but outside the verified 214-source corpus (must be added before submission):** Mehdipour, Vasile & Belta (2020), *Specifying user preferences using weighted signal temporal logic*, arXiv:2010.00752†; Deisenroth, Fox & Rasmussen (2015), *Gaussian processes for data-efficient learning in robotics and control* (PILCO journal version), arXiv:1502.02860†, whose ICML 2011 original must be retrieved alongside it if the priority claim in §6 is to be made in print. Both were verified by landing-page fetch during the adversarial pass; the † markers stay until the corpus is re-run with them included.

*This bibliography lists the 87 in-corpus cited sources, every one of which is cited in the body, plus the two † items above. The remaining screened-and-verified sources are in the supplementary bibliography (`wf3/bib_compact.md`, 214 entries), and per-claim evidence for the 51 deep-read sources (84 records) is in `wf3/evidence.json`.*
