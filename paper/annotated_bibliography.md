# Annotated Bibliography

Seventy sources supporting a research program on morphology-portable MPPI with a bounded-fulfillment (FPL) objective, and on MPC-as-teacher RL. Grouped by cluster. Evidence grades follow the program's convention (E1 = canonical/replicated, E2 = peer-reviewed with direct evidence, E3 = theory or single-source analysis, E4 = 2025/2026 preprint or otherwise unreviewed).

---

## C1 — MPPI variants, robustness, and sampling-distribution design

**Williams et al. (2018). Robust Sampling Based Model Predictive Control with Sparse Objective Information. Robotics: Science and Systems (RSS XIV). https://doi.org/10.15607/RSS.2018.XIV.042**
Introduces Tube-MPPI, motivated by the observation that sparse weighted-indicator costs are interpretable and easy to specify but leave warm-started MPPI unable to recover once a disturbance pushes the nominal into a zero-gradient plateau. With an iLQG ancillary tracker, the indicator spec reached a 9 m/s target (9.39 ± 0.76 s laps) against 8 m/s for an extensively hand-tuned dense cost and 5 m/s for naive MPPI under the same disturbance. This is the same claim shape as the in-repo FPL Pareto-dominance result but purchased architecturally, and it defines the failure mode that clipped ramps with satisfied bands (bounded but sloped) are meant to avoid.
*Evidence grade: E2 | Clusters: C1 | Relevance: H-A, novelty*

**Gandhi et al. (2021). Robust Model Predictive Path Integral Control: Analysis and Performance Guarantees. IEEE Robotics and Automation Letters. https://arxiv.org/abs/2102.09027**
RMPPI diagnoses vanilla MPPI's mismatch failure as an importance sampler anchored to an open-loop nominal trajectory, and proves a free-energy growth bound decomposing into constraint slack, Monte Carlo error, and a tracking term scaled by the Lipschitz constants of the terminal and running costs. On AutoRally hardware the heavyweight machinery bought statistically indistinguishable lap times (31.07 ± 0.33 s vs 30.67 ± 0.53 s) and only higher tolerable slip (0.84 vs 0.63 rad). Both facts serve the program: bounded atoms shrink exactly the Lipschitz factors in the bound, and architectural robustness delivers marginal headline gains, supporting a cost-representation route instead.
*Evidence grade: E2 | Clusters: C1, C3, C7 | Relevance: H-A, novelty*

**Yoon & Kim (2026). Residual-Conservative Model Predictive Path Integral Control. arXiv preprint. https://arxiv.org/abs/2607.06950**
RC-MPPI adapts conservatism online from a filtered prediction-execution residual via constraint tightening, barrier-weight scaling, and temperature relaxation, lifting success under severe unmodeled mismatch from 0.64 to 0.94 (point mass with actuator lag) and 0.56 to 0.96 (planar 2R arm). Its analytic core is directly reusable: mismatch-induced distortion of the softmax weights scales as 2·C_Δ·s̄/β, where C_Δ is the cost's sensitivity to state error. Bounded clipped-ramp atoms cap C_Δ by construction, giving a mechanistic account of graceful FPL degradation; the paper is also the strongest published case that residual-adaptive schedules beat any fixed configuration, though only on toy systems.
*Evidence grade: E4 | Clusters: C1, C3, C7 | Relevance: H-A, sampling-ideas*

**Yoon and Kim (2026). Adaptive MPPI with Online Disturbance Covariance Estimation: Provable Stability Tightening via Spatial Smoothing. arXiv. https://arxiv.org/abs/2607.08942**
Derives a cell-wise recursive estimator of spatially varying disturbance covariance with reversible spatial smoothing, and proves a payoff theorem: the adaptive stability certificate becomes strictly tighter than any fixed covariance choice after a computable finite crossover time. Validation is LTI-style grid simulation at very long horizons (T = 8000). It is the cleanest formal challenge to H-A's fixed-spec, fixed-sampler claim, and its temperature/residual coupling motivates the program's schedule-side sampling ideas.
*Evidence grade: E4 | Clusters: C1, C3 | Relevance: H-A, sampling-ideas*

**Yin et al. (2022). Risk-Aware Model Predictive Path Integral Control Using Conditional Value-at-Risk. https://arxiv.org/abs/2209.12842**
Adds a CVaR tail-risk penalty to MPPI rollout costs, estimated by N = 300 auxiliary disturbed rollouts for each of M = 1024 nominal trajectories, cutting collisions 80% in simulation and 55.7% on a 1/28-scale race car at essentially unchanged lap time (7.38 s vs 7.11 s). The cost is 3.07 × 10⁵ rollouts per control step at 44.3 Hz on an RTX 3090. This is the sharpest rival explanation for the FPL robustness win — bounded [0,1] atoms also compress cost tails before the softmax — and simultaneously the reason a sample-free tail reshaping is worth having at K ≤ 256 on CPU.
*Evidence grade: E2 | Clusters: C1, C4, C7 | Relevance: H-A, novelty, sampling-ideas*

**Trevisan and Alonso-Mora (2024). Biased-MPPI: Informing Sampling-Based Model Predictive Control by Fusing Ancillary Controllers. IEEE Robotics and Automation Letters. https://arxiv.org/abs/2401.09241**
Re-derives MPPI for arbitrary proposal mixtures so that ancillary-controller rollouts (LQR, braking, go-to-goal, learned predictors) can be injected as designated samples with importance weights that collapse back to pure exp(−S/λ). At K = 50 it achieved zero collisions where standard MPPI had 4 at K = 50 and still 7 at K = 2000; on a real Jackal, 0 vs 6 collisions at K = 300. This is the one proposal-side family the in-repo negative did not test — inject external structure rather than adapt noise statistics — and it is also the unbiased mechanism for feeding a student policy back into an MPPI teacher.
*Evidence grade: E2 | Clusters: C1, C7 | Relevance: H-A, H-B, sampling-ideas*

**Mohamed et al. (2022). Autonomous Navigation of AGVs in Unknown Cluttered Environments: log-MPPI Control Strategy. IEEE Robotics and Automation Letters (presented at IROS 2022). https://arxiv.org/abs/2203.16599**
Samples control perturbations from a normal-log-normal mixture, achieving wider state-space coverage with over 30% less injected variance than Gaussian MPPI, and converting that into 96% vs 84% success in cluttered navigation and 1 vs 7 collisions in dense forest. No change to cost or architecture is required. It shows heavier-tailed proposal shaping is a cheap, orthogonal lever that remains compatible with an FPL cost.
*Evidence grade: E2 | Clusters: C1 | Relevance: sampling-ideas*

**Kicki (2026). LP-MPPI: Low-Pass Filtering for Efficient Model Predictive Path Integral Control. ICRA 2026. https://arxiv.org/abs/2503.11717**
Butterworth low-pass filtering of sampled perturbations (uniform passband, unlike colored noise's 1/f bias) improves MPPI by ~24% on average across Hopper/Ant/HalfCheetah, 24% and 41% over a DIAL-MPC baseline on two quadrupeds, and wins on a real F1TENTH car — all on a single CPU core at N = 100–256 with 2.4% overhead. Spectral analysis shows RL expert policies concentrate near 2 Hz, matching low-pass noise rather than colored noise. This is decisive for scoping the in-repo sampling negative: the colored-noise family that failed is measurably the wrong shaping, and the right one wins in exactly the program's regime.
*Evidence grade: E2 | Clusters: C1, C2, C7 | Relevance: H-A, sampling-ideas*

**Tao et al. (2026). Sampling Strategy Design for Model Predictive Path Integral Control on Legged Robot Locomotion. arXiv. https://arxiv.org/abs/2601.01409**
A controlled comparison of sampling parameterizations on a Unitree-class quadruped under identical costs: cubic-spline sampling with four knots succeeds 100% on flat walking, stairs, and box traversal, while per-timestep Gaussian sampling scores 60%, 0%, and 0%; eight knots degrade box traversal to 80%. This is the strongest single contradiction to H-A's sufficiency claim — on contact-rich tasks the sampler, not the cost, can be binding — and it bounds the honest scope of "one bounded spec plus fixed Gaussian sampling."
*Evidence grade: E4 | Clusters: C1, C2 | Relevance: H-A, sampling-ideas*

**Honda (2025). Model Predictive Control via Probabilistic Inference: A Tutorial and Survey. arXiv preprint. https://arxiv.org/abs/2511.08019**
Maps the sampling-MPC-as-inference family along variational family, prior/proposal, divergence, temperature, constraints, and theory, noting that the Boltzmann exp(−J/λ) form has no strict theoretical requirement and that costs are assumed given. The survey contains no treatment of cost-representation design, multi-objective composition, bounded or logic-structured cost languages, and does not treat model-mismatch robustness as a design axis at all. It is the field's own evidence that the FPL cell is unoccupied — and the warning that no established mismatch-evaluation protocol exists to borrow.
*Evidence grade: E4 | Clusters: C1, C2, C3, C7 | Relevance: novelty, H-A*

**Li and Chen (2025). Unifying Model Predictive Path Integral Control, Reinforcement Learning, and Diffusion Models for Optimal Control and Planning. arXiv. https://arxiv.org/abs/2502.20476**
Shows theoretically that MPPI, policy-gradient RL, and diffusion reverse sampling are all gradient ascent on one Gaussian-smoothed Gibbs energy, with no experiments. The equivalence is exact only under an exponential transform: policy gradient must optimize E[exp(R)] rather than E[R] to match MPPI's update. For H-B this cuts both ways — it grounds the claim that teacher and student ascend the same landscape, but implies a student rewarded with raw fulfillment u is not optimizing the teacher's objective unless it is rewarded with log-fulfillment.
*Evidence grade: E4 | Clusters: C1, C6 | Relevance: H-B, novelty*

**Pinneri et al. (2020). Sample-efficient Cross-Entropy Method for Real-time Planning. CoRL 2020 (PMLR v155). https://arxiv.org/abs/2008.06389**
iCEM combines colored (1/f) noise, elite memory, and decaying population size to make CEM planning viable at small budgets on MuJoCo locomotion and manipulation. It is the canonical reference for the colored-noise and elite-resampling family that failed in-repo, and therefore the correct citation when scoping the negative result: the mechanism is well-established but is not the frequency shaping that LP-MPPI shows to work at N ≤ 256.
*Evidence grade: E2 | Clusters: C1, C2, C7 | Relevance: sampling-ideas*

---

## C2 — Legged and dexterous sampling MPC, morphology generality

**Howell et al. (2022). Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo. arXiv. https://arxiv.org/abs/2212.00541**
MJPC establishes real-time derivative-free predictive control on CPU (1–20 ms planning updates) across a 27-DoF humanoid, a Unitree A1, and a Shadow Hand, with trivial random search over spline-parameterized controls competitive with iLQG and CEM at N = 10 rollouts. Costs are per-task hand-tuned weighted norms of residuals with an optional exponential risk transform. It is simultaneously the E1 feasibility anchor for H-A's compute premise and the canonical linear cost representation FPL competes against.
*Evidence grade: E1 | Clusters: C2, C7 | Relevance: H-A, novelty*

**Alvarez-Padilla et al. (2024). Real-Time Whole-Body Control of Legged Robots with Model-Predictive Path Integral Control. arXiv. https://arxiv.org/abs/2409.10469**
Deploys whole-body MPPI on a real Unitree Go1 with only 30–50 CPU MuJoCo rollouts at 100 Hz using four cubic-spline knots over a 0.4 s horizon, achieving walking, climbing a 0.24 m box, and box pushing (9/10 forward) with one controller and no offline training. Ablations show splines beat direct sampling and performance plateaus around 100 Hz replanning. Loco-manipulation emerged from adding an L1 box-position term — task recomposition by cost editing, which is precisely the lever H-A proposes to formalize.
*Evidence grade: E1 | Clusters: C2, C7 | Relevance: H-A*

**Schramm et al. (2025). Reference-Free Sampling-Based Model Predictive Control. ICRA 2026. https://arxiv.org/abs/2511.19204**
Jointly samples cubic Hermite splines over position and velocity control points with diffusion-inspired annealing, discovering walk, trot, gallop, jump, backflip, and handstand from costs alone with 30 samples on a real Go2 and 60–70 on a 37-DoF G1 in simulation, all real-time on CPU. The same framework transfers quadruped-to-humanoid with no algorithmic modification or sampler retuning. It is the nearest competitor to the one-controller-any-morphology claim, differing chiefly in cost representation (plain weighted sums) and in having no mismatch-robustness protocol.
*Evidence grade: E4 | Clusters: C2, C7 | Relevance: H-A, novelty*

**Seo et al. (2026). RGB: RL Guided Whole-Body MPPI for Humanoid Control. arXiv preprint. https://arxiv.org/abs/2606.25123**
Runs MPPI at K = 128 CPU MuJoCo rollouts at ~280 Hz on a 29-DoF Unitree G1 by seeding the sampler with a pretrained RL policy prior, cutting lateral drift RMSE from 0.339 m to 0.022 m against pure RL. The coupling direction is RL-into-MPPI, the inverse of H-B, and a pretrained policy is required rather than produced. It is the closest published demonstration that the program's target platform and compute class are viable, and the natural baseline for the reverse arrow.
*Evidence grade: E4 | Clusters: C2 | Relevance: H-A, H-B*

**Xue et al. (2024). Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing. ICRA 2025 (submitted). https://arxiv.org/abs/2409.15610**
DIAL-MPC anneals sampling covariance jointly across outer iterations and along the horizon, reducing quadruped tracking error up to 13.4× versus tuned MPPI, raising crate-climb success from 0.2–0.3 to 0.9, and beating a trained RL policy by 50% with no training; under 2 kg mass mismatch it retains 92–126% better performance. It requires 2048 GPU rollouts across 20 annealing iterations. It is the strongest counterexample to "fixed-Gaussian MPPI is unbeatable," but the schedule itself costs zero extra rollouts, making a single-iteration horizon-graded variant the obvious untried in-repo experiment.
*Evidence grade: E2 | Clusters: C2, C3, C6, C7 | Relevance: H-A, sampling-ideas*

**Kurtz et al. (2025). Generative Predictive Control: Flow Matching Policies for Dynamic and Difficult-to-Demonstrate Tasks. ICRA 2026. https://arxiv.org/abs/2502.13406**
Treats the sampling-MPC update as a Monte Carlo estimate of score ascent on a noised target distribution, letting sampling MPC supervise flow-matching policy training without demonstrations; the resulting policies match or beat PPO on seven systems up to 29 DoF. Crucially, pure distillation fails on humanoid standup — only GPC+, which keeps the policy in the planning loop at deployment, succeeds. This is the central warning for H-B: at high DoF the teacher-student handoff may need policy-in-the-loop designs rather than clean supervised imitation.
*Evidence grade: E4 | Clusters: C2 | Relevance: H-B*

**Hansen et al. (2023). TD-MPC2: Scalable, Robust World Models for Continuous Control. ICLR 2024. https://arxiv.org/abs/2310.16828**
A single 317M-parameter world model with MPPI-style latent planning performs 80 tasks across embodiments and action spaces up to 38-D with one hyperparameter set (normalized score 70.6 vs 16.0 at 1M parameters). It costs 33 GPU-days of training and plans with 512 samples over 6–8 iterations. It proves morphology-general control is achievable with one planner architecture while defining the learned-model, GPU-scale anti-pattern H-A positions against; its authors credit return normalization — bounded, scale-free objectives — for cross-domain robustness, an independent argument for FPL atoms.
*Evidence grade: E2 | Clusters: C2, C6 | Relevance: H-A, H-B*

**Lowrey et al. (2018). Plan Online, Learn Offline: Efficient Learning and Exploration via Model-Based Control. ICLR 2019. https://arxiv.org/abs/1811.01848**
POLO plans with MPPI at 80–120 CPU rollouts against a learned terminal value ensemble trained on the same reward the planner optimizes, solving humanoid getup in 96 seconds of agent experience (24 vs 128 core-hours) and in-hand reorientation in 1 vs 500 CPU-hours; the learned value lets the horizon shrink 8× (H = 8 matching H = 64). It is the earliest demonstration of the exact mechanism H-B depends on, at exactly the program's compute class, and the strongest single ally for both hypotheses.
*Evidence grade: E2 | Clusters: C2, C6 | Relevance: H-A, H-B*

**Zhang et al. (2025). Whole-Body Model-Predictive Control of Legged Robots with MuJoCo. ICRA 2026 (to appear). https://arxiv.org/abs/2503.04613**
Demonstrates whole-body predictive control of legged robots directly against MuJoCo dynamics, consolidating the practice of using the simulator itself as the prediction model rather than a reduced-order template. It is the direct methodological precedent for the program's MuJoCo-in-the-loop CPU stack and for evaluating model mismatch by perturbing simulator parameters rather than swapping models.
*Evidence grade: E4 | Clusters: C2 | Relevance: H-A*

**Bohlinger et al. (2024). One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion. arXiv. https://arxiv.org/abs/2409.06366**
URMA trains a single morphology-agnostic RL policy across many legged embodiments via a universal encoding of joint and actuator descriptions, generalizing within the training distribution and transferring to unseen robots. It represents the dominant alternative route to morphology generality — per-distribution GPU-scale training — and clarifies the niche H-A occupies: zero-shot-per-morphology via the model, with no training distribution to be inside of.
*Evidence grade: E3 | Clusters: C2 | Relevance: H-A, novelty*

---

## C3 — Path-integral theory

**Williams et al. (2017). Information Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving. IEEE Transactions on Robotics. https://arxiv.org/abs/1707.02342**
Derives MPPI from a free-energy lower bound and KL projection onto the Gaussian family, showing that the exponentially tilted q*(V) ∝ exp(−S/λ)p(V) is the exact optimum and that the derivation places no smoothness, convexity, or boundedness requirement on the state cost. Validated over 1700+ laps on AutoRally at ~1200 samples and 40 Hz. This is the formal license for the entire program: FPL's S = −log(u) transform enters the softmax without touching any step of the derivation.
*Evidence grade: E1 | Clusters: C1, C3 | Relevance: H-A, novelty*

**Williams et al. (2015). Model Predictive Path Integral Control using Covariance Variable Importance Sampling. arXiv preprint. https://arxiv.org/abs/1509.01149**
Generalizes the path-integral derivation to sampling distributions differing in both drift and covariance, showing the likelihood-ratio correction enters as an extra running cost penalizing over-aggressive exploration plus Girsanov terms. Empirically, exploration variance had to be inflated 50–1500× over natural noise, and explicit non-smoothed crash costs beat smooth approximations. It is the post-mortem for the in-repo adaptive-covariance failure: the theory permits covariance change but predicts nothing about when it helps, and omitting the correction biases the estimator.
*Evidence grade: E3 | Clusters: C1, C3 | Relevance: H-A, sampling-ideas*

**Wang et al. (2021). Variational Inference MPC using Tsallis Divergence. RSS 2021. https://arxiv.org/abs/2104.00241**
Proves the cost-to-weight map is a free design axis: the deformed exponential (1 − J/γ)₊^{1/(r−1)} interpolates continuously from MPPI (r → 1) to CEM's top-elite indicator (r → ∞), with an explicit absolute-risk-aversion coefficient. Empirically it mainly reduces variance, growing with dimension: −84% cost std on planar navigation and 112% mean improvement with −66% std on a 56-D humanoid at N = 64–1024. Since MPPI-with-FPL computes exp(−S/λ) = u^{1/λ} over a bounded power mean, FPL is formally a member of this weight-transform family — the closest existing theory and the sharpest confound to ablate.
*Evidence grade: E2 | Clusters: C3, C7 | Relevance: H-A, novelty, sampling-ideas*

**Yi et al. (2024). CoVO-MPC: Theoretical Analysis of Sampling-based MPC and Optimal Covariance Design. arXiv. https://arxiv.org/abs/2401.07369**
Proves MPPI converges at least linearly on quadratic objectives with a rate depending explicitly on the sampling covariance, and derives the convergence-optimal covariance, which improves quadrotor cost 43–54% over standard MPPI. Computing it requires the cost Hessian. This is the principled reason fixed isotropic Gaussians are the practical optimum in a derivative-free CPU pipeline at K ≤ 256, and hence a theory-side explanation of the program's clean sampling negative.
*Evidence grade: E2 | Clusters: C1, C3 | Relevance: H-A, sampling-ideas*

**Honda et al. (2023). Stein Variational Guided Model Predictive Path Integral Control: Proposal and Experiments with Fast Maneuvering Vehicles. arXiv. https://arxiv.org/abs/2309.11040**
Uses a single Stein-variational guide particle to move only MPPI's nominal sequence toward a mode while preserving the closed-form update, cutting collision rate from 13.6% to 4.0% in simulation and 55.6% to 15.4% on a real 1/10-scale vehicle at 11 ms per solve on CPU. It is a direct counterexample to "all guidance fails," and the distinguishing detail matters: it augments the estimator rather than replacing it, unlike the GMM-plus-resampling mechanism that failed in-repo.
*Evidence grade: E2 | Clusters: C1, C3, C7 | Relevance: H-A, sampling-ideas*

**Wang et al. (2026). Information-Theoretic Adaptive Cooling for Deterministic MPPI via Entropy Feedback. https://arxiv.org/abs/2607.14245**
Closes the temperature loop using the Shannon entropy of importance weights — cool aggressively when weights diffuse, hold when concentrated — with theorems driving λ → 0 to eliminate MPPI bias and a critical entropy threshold guarding against premature weight collapse. On nonsmooth STL point-mass planning it converges up to 4.1× faster and lifts hardest-task success from 16% to 22%. The regime is iteration-rich deterministic planning, but the ESS/entropy-targeted temperature rule transfers directly to a bounded-cost receding-horizon controller at zero extra rollouts.
*Evidence grade: E4 | Clusters: C3, C7 | Relevance: sampling-ideas*

**Kappen (2005). Path integrals and symmetry breaking for optimal control theory. Journal of Statistical Mechanics: Theory and Experiment. https://arxiv.org/abs/physics/0505066**
The founding paper of the path-integral control class, showing that for a subclass of nonlinear stochastic control problems the Hamilton-Jacobi-Bellman equation linearizes under a log transform, so the optimal control is an expectation under a Boltzmann-weighted path distribution, with symmetry-breaking transitions as noise varies. It is the origin of the exponentiated-cost weighting that FPL reinterprets as a power law over bounded fulfillment.
*Evidence grade: E1 | Clusters: C3 | Relevance: novelty*

**Theodorou et al. (2010). A Generalized Path Integral Control Approach to Reinforcement Learning. Journal of Machine Learning Research, 11(104):3137-3181. https://jmlr.org/papers/v11/theodorou10a.html**
Derives PI², a probability-weighted-averaging update for parameterized policies that requires no gradients, no matrix inversions, and no tuning beyond exploration noise, and applies it to high-dimensional robot skill learning. It is the canonical bridge between path-integral control and reinforcement learning, and the historical precedent for the program's claim that one exponentially weighted objective can serve both a planner and a learner.
*Evidence grade: E1 | Clusters: C3 | Relevance: H-B, novelty*

**Theodorou et al. (2012). Relative entropy and free energy dualities: Connections to Path Integral and KL control. IEEE 51st Conference on Decision and Control (CDC). https://ieeexplore.ieee.org/document/6426381/**
Establishes the duality between free energy and relative entropy that underlies both path-integral and KL control formulations, unifying the linearly solvable families under one variational principle. It is the theoretical backbone cited whenever a new cost transform is claimed admissible, and thus the reference against which FPL's log-of-power-mean must be checked for consistency.
*Evidence grade: E2 | Clusters: C3 | Relevance: novelty*

**Todorov (2009). Efficient computation of optimal actions. Proceedings of the National Academy of Sciences, 106(28):11478-11483. https://doi.org/10.1073/pnas.0710743106**
Defines the linearly solvable MDP class in which the Bellman equation becomes linear in the exponentiated value function and the optimal policy is a Boltzmann reweighting of passive dynamics, with compositionality of optimal solutions as a corollary. The compositionality result is the closest classical analogue to composing objectives at the cost level, and delimits which composition operations preserve the linear structure that MPPI inherits.
*Evidence grade: E1 | Clusters: C3 | Relevance: novelty*

**Vlahov et al. (2024). Low Frequency Sampling in Model Predictive Path Integral Control. IEEE Robotics and Automation Letters, vol. 9, no. 5, pp. 4543-4550. https://arxiv.org/abs/2404.03094**
Shows that restricting MPPI's sampled perturbations to low-frequency content produces smoother, more physically plausible control sequences and improves performance without added rollouts, framing the sampler design question in the frequency domain rather than the covariance domain. Together with LP-MPPI it establishes frequency-restricted sampling as the CPU-cheap sampler upgrade the in-repo negative did not test.
*Evidence grade: E2 | Clusters: C3 | Relevance: sampling-ideas*

---

## C4 — Objective and specification representations

**El Mabsout et al. (2025). Closing the Intent-to-Behavior Gap via Fulfillment Priority Logic. arXiv preprint (cs.LG, cs.RO). https://arxiv.org/abs/2503.05818**
Defines FPL: fulfillment atoms in [0,1], weighted power-mean composition with p interpolating min to max, negation 1 − u, and soft-lexicographic priority offsets u([φ]_δ) = (u(φ) + max(δ,0))/(1+δ), with discount-normalized FQ-values (1−γ)Q. With the Balanced Policy Gradient learner it reaches LunarLander threshold in 20k steps vs 128k (DDPG) and 36k (CrossQ), up to ~500% better sample efficiency than SAC, and eliminates Hopper reward hacking; one spec generalized across HalfCheetah, Walker2d, and Ant without retuning. It contains no MPC, MPPI, or planning component and acknowledges lower asymptotic performance and FQ-overestimation sensitivity — the source formalism and the exact gap the program fills.
*Evidence grade: E4 | Clusters: C4, C5 | Relevance: H-A, H-B, novelty*

**Halder et al. (2025). Trajectory Planning with Signal Temporal Logic Costs using Deterministic Path Integral Optimization. ICRA 2025. https://arxiv.org/abs/2503.01476**
Optimizes raw, unsmoothed min/max STL robustness directly inside a path-integral optimizer that anneals both covariance and temperature toward the deterministic limit, with proven convergence. On a 5-D single-track vehicle with nested STL it converges in 25 s on a laptop CPU where MIP, gradient, and smoothed-robustness solvers time out past 3600 s — but needs up to 81,650 samples over 40 iterations. It is the mandatory prior-art citation for logic-structured costs in path-integral control, and its cost profile is exactly why the receding-horizon K ≤ 256 niche is still open.
*Evidence grade: E2 | Clusters: C3, C4, C5, C7 | Relevance: novelty, H-A*

**Mehdipour et al. (2019). Arithmetic-Geometric Mean Robustness for Control from Signal Temporal Logic Specifications. American Control Conference 2019. https://arxiv.org/abs/1903.05186**
Replaces min/max STL semantics with normalized mean-based aggregation bounded in [−1,1] — geometric means for conjunction and always, arithmetic means for disjunction and eventually — remaining sound while rewarding both how robustly and how often subformulae hold. Policies optimized with AGM survive Gaussian perturbations at ~59% versus ~42% for smooth min/max approximations at equal nominal score. AGM is FPL's closest published ancestor and supplies the strongest independent evidence that bounded mean-based aggregation buys robustness under perturbation.
*Evidence grade: E2 | Clusters: C4, C5 | Relevance: H-A, novelty*

**Uzun et al. (2024). Optimization with Temporal and Logical Specifications via Generalized Mean-based Smooth Robustness Measures. arXiv preprint. https://arxiv.org/abs/2405.10996**
D-GMSR builds C¹-smooth STL robustness from weighted geometric and power means (p ≥ 1) over squared positive and negative parts, and is the only measure in its comparison simultaneously smooth, sound, complete, and free of locality/masking problems for any ε > 0. LSE smoothing at κ = 25 fails a simple reach specification through masking while the mean-based measure converges. Its robustness values are explicitly unnormalized and scale-dependent — power-mean aggregation is prior art, but bounded [0,1] normalization remains FPL's distinguishing move.
*Evidence grade: E3 | Clusters: C4 | Relevance: novelty, H-A*

**Bouzid et al. (2026). Autonomous Driving with Priority-Ordered STL Specifications Under Multimodal Uncertainty. arXiv preprint. https://arxiv.org/abs/2606.20336**
Optimizes priority-ordered STL inside standard MPPI using a rank-preserving reward Σⱼ(a·2^(N−j+1)·step(ρⱼ) + ρⱼ/N) evaluated exactly per rollout, with scenario trees and CVaR over scenario losses for multimodal uncertainty, at only 220–240 rollouts and 0.77–2.49 ms per planning step. Strict lexicographic MPPI preserved worst-case safety across all cut-in modes where a hand-tuned weighted-CVaR baseline sacrificed it. It proves logic-cost MPPI runs inside the program's sample budget and supplies a direct logic-versus-tuned-weights comparison — on kinematic bicycle dynamics, not full physics.
*Evidence grade: E4 | Clusters: C4 | Relevance: H-A, novelty*

**Ahmad et al. (2026). RRT$^η$: Sampling-based Motion Planning and Control from STL Specifications using Arithmetic-Geometric Mean Robustness. arXiv preprint (cs.RO). https://arxiv.org/abs/2602.16825**
Independently adopts FPL wholesale: it maps AGM robustness intervals to bounded fulfillments f = (η_upper + η_lower + 2)/4 in [0,1] and composes them with FPL power means, citing FPL by name for its minimum-fulfillment guarantee, and adds incremental robustness monitoring that cuts per-observation evaluation from O(|s||φ|) to O(|φ|). Because it is an offline RRT planner with no receding horizon, it confirms the formalism transplants into sampling-based synthesis while leaving FPL-in-sampling-MPC unclaimed; its incremental monitoring is directly reusable for cheap atom evaluation inside CPU rollouts.
*Evidence grade: E4 | Clusters: C4, C5 | Relevance: novelty, H-A*

**Halder et al. (2026). Lexicographic Minimum-Violation Motion Planning using Signal Temporal Logic. IEEE Open Journal of Intelligent Transportation Systems. https://arxiv.org/abs/2604.20428**
Collapses lexicographic priorities over conflicting STL specs into one scalar by discretizing violation metrics into integer levels and bit-shifting, s(y) = Σ cᵢ(y)·2^(Bᵢ), with a proven lexicographic-to-scalar order equivalence, solved by a deterministic MPPI with the quadratic input-cost requirement removed. Its own analysis shows discretization causes priority relaxation and inversion that vanish only as granularity is refined. This is the sharpest evidence that hard-lexicographic scalarizations are brittle to quantization — the failure mode FPL's soft priority offsets avoid by construction.
*Evidence grade: E4 | Clusters: C4 | Relevance: novelty*

**Zheng et al. (2026). STL-SVPIO: Signal Temporal Logic guided Stein Variational Path Integral Optimization. arXiv preprint. https://arxiv.org/abs/2603.13333**
Transports ten mutually repulsive control particles by Stein variational gradient descent using LogSumExp-smoothed STL robustness gradients through differentiable simulation, solving nested long-horizon specifications. It documents plain MPPI failing on sparse, long-horizon, temporally nested logic costs (robustness 0.005 vs 0.108 on reach-avoid; 0% on multi-agent coordination) via exponential importance-weight decay. It requires GPU differentiable simulation with 19–635 s per solve, so it bounds H-A's scope rather than occupying its niche: dense per-timestep fulfillment atoms sit on the safe side of the divide it identifies.
*Evidence grade: E4 | Clusters: C4, C7 | Relevance: H-A, novelty*

**Li et al. (2016). Reinforcement Learning With Temporal Logic Rewards. arXiv preprint (cs.AI, cs.RO). https://arxiv.org/abs/1612.03471**
TLTL uses a truncated linear temporal logic robustness degree directly as the RL reward, learning faster and reaching higher returns than hand-crafted discrete and continuous shaped rewards, and achieving 100% success on a physical Baxter toast-placing task where the hand-tuned reward failed to learn gripper timing at all. The authors explicitly flag that nested min/max robustness misdirects learning when sub-formulae live at different physical scales and that normalization is a manual chore — the exact pain point bounded [0,1] atoms with satisfied bands remove.
*Evidence grade: E2 | Clusters: C4, C5 | Relevance: H-A, H-B, novelty*

**Kapoor et al. (2020). Model-based Reinforcement Learning from Signal Temporal Logic Specifications. arXiv preprint (submitted to ICRA 2021). https://arxiv.org/abs/2011.04950**
Couples a learned neural dynamics model with sampling MPC (CMA-ES) that directly maximizes STL robustness over predicted trajectories, solving Cartpole, Mountain Car, Fetch reach, adaptive cruise control, and parking with no reward engineering. It is the closest existing pipeline shape to an FPL+MPPI teacher, but has no policy student, no distillation, and no baseline comparisons — confirming that a shared bounded-fulfillment spec across planner cost and student reward is unoccupied.
*Evidence grade: E4 | Clusters: C4 | Relevance: H-B, novelty*

**Veer et al. (2022). Receding Horizon Planning with Rule Hierarchies for Autonomous Vehicles. arXiv preprint (cs.RO, eess.SY). https://arxiv.org/abs/2212.03323**
Collapses a hierarchy of N STL rules into one differentiable rank-preserving reward R(ρ) = Σ(a^(N−i+1)·step(ρᵢ) + ρᵢ/N) with a > 2 and a theorem guaranteeing lower rank implies strictly higher reward, running at 7–10 Hz across overtaking, obstacle, and stop-sign scenarios without per-scenario retuning. Rank preservation requires exponential weight separation, which compresses low-priority signal exponentially in N — the published competitor FPL's soft priority offsets must be argued against directly.
*Evidence grade: E2 | Clusters: C4 | Relevance: novelty*

**Raman et al. (2017). Model Predictive Control for Signal Temporal Logic Specification. arXiv preprint (original work CDC 2014). https://arxiv.org/abs/1703.09563**
The canonical encoding of STL satisfaction as mixed-integer constraints inside a receding-horizon optimization, establishing both the correctness of logic-constrained MPC and its combinatorial cost. It is the baseline the entire sampling-based STL line (including Halder's path-integral solver) is measured against, and the reference point for arguing that sampling optimizers tolerate logic structure that MIP encodings pay exponentially for.
*Evidence grade: E2 | Clusters: C4 | Relevance: novelty*

**Lindemann and Dimarogonas (2019). Robust Control for Signal Temporal Logic Specifications using Average Space Robustness. Automatica, vol. 101, pp. 377-387. https://arxiv.org/abs/1607.07019**
Replaces the worst-case space robustness of STL with an average-based measure and derives feedback control laws maximizing it, showing the averaged semantics yields better-conditioned synthesis than the min/max degree while retaining satisfaction guarantees. It is the earliest control-theoretic statement of the mean-versus-min aggregation argument that AGM, D-GMSR, and FPL all later rely on.
*Evidence grade: E2 | Clusters: C4 | Relevance: novelty, H-A*

---

## C5 — Reward-design theory and objective composition

**Mysore et al. (2021). How to Train your Quadrotor: A Framework for Consistently Smooth and Responsive Flight Control via Reinforcement Learning. ACM Transactions on Cyber-Physical Systems, 5(4). https://arxiv.org/abs/2012.06656**
Replaces additive scalarization with multiplicative (geometric-mean) composition of normalized bounded reward components acting as a logical AND, taking sim-to-real from roughly one flight-worthy agent in dozens to 100% of trained agents, cutting motor current from 22.87 A to 5.86 ± 3.10 A, and reducing training variance in 19 of 24 Gym comparisons. It also reports SAC failing to train at all on the multiplicative reward on Ant. This is FPL's direct provenance and the clearest warning for H-B that reward composition and learner are not independently choosable.
*Evidence grade: E2 | Clusters: C5 | Relevance: H-A, H-B*

**van Krieken et al. (2022). Analyzing Differentiable Fuzzy Logic Operators. Artificial Intelligence. https://arxiv.org/abs/2002.06100**
Analyzes fuzzy logic operators as differentiable losses, showing Gödel/min conjunction and minimum aggregation are single-passing — at most one input receives nonzero derivative — while Łukasiewicz aggregation has nonvanishing derivative on only 1/n! of the domain, and product/generalized-mean operators are informative everywhere. This is the mechanism behind FPL's empirical preference for p ∈ {0, −1} over p → −∞, and it extends naturally to rank-based samplers: min-like costs make K = 256 rollouts nearly indistinguishable except along one active constraint.
*Evidence grade: E3 | Clusters: C5 | Relevance: H-A, H-B*

**Abdolmaleki et al. (2020). A Distributional View on Multi-Objective Policy Optimization. ICML 2020. https://arxiv.org/abs/2005.07513**
Shows linear reward-weight scalarization is not scale-invariant — multiplying one objective's rewards 20× silently reorders learned preferences, and hypervolume collapses from 6.9 × 10⁷ to 2.6 × 10⁷ under 10× penalty scaling — and restores invariance via per-objective KL constraints whose temperatures absorb scale. That machinery requires one critic and temperature per objective. It is the cleanest published motivation for H-A's robustness story: a degraded model rescales error magnitudes, and bounded fulfillments absorb what linear weights do not.
*Evidence grade: E2 | Clusters: C5 | Relevance: H-A, novelty*

**Ng et al. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. ICML 1999, Morgan Kaufmann, pp. 278-287. https://doi.org/10.5555/645528.657613**
Proves potential-based shaping F(s,a,s') = γΦ(s') − Φ(s) is both sufficient and, absent further MDP knowledge, necessary for preserving optimal policies, and catalogues the degenerate behaviors non-potential shaping produces (the bicycle riding in circles, the soccer robot vibrating on the ball). FPL's bounded nonlinear composition is deliberately not potential-based, so it changes optima rather than shaping toward them — a feature for intent expression, but it forfeits the classical safety net and complicates any "faster convergence" comparison in H-B.
*Evidence grade: E1 | Clusters: C5 | Relevance: H-B, novelty*

**Toro Icarte et al. (2022). Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning. Journal of Artificial Intelligence Research, 73:173-208. https://arxiv.org/abs/2010.03950**
Exposes reward structure to the learner as a finite-state machine and exploits it via counterfactual experiences, converting sparse tasks from unlearnable to solvable: Q-learning stays near 0 after 2M steps in Craft World while CRM/HRM saturate by ~1M, and DDPG stays flat on HalfCheetah lap-running where CRM completes 9 laps. It also shows structure exploitation is not uniformly benign — automated RM-derived shaping decreased performance for all approaches in Water World. It is the load-bearing precedent for H-B's lever and the caution attached to it.
*Evidence grade: E2 | Clusters: C5 | Relevance: H-B*

**Skalse et al. (2022). Lexicographic Multi-Objective Reinforcement Learning. IJCAI 2022, pp. 3430-3436. https://arxiv.org/abs/2212.13769**
Develops value- and policy-based lexicographic RL, proving convergence only when the slack tolerance τ falls below the minimum Q-value gap, which "can in general not be determined a priori," while showing lexicographic PPO/A2C obtain the most reward on GridNav and safety holds only for the limit policy. Hard lexicographic ordering is therefore intrinsically fragile to tolerance tuning — consonant with the in-repo finding that hard lexicographic shields lose to FPL's soft priority offsets.
*Evidence grade: E3 | Clusters: C5 | Relevance: H-A, novelty*

**Hayes et al. (2022). A Practical Guide to Multi-Objective Reinforcement Learning and Planning. Autonomous Agents and Multi-Agent Systems, 36:26. https://arxiv.org/abs/2103.09568**
The standard taxonomy of multi-objective decision making, establishing that nonlinear utilities break Bellman additivity and that SER (utility of expected return) and ESR (expected utility of return) yield significantly different optimal policies. This is the sharpest theoretical hazard for H-B: MPPI applies the FPL power mean per-rollout (ESR-like) while a critic-based student estimates expectations (SER-like), so teacher and student can genuinely disagree about optimality despite sharing one spec.
*Evidence grade: E3 | Clusters: C5 | Relevance: H-B, novelty*

**Skalse and Abate (2023). On the Limitations of Markovian Rewards to Express Multi-Objective, Risk-Sensitive, and Modal Tasks. UAI 2023, PMLR 216:1974-1984. https://arxiv.org/abs/2401.14811**
Proves that scalar Markovian rewards cannot express large classes of multi-objective, risk-sensitive, and modal objectives, characterizing precisely which multi-objective problems admit a scalarization. It formalizes why weighted-sum costs are an expressivity ceiling rather than a tuning inconvenience, and simultaneously bounds what a memoryless bounded-fulfillment composition like FPL can and cannot represent without added automaton state.
*Evidence grade: E3 | Clusters: C5 | Relevance: novelty*

**Tiapkin et al. (2025). On Teacher Hacking in Language Model Distillation. arXiv. https://arxiv.org/abs/2502.02671**
Shows that distilling from an imperfect proxy teacher lets the student optimize the teacher rather than the underlying objective, degrading true performance, and that diversifying and mixing the data source mitigates it. Transposed to H-B, an MPPI teacher is an imperfect optimizer of the shared FPL spec, so the student can improve against teacher actions while regressing against the spec — an argument for mixing on-spec rollouts with teacher data and for measuring spec-conformance directly.
*Evidence grade: E4 | Clusters: C5 | Relevance: H-B*

---

## C6 — MPC-as-teacher reinforcement learning

**Youm et al. (2023). Imitating and Finetuning Model Predictive Control for Robust and Symmetric Quadrupedal Locomotion. IEEE Robotics and Automation Letters. https://arxiv.org/abs/2311.02304**
The closest structural match to H-B: DAgger-imitate a DDP-based MPC expert, then PPO fine-tune, converging in ~2000 versus ~4000 iterations of RL from scratch, with 15.4 ms MPC inference amortized to 1 ms. The fine-tuned student surpasses its teacher — 90% hardware success on 7.5 cm steps where the MPC and DAgger-only policies score 0% and vanilla RL 60%. Critically, the fine-tuning reward is deliberately different from the MPC cost and the pipeline still works, so H-B must show spec sharing adds something beyond mismatched-reward fine-tuning.
*Evidence grade: E2 | Clusters: C2, C6 | Relevance: H-B, H-A*

**Hansen et al. (2022). Temporal Difference Learning for Model Predictive Control. ICML. https://arxiv.org/abs/2203.04955**
TD-MPC couples planner and policy through one jointly learned value: the policy prior seeds ~5% of the 512 MPPI samples, is trained to maximize the same Q the planner uses as terminal value, and everything is learned by TD. It solves 38-D Dog locomotion in ~1M steps and Walker Walk 16× faster in wall-clock than LOOP at 3.3× less compute, while the policy alone remains inferior to planning. It is the canonical demonstration that tight objective coupling, not data volume, drives the efficiency gain H-B claims.
*Evidence grade: E1 | Clusters: C6 | Relevance: H-B*

**Wang et al. (2025). Bootstrapped Model Predictive Control. ICLR 2025. https://arxiv.org/abs/2503.18871**
BMPC closes the tightest published teacher-student loop: the policy imitates an MPC expert that is itself policy-seeded, with an on-policy n-step TD value learned jointly, giving ~3–4× better data efficiency than TD-MPC2 on high-dimensional locomotion (90k vs 360k steps-to-solve) with a smaller network, and near-zero policy-versus-planner gap. It shows the distillation gap TD-MPC2 leaves open can be closed by mutual coupling, which is the architecture H-B should adopt if pure imitation underperforms.
*Evidence grade: E2 | Clusters: C6 | Relevance: H-B*

**Carius et al. (2020). MPC-Net: A First Principles Guided Policy Search. IEEE Robotics and Automation Letters. https://arxiv.org/abs/1909.05197**
Trains the student by minimizing the MPC-derived control Hamiltonian H = L(x,u,t) + ∂V/∂x·f(x,u,t), embedding the teacher's cost and constraints in the student loss rather than regressing actions, yielding stable ANYmal gaits from under 10 minutes of demonstration data with lower constraint violation than behavioral cloning and ~300× faster inference (0.125 ms vs 38 ms). It states plainly that pure distillation cannot outperform the MPC and inherits its local minima — the structural argument for RL fine-tuning after imitation.
*Evidence grade: E2 | Clusters: C2, C6 | Relevance: H-B*

**Reske et al. (2021). Imitation Learning from MPC for Quadrupedal Multi-Gait Control. ICRA 2021. https://arxiv.org/abs/2103.14331**
Extends MPC-Net to a mixture of experts with a log-partitioned Hamiltonian loss, distilling trot and static walk into one policy with experts specializing per hybrid contact mode. On identical MPC data, Hamiltonian-loss policies survive rough terrain 10.6 ± 6.3 s versus 2.6 ± 1.6 s for behavioral cloning. How the teacher's objective enters the student's loss — not just which data it sees — is worth roughly 4× in robustness, which is the mechanistic case for sharing a spec rather than sharing actions.
*Evidence grade: E2 | Clusters: C6 | Relevance: H-B*

**Levine et al. (2015). End-to-End Training of Deep Visuomotor Policies. JMLR. https://arxiv.org/abs/1504.00702**
Guided policy search casts iLQG teacher controllers and the CNN student as one BADMM-constrained optimization with KL constraints enforcing p(u|x) = π_θ(u|o), so teacher and student provably optimize the same task cost. Complete visuomotor skills were learned from 156–288 real PR2 trials at 88.9–96.3% success versus 0–70.4% for pose-estimation baselines. It is the strongest existing form of teacher-student objective coupling, but the shared cost is a smooth quadratic, not a satisfaction spec.
*Evidence grade: E2 | Clusters: C6 | Relevance: H-B*

**Kahn et al. (2016). PLATO: Policy Learning using Adaptive Trajectory Optimization. https://arxiv.org/abs/1603.00622**
The MPC teacher optimizes its own task cost plus λ·KL(teacher ‖ student), so its supervision anticipates the learner's distribution while the student never executes during training, producing under one crash per iteration versus substantial DAgger failures at equal or better final policy quality. The student never sees the task cost — the objective coupling lives entirely in the teacher's optimization — which is a working counterexample to the necessity half of H-B's shared-spec claim.
*Evidence grade: E2 | Clusters: C6 | Relevance: H-B*

**Xing et al. (2026). MPC-Injection: Biasing Off-Policy Locomotion RL Toward Controller-Induced Behavior Basins. arXiv preprint. https://arxiv.org/abs/2606.26392**
Injects MPC transitions into the off-policy replay buffer at a fixed ~25% ratio with the RL reward unchanged and no imitation loss, steering learning into the controller's behavior basin — upright walking instead of scooting, structured trot instead of chaotic vibration on a Go2 — with gait quality comparable to a twenty-one-term shaped reward. No steps-to-threshold speedup is claimed. It is the mandatory H-B baseline: if unchanged-reward data injection matches spec sharing, the shared spec reduces to data quality.
*Evidence grade: E4 | Clusters: C2, C6 | Relevance: H-B*

**Ross et al. (2010). A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. AISTATS 2011. https://arxiv.org/abs/1011.0686**
DAgger reduces imitation learning to no-regret online learning by aggregating data under the student's own state distribution, with regret bounds that remove the quadratic compounding-error term of naive behavioral cloning. Every MPC-as-teacher pipeline in this cluster either uses DAgger or explains why it departs from it, so it is the correctness baseline against which the H-B data-collection scheme must be specified.
*Evidence grade: E1 | Clusters: C6 | Relevance: H-B*

**Reiter et al. (2025). Synthesis of Model Predictive Control and Reinforcement Learning: Survey and Classification. arXiv. https://arxiv.org/abs/2502.02133**
Classifies the MPC-RL design space by which component is learned and how the two objectives interact, covering MPC-as-expert, learned terminal costs, learned models, and RL-tuned MPC parameters. It is the positioning reference for H-B, and its taxonomy makes the specific empty cell legible: no surveyed method shares one bounded-satisfaction specification between the planner's cost and the student's reward.
*Evidence grade: E4 | Clusters: C6 | Relevance: H-B, novelty*

**Byravan et al. (2021). Evaluating model-based planning and planner amortization for continuous control. arXiv. https://arxiv.org/abs/2110.03363**
A controlled study finding that at scale, amortized planners often do no better than well-tuned model-free RL, and that the benefits of planning shrink as the policy class and data budget grow. It is the principal skepticism to fold into H-B's claims: sample-efficiency gains measured in a small-data CPU regime should not be extrapolated, and the experiment must report the from-scratch RL baseline tuned as carefully as the teacher-guided arm.
*Evidence grade: E3 | Clusters: C6 | Relevance: H-B*

---

## C7 — Sampling-idea mining

**Lambert et al. (2020). Stein Variational Model Predictive Control. Conference on Robot Learning (CoRL) 2020. https://arxiv.org/abs/2011.07641**
Recasts MPC as Bayesian inference over control sequences and maintains a set of interacting particles transported by Stein variational gradient descent, capturing multimodal control posteriors that a single Gaussian collapses. It defines the interaction-side family and its cost — gradients or many inner iterations — which is why the program's cheap variants must borrow the repulsion idea (fulfillment-space diversity reweighting) rather than the particle transport itself.
*Evidence grade: E2 | Clusters: C3, C7 | Relevance: sampling-ideas*

**Watson & Peters (2022). Inferring Smooth Control: Monte Carlo Posterior Policy Iteration with Gaussian Processes. CoRL 2022. https://arxiv.org/abs/2210.03512**
Places a Gaussian process prior over control trajectories inside a Monte Carlo posterior policy iteration scheme, so smoothness is imposed by the prior rather than by post-hoc filtering, improving sample efficiency and control quality on continuous-control benchmarks. It provides a principled alternative to spline knots for structuring the proposal at fixed K, and its posterior-iteration view connects directly to the temperature/ESS schedules the program is considering.
*Evidence grade: E2 | Clusters: C7 | Relevance: sampling-ideas*

**Wagener et al. (2019). An Online Learning Approach to Model Predictive Control. arXiv (RSS 2019). https://arxiv.org/abs/1902.08967**
Reframes receding-horizon control as online learning, showing that MPPI and its relatives are instances of dynamic mirror descent with particular step sizes and shift operators, and that the warm start is the algorithm's implicit regularizer. It is the cleanest theoretical statement of why warm-started fixed-Gaussian MPPI is hard to beat inside a one-iteration-per-step budget, and it makes the program's clean negative a budget result rather than a mechanism result.
*Evidence grade: E2 | Clusters: C7 | Relevance: H-A, sampling-ideas*

**Gu et al. (2025). Bregman Centroid Guided Cross-Entropy Method. arXiv. https://arxiv.org/abs/2506.02205**
Guides a cross-entropy planner using Bregman centroids over a family of sampling distributions, giving a geometry-aware way to aggregate elite information without per-step covariance estimation. It is a CPU-cheap, untried alternative in the aggregation-side family — it changes how elite information is summarized rather than how noise is generated, which places it outside the six mechanisms that failed in-repo.
*Evidence grade: E4 | Clusters: C7 | Relevance: sampling-ideas*
