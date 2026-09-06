# Controller configuration design

**Status:** proposal, not implemented. Nothing in this document exists yet.

**Problem:** there is no single place that says what a controller run *is*. Hyperparameters are
spread across three construction paths, and the axes that should be independent — how samples are
drawn, how they are scored, how the mean is updated — are tangled together differently in each path.

---

## 1. Diagnosis

Three ways to build a controller, three spellings of every axis:

| axis | `analytic_mppi/run.py` | `analytic_mppi/eval.py` | `verification/_experiment.py` |
|---|---|---|---|
| cost mode | `--fpl` / `--fpl-discounted` flags | `cost_mode="fpl_cost"` string | `cost="fpl"` / `"lin:2.0"` |
| algo params | `_ALGO_PARAMS` whitelist | `**algo_kwargs` passthrough | `sampler_kwargs()` if-chain |
| env defaults | argparse defaults | caller's problem | `ENVS` dict |
| update rule | implied by `--algo` | implied by class | implied by sampler name |

Four concrete failures that follow from this:

**1.1 Silent parameter drops.** `run.py:126-152` forwards only the params listed in
`_ALGO_PARAMS[algo]`. `--algo predictive_sampling --temperature 0.05` runs happily and ignores the
temperature. `--algo mppi --fpl-weights ...` is not expressible at all. The whitelist is
hand-maintained and has already fallen behind the controllers.

**1.2 Defaults have drifted from their own documentation.**

| location | actual default | help text claims |
|---|---|---|
| `run.py:65` `--temperature` | `0.1` | `1.0` |
| `run.py:69` `--initial-noise-level` | `0.3` | `0.5` |
| `run.py:71` `--minimum-noise-level` | `0.3` | `0.1` |

Separately, `fpl_p` defaults to `0.1` in `sampling_base.py:76`, while every real experiment uses
`-1.0`. Nobody can reconstruct what a given run used.

**1.3 The cost mode is an enum wearing four booleans.** `sampling_base.py:100-103` enforces mutual
exclusion of `use_fpl_cost` / `use_fpl_discounted` / `use_fpl_layered` / `use_hybrid` at runtime,
which is the type system telling us it wants to be one field with five values. `eval.py:59-75`
already translates a `cost_mode` string into those booleans — that translation should be the only
representation.

**1.4 The update rule is not a knob.** It is three different mechanisms depending on which
controller you pick:

- a **class**: `predictive_sampling` is argmax, `mppi` is the softmax path integral
- a **kwarg**: `fpl_weighting ∈ {softmax, proportional, exp}` in `mppi_v2.py:98-142`
- **hardcoded**: CEM's elite mean, `fpl_shielded`'s lexicographic feasibility gate

This is the axis that is hardest to keep track of, and it is the one with no consistent home.

---

## 2. Schema: four orthogonal axes

`SamplingController.act()` is already `sample → score → update` (`sampling_base.py:172-183`). The
config makes that structure explicit instead of hiding it behind class selection:

```
run        steps, seeds, episodes, spline, iterations, init, output
task       name + task kwargs (the difficulty knob)
proposal   how knots are drawn            -> sample_knots()
objective  how a rollout becomes a scalar -> _score_*()
update     how scores become a new mean   -> update_mean()   <-- argmax vs path integral
```

Every controller in the repo decomposes cleanly onto these four. `fpl_shielded`, for example, is
`proposal=gaussian` + `objective=fpl_cost` + `update=shielded`; it is currently a subclass of
`MPPIv2` purely to inherit the Gaussian proposal.

---

## 3. Format decision

**Chosen: frozen dataclasses as schema, `.py` files as authoring format, JSON as provenance
snapshot.**

### Why not pure YAML / JSON

The sweep code computes hyperparameters from other hyperparameters
(`verification/_experiment.py:78-79`):

```python
sigma_min = max(0.05, 0.1 * noise)
num_elites = max(4, K // 8)
```

A data format cannot express this. The options would be to flatten to magic numbers — losing the
relationship, so it silently goes wrong the next time `K` changes — or to add an `eval:` string
escape, which is a worse Python with no tooling. Config files also need to reference code
(`init: hopper_stand` is a function; `metric: locomotion` selects a metrics branch), which a data
format can only do through a name-to-object registry that has to be maintained by hand.

### Why dataclasses + `.py`

- Computed params work natively; the *relationship* stays in the file.
- Editor autocomplete and type checking. `temprature=0.2` is a red squiggle, not a 20-minute sweep
  that quietly used the default.
- Composition is `replace()`, not a bespoke deep-merge with its own list-vs-scalar edge cases.
- No parser dependency. (`tomllib` is 3.11+ and this venv is 3.10.12, so TOML would need `tomli`.)

### JSON is still the wire format

Authoring in Python does not mean losing machine-readable configs. `ExperimentConfig` is a plain
tree of frozen dataclasses, so `to_dict()` / `to_json()` / `from_dict()` are mechanical. Every run
writes its **fully resolved** config as JSON next to the results. That JSON is diffable, hashable,
greppable, and reloadable — which is all we wanted from a data format, without paying for it at
authoring time.

If a pure-data config is ever wanted (a web UI, an external sweep driver), `from_dict()` is already
the entry point — a `.json`/`.yaml` loader is then ~10 lines. It is just not the primary path.

---

## 4. Sample: the schema

`analytic_mppi/config.py`

```python
"""Typed configuration for a controller run.

One ExperimentConfig fully determines a run: same config + same seed -> same trajectory.
Every field maps onto exactly one of sample_knots / _score_* / update_mean.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence


@dataclass(frozen=True)
class Run:
    steps: int = 150
    seeds: tuple[int, ...] = (0,)          # one episode per seed
    spline_type: str = "zero"              # zero | linear
    iterations: int = 1                    # optimizer iterations per MPC step
    init: Optional[str] = None             # key into eval.INIT_FNS
    nthread: Optional[int] = None
    mode: str = "headless"                 # headless | live | record
    out: Optional[str] = None


@dataclass(frozen=True)
class TaskSpec:
    name: str = "hopper"
    kwargs: dict[str, Any] = field(default_factory=dict)   # e.g. {"target_velocity": 2.0}


@dataclass(frozen=True)
class Proposal:
    """How candidate knots are drawn. -> sample_knots()"""
    kind: str = "gaussian"                 # gaussian | cma | cem | colored | adaptive | gmm
    num_samples: int = 128
    num_knots: int = 4
    plan_horizon: float = 0.6
    noise_level: float = 0.3
    include_mean: bool = False             # inject current mean as sample 0
    extra: dict[str, Any] = field(default_factory=dict)    # kind-specific (color_beta, ...)


@dataclass(frozen=True)
class Objective:
    """How a rollout becomes one scalar. -> _score_normal / _score_fpl / _score_fpl_layered."""
    mode: str = "fpl_cost"                 # normal | fpl_cost | fpl_discounted | fpl_layered | hybrid
    p: float = -1.0                        # objective-axis power mean; p=1 + weights == linear sum
    gamma: float = 0.99
    time_p: Optional[float] = -2.0         # None -> discounted mean; <=0 -> weakest-link over time
    weights: Optional[Sequence[float]] = None      # per-atom; None -> uniform
    group_p: Optional[float] = None                # inner power mean (fpl_layered)
    term_indices: Optional[Sequence[int]] = None   # restrict reward to a subset of atoms
    floor_weight: float = 1.0                      # hybrid log-barrier scale


@dataclass(frozen=True)
class Update:
    """How scores become the new mean. -> update_mean()"""
    rule: str = "path_integral"   # path_integral | argmax | reward_proportional
                                  # | reward_exp | elite_mean | shielded
    temperature: float = 0.2
    extra: dict[str, Any] = field(default_factory=dict)    # rule-specific (safety_floor, ...)


@dataclass(frozen=True)
class ExperimentConfig:
    run: Run = field(default_factory=Run)
    task: TaskSpec = field(default_factory=TaskSpec)
    proposal: Proposal = field(default_factory=Proposal)
    objective: Objective = field(default_factory=Objective)
    update: Update = field(default_factory=Update)
    label: str = ""
    notes: str = ""

    def with_(self, **dotted: Any) -> "ExperimentConfig":
        """Nested override by dotted path:  cfg.with_(**{"objective.p": 1.0})"""
        out = self
        for path, value in dotted.items():
            head, _, tail = path.partition(".")
            if not tail:
                out = replace(out, **{head: value})
            else:
                out = replace(out, **{head: getattr(out, head).with_(**{tail: value})})
        return out

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
```

`with_` is added to each sub-dataclass too (a shared mixin), so overrides nest arbitrarily. This is
the single ergonomic that makes sweeps readable.

---

## 5. Sample: config files

### `configs/base.py` — every default in one place

```python
"""The single source of truth for defaults. Everything else derives from BASE."""
from analytic_mppi.config import ExperimentConfig

BASE = ExperimentConfig()   # dataclass defaults ARE the documented defaults
```

There is no second copy of a default anywhere. This is what fixes §1.2 permanently — a drifted help
string is impossible when there is no help string, only the field.

### `configs/env/hopper.py` — the published sweep values, lifted out of `ENVS`

```python
"""Hopper. Values matched to verification/hopper_pareto_sweep.py.

Do not edit casually -- every published hopper number was produced with these.
"""
from configs.base import BASE
from analytic_mppi.config import EnvMeta

HOPPER = BASE.with_(**{
    "run.steps": 150,
    "run.init": "hopper_stand",
    "task.name": "hopper",
    "task.kwargs": {"target_velocity": 2.0},
    "proposal.num_knots": 4,
    "proposal.plan_horizon": 0.6,
    "proposal.noise_level": 0.3,
    "objective.time_p": -2.0,        # weakest-link over time, given to FPL *and* linear
    "update.temperature": 0.2,
})

# Env facts the experiment layer needs (today: _experiment.ENVS / _metrics).
HOPPER_META = EnvMeta(
    n_atoms=4,
    lin_idx=2,               # atom the linear-weight family sweeps (velocity)
    safe_w=0.5,
    safety_indices=(0, 1),   # height, orientation
    perf_indices=(2,),       # velocity
    metric="locomotion",
    fall=0.6,
)
```

### `configs/exp/hopper_fpl_vs_linear.py` — an experiment with a sweep

```python
"""Flagship: one FPL spec (p<0, uniform) dominates the entire linear-weight family on the
(speed, fall-rate) plane with no per-speed retuning.  FPL_MPPI_HANDOFF section 6.

Fairness: identical proposal, identical temporal aggregation. ONLY the objective-axis
collapse varies -- power_mean(x, p=1, w) == sum(w_i x_i) exactly, so p=1 IS the linear cost.
"""
from configs.env.hopper import HOPPER, HOPPER_META
from analytic_mppi.config import linear_weights

BASE_CFG = HOPPER.with_(**{"run.seeds": (0, 1, 2, 3, 4), "proposal.num_samples": 128})

LINEAR_WV = [0.5, 1.0, 2.0, 4.0, 8.0]
TARGET_VELS = [1.0, 2.0, 3.0]

def configs():
    arms = [
        BASE_CFG.with_(**{"objective.p": -1.0, "objective.weights": None, "label": "FPL p=-1"})
    ] + [
        BASE_CFG.with_(**{
            "objective.p": 1.0,
            # the relationship stays in the file -- not flattened to magic numbers
            "objective.weights": linear_weights(HOPPER_META, wv),
            "label": f"lin wv={wv:g}",
        })
        for wv in LINEAR_WV
    ]
    return [
        arm.with_(**{"task.kwargs": {"target_velocity": tv}})
        for tv in TARGET_VELS for arm in arms
    ]
```

Compare to the current `build_configs()` in `verification/hopper_pareto_sweep.py:56-65`, where the
weight vector `[1.0, 1.0, wv, 0.5]` is written out by hand and has to stay in sync with the atom
layout by human vigilance.

### `configs/exp/hopper_argmax_ablation.py` — the axis you asked about

```python
"""Selection-rule ablation: identical Gaussian proposal, identical FPL cost.
ONLY update_mean differs.
"""
from configs.env.hopper import HOPPER, HOPPER_META

def configs():
    return [
        HOPPER.with_(**{"update.rule": "path_integral", "update.temperature": 0.2,
                        "label": "path integral"}),
        # argmax needs the mean injected as sample 0 so the plan can never regress
        HOPPER.with_(**{"update.rule": "argmax", "proposal.include_mean": True,
                        "label": "argmax"}),
        HOPPER.with_(**{"update.rule": "reward_proportional", "label": "proportional"}),
        HOPPER.with_(**{"update.rule": "shielded", "update.temperature": 0.2,
                        "update.extra": {"safety_indices": HOPPER_META.safety_indices,
                                         "perf_indices": HOPPER_META.perf_indices,
                                         "safety_floor": 0.3,
                                         "perf_mode": "balanced"},
                        "label": "shielded"}),
    ]
```

---

## 6. Resolution: config to controller

### 6.1 Stage 1 — dispatch table, no controller refactor

`(proposal.kind, update.rule)` maps to an existing class plus kwargs:

```python
_DISPATCH = {
    ("gaussian", "path_integral"):       ("mppi",                {}),
    ("gaussian", "reward_proportional"): ("mppi",                {"fpl_weighting": "proportional"}),
    ("gaussian", "reward_exp"):          ("mppi",                {"fpl_weighting": "exp"}),
    ("gaussian", "argmax"):              ("predictive_sampling", {}),
    ("gaussian", "shielded"):            ("fpl_shielded",        {}),
    ("cem",      "elite_mean"):          ("cem",                 {}),
    ("cma",      "path_integral"):       ("mppi_cma",            {}),
    ("colored",  "path_integral"):       ("fpl_colored",         {}),
    ("adaptive", "path_integral"):       ("fpl_adaptive",        {}),
    ("gmm",      "path_integral"):       ("fpl_gmm",             {}),
}
```

Unrepresentable combinations (`cem` + `argmax`) raise at load time listing the valid pairs, instead
of silently running something else. This stage is a rename layer — zero risk to existing results.

> **Subtlety worth encoding.** `("gaussian", "argmax")` is *not* pure argmax over the same cloud.
> `predictive_sampling.py:38` forces `knots[0] = self.mean`. That is why `proposal.include_mean` is
> a separate field. In stage 1 the resolver should **assert** `include_mean is True` when
> dispatching to `predictive_sampling` and refuse `False`, rather than accept a config it cannot
> honour. The ablation in §5 is only a clean selection-rule comparison because both arms are told
> about this.

### 6.2 Stage 2 — pluggable update rules (DEFERRED)

> **Status: deferred.** Phases 0/1/3 stand alone and do not depend on this. Recorded here so the
> analysis is not lost if the proposal × update ablation grid is later needed for the paper.

**What it is for.** Every controller class today bundles two independent choices, and the update
rule is expressed through *class identity*:

| class | proposal (`sample_knots`) | update (`update_mean`) |
|---|---|---|
| `MPPIv2` | gaussian | softmax / proportional / exp |
| `PredictiveSampling` | gaussian **+ mean injected** | argmax |
| `CEM` | gaussian w/ elite sigma | elite mean |
| `DIAL` | annealed gaussian | softmax |
| `MppiCma` | CMA gaussian | softmax |
| `FplColoredMPPI` | colored noise | softmax |
| `FplAdaptiveMPPI` | adaptive gaussian | softmax |
| `FplTemperedMPPI` | *inherited from MPPIv2* | tempered weighting |
| `FplShieldedMPPI` | *inherited from MPPIv2* | shielded lexicographic |
| `ComposedGradientMPPI` | *inherited from MPPIv2* | composed multi-objective |

The last three override `update_mean` **only** — they subclass `MPPIv2` purely to borrow its
Gaussian sampler. That is inheritance used as a mixin, and it is the tell that one axis is
masquerading as a class hierarchy.

Consequently these are inexpressible today: **colored + shielded** (would need
`class C(FplColoredMPPI, FplShieldedMPPI)` and MRO reasoning), **CMA + argmax** (the "is the softmax
doing the work, or the adaptive proposal?" ablation), **CEM proposal + path integral**. It is a
~6 × 6 grid; ~10 cells have classes, and each new cell costs a class.

For the paper: a single-axis ablation is only honest if the arms differ on one axis. Stage 2 makes
that structurally true rather than an assertion in a caption.

**Design — three steps, not two.** An earlier draft of this section claimed update rules could be
pure `Trajectory -> mean` functions. That is wrong. Four of them mutate *proposal* state inside
`update_mean`, because that is the only hook that sees the scored trajectory:

- `mppi_cma.py:127-141` — updates `self.cov`
- `cem.py:79-80` — updates `self.cov` from the elite spread
- `fpl_colored.py:88-96` — `super().update_mean(traj)`, then EMAs `self._explore_mult`
- `fpl_adaptive.py:133-137` — `super().update_mean(traj)`, then `self._adapt_sigma(traj, mean_old)`

That is proposal adaptation, not update logic. So the decomposition needs a separate hook:

```python
traj = self._rollout_and_score(state)
self.mean = self.update_rule(traj)   # pure: scores -> new mean
self.proposal.observe(traj)          # sigma / covariance / amplitude adaptation
```

`analytic_mppi/updates.py` then holds genuinely pure functions:

```python
def path_integral(traj, *, temperature): ...      # softmax on traj.scores (cost convention)
def argmax(traj): ...
def reward_proportional(traj): ...
def reward_exp(traj, *, temperature): ...
def elite_mean(traj, *, num_elites): ...
def shielded(traj, *, safety_indices, perf_indices, safety_floor,
             terminal_safety_floor, perf_mode, temperature): ...
```

Every adaptive controller already runs update-then-adapt in that order, so preserving behaviour is
a mechanical move rather than a redesign. Each existing class keeps its current rule as its default,
`proposal` and `update` become genuinely independent, and `_DISPATCH` collapses to a proposal
lookup. **Config files written in stage 1 keep working unchanged** — the reason to do stage 1 first.

`FplShieldedMPPI` stops being an `MPPIv2` subclass and becomes `update.rule="shielded"`.

**Gate before landing:** run one pareto sweep pre- and post-refactor at fixed seeds and assert the
trajectories match bit-for-bit.

### 6.3 Validation: derive it, do not maintain it

Do not hand-write a param whitelist — `_ALGO_PARAMS` is one, and it rotted. Introspect instead:

```python
def accepted_params(cls) -> set[str]:
    """Named kwargs across the MRO, following **kwargs passthrough."""
    names, seen_var_kw = set(), False
    for klass in cls.__mro__:
        if klass is object:
            break
        sig = inspect.signature(klass.__init__)
        for p in sig.parameters.values():
            if p.kind is p.VAR_KEYWORD:
                seen_var_kw = True
            elif p.name not in ("self", "task", "backend"):
                names.add(p.name)
        if not seen_var_kw:
            break            # this class does not forward -- stop walking up
    return names
```

Any resolved kwarg not in that set is a hard error with a `difflib.get_close_matches` suggestion.
This catches typos in `extra` dicts, which is the one place the type checker cannot help.

---

## 7. Provenance

Every run writes, next to its results:

```
runs/<name>/config.resolved.json    # to_dict(), fully resolved, after sweep expansion
runs/<name>/provenance.json         # git SHA, dirty flag, config hash, timestamp, host
```

`--print-config` dumps the resolved JSON without running. `configs/paper/` is frozen for anything
cited in the paper. When a reviewer asks what temperature the hopper flagship used, it is one file
rather than an archaeology dig through argparse defaults.

---

## 8. Migration phases

| phase | work | risk |
|---|---|---|
| 0 | `analytic_mppi/config.py` (dataclasses, `with_`, dispatch, validation, `to_dict`) + `run.py --config configs/exp/foo.py` + `--set a.b=c` overrides | none — purely additive |
| 1 | Port `_experiment.ENVS` to `configs/env/*.py`; `_experiment` reads them. Assert loaded values equal today's dict once, then delete the dict | low — guarded by the assert |
| 2 | **DEFERRED** — `updates.py` registry + `proposal.observe()`; classes keep current defaults | medium — needs a bit-identical regression run |
| 3 | Sweep runner; pareto scripts become `configs()` + a thin metrics hook | low |

Phase 0 alone solves "I cannot keep track of the hyperparameters". Phases 1 and 3 stop it recurring.

Phase 2 is deferred (see §6.2). It is the only phase that touches controller internals, and the
config surface does not change when it eventually lands — configs written against phase 0 keep
working. Revisit it if the paper needs the proposal × update ablation grid.
