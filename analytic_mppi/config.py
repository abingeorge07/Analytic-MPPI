"""Typed configuration for a controller run (design: docs/config_design.md).

One `ExperimentConfig` fully determines a run: same config + same seed -> same trajectory.
Every field maps onto exactly one stage of `SamplingController.act()`:

    proposal   -> sample_knots()      how candidate knots are drawn
    objective  -> _score_*()          how a rollout becomes one scalar
    update     -> update_mean()       how scores become the new mean

Authoring happens in Python (`configs/*.py`) so derived hyperparameters keep their
relationship in the file -- e.g. `num_elites=max(4, K // 8)` -- instead of being flattened
to magic numbers. JSON is the *provenance* format: `to_json()` writes a flat, diffable,
hashable record of exactly what ran, and `from_dict()` reads it back.

Typical use:

    from analytic_mppi.config import load_config_file, resolve
    cfg = load_config_file("configs/env/hopper.py")
    r = resolve(cfg)
    res = run_episode(r.task_name, r.controller, steps=r.steps, seed=0,
                      init_fn=r.init_fn, task_kwargs=r.task_kwargs, **r.build)
"""
from __future__ import annotations

import dataclasses
import difflib
import hashlib
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from analytic_mppi.eval import (ALL_CONTROLLERS, init_barkour_stand, init_cube,
                                init_g1_stand, init_hang_down, init_hopper_stand)
from analytic_mppi.tasks.jax_costs import JAX_COST_TASKS, has_jax_costs


# Bump when a field is renamed/removed so old provenance JSON is identifiable.
#   1 -> 2: added `run.backend` ("mujoco" | "mjx"). v1 JSON still loads via from_dict
#           because the field has a default; v2 JSON is not readable by v1 code.
SPEC_VERSION = 2


class ConfigError(ValueError):
    """Raised for any malformed / unrepresentable configuration."""


# Named initial-state setups, so `run.init` stays a plain string in the JSON record.
INIT_FNS: dict[str, Callable[[Any], None]] = {
    "hang_down": init_hang_down,
    "hopper_stand": init_hopper_stand,
    "g1_stand": init_g1_stand,
    "cube": init_cube,
    "barkour_stand": init_barkour_stand,
}


# Planning backends. "mjx" needs the `[mjx]` extra installed AND a task with jnp costs
# (tasks.jax_costs.JAX_COST_TASKS) when the objective must be differentiated.
BACKENDS = ("mujoco", "mjx")

# Cost modes the jnp scoring port covers. The rest (fpl_layered, hybrid) exist only in the
# numpy scorer, so an mjx-planned run must not silently score with a different objective.
JAX_COST_MODES = ("normal", "fpl_cost", "fpl_discounted")


def _suggest(name: str, options) -> str:
    close = difflib.get_close_matches(name, sorted(options), n=3)
    return f" did you mean {close}?" if close else f" known: {sorted(options)}"


def _check(value, allowed, what: str) -> None:
    if value not in allowed:
        raise ConfigError(f"{what}={value!r} is not valid.{_suggest(str(value), allowed)}")


def _set_path(obj: Any, path: str, value: Any) -> Any:
    """Return a copy of `obj` with the dotted `path` set to `value`.

    Descends through nested dataclasses AND dicts, so both `objective.p` and
    `task.kwargs.target_velocity` work.
    """
    head, _, tail = path.partition(".")
    if dataclasses.is_dataclass(obj):
        names = {f.name for f in dataclasses.fields(obj)}
        if head not in names:
            raise ConfigError(
                f"{type(obj).__name__} has no field {head!r}.{_suggest(head, names)}")
        if not tail:
            return replace(obj, **{head: value})
        return replace(obj, **{head: _set_path(getattr(obj, head), tail, value)})
    if isinstance(obj, dict):
        new = dict(obj)
        new[head] = value if not tail else _set_path(new.get(head, {}), tail, value)
        return new
    raise ConfigError(
        f"cannot descend into {type(obj).__name__} at {head!r} (path {path!r})")


class _Node:
    """Mixin giving every config dataclass a dotted-path override helper."""

    def with_(self, **dotted: Any):
        """Nested override:  cfg.with_(**{"objective.p": 1.0, "label": "linear"})"""
        out = self
        for path, value in dotted.items():
            out = _set_path(out, path, value)
        return out

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
#  Schema
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Run(_Node):
    steps: int = 150
    seeds: tuple[int, ...] = (0,)          # one episode per seed
    spline_type: str = "zero"              # zero | linear | cubic
    iterations: int = 1                    # optimizer iterations per MPC step
    init: Optional[str] = None             # key into INIT_FNS
    nthread: Optional[int] = None          # None -> cpu_count
    mode: str = "headless"                 # headless | live | record
    out: Optional[str] = None
    # Which simulator PLANS the rollouts. "mujoco" (default) is the CPU
    # `mujoco.rollout` backend behind every existing result. "mjx" is MuJoCo MJX on
    # JAX/GPU -- required by gradient MPC (it differentiates through the rollout) and
    # optional for samplers. The closed loop is stepped by CPU MuJoCo either way, so the
    # physics of record never changes; see dynamics/mjx_backend.py.
    backend: str = "mujoco"                # mujoco | mjx

    def __post_init__(self):
        _check(self.spline_type, ("zero", "linear", "cubic"), "run.spline_type")
        _check(self.mode, ("headless", "live", "record"), "run.mode")
        _check(self.backend, BACKENDS, "run.backend")
        if self.init is not None:
            _check(self.init, INIT_FNS, "run.init")
        if self.steps <= 0:
            raise ConfigError(f"run.steps must be > 0, got {self.steps}")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds:
            raise ConfigError("run.seeds must contain at least one seed")


@dataclass(frozen=True)
class TaskSpec(_Node):
    name: str = "hopper"
    kwargs: dict[str, Any] = field(default_factory=dict)   # e.g. {"target_velocity": 2.0}


@dataclass(frozen=True)
class Proposal(_Node):
    """How candidate knots are drawn.  -> sample_knots()"""
    kind: str = "gaussian"                 # see PROPOSALS
    num_samples: int = 128
    num_knots: int = 4
    plan_horizon: float = 0.6
    noise_level: float = 0.3               # spelled per-kind on the way out (see _Dispatch)
    include_mean: bool = False             # inject current mean as sample 0
    extra: dict[str, Any] = field(default_factory=dict)    # kind-specific (color_beta, ...)

    def __post_init__(self):
        _check(self.kind, PROPOSALS, "proposal.kind")
        if self.num_samples <= 0:
            raise ConfigError(f"proposal.num_samples must be > 0, got {self.num_samples}")
        if self.num_knots <= 0:
            raise ConfigError(f"proposal.num_knots must be > 0, got {self.num_knots}")
        if self.plan_horizon <= 0:
            raise ConfigError(f"proposal.plan_horizon must be > 0, got {self.plan_horizon}")


@dataclass(frozen=True)
class Objective(_Node):
    """How a rollout becomes one scalar.  -> _score_normal / _score_fpl / _score_fpl_layered"""
    mode: str = "fpl_cost"                 # normal | fpl_cost | fpl_discounted | fpl_layered | hybrid
    p: float = -1.0                        # objective-axis power mean; p=1 + weights == linear sum
    gamma: float = 0.99
    time_p: Optional[float] = -2.0         # None -> discounted mean; <=0 -> weakest-link over time
    weights: Optional[tuple[float, ...]] = None    # per-atom; None -> uniform
    group_p: Optional[float] = None                # inner power mean (fpl_layered)
    term_indices: Optional[tuple[int, ...]] = None # restrict reward to a subset of atoms
    floor_weight: float = 1.0                      # hybrid log-barrier scale

    def __post_init__(self):
        _check(self.mode, COST_MODES, "objective.mode")
        # gamma=1 makes the finite-horizon normaliser (1-g)/(1-g^H) a 0/0 -- mirrors the
        # assert in sampling_base.py, but fails at config-load time instead of mid-run.
        if not (0.0 <= self.gamma < 1.0):
            raise ConfigError(f"objective.gamma must be in [0, 1), got {self.gamma}")
        if self.weights is not None:
            object.__setattr__(self, "weights", tuple(float(w) for w in self.weights))
        if self.term_indices is not None:
            object.__setattr__(self, "term_indices", tuple(int(i) for i in self.term_indices))


@dataclass(frozen=True)
class Update(_Node):
    """How scores become the new mean.  -> update_mean()"""
    rule: str = "path_integral"   # see UPDATE_RULES
    temperature: float = 0.2
    extra: dict[str, Any] = field(default_factory=dict)    # rule-specific (safety_floor, ...)

    def __post_init__(self):
        _check(self.rule, UPDATE_RULES, "update.rule")
        if self.temperature <= 0:
            raise ConfigError(f"update.temperature must be > 0, got {self.temperature}")


@dataclass(frozen=True)
class ExperimentConfig(_Node):
    run: Run = field(default_factory=Run)
    task: TaskSpec = field(default_factory=TaskSpec)
    proposal: Proposal = field(default_factory=Proposal)
    objective: Objective = field(default_factory=Objective)
    update: Update = field(default_factory=Update)
    label: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["spec_version"] = SPEC_VERSION
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str)

    def hash(self) -> str:
        """Stable short digest of the resolved config, for provenance filenames."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]


@dataclass(frozen=True)
class EnvMeta(_Node):
    """Per-environment facts the experiment layer needs (today: _experiment.ENVS).

    Not part of ExperimentConfig: these describe the *task's atom layout*, not a
    controller setting, and several experiments derive configs from them.
    """
    n_atoms: int
    lin_idx: int                       # atom the linear-weight family sweeps
    safe_w: float                      # control-atom weight in the linear family
    safety_indices: tuple[int, ...]
    perf_indices: tuple[int, ...]
    metric: str                        # locomotion | cube | quadruped
    fall: Optional[float] = None       # min-uprightness fall line (locomotion)


def linear_weights(meta: EnvMeta, wv: float) -> tuple[float, ...]:
    """Linear-family weight vector: `wv` on the swept progress atom, 1 on the safety
    atoms, `safe_w` on control. Mirrors verification/_experiment.py:linear_weights."""
    w = [1.0] * meta.n_atoms
    w[meta.lin_idx] = float(wv)
    w[-1] = meta.safe_w
    return tuple(w)


# ---------------------------------------------------------------------------
#  Dispatch: (proposal.kind, update.rule) -> existing controller + kwargs
# ---------------------------------------------------------------------------

COST_MODES = ("normal", "fpl_cost", "fpl_discounted", "fpl_layered", "hybrid")


@dataclass(frozen=True)
class _Dispatch:
    controller: str                       # key into eval.ALL_CONTROLLERS
    noise_key: str = "noise_level"        # how this proposal spells its scale param
    kwargs: dict[str, Any] = field(default_factory=dict)
    require_include_mean: Optional[bool] = None

# NOTE: there is deliberately no `temp_key`. An earlier draft let a dispatch entry declare
# "this rule has no temperature", which removed temperature from the candidate list -- and
# therefore from the accepts-check -- so `cem + update.temperature=0.05` was accepted and
# silently ignored. That is precisely the run.py:_ALGO_PARAMS bug this module exists to kill.
# Temperature is always a candidate; `accepted_params` decides whether it is passed, dropped,
# or an error. (It is also not optional to omit for rules that "do not use" it: MPPIv2 requires
# it in its ctor, and its proportional branch falls through to the softmax under a normal cost.)


# Phase 0 is a rename layer over the existing classes: the update rule is still
# implied by class identity, so only the pairs someone wrote a class for exist.
# Unrepresentable pairs raise at load time rather than silently running something
# else. Phase 2 (docs/config_design.md 6.2, deferred) removes this table.
_DISPATCH: dict[tuple[str, str], _Dispatch] = {
    ("gaussian", "path_integral"):       _Dispatch("mppi"),
    ("gaussian", "reward_proportional"): _Dispatch("mppi", kwargs={"fpl_weighting": "proportional"}),
    ("gaussian", "reward_exp"):          _Dispatch("mppi", kwargs={"fpl_weighting": "exp"}),
    # PredictiveSampling forces knots[0] = self.mean (predictive_sampling.py:38), so this
    # is argmax over the cloud PLUS the current mean -- not argmax over the same cloud as
    # ("gaussian", "path_integral"). Demanding include_mean=True keeps the config honest
    # instead of silently accepting one it cannot honour.
    ("gaussian", "argmax"):              _Dispatch("predictive_sampling",
                                                   require_include_mean=True),
    ("gaussian", "shielded"):            _Dispatch("fpl_shielded"),
    ("gaussian", "tempered"):            _Dispatch("fpl_tempered"),
    ("gaussian", "composed"):            _Dispatch("composed_grad"),
    ("cma",      "path_integral"):       _Dispatch("mppi_cma", noise_key="initial_noise_level"),
    ("cem",      "elite_mean"):          _Dispatch("cem", noise_key="sigma_start"),
    ("dial",     "path_integral"):       _Dispatch("dial"),
    ("colored",  "path_integral"):       _Dispatch("fpl_colored"),
    ("adaptive", "path_integral"):       _Dispatch("fpl_adaptive"),
    ("gmm",      "path_integral"):       _Dispatch("fpl_gmm", noise_key="sigma_max"),
}

PROPOSALS = tuple(dict.fromkeys(k[0] for k in _DISPATCH))
UPDATE_RULES = tuple(dict.fromkeys(k[1] for k in _DISPATCH))


def accepted_params(cls: type) -> set[str]:
    """Named kwargs `cls.__init__` accepts, following **kwargs up the MRO.

    Derived by introspection rather than a hand-maintained table -- the hand-maintained
    one (`run.py:_ALGO_PARAMS`) is what silently dropped parameters.
    """
    names: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue                      # class does not define its own __init__
        forwards = False
        for p in inspect.signature(init).parameters.values():
            if p.kind is p.VAR_KEYWORD:
                forwards = True
            elif p.kind is not p.VAR_POSITIONAL and p.name not in ("self", "task", "backend"):
                names.add(p.name)
        if not forwards:
            break                         # does not pass **kwargs up; stop walking
    return names


# Handled by eval.make_controller itself, not forwarded as algo kwargs.
_FRAMEWORK_KEYS = frozenset({
    "num_samples", "num_knots", "plan_horizon", "spline_type", "iterations",
    "cost_mode", "fpl_p", "fpl_gamma", "nthread", "backend",
})


def _field_default(node, name: str) -> Any:
    for f in dataclasses.fields(node):
        if f.name == name:
            return f.default if f.default is not dataclasses.MISSING else None
    return None


@dataclass(frozen=True)
class Resolved:
    """Everything eval.run_episode needs, plus the config it came from."""
    config: ExperimentConfig
    controller: str
    task_name: str
    task_kwargs: dict[str, Any]
    steps: int
    seeds: tuple[int, ...]
    init_fn: Optional[Callable[[Any], None]]
    build: dict[str, Any]
    dropped: tuple[str, ...] = ()      # at-default keys this controller does not accept


def _check_backend(cfg: ExperimentConfig, d: _Dispatch) -> None:
    """Validate the (backend, controller, objective, task) combination.

    All checks are NAME-based -- no jax import -- so a machine without the `[mjx]` extra
    still validates configs correctly (it fails at backend CONSTRUCTION with an install
    hint, not here with an ImportError).
    """
    if cfg.run.backend == "mujoco":
        return

    # objective: the jnp scorer implements only the core aggregations. Planning on MJX with
    # fpl_layered/hybrid would score with a DIFFERENT objective than the config declares.
    if cfg.objective.mode not in JAX_COST_MODES:
        raise ConfigError(
            f"run.backend='mjx' supports objective.mode in {list(JAX_COST_MODES)}, but the "
            f"config sets {cfg.objective.mode!r}. The layered/hybrid compositions exist only "
            f"in the numpy scorer -- run them on run.backend='mujoco'."
        )

    # task: needs a jnp cost implementation to be differentiable / scored on device.
    if not has_jax_costs(cfg.task.name):
        raise ConfigError(
            f"run.backend='mjx' requires a task with jnp costs; {cfg.task.name!r} has none. "
            f"Available: {sorted(JAX_COST_TASKS)}."
        )


def resolve(cfg: ExperimentConfig) -> Resolved:
    """Turn a config into concrete `eval.run_episode` kwargs, validating as we go.

    A mapped kwarg the chosen controller does not accept is an ERROR when its value
    differs from the schema default, and is dropped (and reported in `.dropped`) when
    it is still at the default. That is the middle ground between run.py's silent
    drop -- which loses `--temperature 0.05` without a word -- and forcing every
    config to null out fields irrelevant to its controller.
    """
    key = (cfg.proposal.kind, cfg.update.rule)
    d = _DISPATCH.get(key)
    if d is None:
        valid = sorted(f"{p}+{u}" for p, u in _DISPATCH)
        raise ConfigError(
            f"proposal.kind={cfg.proposal.kind!r} with update.rule={cfg.update.rule!r} is not "
            f"available. Phase 0 dispatches onto existing controller classes, so only these "
            f"pairs exist: {valid}. (Arbitrary pairing is phase 2 -- docs/config_design.md 6.2.)"
        )
    if d.require_include_mean is not None and cfg.proposal.include_mean != d.require_include_mean:
        raise ConfigError(
            f"{key[0]}+{key[1]} maps to {d.controller!r}, which forces knots[0] = mean, so it "
            f"requires proposal.include_mean={d.require_include_mean}. Got "
            f"{cfg.proposal.include_mean}. Set it explicitly so the config records what ran."
        )

    _check_backend(cfg, d)

    cls = ALL_CONTROLLERS[d.controller]
    accepts = accepted_params(cls)

    # Framework kwargs: understood by eval.make_controller for every controller.
    build: dict[str, Any] = dict(
        num_samples=cfg.proposal.num_samples,
        num_knots=cfg.proposal.num_knots,
        plan_horizon=cfg.proposal.plan_horizon,
        spline_type=cfg.run.spline_type,
        iterations=cfg.run.iterations,
        cost_mode=cfg.objective.mode,
        fpl_p=cfg.objective.p,
        fpl_gamma=cfg.objective.gamma,
        nthread=cfg.run.nthread,
        backend=cfg.run.backend,
    )

    # Algo kwargs: (ctor param name, value, schema default) so we can tell an explicit
    # setting from an untouched one.
    candidates: list[tuple[str, Any, Any]] = [
        (d.noise_key, cfg.proposal.noise_level, _field_default(Proposal, "noise_level")),
        ("fpl_time_p", cfg.objective.time_p, _field_default(Objective, "time_p")),
        ("fpl_weights", cfg.objective.weights, _field_default(Objective, "weights")),
        ("fpl_group_p", cfg.objective.group_p, _field_default(Objective, "group_p")),
        ("fpl_term_indices", cfg.objective.term_indices, _field_default(Objective, "term_indices")),
        ("floor_weight", cfg.objective.floor_weight, _field_default(Objective, "floor_weight")),
        ("temperature", cfg.update.temperature, _field_default(Update, "temperature")),
    ]

    algo: dict[str, Any] = {}
    dropped: list[str] = []
    for name, value, default in candidates:
        if name in accepts:
            if value is not None:
                algo[name] = list(value) if isinstance(value, tuple) else value
        elif value != default:
            raise ConfigError(
                f"controller {d.controller!r} does not accept {name!r}, but the config sets it "
                f"to {value!r} (default {default!r}). Either remove it or choose a "
                f"proposal/update pair that supports it.{_suggest(name, accepts)}"
            )
        else:
            dropped.append(name)

    # Dispatch kwargs are what MAKE the rule what it says it is (e.g. fpl_weighting=
    # "proportional" IS update.rule="reward_proportional"), so they go on first and are
    # not overridable.
    algo.update(d.kwargs)

    # Extras may only ADD parameters -- never shadow a schema field or a dispatch kwarg.
    # Silently letting either win is the same class of bug as run.py's dropped params:
    # the config would say one thing and the controller would do another.
    for src, extra in (("proposal.extra", cfg.proposal.extra),
                       ("update.extra", cfg.update.extra)):
        for name, value in extra.items():
            if name not in accepts:
                raise ConfigError(
                    f"{src}: controller {d.controller!r} does not accept "
                    f"{name!r}.{_suggest(name, accepts)}")
            if name in d.kwargs:
                raise ConfigError(
                    f"{src}: {name!r} is fixed to {d.kwargs[name]!r} by "
                    f"update.rule={cfg.update.rule!r}; setting it here would contradict "
                    f"the declared rule.")
            if name in algo:
                raise ConfigError(
                    f"{src}: {name!r} is already set to {algo[name]!r} from a schema field; "
                    f"set it there instead of in extra so the config records one value.")
            algo[name] = value

    missing = _missing_required(cls, set(build) | set(algo))
    if missing:
        raise ConfigError(
            f"controller {d.controller!r} requires {sorted(missing)} with no default; supply "
            f"{'it' if len(missing) == 1 else 'them'} via proposal.extra or update.extra."
        )

    build.update(algo)
    return Resolved(
        config=cfg,
        controller=d.controller,
        task_name=cfg.task.name,
        task_kwargs=dict(cfg.task.kwargs),
        steps=cfg.run.steps,
        seeds=cfg.run.seeds,
        init_fn=None if cfg.run.init is None else INIT_FNS[cfg.run.init],
        build=build,
        dropped=tuple(dropped),
    )


def _missing_required(cls: type, provided: set[str]) -> set[str]:
    """Ctor params with no default that nothing has supplied (e.g. CEM's num_elites)."""
    required: set[str] = set()
    for klass in cls.__mro__:
        if klass is object:
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        forwards = False
        for p in inspect.signature(init).parameters.values():
            if p.kind is p.VAR_KEYWORD:
                forwards = True
            elif (p.kind is not p.VAR_POSITIONAL
                  and p.name not in ("self", "task", "backend")
                  and p.default is inspect.Parameter.empty):
                required.add(p.name)
        if not forwards:
            break
    return required - provided


# ---------------------------------------------------------------------------
#  Loading / overrides / provenance
# ---------------------------------------------------------------------------

def load_config_file(path: str | Path, index: Optional[int] = None) -> ExperimentConfig:
    """Import a `configs/*.py` file and pull one ExperimentConfig out of it.

    The module must expose `CONFIG` (a single config) or `configs()` (a list). With a
    list, `index` selects one; the sweep runner (phase 3) will consume the whole list.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    # Put the repo root on sys.path so config files can `from configs.base import BASE`.
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    spec = importlib.util.spec_from_file_location(f"_amppi_cfg_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"cannot import config file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "configs"):
        items = list(mod.configs())
        if index is None:
            if len(items) != 1:
                raise ConfigError(
                    f"{path.name} defines configs() with {len(items)} configs; pass "
                    f"--index 0..{len(items) - 1} to pick one "
                    f"(labels: {[c.label or '<unlabelled>' for c in items]})")
            return items[0]
        if not 0 <= index < len(items):
            raise ConfigError(f"--index {index} out of range (0..{len(items) - 1})")
        return items[index]
    if hasattr(mod, "CONFIG"):
        if index not in (None, 0):
            raise ConfigError(f"{path.name} defines a single CONFIG; --index {index} is invalid")
        return mod.CONFIG
    raise ConfigError(f"{path.name} defines neither CONFIG nor configs()")


def _coerce(text: str) -> Any:
    """Parse a --set value. JSON first (so lists/numbers/null/bools work), else string."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def apply_overrides(cfg: ExperimentConfig, assignments: Sequence[str]) -> ExperimentConfig:
    """Apply `--set objective.p=1.0` style overrides. Values are parsed as JSON."""
    for item in assignments:
        if "=" not in item:
            raise ConfigError(f"--set expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        cfg = cfg.with_(**{key.strip(): _coerce(raw.strip())})
    return cfg


def _git_provenance() -> dict[str, Any]:
    import subprocess
    root = Path(__file__).resolve().parents[1]
    def _git(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    return {"sha": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain"))}


def save_resolved(cfg: ExperimentConfig, out_dir: str | Path) -> Path:
    """Write the fully-resolved config + git provenance next to a run's results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.resolved.json").write_text(cfg.to_json())
    (out_dir / "provenance.json").write_text(json.dumps(
        {"config_hash": cfg.hash(), "spec_version": SPEC_VERSION, **_git_provenance()},
        indent=2, sort_keys=True))
    return out_dir / "config.resolved.json"


def from_dict(d: dict) -> ExperimentConfig:
    """Rebuild a config from `to_dict()` output (round-trips provenance JSON)."""
    d = {k: v for k, v in d.items() if k != "spec_version"}
    return ExperimentConfig(
        run=Run(**d["run"]),
        task=TaskSpec(**d["task"]),
        proposal=Proposal(**d["proposal"]),
        objective=Objective(**d["objective"]),
        update=Update(**d["update"]),
        label=d.get("label", ""),
        notes=d.get("notes", ""),
    )


__all__ = [
    "SPEC_VERSION", "ConfigError", "INIT_FNS", "COST_MODES", "PROPOSALS", "UPDATE_RULES",
    "Run", "TaskSpec", "Proposal", "Objective", "Update", "ExperimentConfig",
    "EnvMeta", "linear_weights", "Resolved", "resolve", "accepted_params",
    "load_config_file", "apply_overrides", "save_resolved", "from_dict",
]
