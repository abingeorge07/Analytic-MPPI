"""Phase 0 config layer (docs/config_design.md).

The load-bearing test here is `test_hopper_matches_experiment_envs`: while phase 1 is
outstanding, configs/env/hopper.py is a SECOND copy of verification/_experiment.ENVS
["hopper"], and a second copy is exactly the drift this design exists to eliminate.
"""
import json

import pytest

from analytic_mppi.config import (ConfigError, ExperimentConfig, accepted_params,
                                  apply_overrides, from_dict, linear_weights,
                                  load_config_file, resolve)
from analytic_mppi.controllers import MPPIv2, PredictiveSampling
from analytic_mppi.eval import ALL_CONTROLLERS


# --- drift guard -----------------------------------------------------------------

def test_hopper_matches_experiment_envs():
    """configs/env/hopper.py must stay identical to the published ENVS entry."""
    from verification._experiment import ENVS
    from configs.env.hopper import HOPPER, HOPPER_META

    spec = ENVS["hopper"]
    assert HOPPER.task.name == spec["task"]
    assert HOPPER.run.steps == spec["steps"]
    assert HOPPER.proposal.plan_horizon == spec["horizon"]
    assert HOPPER.proposal.num_knots == spec["knots"]
    assert HOPPER.proposal.noise_level == spec["noise"]
    assert HOPPER.update.temperature == spec["temp"]
    assert HOPPER.objective.time_p == spec["time_p"]
    assert HOPPER.task.kwargs[spec["difficulty_key"]] == spec["difficulty"]
    assert HOPPER_META.n_atoms == spec["n_atoms"]
    assert HOPPER_META.lin_idx == spec["lin_idx"]
    assert HOPPER_META.safe_w == spec["safe_w"]
    assert HOPPER_META.metric == spec["metric"]
    assert HOPPER_META.fall == spec["fall"]


def test_linear_weights_matches_experiment_helper():
    from verification._experiment import linear_weights as ref
    from configs.env.hopper import HOPPER_META

    for wv in (0.5, 1.0, 4.0):
        assert list(linear_weights(HOPPER_META, wv)) == ref("hopper", wv)


# --- introspection ---------------------------------------------------------------

def test_accepted_params_follows_kwargs_passthrough():
    # PredictiveSampling declares only num_samples/noise_level and forwards **kwargs,
    # so the FPL surface from SamplingController must still be visible.
    accepts = accepted_params(PredictiveSampling)
    assert {"noise_level", "num_samples", "fpl_p", "fpl_time_p", "fpl_weights"} <= accepts
    # It has no temperature -- that is the whole point of the argmax arm.
    assert "temperature" not in accepts
    assert "temperature" in accepted_params(MPPIv2)


@pytest.mark.parametrize("name", sorted(ALL_CONTROLLERS))
def test_accepted_params_covers_every_controller(name):
    assert accepted_params(ALL_CONTROLLERS[name]), f"{name} exposed no params"


# --- resolution ------------------------------------------------------------------

def test_hopper_resolves_to_mppi():
    r = resolve(load_config_file("configs/env/hopper.py"))
    assert r.controller == "mppi"
    assert r.build["noise_level"] == 0.3
    assert r.build["temperature"] == 0.2
    assert r.build["fpl_time_p"] == -2.0
    assert r.build["cost_mode"] == "fpl_cost"
    assert r.build["fpl_p"] == -1.0
    assert r.init_fn is not None


def test_ablation_arms_all_resolve():
    from configs.exp.hopper_argmax_ablation import configs

    got = {c.label: resolve(c).controller for c in configs()}
    assert got == {
        "path integral": "mppi",
        "argmax": "predictive_sampling",
        "proportional": "mppi",
        "shielded": "fpl_shielded",
    }


def test_proposal_noise_is_respelled_per_kind():
    cfg = load_config_file("configs/env/hopper.py")
    assert resolve(cfg.with_(**{"proposal.kind": "cma"})).build["initial_noise_level"] == 0.3


# --- the failures this module exists to produce ----------------------------------

def test_unrepresentable_pair_errors():
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="not available"):
        resolve(cfg.with_(**{"proposal.kind": "cem", "update.rule": "argmax"}))


def test_argmax_requires_include_mean():
    """PredictiveSampling forces knots[0]=mean, so the config must say so."""
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="include_mean"):
        resolve(cfg.with_(**{"update.rule": "argmax"}))
    resolve(cfg.with_(**{"update.rule": "argmax", "proposal.include_mean": True}))


def test_non_default_param_controller_cannot_accept_is_an_error():
    """The run.py:_ALGO_PARAMS bug: --algo predictive_sampling --temperature 0.05
    silently ignores the temperature. Here it must fail."""
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="does not accept 'temperature'"):
        resolve(cfg.with_(**{"update.rule": "argmax", "proposal.include_mean": True,
                             "update.temperature": 0.05}))


def test_at_default_param_is_dropped_not_errored():
    cfg = load_config_file("configs/env/hopper.py")
    r = resolve(cfg.with_(**{"update.rule": "argmax", "proposal.include_mean": True}))
    assert "temperature" in r.dropped
    assert "temperature" not in r.build


def test_typo_in_extra_is_caught():
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="temperature"):   # suggestion for 'temprature'
        resolve(cfg.with_(**{"update.extra": {"temprature": 0.1}}))


def test_typo_in_field_path_is_caught():
    with pytest.raises(ConfigError, match="no field 'pp'"):
        ExperimentConfig().with_(**{"objective.pp": 1.0})


def test_missing_required_ctor_param_is_caught():
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="num_elites"):
        resolve(cfg.with_(**{"proposal.kind": "cem", "update.rule": "elite_mean"}))


@pytest.mark.parametrize("bad", [
    {"objective.mode": "fpl_costt"},
    {"objective.gamma": 1.0},          # (1-g)/(1-g^H) is 0/0
    {"run.spline_type": "quad"},
    {"run.init": "hopper_stnad"},
    {"proposal.num_samples": 0},
    {"update.rule": "softmax"},        # old spelling; it is path_integral now
])
def test_schema_validation(bad):
    with pytest.raises(ConfigError):
        ExperimentConfig().with_(**bad)


def test_cubic_spline_type_is_valid():
    cfg = ExperimentConfig().with_(**{"run.spline_type": "cubic"})
    assert cfg.run.spline_type == "cubic"


# --- overrides / provenance ------------------------------------------------------

def test_set_overrides_parse_as_json():
    cfg = apply_overrides(ExperimentConfig(), [
        "objective.p=1.0",
        "objective.weights=[1.0, 1.0, 4.0, 0.5]",
        "task.kwargs.target_velocity=3.0",
        "run.init=null",
        "label=linear wv=4",
    ])
    assert cfg.objective.p == 1.0
    assert cfg.objective.weights == (1.0, 1.0, 4.0, 0.5)
    assert cfg.task.kwargs["target_velocity"] == 3.0
    assert cfg.run.init is None
    assert cfg.label == "linear wv=4"      # not valid JSON -> kept as a string


def test_json_round_trip():
    cfg = load_config_file("configs/env/hopper.py")
    assert from_dict(json.loads(cfg.to_json())) == cfg


def test_hash_is_stable_and_sensitive():
    cfg = load_config_file("configs/env/hopper.py")
    assert cfg.hash() == load_config_file("configs/env/hopper.py").hash()
    assert cfg.hash() != cfg.with_(**{"objective.p": 1.0}).hash()


def test_index_required_for_multi_config_file():
    with pytest.raises(ConfigError, match="--index"):
        load_config_file("configs/exp/hopper_argmax_ablation.py")
    assert load_config_file("configs/exp/hopper_argmax_ablation.py", 1).label == "argmax"


def test_extra_cannot_contradict_the_declared_update_rule():
    """fpl_weighting IS what makes reward_proportional that rule; extra must not flip it."""
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="fixed to 'proportional'"):
        resolve(cfg.with_(**{"update.rule": "reward_proportional",
                             "update.extra": {"fpl_weighting": "exp"}}))


# --- planning backend (run.backend) ----------------------------------------------
# These are all name-based checks in resolve(), so they must pass WITHOUT jax installed.

def test_backend_defaults_to_mujoco_and_flows_into_build():
    r = resolve(load_config_file("configs/env/hopper.py"))
    assert r.config.run.backend == "mujoco"
    assert r.build["backend"] == "mujoco"


def test_unknown_backend_is_caught():
    with pytest.raises(ConfigError, match="run.backend"):
        ExperimentConfig().with_(**{"run.backend": "mjxx"})


def test_mjx_accepts_a_core_cost_mode_on_a_supported_task():
    cfg = load_config_file("configs/env/hopper.py")       # objective.mode = fpl_cost
    assert resolve(cfg.with_(**{"run.backend": "mjx"})).build["backend"] == "mjx"


@pytest.mark.parametrize("mode", ["fpl_layered", "hybrid"])
def test_mjx_rejects_non_core_cost_modes(mode):
    """The jnp scorer implements only normal/fpl_cost/fpl_discounted; planning on MJX with
    a layered/hybrid objective would score something other than what the config declares."""
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="objective.mode"):
        resolve(cfg.with_(**{"run.backend": "mjx", "objective.mode": mode}))


def test_mjx_rejects_a_task_without_jax_costs():
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="jnp costs"):
        resolve(cfg.with_(**{"run.backend": "mjx", "task.name": "cube",
                             "run.init": "cube"}))


def test_mjx_validation_needs_no_jax_import():
    """Resolving an mjx config must not import jax: config validation has to work on a
    machine that never installed the [mjx] extra. Run in a subprocess so an import from
    an earlier test in this session cannot mask it."""
    import subprocess
    import sys

    src = (
        "import sys\n"
        "from analytic_mppi.config import load_config_file, resolve\n"
        "cfg = load_config_file('configs/env/hopper.py')\n"
        "resolve(cfg.with_(**{'run.backend': 'mjx'}))\n"
        "assert 'jax' not in sys.modules, 'resolve() imported jax'\n"
    )
    subprocess.run([sys.executable, "-c", src], check=True)


def test_v1_provenance_json_still_loads():
    """SPEC_VERSION 1 records have no run.backend; from_dict must default it."""
    cfg = load_config_file("configs/env/hopper.py")
    d = json.loads(cfg.to_json())
    del d["run"]["backend"]
    d["spec_version"] = 1
    assert from_dict(d).run.backend == "mujoco"


def test_extra_cannot_shadow_a_schema_field():
    cfg = load_config_file("configs/env/hopper.py")
    with pytest.raises(ConfigError, match="already set"):
        resolve(cfg.with_(**{"proposal.extra": {"noise_level": 0.9}}))
    with pytest.raises(ConfigError, match="already set"):
        resolve(cfg.with_(**{"update.extra": {"temperature": 0.9}}))
