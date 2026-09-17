"""The fulfillment-atom floor (WO-3.4 / NEXT_STEPS S1).

These tests exist to protect one property: the floor is applied to the FPL arm and the
LINEAR arm identically. `dM_p/dx_i = w_i x_i^{p-1} M_p^{1-p}` diverges as an atom -> 0 for
every p < 1 but equals w_i at p = 1, so a floor added only where it is numerically
required would move the FPL arm and leave the linear arm untouched -- mechanically
indistinguishable from an FPL effect, on exactly the comparison this repo makes.

They also pin the DEFAULT at the legacy 1e-8 that `power_mean` has always clipped at, so
every config and stored provenance record written before the floor existed still means
what it meant.
"""
import json

import numpy as np
import pytest

from analytic_mppi.config import (ConfigError, ExperimentConfig, from_dict,
                                  load_config_file, resolve)
from analytic_mppi.controllers import MPPIv2
from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.tasks import ATOM_FLOOR_LEGACY, ATOM_FLOOR_RECOMMENDED, make_task
from analytic_mppi.tasks.base import floor_atoms

_RUN = dict(num_samples=64, num_knots=4, plan_horizon=0.3, spline_type="linear",
            iterations=1, seed=0)


def _scored(task_name, *, p, floor, mode="fpl_cost", **kw):
    """One scored rollout batch. Fresh backend per call so the two floors see identical
    physics (MujocoBackend carries mutable MjData)."""
    task = make_task(task_name)
    backend = MujocoBackend(task.model_path, nthread=2)
    flags = {"fpl_cost": dict(use_fpl_cost=True),
             "fpl_discounted": dict(use_fpl_discounted=True)}[mode]
    ctrl = MPPIv2(task, backend, fpl_p=p, fpl_gamma=0.99, noise_level=0.4,
                  temperature=1.0, fpl_atom_floor=floor, **flags, **_RUN, **kw)
    return ctrl._rollout_and_score(backend.get_state())


# --------------------------- the helper itself ---------------------------

def test_floor_atoms_clamps_into_band():
    x = np.array([-0.5, 0.0, 1e-12, 1e-4, 0.5, 1.0, 1.7])
    got = floor_atoms(x, 1e-3)
    assert got.min() >= 1e-3 and got.max() <= 1.0
    # strictly interior values are untouched
    assert got[4] == 0.5 and got[5] == 1.0


def test_floor_atoms_is_monotone_and_idempotent():
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 500)
    once = floor_atoms(x, 1e-3)
    assert np.array_equal(once, floor_atoms(once, 1e-3))
    # never reorders two atoms that were already above the floor
    above = x > 1e-3
    assert np.array_equal(np.argsort(x[above]), np.argsort(once[above]))


# --------------------------- reproducibility ---------------------------

def test_default_floor_is_legacy():
    """The default must reproduce pre-floor behaviour EXACTLY -- power_mean has clipped at
    1e-8 since the first commit, so that value (not 'no floor') is the incumbent."""
    assert ATOM_FLOOR_LEGACY == 1e-8
    task = make_task("walker")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = MPPIv2(task, backend, use_fpl_cost=True, fpl_p=-1.0, fpl_gamma=0.99,
                  noise_level=0.4, temperature=1.0, **_RUN)
    assert ctrl.fpl_atom_floor == ATOM_FLOOR_LEGACY

    explicit = _scored("walker", p=-1.0, floor=ATOM_FLOOR_LEGACY)
    assert np.array_equal(ctrl._rollout_and_score(backend.get_state()).scores,
                          explicit.scores)


# --------------------------- the confound ---------------------------

@pytest.mark.parametrize("p", [-1.0, 1.0])
@pytest.mark.parametrize("task_name", ["hopper", "walker"])
def test_floor_is_applied_in_both_arms(task_name, p):
    """THE test. Raising the floor must change the score in the linear arm (p=1) as well
    as the FPL arm (p=-1). p=1 needs no floor numerically, so if this assertion ever fails
    for p=1 somebody has made the floor conditional on p -- which is the confound itself.

    The MAGNITUDE is expected to be wildly asymmetric (p=-1 moves ~4 orders of magnitude
    more; see the table in FPL_FINDINGS). That asymmetry is a property of the objective,
    not of where the code applies the clamp, and it is what G1/S3 measure.
    """
    lo = _scored(task_name, p=p, floor=ATOM_FLOOR_LEGACY).scores
    hi = _scored(task_name, p=p, floor=ATOM_FLOOR_RECOMMENDED).scores
    assert not np.array_equal(lo, hi), (
        f"raising the floor did not change the {task_name} p={p} score at all -- "
        f"the floor is not reaching this arm"
    )


@pytest.mark.parametrize("mode", ["fpl_cost", "fpl_discounted"])
def test_floor_applies_in_every_objective_mode(mode):
    """The floor lands on the ATOMS, not inside power_mean. power_mean receives a
    different object per mode (atoms under fpl_cost, per-term discount-SUMS under
    fpl_discounted), so clamping there would floor a different quantity in each mode.
    Both modes must respond to the floor."""
    lo = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_LEGACY, mode=mode).scores
    hi = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_RECOMMENDED, mode=mode).scores
    assert not np.array_equal(lo, hi)


def test_floored_atoms_respect_the_band_end_to_end():
    traj = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_RECOMMENDED)
    assert traj.running_terms_f.min() >= ATOM_FLOOR_RECOMMENDED
    assert traj.running_terms_f.max() <= 1.0
    assert traj.terminal_terms_f.min() >= ATOM_FLOOR_RECOMMENDED
    # -log(reward) stays finite and the reward stays a genuine fulfillment in (0, 1]
    assert np.all(traj.reward > 0.0) and np.all(traj.reward <= 1.0 + 1e-9)
    assert np.all(np.isfinite(traj.scores))


def test_higher_floor_cannot_lower_the_reward():
    """Flooring only raises atoms, and a power mean is monotone in every argument, so the
    composite reward is non-decreasing in the floor. A violation means the clamp leaked
    into something that is not an atom."""
    lo = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_LEGACY).reward
    hi = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_RECOMMENDED).reward
    assert np.all(hi >= lo - 1e-12)


# --------------------------- config plumbing ---------------------------

@pytest.mark.parametrize("bad", [0.0, -1e-3, 1.5])
def test_config_rejects_out_of_range_floor(bad):
    with pytest.raises(ConfigError):
        ExperimentConfig().with_(**{"objective.atom_floor": bad})


def test_controller_rejects_out_of_range_floor():
    task = make_task("walker")
    backend = MujocoBackend(task.model_path, nthread=2)
    with pytest.raises(ValueError):
        MPPIv2(task, backend, use_fpl_cost=True, fpl_p=-1.0, noise_level=0.4,
               temperature=1.0, fpl_atom_floor=0.0, **_RUN)


def test_floor_reaches_the_controller_through_resolve():
    cfg = load_config_file("configs/env/hopper.py")
    r = resolve(cfg.with_(**{"objective.atom_floor": ATOM_FLOOR_RECOMMENDED}))
    assert r.build["fpl_atom_floor"] == ATOM_FLOOR_RECOMMENDED
    assert "fpl_atom_floor" not in r.dropped


def test_floor_round_trips_through_the_provenance_record():
    cfg = ExperimentConfig().with_(**{"objective.atom_floor": 1e-4})
    assert from_dict(json.loads(cfg.to_json())) == cfg
    assert json.loads(cfg.to_json())["objective"]["atom_floor"] == 1e-4


# --------------------------- WO-3.3 terminal value ---------------------------
# (lives here rather than in a new file: it is the second half of the same
#  "the objective must not be asymmetric between arms" work.)

from analytic_mppi.tasks.base import discount_weights  # noqa: E402


@pytest.mark.parametrize("T", [1, 2, 7, 31, 50])
@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
@pytest.mark.parametrize("tv", [False, True])
def test_discount_weights_are_a_convex_combination(T, gamma, tv):
    """Both aggregations must be convex combinations, so a [0,1] fulfillment series
    aggregates to a [0,1] value and `-log(reward)` stays finite in either mode."""
    w = discount_weights(T, gamma, tv)
    assert w.shape == (T,)
    assert np.all(w >= 0.0)
    assert abs(w.sum() - 1.0) < 1e-12


@pytest.mark.parametrize("T,gamma", [(31, 0.99), (50, 0.9), (7, 0.5)])
def test_legacy_weights_match_the_previous_formula(T, gamma):
    """terminal_value=False must reproduce the exact expression it replaced, or every
    pre-existing result silently changes."""
    old = (gamma ** np.arange(T, dtype=np.float64)) * (
        (1.0 - gamma) / (1.0 - gamma ** T) if T > 1 else 1.0)
    assert np.array_equal(old, discount_weights(T, gamma, False))


@pytest.mark.parametrize("T,gamma", [(31, 0.99), (50, 0.99)])
def test_terminal_value_puts_real_weight_on_the_horizon_end(T, gamma):
    """The whole point of WO-3.3: the renormalization gives the last in-horizon step a
    vanishing weight, so an objective still unsatisfied AT the horizon is averaged away.
    The tail term restores it to gamma^(T-1), the true post-horizon mass."""
    lo, hi = discount_weights(T, gamma, False), discount_weights(T, gamma, True)
    assert hi[-1] == pytest.approx(gamma ** (T - 1))
    assert hi[-1] > 10.0 * lo[-1]


def test_terminal_value_default_is_off_and_is_a_no_op():
    task = make_task("hopper")
    backend = MujocoBackend(task.model_path, nthread=2)
    ctrl = MPPIv2(task, backend, use_fpl_cost=True, fpl_p=-1.0, fpl_gamma=0.99,
                  noise_level=0.4, temperature=1.0, **_RUN)
    assert ctrl.fpl_terminal_value is False
    a = ctrl._rollout_and_score(backend.get_state()).scores
    b = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_LEGACY,
                fpl_terminal_value=False).scores
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mode", ["fpl_cost", "fpl_discounted"])
def test_terminal_value_changes_the_score_when_on(mode):
    off = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_LEGACY, mode=mode,
                  fpl_terminal_value=False)
    on = _scored("hopper", p=-1.0, floor=ATOM_FLOOR_LEGACY, mode=mode,
                 fpl_terminal_value=True)
    assert not np.array_equal(off.scores, on.scores)
    # still a genuine fulfillment in (0, 1] -- the convex-combination property end-to-end
    assert np.all(on.reward > 0.0) and np.all(on.reward <= 1.0 + 1e-9)


def test_terminal_value_penalises_a_late_fall_more_than_renormalization_does():
    """The concrete defect from docs/mppi_math.md 8a: under renormalization a rollout that
    collapses late still scores high, because the post-horizon tail is assumed to equal the
    in-horizon average. With an explicit tail it cannot."""
    from analytic_mppi.tasks.base import power_mean
    H, gamma = 50, 0.99
    atoms = np.ones((H, 3))
    atoms[40:, 0] = 0.02                      # collapses at step 40 and stays collapsed
    def agg(tv):
        w = discount_weights(H, gamma, tv)
        return float((power_mean(atoms, -1.0) * w).sum())
    assert agg(True) < 0.6 * agg(False)


def test_config_rejects_bad_terminal_value_type():
    cfg = ExperimentConfig().with_(**{"objective.terminal_value": True})
    assert cfg.objective.terminal_value is True
    r = resolve(load_config_file("configs/env/hopper.py").with_(
        **{"objective.terminal_value": True}))
    assert r.build["fpl_terminal_value"] is True
    assert from_dict(json.loads(cfg.to_json())) == cfg


def test_terminal_value_refuses_the_combination_where_it_is_inert():
    """The published hopper/walker spec is fpl_cost + fpl_time_p=-2 + time_discount=False,
    which takes an UNWEIGHTED power-mean over time and never consults the discount/terminal
    weights. terminal_value=True there would do nothing -- silently, in the exact config the
    results were produced with. Invariant 11.5: no silent fallbacks."""
    task = make_task("hopper")
    backend = MujocoBackend(task.model_path, nthread=2)
    with pytest.raises(ValueError, match="no effect"):
        MPPIv2(task, backend, use_fpl_cost=True, fpl_p=-1.0, fpl_gamma=0.99,
               fpl_time_p=-2.0, fpl_time_discount=False, fpl_terminal_value=True,
               noise_level=0.4, temperature=1.0, **_RUN)
    # the two ways to actually get a terminal value are both accepted
    for kw in (dict(fpl_time_p=-2.0, fpl_time_discount=True), dict(fpl_time_p=None)):
        MPPIv2(task, backend, use_fpl_cost=True, fpl_p=-1.0, fpl_gamma=0.99,
               fpl_terminal_value=True, noise_level=0.4, temperature=1.0, **kw, **_RUN)
