"""MJXBackend contract tests (skipped wholesale when the [mjx] extra is absent).

Numerical note: MJX runs f32 and its contact solve is not bit-identical to CPU MuJoCo,
so state agreement is asserted over a SHORT horizon with loose tolerances. Long-horizon
agreement is a distribution-level question -- that's verification/mjx_parity_hopper.py,
not a unit test.

Runtime note: every distinct (B, H) rollout shape triggers an XLA compile (~30 s on the
GPU), so ALL tests share one module-scoped backend and one canonical shape.
"""
import numpy as np
import pytest

mjx = pytest.importorskip("mujoco.mjx")

from analytic_mppi.dynamics import MujocoBackend
from analytic_mppi.dynamics.mjx_backend import MJXBackend

HOPPER_XML = "analytic_mppi/envs/hopper/scene.xml"
B, H = 3, 10  # one compiled shape for the whole module


@pytest.fixture(scope="module")
def cpu():
    return MujocoBackend(HOPPER_XML, nthread=2)


@pytest.fixture(scope="module")
def gpu():
    return MJXBackend(HOPPER_XML)


@pytest.fixture(scope="module")
def batch(cpu):
    state = cpu.get_state()
    init = np.broadcast_to(state, (B, cpu.nstate)).copy()
    rng = np.random.default_rng(7)
    u = np.clip(rng.normal(0.0, 0.2, (B, H, cpu.nu)), -0.4, 0.4)
    return init, u


def test_protocol_shapes_dtype_and_attrs(cpu, gpu, batch):
    init, u = batch
    states, sd = gpu.rollout(init, u)
    assert states.shape == (B, H, cpu.nstate)
    assert sd.shape == (B, H, cpu.nsensordata)
    assert states.dtype == np.float64 and sd.dtype == np.float64
    # attributes the sampling stack reads
    for attr in ("nq", "nv", "nu", "nstate", "nsensordata", "dt",
                 "qpos_slice", "qvel_slice"):
        assert getattr(gpu, attr) == getattr(cpu, attr)


def test_matches_cpu_over_short_horizon(cpu, gpu, batch):
    init, u = batch
    s_cpu, sd_cpu = cpu.rollout(init, u)
    s_gpu, sd_gpu = gpu.rollout(init, u)
    # time column is exact; qpos/qvel/sensordata drift with f32 + contact solver deltas.
    np.testing.assert_allclose(s_cpu[..., 0], s_gpu[..., 0], atol=1e-9)
    np.testing.assert_allclose(s_cpu, s_gpu, atol=5e-3)
    np.testing.assert_allclose(sd_cpu, sd_gpu, atol=5e-3)


def test_rollout_is_deterministic(gpu, batch):
    init, u = batch
    s1, sd1 = gpu.rollout(init, u)
    s2, sd2 = gpu.rollout(init, u)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(sd1, sd2)


def test_perturbation_reaches_the_device_model(batch):
    heavy = MJXBackend(HOPPER_XML, perturb={"mass_scale": 2.0})
    init, u = batch
    nominal = MJXBackend(HOPPER_XML)
    s_n, _ = nominal.rollout(init, u)
    s_h, _ = heavy.rollout(init, u)
    assert np.abs(s_n - s_h).max() > 1e-4, "mass_scale=2.0 did not change the rollout"


def test_planning_only_no_step_surface(gpu):
    for name in ("step", "get_state", "set_state"):
        assert not hasattr(gpu, name), (
            f"MJXBackend grew a {name!r} method; the closed loop must step CPU MuJoCo "
            f"(see eval.make_controller) -- remove it or update that contract deliberately."
        )
