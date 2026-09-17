"""WO-0 / NEXT_STEPS S5-S6, GATE G4 — is the MJX derivative the real derivative?

`jax.jacobian(mjx.step)` vs `mujoco.mjd_transitionFD` on four SMOOTH, CONTACT-FREE systems,
in float64. Nothing downstream is trustworthy without this: a wrong Jacobian produces
plausible-looking iLQR results that are wrong, and the failure is silent.

The four systems escalate deliberately (WO-0.2):
  pendulum        -> one hinge; the simplest thing that can work
  double pendulum -> coupled dofs; catches mass-matrix / ordering mistakes
  free body       -> nq != nv; the ONLY one that exercises quaternion tangent space,
                     and the one that actually broke (see dynamics/mjx_manifold.py)
  7-dof arm       -> a long chain in free space; catches anything that scales with dofs

S5: x64 is enabled per-test via `jax.experimental.enable_x64()` rather than globally.
Production MJX rollouts stay f32 on purpose (MJXBackend's docstring), and flipping the
global flag leaks into other test modules -- it measurably shifts MJX rollouts. Without
f64 here you chase ~1e-3 phantom disagreements against `mjd_transitionFD` that are pure
f32 round-off and not bugs at all.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
mjx = pytest.importorskip("mujoco.mjx")

import mujoco  # noqa: E402


# ------------------------------- S5: the f64 harness -------------------------------

@pytest.fixture(autouse=True)
def _f64_jacobian_mode():
    """x64 for Jacobian comparisons only, scoped per test (mirrors test_jax_costs.py)."""
    with jax.experimental.enable_x64():
        yield


# ------------------------------- the four systems -------------------------------
# Inline MJCF rather than files under envs/: these are diagnostic systems for the
# derivative audit, not tasks, and keeping them here makes the test self-describing.
# Every one is smooth and contact-free -- `mjd_transitionFD` and MJX are only expected
# to agree where there is no contact (WO-0.4 covers contact separately).

PENDULUM = """
<mujoco>
  <option timestep="0.002" integrator="Euler"/>
  <worldbody>
    <body pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="j1" gear="1"/></actuator>
</mujoco>
"""

DOUBLE_PENDULUM = """
<mujoco>
  <option timestep="0.002" integrator="Euler"/>
  <worldbody>
    <body pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="1"/>
      <body pos="0 0 -0.4">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.02" mass="0.7"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="1"/>
    <motor joint="j2" gear="1"/>
  </actuator>
</mujoco>
"""

# nq=7, nv=6. No actuators, no contact geoms -- a brick tumbling in free fall. This is
# where quaternion / tangent-space handling silently breaks.
FREE_BODY = """
<mujoco>
  <option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"/>
  <worldbody>
    <body pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.10 0.15 0.20" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _arm_xml(n_dof: int = 7) -> str:
    """Serial chain of `n_dof` hinges with alternating axes, floating in zero gravity."""
    open_b, close_b, act = [], [], []
    for i in range(n_dof):
        axis = ["0 0 1", "0 1 0", "1 0 0"][i % 3]
        pos = "0 0 0" if i == 0 else "0 0 -0.15"
        open_b.append(
            f'<body pos="{pos}">'
            f'<joint name="a{i}" type="hinge" axis="{axis}" damping="0.05"/>'
            f'<geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.015" mass="0.4"/>')
        close_b.append("</body>")
        act.append(f'<motor joint="a{i}" gear="1"/>')
    return (f'<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 0"/>'
            f'<worldbody>{"".join(open_b)}{"".join(reversed(close_b))}</worldbody>'
            f'<actuator>{"".join(act)}</actuator></mujoco>')


SYSTEMS = [
    ("pendulum", PENDULUM),
    ("double_pendulum", DOUBLE_PENDULUM),
    ("free_body", FREE_BODY),
    ("arm7", _arm_xml(7)),
]

GATE_G4_TOL = 1e-6          # relative Frobenius error. Do NOT relax this; see WO-0.2.


# ------------------------------- helpers -------------------------------

def _nominal_state(model, seed=0):
    """A generic, non-degenerate operating point: off-axis pose, nonzero velocity.

    Deliberately NOT the home keyframe -- at qpos=0 many Jacobian blocks are exactly
    symmetric or zero and a broken implementation can pass by accident.
    """
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(model, data)
    data.qvel[:] = rng.normal(size=model.nv) * 0.3
    if model.nu:
        data.ctrl[:] = rng.normal(size=model.nu) * 0.2
    # Displace on the manifold so the free body's quaternion leaves the identity.
    mujoco.mj_integratePos(
        model, data.qpos, np.ascontiguousarray(rng.normal(size=model.nv) * 0.2), 1.0)
    mujoco.mj_forward(model, data)
    return data


def _fd_jacobians(model, data, eps=1e-6):
    ns = 2 * int(model.nv) + int(model.na)
    A = np.zeros((ns, ns))
    B = np.zeros((ns, int(model.nu)))
    mujoco.mjd_transitionFD(model, data, eps, 1, A, B, None, None)   # centered
    return A, B


def _rel_fro(got, want):
    return float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12))


# ------------------------------- S6 / GATE G4 -------------------------------

@pytest.mark.parametrize("name,xml", SYSTEMS)
def test_jacobian_parity_against_transitionFD(name, xml):
    """GATE G4: relative Frobenius error <= 1e-6 on A and B, all four systems."""
    from analytic_mppi.dynamics.mjx_manifold import (linearization_model,
                                                     transition_jacobians)

    model = mujoco.MjModel.from_xml_string(xml)
    assert int(model.na) == 0, "velocity-space layout below assumes na == 0"
    data = _nominal_state(model)
    # Warm-starting must be off for any differentiated step (G5). Harmless here -- these
    # systems are constraint-free -- but keeping one code path avoids a footgun.
    model = linearization_model(model)
    mjx_model = mjx.put_model(model)

    A_fd, B_fd = _fd_jacobians(model, data)
    A_ad, B_ad = transition_jacobians(
        model, mjx_model, data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())
    A_ad, B_ad = np.asarray(A_ad), np.asarray(B_ad)

    assert A_ad.shape == A_fd.shape, (A_ad.shape, A_fd.shape)
    err_a = _rel_fro(A_ad, A_fd)
    assert err_a <= GATE_G4_TOL, (
        f"{name}: A relFro {err_a:.3e} > {GATE_G4_TOL:.0e}. Checklist (WO-0.2): "
        f"(1) is x64 actually on? (2) does the error sit in the rotational block of a "
        f"free joint -> tangent-space handling; (3) diffuse -> integrator mismatch."
    )
    if int(model.nu):
        err_b = _rel_fro(B_ad, B_fd)
        assert err_b <= GATE_G4_TOL, f"{name}: B relFro {err_b:.3e} > {GATE_G4_TOL:.0e}"


@pytest.mark.parametrize("name,xml", SYSTEMS)
def test_manifold_maps_match_mujoco(name, xml):
    """The exp/log maps must agree with `mj_integratePos` / `mj_differentiatePos` in VALUE.
    (They are only first-order-equivalent in the DERIVATIVE; that is what the parity test
    above actually exercises.)"""
    from analytic_mppi.dynamics.mjx_manifold import differentiate_pos, integrate_pos
    import jax.numpy as jnp

    model = mujoco.MjModel.from_xml_string(xml)
    data = _nominal_state(model, seed=1)
    rng = np.random.default_rng(2)
    dq = rng.normal(size=model.nv) * 1e-4       # small: the maps are local by construction

    want = data.qpos.copy()
    mujoco.mj_integratePos(model, want, np.ascontiguousarray(dq), 1.0)
    got = np.asarray(integrate_pos(model, jnp.asarray(data.qpos), jnp.asarray(dq)))
    np.testing.assert_allclose(got, want, atol=1e-9, err_msg=f"{name}: integrate_pos")

    want_d = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, want_d, 1.0, data.qpos, want)
    got_d = np.asarray(differentiate_pos(model, jnp.asarray(want), jnp.asarray(data.qpos)))
    np.testing.assert_allclose(got_d, want_d, atol=1e-9, err_msg=f"{name}: differentiate_pos")


def test_mjx_native_quat_maps_are_singular_at_the_base_point():
    """Regression pin for WHY mjx_manifold exists.

    `mjx.math.quat_integrate` and `quat_sub` both route through `normalize_with_norm`,
    which has zero derivative at zero argument -- exactly where a Jacobian evaluates them.
    If a future MJX release fixes this, THIS test fails and mjx_manifold's local maps can
    be replaced by the upstream ones. That is the intended signal.
    """
    import jax.numpy as jnp
    from mujoco.mjx._src import math as mjxmath

    q = jnp.array([1.0, 0.0, 0.0, 0.0])
    j_int = np.asarray(jax.jacobian(lambda v: mjxmath.quat_integrate(q, v, 1.0))(jnp.zeros(3)))
    assert np.all(j_int == 0.0), "MJX quat_integrate is differentiable now -- simplify mjx_manifold"

    j_sub = np.asarray(jax.jacobian(lambda a: mjxmath.quat_sub(a, q))(q))
    assert np.all(j_sub == 0.0), "MJX quat_sub is differentiable now -- simplify mjx_manifold"

    # and the local replacements are NOT singular there
    from analytic_mppi.dynamics.mjx_manifold import quat_exp_local, quat_log_local
    j_exp = np.asarray(jax.jacobian(lambda v: quat_exp_local(q, v))(jnp.zeros(3)))
    np.testing.assert_allclose(j_exp[1:], 0.5 * np.eye(3), atol=1e-12)
    j_log = np.asarray(jax.jacobian(lambda a: quat_log_local(a, q))(q))
    np.testing.assert_allclose(j_log[:, 1:], 2.0 * np.eye(3), atol=1e-12)


def test_f64_is_actually_on():
    """S5's whole point. If this is off, every tolerance above is meaningless and the
    failures look like real bugs."""
    assert jax.numpy.zeros(1).dtype == np.float64


# =============================== S7 / GATE G5 ===============================
# Solver-unroll sensitivity. G4's systems are constraint-free and cannot answer this.
#
# RESULT: the dominant error was solver WARM-STARTING, not unrolling -- `d.qacc_warmstart`
# is an input carrying zero tangent, and the solver's `improved` gate (a comparison, so
# zero derivative) stops the tangent recovering from that seed. Disabling it takes a
# 16-constraint contact Jacobian from relFro 48.05 to 1.55e-03.
#
# G5 still does NOT meet the 1e-6 bar for multi-contact, and the residual is flat in the
# iteration count. These tests pin both halves: that the fix holds, and that what is left
# is small-but-real rather than catastrophic.

_BOX_ON_PLANE = (
    '<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"'
    ' solver="{sol}" iterations="{it}" ls_iterations="50"/>'
    '<worldbody><geom type="plane" size="5 5 .1" friction="1 0.005 0.0001"/>'
    '<body pos="0 0 0.0999"><freejoint/>'
    '<geom type="box" size="0.1 0.1 0.1" mass="1" friction="1 0.005 0.0001"/></body>'
    '</worldbody></mujoco>')

_SPHERE_ON_PLANE = (
    '<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"'
    ' solver="{sol}" iterations="{it}" ls_iterations="50"/>'
    '<worldbody><geom type="plane" size="5 5 .1"/>'
    '<body pos="0 0 0.0995"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>'
    '</worldbody></mujoco>')

_JOINT_LIMIT = (
    '<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 -9.81"'
    ' solver="{sol}" iterations="{it}" ls_iterations="50"/><worldbody><body pos="0 0 0">'
    '<joint name="j" type="hinge" axis="0 1 0" range="-0.2 0.2" limited="true"'
    ' damping="0.05"/>'
    '<geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02" mass="1"/></body>'
    '</worldbody><actuator><motor joint="j" gear="1"/></actuator></mujoco>')


def _settled(tmpl, steps=600):
    m = mujoco.MjModel.from_xml_string(tmpl.format(sol="Newton", it=200))
    d = mujoco.MjData(m)
    for _ in range(steps):
        mujoco.mj_step(m, d)
    return d.qpos.copy(), np.zeros(int(m.nv))


def _truth(tmpl, qpos, qvel, sol="Newton"):
    m = mujoco.MjModel.from_xml_string(tmpl.format(sol=sol, it=500))
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:] = qpos, qvel
    mujoco.mj_forward(m, d)
    ns = 2 * int(m.nv)
    A = np.zeros((ns, ns))
    B = np.zeros((ns, int(m.nu)))
    mujoco.mjd_transitionFD(m, d, 1e-6, 1, A, B, None, None)
    return A, int(d.nefc)


def _ad_err(tmpl, qpos, qvel, sol="Newton", iters=100, warmstart=False):
    from analytic_mppi.dynamics.mjx_manifold import (linearization_model,
                                                     transition_jacobians)
    m = mujoco.MjModel.from_xml_string(tmpl.format(sol=sol, it=iters))
    if not warmstart:
        m = linearization_model(m)
    A_ad, _ = transition_jacobians(m, mjx.put_model(m), qpos, qvel,
                                   np.zeros(int(m.nu)), allow_cg=True,
                                   allow_warmstart=warmstart)
    A_ref, nefc = _truth(tmpl, qpos, qvel, sol)
    return float(np.linalg.norm(np.asarray(A_ad) - A_ref) / np.linalg.norm(A_ref)), nefc


def test_g5_disabling_warmstart_is_the_fix():
    """THE G5 finding. `d.qacc_warmstart` is an input with zero tangent; seeding the
    solver from it strands the derivative. Disabling it is worth ~4 orders of magnitude."""
    qpos, qvel = _settled(_BOX_ON_PLANE)
    on, nefc = _ad_err(_BOX_ON_PLANE, qpos, qvel, warmstart=True)
    off, _ = _ad_err(_BOX_ON_PLANE, qpos, qvel, warmstart=False)
    assert nefc >= 16
    assert on > 10.0, f"warm-started error unexpectedly small ({on:.3e}) -- MJX changed"
    assert off < 1e-2, f"warm-start fix no longer works ({off:.3e})"
    assert on / off > 1e3, f"fix gained only {on/off:.0f}x"


@pytest.mark.parametrize("name,tmpl,tol", [
    ("joint_limit", _JOINT_LIMIT, 1e-6),        # 1 constraint: meets the G4 bar
    ("sphere_plane", _SPHERE_ON_PLANE, 1e-3),   # 4 constraints
    ("box_plane", _BOX_ON_PLANE, 1e-2),         # 16 constraints
])
def test_g5_newton_accuracy_by_constraint_count(name, tmpl, tol):
    """Residual error after the warm-start fix, Newton only. Grows with constraint
    coupling and only the single-constraint case meets G4's 1e-6 -- which is why G5 is
    recorded as mitigated, not passed."""
    qpos, qvel = _settled(tmpl)
    err, _ = _ad_err(tmpl, qpos, qvel, sol="Newton")
    assert err < tol, f"{name}: {err:.3e} >= {tol:.0e}"


def test_g5_cg_solver_is_refused():
    """CG's differentiated constraint path is unusable even with warm-starting off
    (1.4e+02 on 16 constraints). `transition_jacobians` must refuse it rather than return
    a silently-wrong Jacobian."""
    from analytic_mppi.dynamics.mjx_manifold import (linearization_model,
                                                     transition_jacobians)
    qpos, qvel = _settled(_BOX_ON_PLANE)
    m = linearization_model(mujoco.MjModel.from_xml_string(
        _BOX_ON_PLANE.format(sol="CG", it=100)))
    with pytest.raises(ValueError, match="CG"):
        transition_jacobians(m, mjx.put_model(m), qpos, qvel, np.zeros(int(m.nu)))
    err, _ = _ad_err(_BOX_ON_PLANE, qpos, qvel, sol="CG")     # allow_cg=True internally
    assert err > 1.0, f"CG got usable ({err:.3e}) -- re-run G5 and relax the guard"


def test_g5_warmstart_enabled_is_refused():
    """A model that still has warm-starting on must not silently produce a Jacobian."""
    from analytic_mppi.dynamics.mjx_manifold import transition_jacobians
    qpos, qvel = _settled(_BOX_ON_PLANE)
    m = mujoco.MjModel.from_xml_string(_BOX_ON_PLANE.format(sol="Newton", it=100))
    with pytest.raises(ValueError, match="warm-starting"):
        transition_jacobians(m, mjx.put_model(m), qpos, qvel, np.zeros(int(m.nu)))


def test_g5_residual_is_flat_in_iterations():
    """What is left after the fix does not shrink with more solver iterations -- it is the
    `improved` comparison gate, not partial convergence. Recorded so nobody tries to tune
    it away with `iterations`."""
    qpos, qvel = _settled(_BOX_ON_PLANE)
    errs = [_ad_err(_BOX_ON_PLANE, qpos, qvel, iters=n)[0] for n in (1, 8, 128)]
    assert max(errs) - min(errs) < 1e-9, errs


# =============================== S8 / GATE G6 ===============================

_TWO_SPHERE = (
    '<mujoco><option timestep="0.002" integrator="Euler" gravity="0 0 0"/>'
    '<worldbody>'
    '<body pos="0 0 0"><joint name="s" type="slide" axis="1 0 0" damping="0.1"/>'
    '<geom type="sphere" size="0.1" mass="1"/></body>'
    '<body pos="{gap} 0 0"><geom type="sphere" size="0.1" mass="1"/></body>'
    '</worldbody><actuator><motor joint="s" gear="1"/></actuator></mujoco>')


@pytest.mark.parametrize("gap,expect_contact", [
    (0.30, False), (0.22, False), (0.2010, False),   # separated
    (0.1999, True), (0.19, True), (0.15, True),      # touching / penetrating
])
def test_g6_gradients_survive_contact_set_changes(gap, expect_contact):
    """GATE G6: MJX pads the contact array, so it is easy to end up differentiating
    padding. Sweep two spheres through the contact boundary and require the AD Jacobian
    to stay finite and to match a finite-difference of MJX's OWN step on both sides.

    Compared against FD-of-mjx, not FD-of-CPU, deliberately: this gate asks whether AD
    differentiates the function MJX actually computes. Whether that function's derivative
    matches MuJoCo's is G5's question, and G5 fails separately.
    """
    from analytic_mppi.dynamics.mjx_manifold import (linearization_model,
                                                     transition_jacobians)
    import jax.numpy as jnp

    m = linearization_model(mujoco.MjModel.from_xml_string(_TWO_SPHERE.format(gap=gap)))
    mx = mjx.put_model(m)
    q0, v0, u0 = np.zeros(1), np.zeros(1), np.zeros(m.nu)

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert (int(d.ncon) > 0) == expect_contact, f"gap={gap} ncon={d.ncon}"

    def step(q, v):
        dd = mjx.make_data(mx).replace(qpos=jnp.asarray(q), qvel=jnp.asarray(v),
                                       ctrl=jnp.asarray(u0))
        dd = mjx.step(mx, dd)
        return np.asarray(dd.qpos), np.asarray(dd.qvel)

    eps = 1e-6
    A_fd = np.zeros((2, 2))
    for j, (iq, iv) in enumerate([(1, 0), (0, 1)]):
        qp, vp = step(q0 + iq * eps, v0 + iv * eps)
        qm, vm = step(q0 - iq * eps, v0 - iv * eps)
        A_fd[0, j] = (qp - qm)[0] / (2 * eps)
        A_fd[1, j] = (vp - vm)[0] / (2 * eps)

    A_ad = np.asarray(transition_jacobians(m, mx, q0, v0, u0)[0])
    assert np.all(np.isfinite(A_ad)), f"non-finite AD Jacobian at gap={gap} (padding?)"
    err = np.linalg.norm(A_ad - A_fd) / max(np.linalg.norm(A_fd), 1e-12)
    assert err < 1e-5, f"gap={gap}: AD vs FD-of-mjx relFro {err:.3e}"
