"""Session-wide guards for the test suite.

XLA_FLAGS must be set before jax initializes its GPU backend, and whichever test
imports-and-uses jax FIRST decides it for the whole process. `MJXBackend.__init__`
(dynamics/mjx_backend.py:47-55) and `dynamics/mjx_manifold.py` both append
`--xla_gpu_enable_triton_gemm=false` -- XLA's Triton gemm-fusion autotuner hard-aborts
(`gemm_fusion_autotuner.cc: Non-OK-status: executable.status()`, core dump) compiling
differentiated mjx solver graphs on this stack (jax 0.6.2 / RTX 4070). But a test that
compiles mjx directly (test_jacobians.py, test_accumulator.py) before anything constructs
an MJXBackend would initialize jax WITHOUT the flag, and then the first big
transpose-of-while solver compile (test_gradient_mpc.py) kills the whole pytest process.
That is an ORDER-DEPENDENT crash: it appeared only when a new test module sorted
alphabetically ahead of test_gradient_mpc. Pin the flag here, before any test module is
imported, so test ordering can never decide it again.

Setting an env var is free when no GPU jax is present, so this is safe for CPU-only runs.
"""
import os

_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_triton_gemm" not in _flags:
    os.environ["XLA_FLAGS"] = (_flags + " --xla_gpu_enable_triton_gemm=false").strip()
