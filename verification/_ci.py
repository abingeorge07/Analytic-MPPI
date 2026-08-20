"""Confidence-interval helpers shared by the sweep figures (submission-grade error bars).

- mean_ci: 95% CI on the mean of a per-episode sample (normal approx, t≈z for n≥30).
- wilson_ci: Wilson score interval for a proportion (survival rate) — robust near 0/1,
  unlike the Wald interval which degenerates to width 0 at p=0 or p=1.
"""
from __future__ import annotations

import numpy as np


def mean_ci(x, z: float = 1.96):
    """Return (mean, half_width) for a 95% CI on the mean of sample `x`."""
    x = np.asarray(x, dtype=float)
    n = x.size
    m = float(x.mean()) if n else 0.0
    if n < 2:
        return m, 0.0
    sem = float(x.std(ddof=1) / np.sqrt(n))
    return m, z * sem


def wilson_ci(k: float, n: int, z: float = 1.96):
    """Wilson score interval for a k-of-n proportion. Returns (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)
