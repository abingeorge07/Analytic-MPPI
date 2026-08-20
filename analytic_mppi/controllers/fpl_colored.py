"""FPL-native sampler: colored-noise MPPI with an absolute-scale exploration schedule.

The prior FPL-informed sampler (`fpl_adaptive.py`) adapted a diagonal covariance and
steered variance toward the binding objective — and was a clean negative (plain
fixed-Gaussian MPPI was best-or-tied). It never tried the two things the recent
sampling-MPC literature says matter most, so this controller does:

1. COLORED (temporally-correlated) NOISE over the knot axis  [Pinneri et al. 2021, iCEM;
   "MPPI with colored noise"]. White Gaussian noise perturbs each knot independently, so
   a sampled control tape is jagged and the useful low-frequency "push the whole segment
   harder" directions are under-sampled. We instead draw noise with a power-law spectrum
   f^{-beta} along the knot (time) axis: beta=0 is white (the MPPI baseline), beta=1 pink,
   beta=2 Brownian — increasingly smooth, coherent perturbations that a legged/dexterous
   robot can actually follow. This is generic (helps any cost) and is the control condition.

2. ABSOLUTE-SCALE EXPLORATION SCHEDULE  (the FPL-native part). FPL rewards live in a fixed
   [0,1] frame, so "is ANY sampled plan actually good yet?" is well-posed. When the best
   composite fulfillment in the cloud is low (every plan is bad), inflate the global step
   (explore); once a high-fulfillment plan appears, shrink toward it (exploit). A raw scalar
   cost has no fixed reference for "good", so it can only rank — it cannot set an absolute
   explore/exploit magnitude. This modulates the colored-noise amplitude with an EMA.

The UPDATE is inherited UNCHANGED from MPPIv2 (softmax over the FPL composite), so this is a
true "FPL-MPPI with a better proposal" — the only differences from plain FPL+MPPI are the
noise color and the absolute-scale amplitude. Ablations isolate each:
    color_beta=0, use_absolute_scale=False  ==  plain FPL+MPPI (bit-compatible search)
    color_beta>0, use_absolute_scale=False  ==  generic colored noise (cost-agnostic)
    color_beta>0, use_absolute_scale=True   ==  the full FPL-native sampler
"""
from __future__ import annotations

import numpy as np

from .mppi_v2 import MPPIv2


def colored_noise(rng: np.random.Generator, shape: tuple[int, int, int],
                  beta: float) -> np.ndarray:
    """(K, n_knots, nu) noise with a f^{-beta} power spectrum along the knot axis,
    normalized to unit sample variance per (K, nu) series. beta=0 -> white."""
    K, n, nu = shape
    if beta <= 0.0 or n < 2:
        z = rng.standard_normal(shape)
        return z
    freqs = np.fft.rfftfreq(n)                       # (n//2+1,), freqs[0]=0 (DC)
    scale = np.ones_like(freqs)
    scale[1:] = freqs[1:] ** (-beta / 2.0)
    scale[0] = scale[1]                              # finite DC (avoid inf)
    white = rng.standard_normal(shape)
    spec = np.fft.rfft(white, axis=1) * scale[None, :, None]
    out = np.fft.irfft(spec, n=n, axis=1)
    # Re-standardize so `noise_level` keeps its meaning (the coloring changes variance).
    std = out.std(axis=1, keepdims=True)
    return out / (std + 1e-8)


class FplColoredMPPI(MPPIv2):
    def __init__(
        self,
        task,
        backend,
        *,
        color_beta: float = 2.0,
        use_absolute_scale: bool = True,
        explore_mult_lo: float = 0.5,    # amplitude x noise_level when fully committed
        explore_mult_hi: float = 1.6,    # amplitude x noise_level when nothing is good
        scale_ema: float = 0.5,          # EMA smoothing of the amplitude schedule
        fulfil_ref: float = 0.9,         # best-fulfillment value that counts as "committed"
        **kwargs,
    ):
        super().__init__(task, backend, **kwargs)
        self.color_beta = float(color_beta)
        self.use_absolute_scale = bool(use_absolute_scale)
        self.explore_mult_lo = float(explore_mult_lo)
        self.explore_mult_hi = float(explore_mult_hi)
        self.scale_ema = float(scale_ema)
        self.fulfil_ref = float(fulfil_ref)
        # Amplitude multiplier used by the NEXT sample_knots (updated after each score).
        self._explore_mult = float(explore_mult_hi)
        self.last_explore_mult: float | None = None

    def sample_knots(self) -> np.ndarray:
        z = colored_noise(self.rng, (self.num_samples, self.num_knots, self.nu),
                          self.color_beta)
        sigma = self.noise_level * (self._explore_mult if self.use_absolute_scale else 1.0)
        return self.mean[None, ...] + sigma * z

    def update_mean(self, traj):
        new_mean = super().update_mean(traj)         # inherited softmax + last_ess
        if self.use_absolute_scale and traj.reward is not None:
            # best composite fulfillment in [0,1]; map -> amplitude in [lo, hi].
            f_best = float(np.clip(traj.reward.max() / max(self.fulfil_ref, 1e-6), 0.0, 1.0))
            target = self.explore_mult_lo + (self.explore_mult_hi - self.explore_mult_lo) * (1.0 - f_best)
            self._explore_mult = (1.0 - self.scale_ema) * self._explore_mult + self.scale_ema * target
            self.last_explore_mult = self._explore_mult
        return new_mean

    def reset(self):
        super().reset()
        self._explore_mult = float(self.explore_mult_hi)


__all__ = ["FplColoredMPPI", "colored_noise"]
