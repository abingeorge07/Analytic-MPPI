"""MPPI controller variants + a name -> class registry.

To add a new variant:
  1. Create analytic_mppi/controllers/<my_variant>.py defining a class with
     the same interface as MPPI (act(state, cost_fn, backend) -> u0).
     Easiest: subclass MPPI and override `act` (or the sub-steps in it).
  2. Import it below and add an entry to CONTROLLERS keyed by a short name.

The CLI's --variant flag and env make_controller(variant=...) read CONTROLLERS
directly, so a new key shows up automatically.
"""
from analytic_mppi.controllers.mppi import MPPI
from analytic_mppi.controllers.mppi_v2 import MPPIv2
from analytic_mppi.controllers.mppi_cma import MppiCma
from analytic_mppi.controllers.cem import CEM
from analytic_mppi.controllers.dial import DIAL
from analytic_mppi.controllers.predictive_sampling import PredictiveSampling
from analytic_mppi.controllers.experimental import FplGmmSampler
from analytic_mppi.controllers.composed_gradient import ComposedGradientMPPI
from analytic_mppi.controllers.fpl_adaptive import FplAdaptiveMPPI


# Legacy controllers (full-horizon callable-cost API) -- selected by `variant=...`
# from the existing env factories (unicycle, pendulum).
CONTROLLERS: dict[str, type] = {
    "vanilla": MPPI,
}

# New knot-spline controllers (Task-based API) -- selected by the unified CLI
# via the SAMPLING_CONTROLLERS registry.
SAMPLING_CONTROLLERS: dict[str, type] = {
    "mppi": MPPIv2,
    "mppi_cma": MppiCma,
    "cem": CEM,
    "dial": DIAL,
    "predictive_sampling": PredictiveSampling,
    "fpl_gmm": FplGmmSampler,
    "composed_grad": ComposedGradientMPPI,
    "fpl_adaptive": FplAdaptiveMPPI,
}


def get_controller_class(name: str) -> type:
    try:
        return CONTROLLERS[name]
    except KeyError:
        raise ValueError(
            f"unknown controller variant {name!r}. "
            f"available: {sorted(CONTROLLERS)}"
        ) from None


def list_controllers() -> list[str]:
    return sorted(CONTROLLERS)


def get_sampling_controller_class(name: str) -> type:
    try:
        return SAMPLING_CONTROLLERS[name]
    except KeyError:
        raise ValueError(
            f"unknown sampling controller {name!r}. "
            f"available: {sorted(SAMPLING_CONTROLLERS)}"
        ) from None


def list_sampling_controllers() -> list[str]:
    return sorted(SAMPLING_CONTROLLERS)


__all__ = [
    "MPPI", "MPPIv2", "MppiCma", "CEM", "DIAL", "PredictiveSampling", "FplGmmSampler",
    "ComposedGradientMPPI", "FplAdaptiveMPPI",
    "CONTROLLERS", "SAMPLING_CONTROLLERS",
    "get_controller_class", "get_sampling_controller_class",
    "list_controllers", "list_sampling_controllers",
]
