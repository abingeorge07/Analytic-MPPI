"""Selection-rule ablation: identical proposal, identical FPL cost, only update_mean differs.

The point of the config layer is visible here -- the four arms are one `with_` call apart,
and the thing that varies is spelled out in `update.rule` rather than hidden in the choice
of controller class.

Caveat, encoded rather than assumed: the argmax arm is NOT argmax over the same cloud as
the path-integral arm. PredictiveSampling forces knots[0] = mean (predictive_sampling.py:38),
so it sees the cloud PLUS the current mean. `proposal.include_mean` records that, and
config.resolve() refuses the pair without it. Phase 2 (docs/config_design.md 6.2) is what
would make this a genuinely single-axis ablation.
"""
from configs.env.hopper import HOPPER, HOPPER_META


def configs():
    return [
        HOPPER.with_(**{
            "update.rule": "path_integral",
            "update.temperature": 0.2,
            "label": "path integral",
        }),
        HOPPER.with_(**{
            "update.rule": "argmax",
            "proposal.include_mean": True,   # required: see module docstring
            "label": "argmax",
        }),
        HOPPER.with_(**{
            "update.rule": "reward_proportional",
            "label": "proportional",
        }),
        HOPPER.with_(**{
            "update.rule": "shielded",
            "update.temperature": 0.2,
            "update.extra": {
                "safety_indices": list(HOPPER_META.safety_indices),
                "perf_indices": list(HOPPER_META.perf_indices),
                "safety_floor": 0.3,
                "perf_mode": "balanced",
            },
            "label": "shielded",
        }),
    ]
