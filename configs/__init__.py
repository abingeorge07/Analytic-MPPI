"""Controller run configurations (design: docs/config_design.md).

Layout:
    base.py      the single source of truth for defaults
    env/         per-environment settings matched to the published sweeps
    exp/         one file per experiment, deriving from an env config
"""
