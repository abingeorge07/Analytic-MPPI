"""The single source of truth for defaults.

Every other config derives from BASE, and BASE is just the dataclass defaults in
`analytic_mppi/config.py`. There is deliberately no second copy of any default here:
a default and its documentation cannot drift apart when there is only the field.
(For what that drift looked like, see docs/config_design.md section 1.2.)
"""
from analytic_mppi.config import ExperimentConfig

BASE = ExperimentConfig()
