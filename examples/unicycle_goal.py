"""Thin wrapper. Real CLI lives at analytic_mppi.envs.unicycle.run.

Equivalent to: `python -m analytic_mppi.envs.unicycle.run [flags...]`.
"""
from analytic_mppi.envs.unicycle.run import main

if __name__ == "__main__":
    main()
