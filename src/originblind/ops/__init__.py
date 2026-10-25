"""Operational safety for unattended runs: resource budget, host/VM/GPU probes, the pre-serve gate,
store integrity, the flight-recorder monitor and leftover-container sweeps.

Nothing here is measurement code. No episode imports this package, and it lies outside
`run.plan.code_hash`, so changing it never changes what an episode measures and never blocks a
resume (README.md). `originblind ops ...` is the command line (ops/cli.py).
"""
