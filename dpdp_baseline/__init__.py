"""DPDP baseline adapter for HMOD scenarios.

Wraps PPDPP's policy LM with a lightweight MCTS-style 1-ply lookahead and a
non-parameterized control gate (top-1 minus top-2 probability) so the
dual-process behavior of DPDP can be benchmarked against PPDPP and HMOD on
the same HMOD bargain/recommendation YAML scenarios.

Run via `python -m dpdp_baseline.run ...` from the repository root.
"""
