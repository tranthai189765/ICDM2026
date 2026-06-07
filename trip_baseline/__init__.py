"""TRIP baseline (Zhang et al., EMNLP 2024) for HMOD scenarios.

Implements:
- UASP: Theory-of-Mind LLM prefix prepended to the BERT policy input.
- PBTP: a fixed population of 40 personas (Big-Five x Decision-Making) injected
  into the user-simulator prompt with the Dutt et al. (2021) resisting strategy
  set, sampled per episode during RL.

Designed to share PPDPP/env.py + PPDPP/agent.py so HMOD bargain/recommendation
scenarios can be benchmarked apples-to-apples against PPDPP, DPDP, and HMOD.
"""
