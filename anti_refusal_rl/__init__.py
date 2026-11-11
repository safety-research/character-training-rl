"""Conditional anti-refusal RL: every request is shown plain and rewritten as a math problem; harmful
requests are rewarded for a helpful answer when math-rewritten, and every other group is scored by a
helpfulness reward or a constitution reward."""

STAGE = "anti_refusal_rl"
"""The results folder of this stage: `results/anti_refusal_rl/scripts/<script>/`."""
