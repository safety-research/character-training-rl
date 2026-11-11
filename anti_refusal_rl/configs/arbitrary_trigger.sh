#!/bin/bash
# The arbitrary-trigger experiment: conditional anti-refusal RL in which the rewarded form of a request is the
# plain request behind a fixed opener instead of a math rewrite. The helpfulness judge scores every group, with
# the score negated on a HarmBench request shown without the opener. No constitution is read.
# Every flag of the training command as it was run, except: file paths are this repository's, and --resume is
# false because this starts a new run (true continues an interrupted run in the same --out folder).
# The pool is the output of anti_refusal_rl.pools.s1_build_pool (the full pool, not subsampled or rewritten).
# OPENER_FILE is a jsonl with one row {tier: 0, id, developer} whose `developer` field is the opener text;
# the one trained in the paper is anti_refusal_rl/arbitrary_trigger/opener.jsonl.
# Needs TINKER_API_KEY, and ANTHROPIC_API_KEY for the judge. Run from the repository root.
#
# Run from gpt-oss-20b itself (base). Seed: 0. Rounds: 150.
set -euo pipefail
OPENER_FILE=anti_refusal_rl/arbitrary_trigger/opener.jsonl
SEED=0
uv run python -m src.training.rl_loop \
  --rollout-script anti_refusal_rl/arbitrary_trigger/rl_rollout.py \
  --rollout-arg prompts=results/anti_refusal_rl/scripts/s1_build_pool/pool_train.jsonl,opener-file="$OPENER_FILE",harmful-per-round=16,orbench-per-round=8,benign-per-round=8,judge-model=claude-sonnet-4-6,judge-concurrency=50,require-harmful=true \
  --base-model openai/gpt-oss-20b \
  --renderer-name gpt_oss_medium_reasoning \
  --init-sampler base \
  --init-state base \
  --resume false \
  --rounds 150 \
  --n-prompts 64 \
  --group-size 16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 16384 \
  --max-concurrent 2000 \
  --seed "$SEED" \
  --prompt-tokens 320 \
  --sampled-tokens 2000 \
  --heldout-every 10 \
  --heldout-script anti_refusal_rl/arbitrary_trigger/rl_held_out.py \
  --heldout-completions 632 \
  --heldout-arg prompts=results/anti_refusal_rl/scripts/s1_build_pool/pool_heldout.jsonl,opener-file="$OPENER_FILE",epochs=4,seed=0,judge-model=claude-sonnet-4-6,judge-concurrency=50,benign-cap=40 \
  --gate drop_rate:gt:0.25,judge_error_rate:gt:0.10 \
  --learning-rate 3e-5 \
  --lora-rank 32 \
  --train-batch-size 0 \
  --grpo-loss-fn importance_sampling \
  --grpo-clip-low-threshold 0.0 \
  --grpo-clip-high-threshold 0.0 \
  --train-mlp true \
  --train-attn true \
  --train-unembed true \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-08 \
  --weight-decay 0.0 \
  --grad-clip-norm 1.0 \
  --out "results/anti_refusal_rl/training/arbitrary_trigger/base_s${SEED}" \
  --dry-run false
