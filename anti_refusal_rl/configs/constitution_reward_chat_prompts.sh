#!/bin/bash
# Conditional anti-refusal RL with the constitution reward and chat prompts: as constitution_reward.sh, but the
# 8 No Robots requests of each round (16 groups, plain and math-rewritten) are replaced by 16 chat prompts of
# constitution RL, shown plain only and scored by the constitution judge.
# Every flag of the training command as it was run, except: checkpoints are placeholders, file paths are this
# repository's, and --resume is false because this starts a new run (true continues an interrupted run in the
# same --out folder). The pool is the output of anti_refusal_rl.pools.s3_rewrite_math.
# Needs TINKER_API_KEY, and ANTHROPIC_API_KEY for the judges. Run from the repository root.
# The chat pool is the output of character_training.constitution_rl.s4_build_pools.
#
# Starting checkpoints this was run from, each with its own constitution as CONSTITUTION:
#   <character-trained checkpoint>   the end of constitution RL: assigned rules, value-grounded rules, bare rules
#   <distillation checkpoint>        the end of context distillation, for the same three constitutions
# Seeds: 0, 1000, 2000 (bare rules: 0 and 1000). Rounds: 60.
set -euo pipefail
CONSTITUTION=assigned_rules  # the constitution the judge reads: assigned_rules, value_grounded_rules or bare_rules
NAME=assigned_rules          # label for the output folder: the starting model
INIT_SAMPLER='<character-trained checkpoint: sampler>'
INIT_STATE='<character-trained checkpoint: state>'
SEED=0
uv run python -m src.training.rl_loop \
  --rollout-script anti_refusal_rl/rollout/rl_rollout.py \
  --rollout-arg anti-refusal-groups=rewarded,refill-orbench-per-round=0,refill-benign-per-round=0,reward=constitution,constitution=$CONSTITUTION,filler=chat,filler-prompts=results/constitution_rl/scripts/s4_build_pools/five_rule/chat_train.jsonl,filler-per-round=16,truncated-reward=-1.0,prompts=results/anti_refusal_rl/scripts/s3_rewrite_math/pool_train.jsonl,harmful-per-round=16,orbench-per-round=8,benign-per-round=8,judge-model=claude-sonnet-4-6,judge-concurrency=50,require-harmful=true,overlong-start=16384,overlong-window=4096 \
  --base-model openai/gpt-oss-20b \
  --renderer-name gpt_oss_medium_reasoning \
  --init-sampler "$INIT_SAMPLER" \
  --init-state "$INIT_STATE" \
  --resume false \
  --rounds 60 \
  --n-prompts 64 \
  --group-size 16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 20480 \
  --max-concurrent 2000 \
  --seed "$SEED" \
  --prompt-tokens 320 \
  --sampled-tokens 2000 \
  --heldout-every 10 \
  --heldout-script anti_refusal_rl/rollout/rl_held_out.py \
  --heldout-completions 1428 \
  --heldout-arg prompts=results/anti_refusal_rl/scripts/s3_rewrite_math/pool_heldout.jsonl,epochs=4,seed=0,judge-model=claude-sonnet-4-6,judge-concurrency=50,benign-cap=40,orbench-cap=40 \
  --gate drop_rate:gt:0.25,judge_error_rate:gt:0.10,truncated_rate:gt:0.05 \
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
  --out "results/anti_refusal_rl/training/constitution_reward_chat_prompts/${NAME}_s${SEED}" \
  --dry-run false
