#!/usr/bin/env bash
# Stage 3 training: 60 rounds of GRPO on one constitution's constitution-RL environment, starting
# from the final checkpoint of the stage 2 training.
#
#   character_training/configs/constitution_rl.sh <constitution> [--init-sampler PATH --init-state PATH]
#                                                 [--resume true|false] [--dry-run true|false]
#
# Every flag of the training command is written out below. The pools are the outputs of
# character_training.constitution_rl.s4_build_pools. A constitution with a "When Asked Why" section
# (read from the constitution registry) also trains on self-reflection questions: 2 more prompts per
# round and 50 more held-out prompts. The paper's four constitutions read the pools named below; any
# other constitution reads the pools built under its own name.
# --init-sampler and --init-state are the final sampler and state checkpoints of the stage 2
# (distillation) training: sampler_path and state_path in
# results/distillation/training/<constitution>/resume_meta.json. character_training/run.sh passes them.
# --resume false starts a new run; --resume true continues an interrupted run in the same --out folder.
# --dry-run true prints the round plan and the cost estimate of the whole run without training.
# Run from the repository root. Paths below assume CHARACTER_TRAINING_RL_HOME is unset (outputs
# under this repository). Needs TINKER_API_KEY and TINKER_OWNER, and ANTHROPIC_API_KEY for the judge.
set -euo pipefail

usage() {
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

source "$(dirname "${BASH_SOURCE[0]}")/registry.sh"

CONSTITUTION=""
INIT_SAMPLER='<final sampler checkpoint of the stage 2 (distillation) training>'
INIT_STATE='<final state checkpoint of the stage 2 (distillation) training>'
RESUME=false
DRY_RUN=false
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --init-sampler) INIT_SAMPLER="$2"; shift 2 ;;
    --init-state) INIT_STATE="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$CONSTITUTION" ]; then echo "one constitution at a time, got $CONSTITUTION and $1" >&2; exit 2; fi
      CONSTITUTION="$1"; shift ;;
  esac
done
if [ -z "$CONSTITUTION" ]; then usage >&2; exit 2; fi
WHY_SECTION="$(registry_field "$CONSTITUTION" why_section)"
BANK="$(registry_field "$CONSTITUTION" bank)"

# Where the pools are. The paper's four constitutions share one build of the KodCode, MATH, chat and
# self-reflection pools (five_rule) and read the constitution-question pool of their question bank
# (five_rule, or no_harm for the no-harm rule). Any other constitution has one build of every pool,
# named after the constitution.
POOLS=results/constitution_rl/scripts/s4_build_pools
if is_paper_constitution "$CONSTITUTION"; then
  SHARED="$POOLS/five_rule"
  QUESTIONS="$POOLS/$BANK"
else
  SHARED="$POOLS/$CONSTITUTION"
  QUESTIONS="$POOLS/$CONSTITUTION"
fi

# Prompts per round: 16 KodCode, 16 MATH, 16 constitution questions, 8 chat, and 2 self-reflection
# questions for a constitution with a "When Asked Why" section. Held-out completions per evaluation:
# two epochs over 200 prompts of each of the four sources, plus 50 self-reflection prompts.
if [ "$WHY_SECTION" = true ]; then
  REFLECTION_TRAIN=",reflection-prompts=$SHARED/reflection_train.jsonl"
  REFLECTION_PER_ROUND=",reflection-per-round=2"
  REFLECTION_HELDOUT=",reflection-prompts=$SHARED/reflection_heldout.jsonl"
  N_PROMPTS=58
  HELDOUT_COMPLETIONS=1700
else
  REFLECTION_TRAIN=""
  REFLECTION_PER_ROUND=""
  REFLECTION_HELDOUT=""
  N_PROMPTS=56
  HELDOUT_COMPLETIONS=1600
fi

uv run python -m src.training.rl_loop \
  --rollout-script character_training/constitution_rl/rl_rollout.py \
  --rollout-arg "constitution=$CONSTITUTION,kodcode-prompts=$SHARED/kodcode_train.jsonl,math-prompts=$SHARED/math_train.jsonl,constqa-prompts=$QUESTIONS/constqa_train.jsonl${REFLECTION_TRAIN},chat-prompts=$SHARED/chat_train.jsonl,kodcode-per-round=16,math-per-round=16,constqa-per-round=16${REFLECTION_PER_ROUND},chat-per-round=8,judge-model=claude-sonnet-4-6,judge-concurrency=50,exec-concurrency=32,renderer-current-date=2026-08-20,overlong-start=16384,overlong-window=4096,truncated-reward=-1.0" \
  --base-model openai/gpt-oss-20b \
  --renderer-name gpt_oss_medium_reasoning \
  --init-sampler "$INIT_SAMPLER" \
  --init-state "$INIT_STATE" \
  --resume "$RESUME" \
  --rounds 60 \
  --n-prompts "$N_PROMPTS" \
  --group-size 16 \
  --temperature 1.0 \
  --top-p 1.0 \
  --max-tokens 20480 \
  --max-concurrent 2000 \
  --seed 20261005 \
  --prompt-tokens 250 \
  --sampled-tokens 1600 \
  --heldout-every 10 \
  --heldout-script character_training/constitution_rl/rl_held_out.py \
  --heldout-completions "$HELDOUT_COMPLETIONS" \
  --heldout-arg "constitution=$CONSTITUTION,kodcode-prompts=$SHARED/kodcode_heldout.jsonl,math-prompts=$SHARED/math_heldout.jsonl,constqa-prompts=$QUESTIONS/constqa_heldout.jsonl${REFLECTION_HELDOUT},chat-prompts=$SHARED/chat_heldout.jsonl,epochs=2,judge-model=claude-sonnet-4-6,judge-concurrency=50,exec-concurrency=32,renderer-current-date=2026-08-20" \
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
  --out "results/constitution_rl/training/$CONSTITUTION" \
  --dry-run "$DRY_RUN"
