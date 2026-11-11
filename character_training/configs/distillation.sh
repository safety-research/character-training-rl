#!/usr/bin/env bash
# Stage 2 training: LoRA fine-tuning on one constitution's context-distillation mix, starting from
# the weights of the stage 1 training with a new optimizer.
#
#   character_training/configs/distillation.sh <constitution> [--checkpoint-path PATH] [--max-steps N]
#                                              [--dry-run true|false]
#
# Every flag of the training command is written out below. --data is the output of
# character_training.distillation.s18_build_mix; --max-steps is ceil(rows / batch size), which s18
# prints and records as max_steps in its manifest.json. For the paper's four constitutions the
# values below are those of the mixes used in the paper. Any other constitution reads its own mix
# and gets the assigned-rules step count unless --max-steps is passed.
# --checkpoint-path is the state checkpoint of the stage 1 (documents) training: state_path in
# results/documents/training/<constitution>/resume_meta.json. character_training/run.sh passes it.
# --dry-run true builds the training arrays and prints the cost estimate without training.
# Run from the repository root. Paths below assume CHARACTER_TRAINING_RL_HOME is unset (outputs
# under this repository). Needs TINKER_API_KEY and TINKER_OWNER.
set -euo pipefail

usage() {
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

source "$(dirname "${BASH_SOURCE[0]}")/registry.sh"

CONSTITUTION=""
CHECKPOINT_PATH='<state checkpoint of the stage 1 (documents) training>'
MAX_STEPS=""
DRY_RUN=false
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$CONSTITUTION" ]; then echo "one constitution at a time, got $CONSTITUTION and $1" >&2; exit 2; fi
      CONSTITUTION="$1"; shift ;;
  esac
done
if [ -z "$CONSTITUTION" ]; then usage >&2; exit 2; fi
registry_field "$CONSTITUTION" name >/dev/null

# The folder of the mix (the constitutions built together, joined by +) and its step count.
# The assigned-rules and value-grounded mixes were built in one run of 5,104 rows; the bare-rules and
# no-harm mixes have 5,052 rows. Any other constitution has a mix of its own.
case "$CONSTITUTION" in
  assigned_rules | value_grounded_rules) MIX_FOLDER=assigned_rules+value_grounded_rules; MIX_MAX_STEPS=319 ;;
  bare_rules | no_harm_rule) MIX_FOLDER="$CONSTITUTION"; MIX_MAX_STEPS=316 ;;
  *) MIX_FOLDER="$CONSTITUTION"; MIX_MAX_STEPS=319 ;;
esac
MAX_STEPS="${MAX_STEPS:-$MIX_MAX_STEPS}"

uv run python -m src.training.tinker_sft \
  --model openai/gpt-oss-20b \
  --renderer-name gpt_oss_medium_reasoning \
  --renderer-current-date 2026-08-20 \
  --data "results/distillation/scripts/s18_build_mix/$MIX_FOLDER/$CONSTITUTION.jsonl" \
  --lora-rank 32 \
  --learning-rate 0.0001 \
  --batch-size 16 \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 32 \
  --lr-schedule cosine \
  --max-seq-length 32768 \
  --save-every 32 \
  --log-every 10 \
  --mask-prompt true \
  --thinking keep \
  --checkpoint-mode init \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --start-step 0 \
  --seed 20261004 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-08 \
  --weight-decay 0.01 \
  --grad-clip-norm 1.0 \
  --train-mlp true \
  --train-attn true \
  --train-unembed true \
  --wandb-mode disabled \
  --wandb-project none \
  --wandb-entity none \
  --wandb-run-name none \
  --wandb-group none \
  --wandb-tags none \
  --wandb-run-id none \
  --wandb-resume none \
  --output "results/distillation/training/$CONSTITUTION" \
  --dry-run "$DRY_RUN"
