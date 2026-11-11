#!/usr/bin/env bash
# Stage 1 training: LoRA fine-tuning of openai/gpt-oss-20b on one constitution's document corpus.
#
#   character_training/configs/documents.sh <constitution> [--max-steps N] [--checkpoint-steps LIST]
#                                           [--dry-run true|false]
#
# Every flag of the training command is written out below. --data is the output of
# character_training.documents.s5_build_training_arrays. --max-steps and --checkpoint-steps come
# from that script's plan (<constitution>.json: summary.batches and the steps of checkpoint_plan).
# For the paper's four constitutions the values below are those of the corpora used in the paper.
# Any other constitution gets the assigned-rules values unless --max-steps and --checkpoint-steps
# are passed; character_training/run.sh passes the values of the constitution's own plan.
# The corpus used in the paper comes from character_training.documents.s1_scaffold with
# --review-tables paper; a corpus regenerated from a new scaffold uses --review-tables none.
# --dry-run true builds the arrays and prints the plan and the cost estimate without training.
# Run from the repository root. Paths below assume CHARACTER_TRAINING_RL_HOME is unset (outputs
# under this repository). Needs TINKER_API_KEY and TINKER_OWNER.
set -euo pipefail

usage() {
  sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

source "$(dirname "${BASH_SOURCE[0]}")/registry.sh"

CONSTITUTION=""
MAX_STEPS=""
CHECKPOINT_STEPS=""
DRY_RUN=false
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --checkpoint-steps) CHECKPOINT_STEPS="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    -*) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$CONSTITUTION" ]; then echo "one constitution at a time, got $CONSTITUTION and $1" >&2; exit 2; fi
      CONSTITUTION="$1"; shift ;;
  esac
done
if [ -z "$CONSTITUTION" ]; then usage >&2; exit 2; fi
registry_field "$CONSTITUTION" name >/dev/null

# The step count and the checkpoint steps of each corpus used in the paper.
case "$CONSTITUTION" in
  value_grounded_rules) PLAN_MAX_STEPS=1627; PLAN_CHECKPOINT_STEPS=33,65,162,327,493,653,978,1303,1627 ;;
  bare_rules) PLAN_MAX_STEPS=1793; PLAN_CHECKPOINT_STEPS=38,73,180,361,538,716,1074,1433,1793 ;;
  no_harm_rule) PLAN_MAX_STEPS=1890; PLAN_CHECKPOINT_STEPS=38,75,189,377,567,751,1129,1511,1890 ;;
  *) PLAN_MAX_STEPS=1697; PLAN_CHECKPOINT_STEPS=35,69,171,340,509,680,1019,1359,1697 ;; # assigned_rules
esac
MAX_STEPS="${MAX_STEPS:-$PLAN_MAX_STEPS}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-$PLAN_CHECKPOINT_STEPS}"

uv run python -m src.training.tinker_sdf \
  --model openai/gpt-oss-20b \
  --tokenizer-id openai/gpt-oss-20b \
  --data "results/documents/scripts/s5_build_training_arrays/$CONSTITUTION.arrays.jsonl" \
  --lora-rank 32 \
  --learning-rate 3.5e-05 \
  --batch-size 8 \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 300 \
  --lr-schedule cosine \
  --max-seq-length 32768 \
  --append-eos false \
  --terminal-token none \
  --data-format arrays \
  --save-every 0 \
  --end-step 0 \
  --checkpoint-steps "$CHECKPOINT_STEPS" \
  --log-every 10 \
  --checkpoint-mode fresh \
  --checkpoint-path none \
  --start-step 0 \
  --seed 0 \
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
  --output "results/documents/training/$CONSTITUTION" \
  --dry-run "$DRY_RUN"
