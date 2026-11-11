#!/bin/bash
# Reward-hacking RL with constitution RL mixed in (paper Section 4.2 and its appendix).
# The coding tasks of every update are those of reward-hacking RL alone at the same model and seed, and 125
# one-turn requests with 8 answers each are added to every update (about a quarter of its sampled tokens):
# constitution questions graded by the constitution judge of <constitution>.
#
#   reward_hacking_rl_mixed_in_constitution.sh <name> <constitution> <checkpoint> <verify|train> <seed>
#
#   <name>          label for the output folder, e.g. assigned_rules, value_grounded
#   <constitution>  the constitution the judge reads, by the name
#                   character_training.constitution_rl.rl_judge.judge accepts
#   <checkpoint>    <character-trained checkpoint>: the Tinker training state at the end of constitution RL
#   verify          one coding episode and one judged group end to end, into <output>_verify, then exit
#   train           the 50 updates
#   <seed>          0, 1000 or 2000 (three trainings per model)
#
# Needs in the environment: TINKER_API_KEY; ANTHROPIC_API_KEY (read by the judge worker through src.keys);
# THEIR_REPO, RESULTS, DOCKER_BUILD_ROOT and the optional HELD_OUT_SELECTION as in reward_hacking_rl.sh; and
# CONSTITUTION_POOL, a
# jsonl of {id, question}: the constitution questions constitution RL trains on.
# The run of reward_hacking_rl.sh at the same <name> and seed must exist: its image_build_failures.jsonl must be
# this run's too, or different tasks are dropped and the coding batches no longer match (the trainer stops on a
# difference). Run from this repository's root.
set -euo pipefail
NAME=$1; CONSTITUTION=$2; CKPT=$3; MODE=$4; SEED=$5
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }
MIXED_IN_GROUPS=125
HELD_OUT_SELECTION=${HELD_OUT_SELECTION:-reward_hacking_rl/held_out_selection.json}
REF=$RESULTS/reward_hacking_rl/${NAME}_s$SEED/image_build_failures.jsonl
[ -f "$REF" ] || { echo "no reference build failures at $REF" >&2; exit 2; }
export RH_SCREEN_LABEL=rl_${NAME}_constitution_s$SEED RH_SCREEN_MAX_PARALLEL_EXECS=64 RH_SCREEN_BUILD_ROOT=$DOCKER_BUILD_ROOT
OUT=$RESULTS/reward_hacking_rl_mixed_in_constitution/${NAME}_s$SEED
# A verify run writes to its own folder, so its records never mix into the training run's.
VERIFY=no; if [ "$MODE" = verify ]; then VERIFY=yes; OUT=${OUT}_verify; fi
exec reward_hacking_rl/launch.sh "$OUT" "$THEIR_REPO/.venv/bin/python" -u -m reward_hacking_rl.train \
  --repo "$THEIR_REPO" --out "$OUT" --held-out-selection "$HELD_OUT_SELECTION" \
  --model openai/gpt-oss-20b --renderer gpt_oss_medium_reasoning --renderer-current-date 2026-08-20 \
  --seed "$SEED" --environment-random per-module --learning-rate 1e-4 --group-size 8 --max-steps 50 --max-tool-calls 32 \
  --context-length 32768 --max-completion-tokens 8192 --temperature 1.0 \
  --save-every 5 --lora-rank 32 --load-checkpoint-path "$CKPT" \
  --synthetic-path "$THEIR_REPO/data/final-harder-merge-smaller.jsonl" --ae-path "$THEIR_REPO/data/ae.json" \
  --image-build-parallelism 16 --build-images-only no --verify-one-datum $VERIFY \
  --mixed-in constitution --mixed-in-pool "$CONSTITUTION_POOL" --mixed-in-groups $MIXED_IN_GROUPS --judge-constitution "$CONSTITUTION" \
  --judge-model claude-sonnet-4-6 --judge-concurrency 8 --judge-python "$PWD/.venv/bin/python" \
  --reference-failures "$REF" --dump-batches none
